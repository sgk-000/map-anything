from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
import importlib.util
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

import mapanything.utils.kitti_odom_inference as ko
from mapanything.utils.geometry import get_rays_in_camera_frame


def _load_cli_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "kitti_odom_inference.py"
    spec = importlib.util.spec_from_file_location("kitti_odom_inference_cli", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_rgb_image(path: Path, width: int = 200, height: int = 100, value: int = 32) -> None:
    image = np.full((height, width, 3), value, dtype=np.uint8)
    Image.fromarray(image).save(path)


def _write_kitti_sequence(root: Path, sequence_name: str, num_frames: int = 3) -> Path:
    sequence_dir = root / "sequences" / sequence_name
    (sequence_dir / "image_2").mkdir(parents=True, exist_ok=True)
    (sequence_dir / "image_3").mkdir(parents=True, exist_ok=True)

    calib_text = "\n".join(
        [
            "P0: 1 0 0 0 0 1 0 0 0 0 1 0",
            "P1: 1 0 0 0 0 1 0 0 0 0 1 0",
            "P2: 700 0 600 10 0 700 180 0 0 0 1 0",
            "P3: 710 0 610 -20 0 710 190 0 0 0 1 0",
        ]
    )
    (sequence_dir / "calib.txt").write_text(calib_text, encoding="utf-8")

    for frame_idx in range(num_frames):
        frame_name = f"{frame_idx:06d}.png"
        _write_rgb_image(sequence_dir / "image_2" / frame_name, value=32 + frame_idx)
        _write_rgb_image(sequence_dir / "image_3" / frame_name, value=64 + frame_idx)

    return sequence_dir


class StubMapAnythingModel:
    def __init__(self) -> None:
        self.infer_calls: list[dict[str, object]] = []

    def infer(self, views, **kwargs):
        self.infer_calls.append(kwargs)
        predictions = []
        for view in views:
            _, _, height, width = view["img"].shape
            predictions.append(
                {
                    "depth_z": torch.ones(1, height, width, 1),
                    "depth_along_ray": torch.full((1, height, width, 1), 2.0),
                    "conf": torch.full((1, height, width), 0.8),
                    "intrinsics": view["intrinsics"].clone(),
                    "camera_poses": torch.eye(4).unsqueeze(0),
                }
            )
        return predictions


class StubWrapperModel:
    def __call__(self, views):
        outputs = []
        for view in views:
            _, _, height, width = view["img"].shape
            intrinsics = view["intrinsics"]
            _, ray_directions = get_rays_in_camera_frame(
                intrinsics=intrinsics,
                height=height,
                width=width,
                normalize_to_unit_sphere=True,
            )
            depth_z = torch.ones(1, height, width, 1)
            pts3d_cam = ray_directions / ray_directions[..., 2:3] * depth_z
            depth_along_ray = torch.norm(pts3d_cam, dim=-1, keepdim=True)
            outputs.append(
                {
                    "pts3d_cam": pts3d_cam,
                    "ray_directions": ray_directions,
                    "depth_along_ray": depth_along_ray,
                    "cam_trans": torch.zeros(1, 3),
                    "cam_quats": torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
                    "conf": torch.full((1, height, width), 0.6),
                }
            )
        return outputs


class KittiOdomInferenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temp_dir.name)
        self.dataset_root = self.tmp_path / "dataset"
        _write_kitti_sequence(self.dataset_root, "09", num_frames=3)
        _write_kitti_sequence(self.dataset_root, "10", num_frames=2)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_discover_sequence_dirs(self) -> None:
        discovered = ko.discover_sequence_dirs(self.dataset_root, None)
        self.assertEqual([path.name for path in discovered], ["09", "10"])

        subset = ko.discover_sequence_dirs(self.dataset_root, ["10"])
        self.assertEqual([path.name for path in subset], ["10"])

    def test_parse_kitti_odom_intrinsics_for_both_cameras(self) -> None:
        sequence_dir = self.dataset_root / "sequences" / "09"

        intrinsics_left = ko.parse_kitti_odom_intrinsics(sequence_dir, "image_2")
        intrinsics_right = ko.parse_kitti_odom_intrinsics(sequence_dir, "image_3")

        np.testing.assert_allclose(
            intrinsics_left,
            np.array([[700.0, 0.0, 600.0], [0.0, 700.0, 180.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        )
        np.testing.assert_allclose(
            intrinsics_right,
            np.array([[710.0, 0.0, 610.0], [0.0, 710.0, 190.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        )

    def test_build_batch_ranges_handles_short_and_overlapping_final_windows(self) -> None:
        self.assertEqual(ko.build_batch_ranges(num_images=3, window_size=5), [(0, 3)])
        self.assertEqual(
            ko.build_batch_ranges(num_images=250, window_size=100),
            [(0, 100), (100, 200), (150, 250)],
        )

    def test_output_dir_for_sequence_includes_model_and_camera(self) -> None:
        output_dir = ko.output_dir_for_sequence(
            output_root=self.tmp_path,
            model_name="da3_nested",
            window_size=100,
            resolution_label="res_504",
            camera="image_3",
            sequence_name="09",
        )

        self.assertEqual(
            output_dir,
            self.tmp_path / "da3_nested" / "window_100_res_504" / "image_3" / "09",
        )

    def test_preprocess_batch_views_resizes_images_and_intrinsics(self) -> None:
        image_path = self.dataset_root / "sequences" / "09" / "image_2" / "000000.png"
        intrinsics = ko.parse_kitti_odom_intrinsics(
            self.dataset_root / "sequences" / "09",
            "image_2",
        )
        raw_views = ko.build_raw_views_for_batch([image_path], intrinsics)

        default_config = ko.KittiOdomInferenceConfig(model="mapanything", device="cpu")
        long_side_config = ko.KittiOdomInferenceConfig(
            model="mapanything",
            device="cpu",
            long_side_resolution=560,
        )

        map_views = ko.preprocess_batch_views(
            raw_views, ko.MODEL_SPECS["mapanything"], default_config
        )
        da3_views = ko.preprocess_batch_views(
            raw_views, ko.MODEL_SPECS["da3_nested"], default_config
        )
        pi3x_views = ko.preprocess_batch_views(
            raw_views, ko.MODEL_SPECS["pi3x"], default_config
        )
        long_side_views = ko.preprocess_batch_views(
            raw_views, ko.MODEL_SPECS["mapanything"], long_side_config
        )

        self.assertEqual(tuple(map_views[0]["img"].shape[-2:]), (252, 518))
        self.assertEqual(tuple(da3_views[0]["img"].shape[-2:]), (238, 504))
        self.assertEqual(tuple(long_side_views[0]["img"].shape[-2:]), (280, 560))
        self.assertEqual(map_views[0]["data_norm_type"], ["dinov2"])
        self.assertEqual(pi3x_views[0]["data_norm_type"], ["identity"])
        self.assertEqual(tuple(map_views[0]["intrinsics"].shape), (1, 3, 3))
        self.assertFalse(np.allclose(map_views[0]["intrinsics"][0].numpy(), intrinsics))

    def test_move_views_to_device_preserves_metadata_and_moves_tensors(self) -> None:
        views = [
            {
                "img": torch.ones(1, 3, 4, 5),
                "intrinsics": torch.eye(3).unsqueeze(0),
                "camera_poses": (torch.tensor([[0.0, 0.0, 0.0, 1.0]]), torch.zeros(1, 3)),
                "data_norm_type": ["identity"],
                "instance": "000000.png",
                "idx": 0,
            }
        ]

        moved_views = ko.move_views_to_device(views, torch.device("cpu"))

        self.assertEqual(moved_views[0]["img"].device.type, "cpu")
        self.assertEqual(moved_views[0]["intrinsics"].device.type, "cpu")
        self.assertEqual(moved_views[0]["camera_poses"][0].device.type, "cpu")
        self.assertEqual(moved_views[0]["camera_poses"][1].device.type, "cpu")
        self.assertEqual(moved_views[0]["data_norm_type"], ["identity"])
        self.assertEqual(moved_views[0]["instance"], "000000.png")
        self.assertEqual(moved_views[0]["idx"], 0)

    def test_model_adapters_normalize_to_shared_schema(self) -> None:
        image_paths = [
            self.dataset_root / "sequences" / "09" / "image_2" / "000000.png",
            self.dataset_root / "sequences" / "09" / "image_2" / "000001.png",
        ]
        intrinsics = ko.parse_kitti_odom_intrinsics(
            self.dataset_root / "sequences" / "09",
            "image_2",
        )
        raw_views = ko.build_raw_views_for_batch(image_paths, intrinsics)
        device = torch.device("cpu")

        map_config = ko.KittiOdomInferenceConfig(model="mapanything", device="cpu", use_amp=False)
        map_views = ko.preprocess_batch_views(
            raw_views, ko.MODEL_SPECS["mapanything"], map_config
        )
        map_model = StubMapAnythingModel()
        map_predictions = ko.run_model_inference(
            "mapanything", map_model, map_views, device, map_config
        )
        map_serialized = ko.normalize_predictions_for_saving(map_predictions)

        wrapper_config = ko.KittiOdomInferenceConfig(model="pi3x", device="cpu", use_amp=False)
        wrapper_views = ko.preprocess_batch_views(
            raw_views, ko.MODEL_SPECS["pi3x"], wrapper_config
        )
        wrapper_predictions = ko.run_model_inference(
            "pi3x", StubWrapperModel(), wrapper_views, device, wrapper_config
        )
        wrapper_serialized = ko.normalize_predictions_for_saving(wrapper_predictions)

        expected_keys = {
            "depth_z",
            "depth_along_ray",
            "conf",
            "intrinsics",
            "camera_poses",
            "valid_mask",
        }
        self.assertEqual(set(map_serialized), expected_keys)
        self.assertEqual(set(wrapper_serialized), expected_keys)
        self.assertEqual(map_serialized["depth_z"].shape[0], 2)
        self.assertEqual(wrapper_serialized["camera_poses"].shape, (2, 4, 4))
        self.assertIs(map_model.infer_calls[0]["apply_mask"], False)
        self.assertIs(map_model.infer_calls[0]["mask_edges"], False)

        save_path = self.tmp_path / "batch_output.npz"
        ko.save_batch_output(
            save_path=save_path,
            model_name="mapanything",
            sequence_name="09",
            camera="image_2",
            frame_ids=["000000", "000001"],
            window_start=0,
            window_end=2,
            predictions=map_serialized,
        )

        loaded = np.load(save_path)
        self.assertEqual(
            set(loaded.files),
            expected_keys | {
                "model",
                "sequence",
                "camera",
                "frame_ids",
                "window_start",
                "window_end",
            },
        )

    def test_cli_smoke_writes_outputs_and_manifest(self) -> None:
        stub_model = StubMapAnythingModel()
        stub_spec = replace(
            ko.MODEL_SPECS["mapanything"],
            loader=lambda config, device: stub_model,
        )
        cli = _load_cli_module()

        output_root = self.tmp_path / "outputs"
        with patch.dict(ko.MODEL_SPECS, {"mapanything": stub_spec}, clear=False):
            cli.main(
                [
                    "--model",
                    "mapanything",
                    "--dataset-root",
                    str(self.dataset_root),
                    "--sequences",
                    "09",
                    "--num-images",
                    "2",
                    "--long-side-resolution",
                    "560",
                    "--device",
                    "cpu",
                    "--no-amp",
                    "--output-root",
                    str(output_root),
                ]
            )

        sequence_output_dir = (
            output_root / "mapanything" / "window_2_long_side_560" / "image_2" / "09"
        )
        self.assertTrue((sequence_output_dir / "run_manifest.json").exists())
        self.assertTrue((sequence_output_dir / "0.npz").exists())
        self.assertTrue((sequence_output_dir / "1.npz").exists())

        first_batch = np.load(sequence_output_dir / "0.npz")
        second_batch = np.load(sequence_output_dir / "1.npz")

        self.assertEqual(first_batch["frame_ids"].tolist(), ["000000", "000001"])
        self.assertEqual(second_batch["frame_ids"].tolist(), ["000001", "000002"])
        self.assertEqual(first_batch["depth_z"].shape[0], 2)


if __name__ == "__main__":
    unittest.main()
