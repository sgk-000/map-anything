from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

import mapanything.utils.map_free_inference as mf


def _load_cli_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "map_free_inference.py"
    spec = importlib.util.spec_from_file_location("map_free_inference_cli", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_rgb_image(path: Path, width: int = 200, height: int = 100, value: int = 32) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((height, width, 3), value, dtype=np.uint8)
    Image.fromarray(image).save(path)


def _write_map_free_scene(
    root: Path,
    scene_name: str,
    seq_frames: dict[str, list[int]],
) -> Path:
    scene_dir = root / "train" / scene_name
    intrinsics_lines: list[str] = []
    poses_lines: list[str] = []

    for sequence_name, frame_ids in seq_frames.items():
        sequence_offset = 100 if sequence_name == "seq1" else 0
        for frame_id in frame_ids:
            relative_frame_path = f"{sequence_name}/frame_{frame_id:05d}.jpg"
            _write_rgb_image(
                scene_dir / relative_frame_path,
                value=32 + sequence_offset + frame_id,
            )
            fx = 500.0 + sequence_offset + frame_id
            fy = 501.0 + sequence_offset + frame_id
            intrinsics_lines.append(
                f"{relative_frame_path} {fx} {fy} 100.0 50.0 200 100"
            )
            poses_lines.append(
                f"{relative_frame_path} 1.0 0.0 0.0 0.0 {frame_id}.0 0.0 0.0"
            )

    (scene_dir / "intrinsics.txt").write_text(
        "\n".join(intrinsics_lines),
        encoding="utf-8",
    )
    (scene_dir / "poses.txt").write_text("\n".join(poses_lines), encoding="utf-8")
    (scene_dir / "poses_device.txt").write_text(
        "\n".join(poses_lines),
        encoding="utf-8",
    )
    np.savez(
        scene_dir / "overlaps.npz",
        idxs=np.zeros((0, 4), dtype=np.uint32),
        overlaps=np.zeros((0,), dtype=np.float32),
    )
    return scene_dir


def _write_image_list_csv(
    path: Path,
    dataset_root: Path,
    rows: list[tuple[str, str]],
) -> Path:
    lines = ["scene_id,seq,seq_idx,frame_idx,rel_path,image_path"]
    for scene_id, rel_path in rows:
        seq, filename = rel_path.split("/")
        seq_idx = 0 if seq == "seq0" else 1
        frame_idx = int(filename.removeprefix("frame_").removesuffix(".jpg"))
        image_path = dataset_root / "train" / scene_id / rel_path
        lines.append(
            f"{scene_id},{seq},{seq_idx},{frame_idx},{rel_path},{image_path}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


class StubMapAnythingModel:
    def __init__(self) -> None:
        self.infer_calls: list[dict[str, object]] = []

    def infer(self, views, **kwargs):
        self.infer_calls.append(
            {
                "kwargs": kwargs,
                "instances": [view["instance"] for view in views],
                "idxs": [view["idx"] for view in views],
            }
        )
        predictions = []
        for view in views:
            _, _, height, width = view["img"].shape
            camera_pose = torch.eye(4)
            camera_pose[0, 3] = float(view["idx"])
            predictions.append(
                {
                    "depth_z": torch.ones(1, height, width, 1),
                    "depth_along_ray": torch.full((1, height, width, 1), 2.0),
                    "conf": torch.full((1, height, width), 0.8),
                    "intrinsics": view["intrinsics"].clone(),
                    "camera_poses": camera_pose.unsqueeze(0),
                }
            )
        return predictions


class MapFreeInferenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temp_dir.name)
        self.dataset_root = self.tmp_path / "dataset"
        self.scene0 = _write_map_free_scene(
            self.dataset_root,
            "s00000",
            {"seq0": [0, 2], "seq1": [0, 3]},
        )
        self.scene1 = _write_map_free_scene(
            self.dataset_root,
            "s00001",
            {"seq0": [0, 2], "seq1": [0]},
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_discover_scene_dirs_filters_selected_scenes(self) -> None:
        discovered = mf.discover_scene_dirs(self.dataset_root, "train", None)
        self.assertEqual([path.name for path in discovered], ["s00000", "s00001"])

        subset = mf.discover_scene_dirs(self.dataset_root, "train", ["s00001"])
        self.assertEqual([path.name for path in subset], ["s00001"])

        with self.assertRaises(ValueError):
            mf.discover_scene_dirs(self.dataset_root, "train", ["missing"])

    def test_parse_map_free_intrinsics_validates_entries(self) -> None:
        intrinsics = mf.parse_map_free_intrinsics(self.scene0)

        np.testing.assert_allclose(
            intrinsics["seq0/frame_00002.jpg"],
            np.array(
                [[502.0, 0.0, 100.0], [0.0, 503.0, 50.0], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            ),
        )
        np.testing.assert_allclose(
            intrinsics["seq1/frame_00003.jpg"],
            np.array(
                [[603.0, 0.0, 100.0], [0.0, 604.0, 50.0], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            ),
        )

        bad_scene = self.dataset_root / "train" / "bad_scene"
        bad_scene.mkdir(parents=True)
        (bad_scene / "intrinsics.txt").write_text(
            "seq0/frame_00000.jpg 1 2 3\n",
            encoding="utf-8",
        )
        with self.assertRaises(ValueError):
            mf.parse_map_free_intrinsics(bad_scene)

        with self.assertRaises(KeyError):
            mf.validate_intrinsics_for_images(
                self.scene0,
                [self.scene0 / "seq0" / "frame_00000.jpg"],
                {},
            )

    def test_load_scene_image_paths_and_raw_views_preserve_relative_ids(self) -> None:
        image_paths = mf.load_scene_image_paths(self.scene0)
        relative_paths = [mf.scene_relative_path(self.scene0, path) for path in image_paths]

        self.assertEqual(
            relative_paths,
            [
                "seq0/frame_00000.jpg",
                "seq0/frame_00002.jpg",
                "seq1/frame_00000.jpg",
                "seq1/frame_00003.jpg",
            ],
        )

        intrinsics = mf.parse_map_free_intrinsics(self.scene0)
        raw_views = mf.build_raw_views_for_batch(
            image_paths,
            scene_dir=self.scene0,
            intrinsics_by_frame=intrinsics,
            index_offset=10,
        )

        self.assertEqual([view["instance"] for view in raw_views], relative_paths)
        self.assertEqual([view["idx"] for view in raw_views], [10, 11, 12, 13])
        np.testing.assert_allclose(raw_views[1]["intrinsics"], intrinsics["seq0/frame_00002.jpg"])

    def test_load_image_list_csv_groups_scenes_and_preserves_order(self) -> None:
        csv_path = _write_image_list_csv(
            self.tmp_path / "image_list.csv",
            self.dataset_root,
            [
                ("s00000", "seq1/frame_00003.jpg"),
                ("s00000", "seq0/frame_00000.jpg"),
                ("s00001", "seq1/frame_00000.jpg"),
            ],
        )

        image_paths_by_scene = mf.load_image_list_csv(
            self.dataset_root,
            "train",
            csv_path,
        )

        self.assertEqual(sorted(image_paths_by_scene), ["s00000", "s00001"])
        self.assertEqual(
            [
                mf.scene_relative_path(self.scene0, path)
                for path in image_paths_by_scene["s00000"]
            ],
            ["seq1/frame_00003.jpg", "seq0/frame_00000.jpg"],
        )
        selected = mf.select_image_list_for_scene_dirs(
            [self.scene1],
            image_paths_by_scene,
            csv_path,
        )
        self.assertEqual(
            [mf.scene_relative_path(self.scene1, path) for path in selected["s00001"]],
            ["seq1/frame_00000.jpg"],
        )

        with self.assertRaises(ValueError):
            mf.select_image_list_for_scene_dirs([self.scene0, self.scene1], {"s00000": []}, csv_path)

    def test_load_image_list_csv_rejects_invalid_rows(self) -> None:
        missing_column_csv = self.tmp_path / "missing_column.csv"
        missing_column_csv.write_text(
            "scene_id,seq,seq_idx,frame_idx,rel_path\n"
            "s00000,seq0,0,0,seq0/frame_00000.jpg\n",
            encoding="utf-8",
        )
        with self.assertRaises(ValueError):
            mf.load_image_list_csv(self.dataset_root, "train", missing_column_csv)

        bad_rel_path_csv = self.tmp_path / "bad_rel_path.csv"
        bad_rel_path_csv.write_text(
            "scene_id,seq,seq_idx,frame_idx,rel_path,image_path\n"
            f"s00000,seq0,0,0,seq0/frame_00001.jpg,{self.scene0 / 'seq0' / 'frame_00000.jpg'}\n",
            encoding="utf-8",
        )
        with self.assertRaises(ValueError):
            mf.load_image_list_csv(self.dataset_root, "train", bad_rel_path_csv)

        missing_image_csv = _write_image_list_csv(
            self.tmp_path / "missing_image.csv",
            self.dataset_root,
            [("s00000", "seq0/frame_00042.jpg")],
        )
        with self.assertRaises(FileNotFoundError):
            mf.load_image_list_csv(self.dataset_root, "train", missing_image_csv)

        _write_rgb_image(self.scene0 / "seq1" / "frame_00005.jpg")
        missing_intrinsics_csv = _write_image_list_csv(
            self.tmp_path / "missing_intrinsics.csv",
            self.dataset_root,
            [("s00000", "seq1/frame_00005.jpg")],
        )
        image_paths_by_scene = mf.load_image_list_csv(
            self.dataset_root,
            "train",
            missing_intrinsics_csv,
        )
        with self.assertRaises(KeyError):
            mf.validate_image_list_intrinsics_for_scene_dirs(
                [self.scene0],
                image_paths_by_scene,
            )

    def test_build_batch_ranges_supports_full_scene_and_shifted_tail(self) -> None:
        self.assertEqual(mf.build_batch_ranges(num_images=4, window_size=0), [(0, 4)])
        self.assertEqual(
            mf.build_batch_ranges(num_images=3, window_size=2),
            [(0, 2), (1, 3)],
        )
        self.assertEqual(
            mf.build_batch_output_filenames([(0, 4)], window_size=0),
            ["0.npz"],
        )

    def test_output_dir_for_scene_uses_split_and_window_label(self) -> None:
        output_dir = mf.output_dir_for_scene(
            output_root=self.tmp_path,
            model_name="mapanything",
            window_size=0,
            resolution_label="res_518",
            split="train",
            scene_name="s00000",
        )

        self.assertEqual(
            output_dir,
            self.tmp_path / "mapanything" / "window_all_res_518" / "train" / "s00000",
        )

    def test_model_adapter_and_save_output_use_kitti_raw_schema(self) -> None:
        image_paths = mf.load_scene_image_paths(self.scene0)[:2]
        intrinsics = mf.parse_map_free_intrinsics(self.scene0)
        raw_views = mf.build_raw_views_for_batch(
            image_paths,
            scene_dir=self.scene0,
            intrinsics_by_frame=intrinsics,
        )
        config = mf.MapFreeInferenceConfig(model="mapanything", device="cpu", use_amp=False)
        processed_views = mf.preprocess_batch_views(
            raw_views,
            mf.MODEL_SPECS["mapanything"],
            config,
        )

        stub_model = StubMapAnythingModel()
        predictions = mf.run_model_inference(
            "mapanything",
            stub_model,
            processed_views,
            torch.device("cpu"),
            config,
        )
        serialized = mf.normalize_predictions_for_saving(predictions)

        self.assertEqual(set(serialized), {"camera_poses"})
        self.assertEqual(serialized["camera_poses"].shape, (2, 4, 4))
        self.assertIs(stub_model.infer_calls[0]["kwargs"]["memory_efficient_inference"], True)
        self.assertIs(stub_model.infer_calls[0]["kwargs"]["apply_mask"], False)
        self.assertIs(stub_model.infer_calls[0]["kwargs"]["mask_edges"], False)
        self.assertIs(stub_model.infer_calls[0]["kwargs"]["use_amp"], False)

        save_path = self.tmp_path / "batch_output.npz"
        mf.save_batch_output(
            save_path=save_path,
            model_name="mapanything",
            split_name="train",
            scene_name="s00000",
            frame_ids=["seq0/frame_00000.jpg", "seq0/frame_00002.jpg"],
            window_start=0,
            window_end=2,
            predictions=serialized,
        )

        loaded = np.load(save_path)
        self.assertEqual(
            set(loaded.files),
            {
                "model",
                "date",
                "drive",
                "camera",
                "frame_ids",
                "window_start",
                "window_end",
                "poses",
            },
        )
        self.assertEqual(loaded["date"].item(), "train")
        self.assertEqual(loaded["drive"].item(), "s00000")
        self.assertEqual(loaded["camera"].item(), "all")
        self.assertEqual(loaded["poses"].shape, (2, 4, 4))

    def test_cli_smoke_writes_full_scene_output_and_manifest(self) -> None:
        stub_model = StubMapAnythingModel()
        stub_spec = replace(
            mf.MODEL_SPECS["mapanything"],
            loader=lambda config, device: stub_model,
        )
        cli = _load_cli_module()

        output_root = self.tmp_path / "outputs"
        with patch.dict(mf.MODEL_SPECS, {"mapanything": stub_spec}, clear=False):
            cli.main(
                [
                    "--model",
                    "mapanything",
                    "--dataset-root",
                    str(self.dataset_root),
                    "--scenes",
                    "s00000",
                    "--device",
                    "cpu",
                    "--no-amp",
                    "--output-root",
                    str(output_root),
                ]
            )

        scene_output_dir = (
            output_root / "mapanything" / "window_all_res_518" / "train" / "s00000"
        )
        self.assertTrue((scene_output_dir / "run_manifest.json").exists())
        self.assertTrue((scene_output_dir / "0.npz").exists())
        self.assertFalse((scene_output_dir / "1.npz").exists())

        manifest = (scene_output_dir / "run_manifest.json").read_text(encoding="utf-8")
        self.assertIn('"date": "train"', manifest)
        self.assertIn('"drive": "s00000"', manifest)
        self.assertIn('"camera": "all"', manifest)

        batch = np.load(scene_output_dir / "0.npz")
        self.assertEqual(
            batch["frame_ids"].tolist(),
            [
                "seq0/frame_00000.jpg",
                "seq0/frame_00002.jpg",
                "seq1/frame_00000.jpg",
                "seq1/frame_00003.jpg",
            ],
        )
        self.assertEqual(batch["poses"].shape, (4, 4, 4))
        self.assertEqual(len(stub_model.infer_calls), 1)
        self.assertEqual(stub_model.infer_calls[0]["idxs"], [0, 1, 2, 3])

    def test_cli_image_list_csv_writes_only_listed_frames_and_manifest(self) -> None:
        stub_model = StubMapAnythingModel()
        stub_spec = replace(
            mf.MODEL_SPECS["mapanything"],
            loader=lambda config, device: stub_model,
        )
        cli = _load_cli_module()
        csv_path = _write_image_list_csv(
            self.tmp_path / "selected_frames.csv",
            self.dataset_root,
            [
                ("s00000", "seq1/frame_00003.jpg"),
                ("s00000", "seq0/frame_00000.jpg"),
            ],
        )

        output_root = self.tmp_path / "csv_outputs"
        with patch.dict(mf.MODEL_SPECS, {"mapanything": stub_spec}, clear=False):
            cli.main(
                [
                    "--model",
                    "mapanything",
                    "--dataset-root",
                    str(self.dataset_root),
                    "--scenes",
                    "s00000",
                    "--image-list-csv",
                    str(csv_path),
                    "--device",
                    "cpu",
                    "--no-amp",
                    "--output-root",
                    str(output_root),
                ]
            )

        scene_output_dir = (
            output_root / "mapanything" / "window_all_res_518" / "train" / "s00000"
        )
        manifest = json.loads(
            (scene_output_dir / "run_manifest.json").read_text(encoding="utf-8")
        )
        self.assertTrue(manifest["image_list_mode"])
        self.assertEqual(manifest["image_list_csv"], str(csv_path))
        self.assertEqual(manifest["num_listed_frames"], 2)
        self.assertEqual(manifest["first_frame_id"], "seq1/frame_00003.jpg")
        self.assertEqual(manifest["last_frame_id"], "seq0/frame_00000.jpg")

        batch = np.load(scene_output_dir / "0.npz")
        self.assertEqual(
            set(batch.files),
            {
                "model",
                "date",
                "drive",
                "camera",
                "frame_ids",
                "window_start",
                "window_end",
                "poses",
            },
        )
        self.assertEqual(
            batch["frame_ids"].tolist(),
            ["seq1/frame_00003.jpg", "seq0/frame_00000.jpg"],
        )
        self.assertEqual(batch["poses"].shape, (2, 4, 4))
        self.assertEqual(stub_model.infer_calls[0]["instances"], batch["frame_ids"].tolist())

    def test_resume_uses_image_list_frame_count(self) -> None:
        stub_model = StubMapAnythingModel()
        loader_called = False

        def loader(config, device):
            nonlocal loader_called
            loader_called = True
            return stub_model

        stub_spec = replace(mf.MODEL_SPECS["mapanything"], loader=loader)
        cli = _load_cli_module()
        csv_path = _write_image_list_csv(
            self.tmp_path / "resume_frames.csv",
            self.dataset_root,
            [
                ("s00000", "seq0/frame_00000.jpg"),
                ("s00000", "seq1/frame_00003.jpg"),
            ],
        )

        output_root = self.tmp_path / "resume_outputs"
        scene_output_dir = (
            output_root / "mapanything" / "window_2_res_518" / "train" / "s00000"
        )
        scene_output_dir.mkdir(parents=True)
        (scene_output_dir / "0.npz").touch()

        with patch.dict(mf.MODEL_SPECS, {"mapanything": stub_spec}, clear=False):
            cli.main(
                [
                    "--model",
                    "mapanything",
                    "--dataset-root",
                    str(self.dataset_root),
                    "--scenes",
                    "s00000",
                    "--image-list-csv",
                    str(csv_path),
                    "--num-images",
                    "2",
                    "--resume",
                    "--device",
                    "cpu",
                    "--no-amp",
                    "--output-root",
                    str(output_root),
                ]
            )

        self.assertFalse(loader_called)
        self.assertFalse((scene_output_dir / "1.npz").exists())
        self.assertEqual(stub_model.infer_calls, [])

    def test_cli_window_size_writes_shifted_final_window(self) -> None:
        stub_model = StubMapAnythingModel()
        stub_spec = replace(
            mf.MODEL_SPECS["mapanything"],
            loader=lambda config, device: stub_model,
        )
        cli = _load_cli_module()

        output_root = self.tmp_path / "window_outputs"
        with patch.dict(mf.MODEL_SPECS, {"mapanything": stub_spec}, clear=False):
            cli.main(
                [
                    "--model",
                    "mapanything",
                    "--dataset-root",
                    str(self.dataset_root),
                    "--scenes",
                    "s00001",
                    "--num-images",
                    "2",
                    "--device",
                    "cpu",
                    "--no-amp",
                    "--output-root",
                    str(output_root),
                ]
            )

        scene_output_dir = output_root / "mapanything" / "window_2_res_518" / "train" / "s00001"
        first_batch = np.load(scene_output_dir / "0.npz")
        second_batch = np.load(scene_output_dir / "1.npz")

        self.assertEqual(
            first_batch["frame_ids"].tolist(),
            ["seq0/frame_00000.jpg", "seq0/frame_00002.jpg"],
        )
        self.assertEqual(
            second_batch["frame_ids"].tolist(),
            ["seq0/frame_00002.jpg", "seq1/frame_00000.jpg"],
        )
        self.assertEqual(stub_model.infer_calls[0]["idxs"], [0, 1])
        self.assertEqual(stub_model.infer_calls[1]["idxs"], [1, 2])

    def test_skip_failures_continues_after_failed_scene(self) -> None:
        (self.scene0 / "intrinsics.txt").unlink()
        stub_model = StubMapAnythingModel()
        stub_spec = replace(
            mf.MODEL_SPECS["mapanything"],
            loader=lambda config, device: stub_model,
        )
        cli = _load_cli_module()

        output_root = self.tmp_path / "skip_failure_outputs"
        cli_args = [
            "--model",
            "mapanything",
            "--dataset-root",
            str(self.dataset_root),
            "--scenes",
            "s00000",
            "s00001",
            "--device",
            "cpu",
            "--no-amp",
            "--output-root",
            str(output_root),
        ]

        with patch.dict(mf.MODEL_SPECS, {"mapanything": stub_spec}, clear=False):
            with self.assertRaises(FileNotFoundError):
                cli.main(cli_args)

            cli.main([*cli_args, "--skip-failures"])

        failed_scene_output_dir = (
            output_root / "mapanything" / "window_all_res_518" / "train" / "s00000"
        )
        completed_scene_output_dir = (
            output_root / "mapanything" / "window_all_res_518" / "train" / "s00001"
        )

        self.assertFalse((failed_scene_output_dir / "0.npz").exists())
        self.assertTrue((completed_scene_output_dir / "0.npz").exists())
        completed_batch = np.load(completed_scene_output_dir / "0.npz")
        self.assertEqual(
            completed_batch["frame_ids"].tolist(),
            [
                "seq0/frame_00000.jpg",
                "seq0/frame_00002.jpg",
                "seq1/frame_00000.jpg",
            ],
        )


if __name__ == "__main__":
    unittest.main()
