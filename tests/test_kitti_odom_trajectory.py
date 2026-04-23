from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

import mapanything.utils.kitti_odom_trajectory as kot


def _pose(tx: float, ty: float = 0.0, tz: float = 0.0) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = np.array([tx, ty, tz], dtype=np.float64)
    return pose


def _write_batch(
    path: Path,
    *,
    model: str = "mapanything",
    sequence: str = "09",
    camera: str = "image_2",
    frame_ids: list[str],
    window_start: int,
    window_end: int,
    camera_poses: np.ndarray,
    omit_keys: tuple[str, ...] = (),
) -> None:
    payload = {
        "model": np.array(model),
        "sequence": np.array(sequence),
        "camera": np.array(camera),
        "frame_ids": np.asarray(frame_ids),
        "window_start": np.int64(window_start),
        "window_end": np.int64(window_end),
        "camera_poses": np.asarray(camera_poses, dtype=np.float64),
    }
    for key in omit_keys:
        payload.pop(key, None)
    np.savez(path, **payload)


class KittiOdomTrajectoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temp_dir.name)
        self.sequence_results_dir = self.tmp_path / "mapanything" / "window_3_res_518" / "image_2" / "09"
        self.sequence_results_dir.mkdir(parents=True, exist_ok=True)

        (self.sequence_results_dir / "run_manifest.json").write_text(
            '{"config": {"window_size": 3}, "model_spec": {"resolution": 518}}',
            encoding="utf-8",
        )

        batch0_poses = np.stack([_pose(0.0), _pose(1.0), _pose(2.0)], axis=0)
        batch1_poses = np.stack([_pose(10.0), _pose(11.0), _pose(12.0)], axis=0)
        _write_batch(
            self.sequence_results_dir / "0.npz",
            frame_ids=["000000", "000001", "000002"],
            window_start=0,
            window_end=3,
            camera_poses=batch0_poses,
        )
        _write_batch(
            self.sequence_results_dir / "1.npz",
            frame_ids=["000002", "000003", "000004"],
            window_start=2,
            window_end=5,
            camera_poses=batch1_poses,
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_resolve_sequence_results_dir_accepts_directory_and_batch_path(self) -> None:
        resolved_from_dir = kot.resolve_sequence_results_dir(self.sequence_results_dir)
        resolved_from_batch = kot.resolve_sequence_results_dir(
            self.sequence_results_dir / "0.npz"
        )

        self.assertEqual(resolved_from_dir, self.sequence_results_dir)
        self.assertEqual(resolved_from_batch, self.sequence_results_dir)

        loaded = kot.load_sequence_results(self.sequence_results_dir / "0.npz")
        self.assertEqual(loaded.sequence_results_dir, self.sequence_results_dir)
        self.assertEqual(len(loaded.batch_results), 2)
        self.assertEqual(loaded.manifest["config"]["window_size"], 3)

    def test_load_batch_result_validates_required_keys_and_shape_consistency(self) -> None:
        missing_key_path = self.sequence_results_dir / "bad_missing.npz"
        _write_batch(
            missing_key_path,
            frame_ids=["000000"],
            window_start=0,
            window_end=1,
            camera_poses=np.stack([_pose(0.0)], axis=0),
            omit_keys=("camera_poses",),
        )
        with self.assertRaises(KeyError):
            kot.load_batch_result(missing_key_path)

        bad_shape_path = self.sequence_results_dir / "2.npz"
        _write_batch(
            bad_shape_path,
            frame_ids=["000005", "000006"],
            window_start=5,
            window_end=7,
            camera_poses=np.zeros((2, 3, 4), dtype=np.float64),
        )
        with self.assertRaises(ValueError):
            kot.load_batch_result(bad_shape_path)

    def test_connect_batch_camera_poses_reconstructs_overlapping_sequence(self) -> None:
        loaded = kot.load_sequence_results(self.sequence_results_dir)
        connected_poses, filled_mask, connected_frame_ids = kot.connect_batch_camera_poses(
            loaded.batch_results
        )
        connected_centers = kot.camera_centers_from_poses(connected_poses)

        self.assertTrue(np.all(filled_mask))
        self.assertEqual(connected_frame_ids.tolist(), ["000000", "000001", "000002", "000003", "000004"])
        np.testing.assert_allclose(
            connected_centers,
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                    [3.0, 0.0, 0.0],
                    [4.0, 0.0, 0.0],
                ],
                dtype=np.float64,
            ),
        )

    def test_connect_batch_camera_poses_falls_back_to_previous_frame_for_contiguous_windows(self) -> None:
        contiguous_dir = self.tmp_path / "mapanything" / "window_2_res_518" / "image_2" / "10"
        contiguous_dir.mkdir(parents=True, exist_ok=True)
        _write_batch(
            contiguous_dir / "0.npz",
            sequence="10",
            frame_ids=["000000", "000001"],
            window_start=0,
            window_end=2,
            camera_poses=np.stack([_pose(0.0), _pose(1.0)], axis=0),
        )
        _write_batch(
            contiguous_dir / "1.npz",
            sequence="10",
            frame_ids=["000002", "000003"],
            window_start=2,
            window_end=4,
            camera_poses=np.stack([_pose(10.0), _pose(11.0)], axis=0),
        )

        loaded = kot.load_sequence_results(contiguous_dir)
        connected_poses, filled_mask, connected_frame_ids = kot.connect_batch_camera_poses(
            loaded.batch_results
        )
        connected_centers = kot.camera_centers_from_poses(connected_poses)

        self.assertTrue(np.all(filled_mask))
        self.assertEqual(connected_frame_ids.tolist(), ["000000", "000001", "000002", "000003"])
        np.testing.assert_allclose(
            connected_centers,
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                ],
                dtype=np.float64,
            ),
        )

    def test_camera_centers_from_poses_extracts_translations(self) -> None:
        poses = np.stack([_pose(1.0, 2.0, 3.0), _pose(-1.0, 0.5, 4.0)], axis=0)
        centers = kot.camera_centers_from_poses(poses)
        np.testing.assert_allclose(
            centers,
            np.array([[1.0, 2.0, 3.0], [-1.0, 0.5, 4.0]], dtype=np.float64),
        )

    def test_umeyama_alignment_recovers_similarity_transform(self) -> None:
        predicted = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 1.0, 0.0],
                [3.0, 1.0, 1.0],
            ],
            dtype=np.float64,
        )
        rotation = np.array(
            [
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        scale = 2.5
        translation = np.array([5.0, -3.0, 4.0], dtype=np.float64)
        gt = kot.apply_similarity_transform(predicted, scale, rotation, translation)

        aligned, estimated_scale = kot.align_predicted_trajectory_to_gt(
            predicted_trajectory=predicted,
            gt_trajectory=gt,
            filled_mask=np.ones(len(predicted), dtype=bool),
        )

        self.assertAlmostEqual(estimated_scale, scale)
        np.testing.assert_allclose(aligned, gt, atol=1e-6)

    def test_start_point_alignment_only_translates_trajectory(self) -> None:
        predicted = np.array(
            [
                [1.0, 2.0, 3.0],
                [2.0, 2.5, 3.0],
                [4.0, 3.0, 5.0],
            ],
            dtype=np.float64,
        )
        gt = np.array(
            [
                [10.0, 20.0, 30.0],
                [99.0, 99.0, 99.0],
                [98.0, 98.0, 98.0],
            ],
            dtype=np.float64,
        )
        filled_mask = np.array([True, False, True], dtype=bool)

        aligned = kot.align_predicted_trajectory_start_point(
            predicted_trajectory=predicted,
            gt_trajectory=gt,
            filled_mask=filled_mask,
        )

        expected_translation = gt[0] - predicted[0]
        np.testing.assert_allclose(aligned, predicted + expected_translation)
        np.testing.assert_allclose(aligned[0], gt[0])

    def test_gt_missing_fallback_still_leaves_plottable_estimated_trajectory(self) -> None:
        loaded = kot.load_sequence_results(self.sequence_results_dir)
        connected_poses, filled_mask, _ = kot.connect_batch_camera_poses(loaded.batch_results)
        estimated_trajectory = kot.camera_centers_from_poses(connected_poses)

        self.assertIsNone(kot.load_optional_kitti_gt_poses(None, "09"))
        self.assertIsNone(kot.load_optional_kitti_gt_poses(self.tmp_path / "missing_dataset", "09"))
        self.assertEqual(estimated_trajectory.shape, (5, 3))
        self.assertTrue(np.isfinite(estimated_trajectory[filled_mask]).all())


if __name__ == "__main__":
    unittest.main()
