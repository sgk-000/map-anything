from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


REQUIRED_BATCH_KEYS = (
    "model",
    "sequence",
    "camera",
    "frame_ids",
    "window_start",
    "window_end",
    "camera_poses",
)


@dataclass(frozen=True)
class KittiOdomBatchResult:
    path: Path
    batch_index: int
    model: str
    sequence: str
    camera: str
    frame_ids: tuple[str, ...]
    window_start: int
    window_end: int
    camera_poses: np.ndarray


@dataclass(frozen=True)
class LoadedKittiOdomResults:
    sequence_results_dir: Path
    batch_results: tuple[KittiOdomBatchResult, ...]
    manifest: dict[str, Any] | None


def _load_scalar(batch: np.lib.npyio.NpzFile, key: str) -> Any:
    value = np.asarray(batch[key])
    if value.shape != ():
        raise ValueError(f"Expected scalar value for '{key}', got shape {value.shape}.")
    return value.item()


def _batch_index_from_path(batch_path: Path) -> int:
    try:
        return int(batch_path.stem)
    except ValueError as exc:
        raise ValueError(
            f"Expected numeric batch filename stem, but got '{batch_path.stem}'."
        ) from exc


def to_homogeneous_matrices(matrices: np.ndarray) -> np.ndarray:
    matrices = np.asarray(matrices, dtype=np.float64)
    if matrices.ndim != 3:
        raise ValueError(f"Expected matrices with 3 dimensions, got shape {matrices.shape}.")
    if matrices.shape[-2:] == (4, 4):
        return matrices.copy()
    if matrices.shape[-2:] != (3, 4):
        raise ValueError(
            f"Expected matrices with shape (N, 3, 4) or (N, 4, 4), got {matrices.shape}."
        )

    homogeneous = np.repeat(np.eye(4, dtype=np.float64)[None, ...], matrices.shape[0], axis=0)
    homogeneous[:, :3, :4] = matrices
    return homogeneous


def resolve_sequence_results_dir(results_path: str | Path) -> Path:
    results_path = Path(results_path)
    if not results_path.exists():
        raise FileNotFoundError(f"Results path not found: {results_path}")
    if results_path.is_dir():
        return results_path
    if results_path.suffix != ".npz":
        raise ValueError(
            "results_path must point to either a batch '.npz' file or a sequence results directory."
        )
    return results_path.parent


def get_sorted_batch_paths(sequence_results_dir: Path) -> list[Path]:
    batch_paths = sorted(
        sequence_results_dir.glob("*.npz"),
        key=_batch_index_from_path,
    )
    if not batch_paths:
        raise FileNotFoundError(f"No batch '.npz' files were found in {sequence_results_dir}.")
    return batch_paths


def load_run_manifest(sequence_results_dir: Path) -> dict[str, Any] | None:
    manifest_path = sequence_results_dir / "run_manifest.json"
    if not manifest_path.exists():
        return None
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def load_batch_result(batch_path: Path) -> KittiOdomBatchResult:
    with np.load(batch_path) as batch:
        missing_keys = [key for key in REQUIRED_BATCH_KEYS if key not in batch]
        if missing_keys:
            raise KeyError(f"{batch_path.name} is missing required keys: {missing_keys}")

        batch_index = _batch_index_from_path(batch_path)
        model = str(_load_scalar(batch, "model"))
        sequence = str(_load_scalar(batch, "sequence"))
        camera = str(_load_scalar(batch, "camera"))
        window_start = int(_load_scalar(batch, "window_start"))
        window_end = int(_load_scalar(batch, "window_end"))

        frame_ids_array = np.asarray(batch["frame_ids"])
        if frame_ids_array.ndim != 1:
            raise ValueError(
                f"Expected 'frame_ids' to be 1D in {batch_path.name}, got shape {frame_ids_array.shape}."
            )
        frame_ids = tuple(str(frame_id) for frame_id in frame_ids_array.tolist())

        camera_poses = np.asarray(batch["camera_poses"], dtype=np.float64)
        if camera_poses.ndim != 3 or camera_poses.shape[-2:] != (4, 4):
            raise ValueError(
                f"Expected 'camera_poses' shape (N, 4, 4) in {batch_path.name}, "
                f"got {camera_poses.shape}."
            )

    if window_start < 0:
        raise ValueError(f"window_start must be non-negative in {batch_path.name}.")
    if window_end <= window_start:
        raise ValueError(
            f"window_end must be greater than window_start in {batch_path.name}, "
            f"got ({window_start}, {window_end})."
        )

    expected_batch_len = window_end - window_start
    if len(frame_ids) != expected_batch_len:
        raise ValueError(
            f"{batch_path.name} has {len(frame_ids)} frame ids, but expected {expected_batch_len}."
        )
    if len(camera_poses) != expected_batch_len:
        raise ValueError(
            f"{batch_path.name} has {len(camera_poses)} camera poses, but expected {expected_batch_len}."
        )

    return KittiOdomBatchResult(
        path=batch_path,
        batch_index=batch_index,
        model=model,
        sequence=sequence,
        camera=camera,
        frame_ids=frame_ids,
        window_start=window_start,
        window_end=window_end,
        camera_poses=camera_poses,
    )


def _validate_sequence_batch_results(
    batch_results: list[KittiOdomBatchResult],
) -> None:
    if not batch_results:
        raise ValueError("batch_results cannot be empty.")

    first_batch = batch_results[0]
    for batch in batch_results[1:]:
        if batch.model != first_batch.model:
            raise ValueError(
                f"Inconsistent model names across batches: '{first_batch.model}' vs '{batch.model}'."
            )
        if batch.sequence != first_batch.sequence:
            raise ValueError(
                f"Inconsistent sequence ids across batches: '{first_batch.sequence}' vs '{batch.sequence}'."
            )
        if batch.camera != first_batch.camera:
            raise ValueError(
                f"Inconsistent camera ids across batches: '{first_batch.camera}' vs '{batch.camera}'."
            )


def load_sequence_results(results_path: str | Path) -> LoadedKittiOdomResults:
    sequence_results_dir = resolve_sequence_results_dir(results_path)
    batch_paths = get_sorted_batch_paths(sequence_results_dir)
    batch_results = [load_batch_result(batch_path) for batch_path in batch_paths]
    _validate_sequence_batch_results(batch_results)
    manifest = load_run_manifest(sequence_results_dir)
    return LoadedKittiOdomResults(
        sequence_results_dir=sequence_results_dir,
        batch_results=tuple(batch_results),
        manifest=manifest,
    )


def connect_batch_camera_poses(
    batch_results: list[KittiOdomBatchResult] | tuple[KittiOdomBatchResult, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    batch_results = list(batch_results)
    _validate_sequence_batch_results(batch_results)

    num_frames = max(batch.window_end for batch in batch_results)
    connected_camera_poses = np.full((num_frames, 4, 4), np.nan, dtype=np.float64)
    filled_mask = np.zeros(num_frames, dtype=bool)
    connected_frame_ids = np.full(num_frames, "", dtype=object)

    for batch_idx, batch in enumerate(batch_results):
        batch_camera_poses = to_homogeneous_matrices(batch.camera_poses)

        if batch_idx == 0:
            batch_to_global = np.linalg.inv(batch_camera_poses[0])
        else:
            if filled_mask[batch.window_start]:
                anchor_pose = connected_camera_poses[batch.window_start]
                anchor_frame_id = connected_frame_ids[batch.window_start]
                if anchor_frame_id and anchor_frame_id != batch.frame_ids[0]:
                    raise ValueError(
                        f"Anchor frame id mismatch for {batch.path.name}: expected '{anchor_frame_id}', "
                        f"got '{batch.frame_ids[0]}'."
                    )
            elif batch.window_start > 0 and filled_mask[batch.window_start - 1]:
                anchor_pose = connected_camera_poses[batch.window_start - 1]
            else:
                raise RuntimeError(
                    f"No connected anchor pose is available for {batch.path.name} at "
                    f"window_start={batch.window_start}."
                )
            batch_to_global = anchor_pose @ np.linalg.inv(batch_camera_poses[0])

        connected_batch_poses = batch_to_global[None, ...] @ batch_camera_poses
        for frame_offset, global_index in enumerate(range(batch.window_start, batch.window_end)):
            frame_id = batch.frame_ids[frame_offset]
            if not filled_mask[global_index]:
                connected_camera_poses[global_index] = connected_batch_poses[frame_offset]
                connected_frame_ids[global_index] = frame_id
                filled_mask[global_index] = True
                continue

            if connected_frame_ids[global_index] != frame_id:
                raise ValueError(
                    f"Frame id mismatch at global frame {global_index}: "
                    f"existing '{connected_frame_ids[global_index]}', new '{frame_id}'."
                )

    return connected_camera_poses, filled_mask, connected_frame_ids


def camera_centers_from_poses(camera_poses: np.ndarray) -> np.ndarray:
    homogeneous_poses = to_homogeneous_matrices(camera_poses)
    return homogeneous_poses[:, :3, 3].copy()


def load_kitti_gt_poses(dataset_root: str | Path, sequence_id: str) -> np.ndarray:
    gt_path = Path(dataset_root) / "poses" / f"{sequence_id}.txt"
    if not gt_path.exists():
        raise FileNotFoundError(f"GT pose file not found: {gt_path}")
    gt_poses = np.loadtxt(gt_path, dtype=np.float64)
    if gt_poses.ndim == 1:
        gt_poses = gt_poses[None, ...]
    return gt_poses.reshape(-1, 3, 4)


def load_optional_kitti_gt_poses(
    dataset_root: str | Path | None,
    sequence_id: str,
) -> np.ndarray | None:
    if dataset_root is None:
        return None
    gt_path = Path(dataset_root) / "poses" / f"{sequence_id}.txt"
    if not gt_path.exists():
        return None
    return load_kitti_gt_poses(dataset_root, sequence_id)


def estimate_umeyama_similarity(
    source_points: np.ndarray,
    target_points: np.ndarray,
    estimate_scale: bool = True,
) -> tuple[float, np.ndarray, np.ndarray]:
    source_points = np.asarray(source_points, dtype=np.float64)
    target_points = np.asarray(target_points, dtype=np.float64)

    if source_points.shape != target_points.shape:
        raise ValueError(
            f"source_points and target_points must have the same shape, got "
            f"{source_points.shape} and {target_points.shape}."
        )
    if source_points.ndim != 2 or source_points.shape[1] != 3:
        raise ValueError(
            f"Expected point arrays with shape (N, 3), got {source_points.shape}."
        )
    if len(source_points) < 2:
        raise ValueError("At least two points are required for Umeyama alignment.")

    source_mean = source_points.mean(axis=0)
    target_mean = target_points.mean(axis=0)
    source_centered = source_points - source_mean
    target_centered = target_points - target_mean

    covariance = (target_centered.T @ source_centered) / len(source_points)
    u_matrix, singular_values, v_transpose = np.linalg.svd(covariance)

    correction = np.eye(3, dtype=np.float64)
    if np.linalg.det(u_matrix) * np.linalg.det(v_transpose) < 0:
        correction[-1, -1] = -1.0

    rotation = u_matrix @ correction @ v_transpose

    if estimate_scale:
        source_variance = np.mean(np.sum(source_centered**2, axis=1))
        if source_variance <= 0:
            raise ValueError("Source points have zero variance; cannot estimate scale.")
        scale = float(np.sum(singular_values * np.diag(correction)) / source_variance)
    else:
        scale = 1.0

    translation = target_mean - scale * (rotation @ source_mean)
    return scale, rotation, translation


def apply_similarity_transform(
    points: np.ndarray,
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    rotation = np.asarray(rotation, dtype=np.float64)
    translation = np.asarray(translation, dtype=np.float64)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected points with shape (N, 3), got {points.shape}.")
    if rotation.shape != (3, 3):
        raise ValueError(f"Expected rotation with shape (3, 3), got {rotation.shape}.")
    if translation.shape != (3,):
        raise ValueError(f"Expected translation with shape (3,), got {translation.shape}.")

    return (scale * (rotation @ points.T)).T + translation


def align_predicted_trajectory_start_point(
    predicted_trajectory: np.ndarray,
    gt_trajectory: np.ndarray,
    filled_mask: np.ndarray,
) -> np.ndarray:
    predicted_trajectory = np.asarray(predicted_trajectory, dtype=np.float64)
    gt_trajectory = np.asarray(gt_trajectory, dtype=np.float64)
    filled_mask = np.asarray(filled_mask, dtype=bool)

    if predicted_trajectory.ndim != 2 or predicted_trajectory.shape[1] != 3:
        raise ValueError(
            f"Expected predicted_trajectory shape (N, 3), got {predicted_trajectory.shape}."
        )
    if gt_trajectory.ndim != 2 or gt_trajectory.shape[1] != 3:
        raise ValueError(
            f"Expected gt_trajectory shape (N, 3), got {gt_trajectory.shape}."
        )
    if filled_mask.shape != (len(predicted_trajectory),):
        raise ValueError(
            f"Expected filled_mask shape ({len(predicted_trajectory)},), got {filled_mask.shape}."
        )
    if len(predicted_trajectory) > len(gt_trajectory):
        raise ValueError(
            f"Predicted trajectory length ({len(predicted_trajectory)}) exceeds GT length ({len(gt_trajectory)})."
        )

    finite_mask = np.isfinite(predicted_trajectory).all(axis=1)
    alignment_mask = filled_mask & finite_mask
    if not alignment_mask.any():
        raise ValueError("At least one valid predicted point is required for start-point alignment.")

    gt_subset = gt_trajectory[: len(predicted_trajectory)]
    first_valid_index = int(np.flatnonzero(alignment_mask)[0])
    translation = gt_subset[first_valid_index] - predicted_trajectory[first_valid_index]
    return predicted_trajectory + translation


def align_predicted_trajectory_to_gt(
    predicted_trajectory: np.ndarray,
    gt_trajectory: np.ndarray,
    filled_mask: np.ndarray,
    match_start: bool = True,
) -> tuple[np.ndarray, float]:
    predicted_trajectory = np.asarray(predicted_trajectory, dtype=np.float64)
    gt_trajectory = np.asarray(gt_trajectory, dtype=np.float64)
    filled_mask = np.asarray(filled_mask, dtype=bool)

    if predicted_trajectory.ndim != 2 or predicted_trajectory.shape[1] != 3:
        raise ValueError(
            f"Expected predicted_trajectory shape (N, 3), got {predicted_trajectory.shape}."
        )
    if gt_trajectory.ndim != 2 or gt_trajectory.shape[1] != 3:
        raise ValueError(
            f"Expected gt_trajectory shape (N, 3), got {gt_trajectory.shape}."
        )
    if filled_mask.shape != (len(predicted_trajectory),):
        raise ValueError(
            f"Expected filled_mask shape ({len(predicted_trajectory)},), got {filled_mask.shape}."
        )
    if len(predicted_trajectory) > len(gt_trajectory):
        raise ValueError(
            f"Predicted trajectory length ({len(predicted_trajectory)}) exceeds GT length ({len(gt_trajectory)})."
        )

    finite_mask = np.isfinite(predicted_trajectory).all(axis=1)
    alignment_mask = filled_mask & finite_mask
    if alignment_mask.sum() < 2:
        raise ValueError("At least two valid predicted points are required for GT alignment.")

    gt_subset = gt_trajectory[: len(predicted_trajectory)]
    scale, rotation, translation = estimate_umeyama_similarity(
        predicted_trajectory[alignment_mask],
        gt_subset[alignment_mask],
        estimate_scale=True,
    )
    aligned_trajectory = apply_similarity_transform(
        predicted_trajectory,
        scale,
        rotation,
        translation,
    )

    if match_start:
        first_valid_index = int(np.flatnonzero(alignment_mask)[0])
        aligned_trajectory += (
            gt_subset[first_valid_index] - aligned_trajectory[first_valid_index]
        )

    return aligned_trajectory, float(scale)


__all__ = [
    "KittiOdomBatchResult",
    "LoadedKittiOdomResults",
    "align_predicted_trajectory_start_point",
    "align_predicted_trajectory_to_gt",
    "apply_similarity_transform",
    "camera_centers_from_poses",
    "connect_batch_camera_poses",
    "estimate_umeyama_similarity",
    "get_sorted_batch_paths",
    "load_batch_result",
    "load_kitti_gt_poses",
    "load_optional_kitti_gt_poses",
    "load_run_manifest",
    "load_sequence_results",
    "resolve_sequence_results_dir",
    "to_homogeneous_matrices",
]
