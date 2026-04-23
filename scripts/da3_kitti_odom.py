from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from depth_anything_3.api import DepthAnything3


# num_image=300 -> long_edge=504
# num_image=250 -> long_edge=550
# num_image=200 -> long_edge=630
# num_image=175 -> long_edge=690
# num_image=150 -> long_edge=730
# num_image=100 -> long_edge=720
# num_image=75 -> long_edge=850
# num_image=30 -> long_edge=1224



DEFAULT_DATASET_ROOT = Path("/home/kobayashi/dataset/kitti_odom/dataset")
DEFAULT_OUTPUT_ROOT_BASE = Path("/home/kobayashi/dataset/kitti_odom")
DEFAULT_MODEL_ID = "depth-anything/DA3NESTED-GIANT-LARGE"
SUPPORTED_CAMERAS = ("image_2",)
# SUPPORTED_CAMERAS = ("image_2", "image_3")
PROJECTION_KEY_BY_CAMERA = {
    "image_2": "P2",
    "image_3": "P3",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Depth Anything 3 inference on the KITTI odometry dataset."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="KITTI odometry dataset root containing the sequences directory.",
    )
    parser.add_argument(
        "--sequences",
        nargs="*",
        default=None,
        help="Optional list of KITTI odometry sequence IDs. Defaults to all discovered sequences.",
    )
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=list(SUPPORTED_CAMERAS),
        choices=SUPPORTED_CAMERAS,
        help="Camera folders to process.",
    )
    parser.add_argument("--gpu-id", type=int, default=1, help="CUDA device index.")
    parser.add_argument(
        "--num-images",
        type=int,
        default=100,
        help="Number of images per inference window.",
    )
    parser.add_argument(
        "--long-edge",
        type=int,
        default=720,
        help="Processing resolution passed to Depth Anything 3.",
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help="Hugging Face model identifier to load.",
    )
    parser.add_argument(
        "--output-root-base",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT_BASE,
        help="Base directory for camera-specific output roots.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing batch outputs instead of skipping them.",
    )
    return parser.parse_args()


def parse_kitti_odom_intrinsics(sequence_dir: Path, camera: str) -> np.ndarray:
    if camera not in PROJECTION_KEY_BY_CAMERA:
        raise ValueError(f"Unsupported camera '{camera}'. Expected one of {SUPPORTED_CAMERAS}.")

    calib_path = sequence_dir / "calib.txt"
    if not calib_path.exists():
        raise FileNotFoundError(f"Missing calibration file: {calib_path}")

    calib_entries: dict[str, str] = {}
    with calib_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            key, values = line.split(":", maxsplit=1)
            calib_entries[key] = values.strip()

    projection_key = PROJECTION_KEY_BY_CAMERA[camera]
    if projection_key not in calib_entries:
        raise KeyError(f"Missing {projection_key} in calibration file: {calib_path}")

    projection_values = np.fromstring(calib_entries[projection_key], sep=" ", dtype=np.float32)
    if projection_values.size != 12:
        raise ValueError(
            f"Expected 12 values for {projection_key} in {calib_path}, "
            f"but found {projection_values.size}."
        )

    projection = projection_values.reshape(3, 4)
    return projection[:, :3].copy()


def discover_sequence_dirs(dataset_root: Path, selected_sequences: list[str] | None) -> list[Path]:
    sequences_root = dataset_root / "sequences"
    if not sequences_root.exists():
        raise FileNotFoundError(f"Missing sequences directory: {sequences_root}")

    available = {
        sequence_dir.name: sequence_dir
        for sequence_dir in sorted(sequences_root.iterdir())
        if sequence_dir.is_dir()
    }
    if not available:
        raise ValueError(f"No sequence directories found under {sequences_root}")

    if not selected_sequences:
        return [available[name] for name in sorted(available)]

    missing = [sequence for sequence in selected_sequences if sequence not in available]
    if missing:
        raise ValueError(
            f"Requested sequences not found under {sequences_root}: {', '.join(missing)}"
        )

    return [available[sequence] for sequence in selected_sequences]


def build_batch_ranges(num_images: int, window_size: int) -> list[tuple[int, int]]:
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, but got {window_size}")
    if num_images <= 0:
        return []
    if num_images <= window_size:
        return [(0, num_images)]

    batch_ranges: list[tuple[int, int]] = []
    start = 0
    while start + window_size <= num_images:
        batch_ranges.append((start, start + window_size))
        start += window_size

    if num_images % window_size != 0:
        batch_ranges.append((num_images - window_size, num_images))

    return batch_ranges


def output_dir_for_sequence(
    output_root_base: Path,
    num_images: int,
    camera: str,
    sequence_name: str,
    long_edge: int,
) -> Path:
    return output_root_base / f"depth_anything3_num_img{num_images}_long_edge{long_edge}" / sequence_name


def run_sequence_camera_job(
    model: DepthAnything3,
    sequence_dir: Path,
    camera: str,
    output_root_base: Path,
    num_images: int,
    long_edge: int,
    overwrite: bool,
) -> None:
    camera_dir = sequence_dir / camera
    if not camera_dir.exists():
        raise FileNotFoundError(f"Missing camera directory: {camera_dir}")

    image_paths = sorted(camera_dir.glob("*.png"))
    if not image_paths:
        raise ValueError(f"No PNG images found in {camera_dir}")

    intrinsics = parse_kitti_odom_intrinsics(sequence_dir, camera)
    save_dir = output_dir_for_sequence(output_root_base, num_images, camera, sequence_dir.name, long_edge)
    save_dir.mkdir(parents=True, exist_ok=True)

    batch_ranges = build_batch_ranges(len(image_paths), num_images)
    batch_iterator = tqdm(
        enumerate(batch_ranges),
        total=len(batch_ranges),
        desc=f"{sequence_dir.name}/{camera}",
        leave=False,
    )
    for batch_idx, (start, end) in batch_iterator:
        save_path = save_dir / f"{batch_idx}.npz"
        if save_path.exists() and not overwrite:
            continue

        batch_image_paths = [str(path) for path in image_paths[start:end]]
        batch_intrinsics = np.stack([intrinsics] * len(batch_image_paths), axis=0)
        prediction = model.inference(
            batch_image_paths,
            intrinsics=batch_intrinsics,
            process_res=long_edge,
        )
        np.savez(
            save_path,
            depth=prediction.depth,
            conf=prediction.conf,
            extrinsics=prediction.extrinsics,
            intrinsics=prediction.intrinsics,
        )


def main() -> None:
    args = parse_args()

    if args.num_images <= 0:
        raise ValueError(f"--num-images must be positive, but got {args.num_images}")
    if args.long_edge <= 0:
        raise ValueError(f"--long-edge must be positive, but got {args.long_edge}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script, but no CUDA device is available.")

    sequence_dirs = discover_sequence_dirs(args.dataset_root, args.sequences)
    jobs = [(sequence_dir, camera) for sequence_dir in sequence_dirs for camera in args.cameras]

    device = torch.device(f"cuda:{args.gpu_id}")
    model = DepthAnything3.from_pretrained(args.model_id)
    model = model.to(device=device)

    for sequence_dir, camera in tqdm(jobs, desc="sequence/camera jobs"):
        run_sequence_camera_job(
            model=model,
            sequence_dir=sequence_dir,
            camera=camera,
            output_root_base=args.output_root_base,
            num_images=args.num_images,
            long_edge=args.long_edge,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
