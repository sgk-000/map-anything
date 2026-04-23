# Usage: uv run scripts/kitti_odom_inference.py --model mapanything --dataset-root ~/dataset/kitti_odom/dataset/ --sequences --window-size 200

from __future__ import annotations

import argparse
from pathlib import Path

from mapanything.utils.kitti_odom_inference import (
    DEFAULT_DATASET_ROOT,
    DEFAULT_MAPANYTHING_MODEL_ID,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_WINDOW_SIZE,
    KittiOdomInferenceConfig,
    SUPPORTED_CAMERAS,
    SUPPORTED_MODELS,
    run_kitti_odom_inference,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run unified inference on the KITTI Odometry dataset."
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=SUPPORTED_MODELS,
        help="Model to run through the unified KITTI Odometry interface.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="KITTI Odometry dataset root containing the sequences directory.",
    )
    parser.add_argument(
        "--sequences",
        nargs="*",
        default=None,
        help="Optional list of KITTI Odometry sequence IDs. Defaults to all discovered sequences.",
    )
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=["image_2"],
        choices=SUPPORTED_CAMERAS,
        help="Camera folders to process.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=DEFAULT_WINDOW_SIZE,
        help="Number of images per inference window.",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=None,
        help="Number of images processed by a model once. Alias for --window-size.",
    )
    parser.add_argument(
        "--long-side-resolution",
        type=int,
        default=None,
        help="Optional resize override that sets the image long side before inference.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device string to use (for example: auto, cuda, cuda:0, cpu).",
    )
    parser.add_argument(
        "--machine",
        default="default",
        help="Hydra machine config name used for non-MapAnything models.",
    )
    parser.add_argument(
        "--mapanything-model-id",
        default=DEFAULT_MAPANYTHING_MODEL_ID,
        help="Hugging Face model identifier used when --model=mapanything.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory for standardized inference outputs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing batch outputs instead of skipping them.",
    )
    parser.add_argument(
        "--use-amp",
        dest="use_amp",
        action="store_true",
        default=True,
        help="Enable automatic mixed precision.",
    )
    parser.add_argument(
        "--no-amp",
        dest="use_amp",
        action="store_false",
        help="Disable automatic mixed precision.",
    )
    parser.add_argument(
        "--amp-dtype",
        choices=("bf16", "fp16", "fp32"),
        default="bf16",
        help="AMP dtype to request when mixed precision is enabled.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    effective_window_size = (
        args.num_images if args.num_images is not None else args.window_size
    )
    config = KittiOdomInferenceConfig(
        model=args.model,
        dataset_root=args.dataset_root,
        sequences=args.sequences,
        cameras=args.cameras,
        window_size=effective_window_size,
        long_side_resolution=args.long_side_resolution,
        device=args.device,
        machine=args.machine,
        mapanything_model_id=args.mapanything_model_id,
        output_root=args.output_root,
        overwrite=args.overwrite,
        use_amp=args.use_amp,
        amp_dtype=args.amp_dtype,
    )
    run_kitti_odom_inference(config)


if __name__ == "__main__":
    main()
