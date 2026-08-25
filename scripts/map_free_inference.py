# Usage: uv run scripts/map_free_inference.py --model mapanything --dataset-root ~/dataset/map_free --scenes s00000

from __future__ import annotations

import argparse
from pathlib import Path

from mapanything.utils.map_free_inference import (
    DEFAULT_DATASET_ROOT,
    DEFAULT_MAPANYTHING_MODEL_ID,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SPLIT,
    DEFAULT_WINDOW_SIZE,
    MapFreeInferenceConfig,
    run_map_free_inference,
    SUPPORTED_SPLITS,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MapAnything inference on the Map-Free dataset."
    )
    parser.add_argument(
        "--model",
        default="mapanything",
        help="Model to run through the Map-Free inference interface.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Map-Free dataset root containing train/val/test split directories.",
    )
    parser.add_argument(
        "--split",
        choices=SUPPORTED_SPLITS,
        default=DEFAULT_SPLIT,
        help="Map-Free split to process.",
    )
    parser.add_argument(
        "--scenes",
        nargs="*",
        default=None,
        help="Optional list of Map-Free scene IDs. Defaults to all discovered scenes.",
    )
    parser.add_argument(
        "--image-list-csv",
        type=Path,
        default=None,
        help=(
            "Optional CSV of Map-Free frames to process. When set, each scene uses "
            "only the listed seq0/seq1 frames instead of all discovered images."
        ),
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=DEFAULT_WINDOW_SIZE,
        help="Number of images per inference window. Use 0 to process a whole scene at once.",
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
    output_mode_group = parser.add_mutually_exclusive_group()
    output_mode_group.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing batch outputs instead of skipping them.",
    )
    output_mode_group.add_argument(
        "--resume",
        action="store_true",
        help="Skip scene jobs whose expected batch outputs already exist.",
    )
    parser.add_argument(
        "--skip-failures",
        action="store_true",
        help="Continue processing remaining scenes when a scene fails.",
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
    config = MapFreeInferenceConfig(
        model=args.model,
        dataset_root=args.dataset_root,
        split=args.split,
        scenes=args.scenes,
        image_list_csv=args.image_list_csv,
        window_size=effective_window_size,
        long_side_resolution=args.long_side_resolution,
        device=args.device,
        mapanything_model_id=args.mapanything_model_id,
        output_root=args.output_root,
        overwrite=args.overwrite,
        resume=args.resume,
        skip_failures=args.skip_failures,
        use_amp=args.use_amp,
        amp_dtype=args.amp_dtype,
    )
    run_map_free_inference(config)


if __name__ == "__main__":
    main()
