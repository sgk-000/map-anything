# usage: uv run scripts/map_free_inference.py --model mapanything --dataset-root /home/kobayashi/dataset/map_free --scenes s00000

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from mapanything.models import init_model_from_config, MapAnything
from mapanything.utils.device import (
    get_amp_dtype,
    get_autocast_device_type,
    get_device,
    to_device,
)
from mapanything.utils.image import preprocess_inputs
from mapanything.utils.inference import postprocess_model_outputs_for_inference

DEFAULT_DATASET_ROOT = Path("/mnt/ssd2/map_free")
DEFAULT_OUTPUT_ROOT = DEFAULT_DATASET_ROOT / "mapanything_inference_outputs"
DEFAULT_MAPANYTHING_MODEL_ID = "facebook/map-anything"
DEFAULT_SPLIT = "train"
DEFAULT_WINDOW_SIZE = 0
MAP_FREE_CAMERA_NAME = "all"
SUPPORTED_MODELS = ("mapanything", "pi3x")
SUPPORTED_SPLITS = ("train", "val", "test")
SUPPORTED_SEQUENCES = ("seq0", "seq1")
IMAGE_LIST_REQUIRED_COLUMNS = frozenset(
    {"scene_id", "seq", "seq_idx", "frame_idx", "rel_path", "image_path"}
)


@dataclass(frozen=True)
class MapFreeInferenceConfig:
    model: str
    dataset_root: Path = DEFAULT_DATASET_ROOT
    split: str = DEFAULT_SPLIT
    scenes: list[str] | None = None
    image_list_csv: Path | None = None
    window_size: int = DEFAULT_WINDOW_SIZE
    long_side_resolution: int | None = None
    device: str = "auto"
    machine: str = "default"
    mapanything_model_id: str = DEFAULT_MAPANYTHING_MODEL_ID
    output_root: Path = DEFAULT_OUTPUT_ROOT
    overwrite: bool = False
    resume: bool = False
    skip_failures: bool = False
    use_amp: bool = True
    amp_dtype: str = "bf16"

    def __post_init__(self) -> None:
        if self.model not in SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model '{self.model}'. Expected one of {SUPPORTED_MODELS}."
            )

        if self.split not in SUPPORTED_SPLITS:
            raise ValueError(
                f"Unsupported split '{self.split}'. Expected one of {SUPPORTED_SPLITS}."
            )

        if self.window_size < 0:
            raise ValueError(
                "window_size must be non-negative. Use 0 for full-scene inference, "
                f"but received {self.window_size}."
            )

        if self.long_side_resolution is not None and self.long_side_resolution <= 0:
            raise ValueError(
                "long_side_resolution must be positive when provided, "
                f"but received {self.long_side_resolution}."
            )

        if self.amp_dtype not in {"bf16", "fp16", "fp32"}:
            raise ValueError(
                f"amp_dtype must be one of ('bf16', 'fp16', 'fp32'), got '{self.amp_dtype}'."
            )

        if self.resume and self.overwrite:
            raise ValueError("resume and overwrite cannot both be enabled at the same time.")

        object.__setattr__(self, "dataset_root", Path(self.dataset_root))
        object.__setattr__(self, "output_root", Path(self.output_root))
        if self.image_list_csv is not None:
            object.__setattr__(self, "image_list_csv", Path(self.image_list_csv))
        if self.scenes is not None:
            object.__setattr__(self, "scenes", list(self.scenes))


@dataclass(frozen=True)
class ModelSpec:
    name: str
    resolution: int
    norm_type: str
    patch_size: int
    loader: Callable[[MapFreeInferenceConfig, torch.device], torch.nn.Module]
    adapter: Callable[
        [torch.nn.Module, list[dict[str, Any]], torch.device, MapFreeInferenceConfig],
        list[dict[str, torch.Tensor]],
    ]


def load_mapanything_model(
    config: MapFreeInferenceConfig,
    device: torch.device,
) -> torch.nn.Module:
    model = MapAnything.from_pretrained(config.mapanything_model_id).to(device)
    model.eval()
    return model


def load_hydra_model(
    model_name: str,
    config: MapFreeInferenceConfig,
    device: torch.device,
) -> torch.nn.Module:
    model = init_model_from_config(model_name, device=device, machine=config.machine)
    model.eval()
    return model


def mapanything_adapter(
    model: torch.nn.Module,
    views: list[dict[str, Any]],
    device: torch.device,
    config: MapFreeInferenceConfig,
) -> list[dict[str, torch.Tensor]]:
    amp_enabled = config.use_amp and device.type != "cpu"
    minibatch_size = 1 if device.type == "cuda" else None
    with torch.inference_mode():
        return model.infer(
            views,
            memory_efficient_inference=True,
            minibatch_size=minibatch_size,
            use_amp=amp_enabled,
            amp_dtype=config.amp_dtype,
            apply_mask=False,
            mask_edges=False,
        )


def wrapper_adapter(
    model: torch.nn.Module,
    views: list[dict[str, Any]],
    device: torch.device,
    config: MapFreeInferenceConfig,
) -> list[dict[str, torch.Tensor]]:
    amp_enabled = config.use_amp and device.type != "cpu"
    amp_dtype = get_amp_dtype(device, config.amp_dtype) if amp_enabled else torch.float32
    autocast_device_type = get_autocast_device_type(device)
    device_views = move_views_to_device(views, device)

    with torch.inference_mode():
        with torch.autocast(
            autocast_device_type,
            enabled=amp_enabled,
            dtype=amp_dtype,
        ):
            raw_outputs = model(device_views)

    return postprocess_model_outputs_for_inference(
        raw_outputs=raw_outputs,
        input_views=device_views,
        apply_mask=False,
        mask_edges=False,
    )


MODEL_SPECS: dict[str, ModelSpec] = {
    "mapanything": ModelSpec(
        name="mapanything",
        resolution=518,
        norm_type="dinov2",
        patch_size=14,
        loader=load_mapanything_model,
        adapter=mapanything_adapter,
    ),
    "pi3x": ModelSpec(
        name="pi3x",
        resolution=518,
        norm_type="identity",
        patch_size=14,
        loader=lambda config, device: load_hydra_model("pi3x", config, device),
        adapter=wrapper_adapter,
    ),
}


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return get_device()
    return torch.device(device_arg)


def discover_scene_dirs(
    dataset_root: Path,
    split: str,
    selected_scenes: list[str] | None,
) -> list[Path]:
    split_root = dataset_root / split
    if not split_root.exists():
        raise FileNotFoundError(f"Missing Map-Free split directory: {split_root}")

    available = {
        scene_dir.name: scene_dir
        for scene_dir in sorted(split_root.iterdir())
        if scene_dir.is_dir()
    }
    if not available:
        raise ValueError(f"No scene directories found under {split_root}")

    if not selected_scenes:
        return [available[name] for name in sorted(available)]

    missing = [scene for scene in selected_scenes if scene not in available]
    if missing:
        raise ValueError(
            f"Requested scenes not found under {split_root}: {', '.join(missing)}"
        )

    return [available[scene] for scene in selected_scenes]


def scene_relative_path(scene_dir: Path, path: Path) -> str:
    return path.relative_to(scene_dir).as_posix()


def load_scene_image_paths(scene_dir: Path) -> list[Path]:
    image_paths: list[Path] = []
    for sequence_name in SUPPORTED_SEQUENCES:
        sequence_dir = scene_dir / sequence_name
        if sequence_dir.is_dir():
            image_paths.extend(sequence_dir.glob("*.jpg"))

    return sorted(image_paths, key=lambda path: scene_relative_path(scene_dir, path))


def _required_csv_value(row: dict[str, str | None], column: str, line_number: int) -> str:
    value = row.get(column)
    if value is None or value.strip() == "":
        raise ValueError(f"Missing value for '{column}' on image-list CSV line {line_number}.")
    return value.strip()


def _validate_image_list_rel_path(
    rel_path: str,
    *,
    csv_path: Path,
    line_number: int,
) -> None:
    pure = PurePosixPath(rel_path)
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError(
            f"Invalid Map-Free relative path at {csv_path}:{line_number}: {rel_path!r}"
        )
    if len(pure.parts) != 2 or pure.parent.name not in SUPPORTED_SEQUENCES:
        raise ValueError(
            "Expected Map-Free image-list path like 'seq0/frame_00000.jpg' at "
            f"{csv_path}:{line_number}, got {rel_path!r}."
        )
    if pure.suffix.lower() != ".jpg":
        raise ValueError(
            f"Expected a .jpg Map-Free image path at {csv_path}:{line_number}, "
            f"got {rel_path!r}."
        )


def load_image_list_csv(
    dataset_root: Path,
    split: str,
    image_list_csv: Path,
) -> dict[str, list[Path]]:
    if not image_list_csv.is_file():
        raise FileNotFoundError(f"Missing Map-Free image-list CSV: {image_list_csv}")

    image_paths_by_scene: dict[str, list[Path]] = {}
    seen_frames: set[tuple[str, str]] = set()
    with image_list_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing_columns = sorted(IMAGE_LIST_REQUIRED_COLUMNS.difference(fieldnames))
        if missing_columns:
            raise ValueError(
                f"Map-Free image-list CSV is missing required columns {missing_columns}: "
                f"{image_list_csv}"
            )

        for row in reader:
            line_number = reader.line_num
            scene_id = _required_csv_value(row, "scene_id", line_number)
            seq = _required_csv_value(row, "seq", line_number)
            rel_path = _required_csv_value(row, "rel_path", line_number)
            csv_image_path = _required_csv_value(row, "image_path", line_number)
            _validate_image_list_rel_path(
                rel_path,
                csv_path=image_list_csv,
                line_number=line_number,
            )

            if seq not in SUPPORTED_SEQUENCES:
                raise ValueError(
                    f"Unsupported Map-Free sequence at {image_list_csv}:{line_number}: "
                    f"{seq!r}. Expected one of {SUPPORTED_SEQUENCES}."
                )

            try:
                seq_idx = int(_required_csv_value(row, "seq_idx", line_number))
                frame_idx = int(_required_csv_value(row, "frame_idx", line_number))
            except ValueError as exc:
                raise ValueError(
                    f"Invalid integer sequence/frame index at {image_list_csv}:{line_number}."
                ) from exc

            expected_seq_idx = SUPPORTED_SEQUENCES.index(seq)
            if seq_idx != expected_seq_idx:
                raise ValueError(
                    f"Map-Free image-list seq_idx mismatch at {image_list_csv}:{line_number}: "
                    f"seq={seq!r} expects {expected_seq_idx}, got {seq_idx}."
                )

            expected_rel_path = f"{seq}/frame_{frame_idx:05d}.jpg"
            if rel_path != expected_rel_path:
                raise ValueError(
                    f"Map-Free image-list rel_path mismatch at {image_list_csv}:{line_number}: "
                    f"expected {expected_rel_path!r}, got {rel_path!r}."
                )

            frame_key = (scene_id, rel_path)
            if frame_key in seen_frames:
                raise ValueError(
                    f"Duplicate Map-Free image-list frame at {image_list_csv}:{line_number}: "
                    f"{scene_id}/{rel_path}"
                )
            seen_frames.add(frame_key)

            image_path = dataset_root / split / scene_id / rel_path
            if not image_path.is_file():
                raise FileNotFoundError(
                    "Map-Free image listed in CSV does not exist: "
                    f"{image_path} (CSV image_path={csv_image_path})"
                )

            image_paths_by_scene.setdefault(scene_id, []).append(image_path)

    if not image_paths_by_scene:
        raise ValueError(f"No Map-Free image rows found in image-list CSV: {image_list_csv}")

    return image_paths_by_scene


def select_image_list_for_scene_dirs(
    scene_dirs: list[Path],
    image_paths_by_scene: dict[str, list[Path]],
    image_list_csv: Path,
) -> dict[str, list[Path]]:
    missing = [scene_dir.name for scene_dir in scene_dirs if scene_dir.name not in image_paths_by_scene]
    if missing:
        preview = ", ".join(missing[:10])
        suffix = "" if len(missing) <= 10 else f", ... ({len(missing)} total)"
        raise ValueError(
            f"Requested Map-Free scene(s) are missing from image-list CSV "
            f"{image_list_csv}: {preview}{suffix}"
        )
    return {scene_dir.name: image_paths_by_scene[scene_dir.name] for scene_dir in scene_dirs}


def validate_image_list_intrinsics_for_scene_dirs(
    scene_dirs: list[Path],
    image_paths_by_scene: dict[str, list[Path]],
) -> None:
    for scene_dir in scene_dirs:
        intrinsics_by_frame = parse_map_free_intrinsics(scene_dir)
        validate_intrinsics_for_images(
            scene_dir,
            image_paths_by_scene[scene_dir.name],
            intrinsics_by_frame,
        )


def parse_map_free_intrinsics(scene_dir: Path) -> dict[str, np.ndarray]:
    intrinsics_path = scene_dir / "intrinsics.txt"
    if not intrinsics_path.exists():
        raise FileNotFoundError(f"Missing Map-Free intrinsics file: {intrinsics_path}")

    intrinsics_by_frame: dict[str, np.ndarray] = {}
    with intrinsics_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) != 7:
                raise ValueError(
                    f"Malformed intrinsics line {line_number} in {intrinsics_path}: "
                    "expected 'frame_path fx fy cx cy frame_width frame_height'."
                )

            frame_path = parts[0]
            try:
                fx, fy, cx, cy, frame_width, frame_height = map(float, parts[1:])
            except ValueError as exc:
                raise ValueError(
                    f"Malformed numeric intrinsics on line {line_number} in {intrinsics_path}."
                ) from exc

            if frame_width <= 0 or frame_height <= 0:
                raise ValueError(
                    f"Invalid frame size on line {line_number} in {intrinsics_path}: "
                    f"{frame_width:g}x{frame_height:g}."
                )

            if frame_path in intrinsics_by_frame:
                raise ValueError(
                    f"Duplicate intrinsics entry for '{frame_path}' in {intrinsics_path}."
                )

            intrinsics_by_frame[frame_path] = np.array(
                [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            )

    if not intrinsics_by_frame:
        raise ValueError(f"No intrinsics entries found in {intrinsics_path}")

    return intrinsics_by_frame


def validate_intrinsics_for_images(
    scene_dir: Path,
    image_paths: list[Path],
    intrinsics_by_frame: dict[str, np.ndarray],
) -> None:
    missing = [
        scene_relative_path(scene_dir, image_path)
        for image_path in image_paths
        if scene_relative_path(scene_dir, image_path) not in intrinsics_by_frame
    ]
    if missing:
        preview = ", ".join(missing[:10])
        suffix = "" if len(missing) <= 10 else f", ... ({len(missing)} total)"
        raise KeyError(
            f"Missing intrinsics for Map-Free frame(s) in {scene_dir}: {preview}{suffix}"
        )


def parse_map_free_poses(
    scene_dir: Path,
    filename: str = "poses.txt",
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    poses_path = scene_dir / filename
    if not poses_path.exists():
        raise FileNotFoundError(f"Missing Map-Free poses file: {poses_path}")

    poses_by_frame: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    with poses_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) != 8:
                raise ValueError(
                    f"Malformed pose line {line_number} in {poses_path}: "
                    "expected 'frame_path qw qx qy qz tx ty tz'."
                )

            frame_path = parts[0]
            try:
                pose_values = np.asarray(parts[1:], dtype=np.float64)
            except ValueError as exc:
                raise ValueError(
                    f"Malformed numeric pose on line {line_number} in {poses_path}."
                ) from exc

            quaternion = pose_values[:4]
            translation = pose_values[4:]
            if np.isclose(np.linalg.norm(quaternion), 0.0):
                raise ValueError(
                    f"Invalid zero-norm quaternion on line {line_number} in {poses_path}."
                )

            if frame_path in poses_by_frame:
                raise ValueError(f"Duplicate pose entry for '{frame_path}' in {poses_path}.")

            poses_by_frame[frame_path] = (quaternion, translation)

    return poses_by_frame


def build_batch_ranges(num_images: int, window_size: int) -> list[tuple[int, int]]:
    if window_size < 0:
        raise ValueError(f"window_size must be non-negative, but got {window_size}")
    if num_images <= 0:
        return []
    if window_size == 0 or num_images <= window_size:
        return [(0, num_images)]

    batch_ranges: list[tuple[int, int]] = []
    start = 0
    while start + window_size <= num_images:
        batch_ranges.append((start, start + window_size))
        start += window_size

    if num_images % window_size != 0:
        batch_ranges.append((num_images - window_size, num_images))

    return batch_ranges


def build_batch_output_filenames(
    batch_ranges: list[tuple[int, int]],
    window_size: int,
) -> list[str]:
    if window_size < 0:
        raise ValueError(f"window_size must be non-negative, but got {window_size}")

    return [f"{batch_idx}.npz" for batch_idx, _ in enumerate(batch_ranges)]


def expected_output_paths_for_job(
    save_dir: Path,
    num_images: int,
    window_size: int,
) -> list[Path]:
    batch_ranges = build_batch_ranges(num_images, window_size)
    batch_filenames = build_batch_output_filenames(batch_ranges, window_size)
    return [save_dir / filename for filename in batch_filenames]


def should_resume_skip_job(
    save_dir: Path,
    num_images: int,
    window_size: int,
) -> bool:
    expected_paths = expected_output_paths_for_job(save_dir, num_images, window_size)
    return bool(expected_paths) and all(path.exists() for path in expected_paths)


def get_resolution_label(
    model_spec: ModelSpec,
    config: MapFreeInferenceConfig,
) -> str:
    if config.long_side_resolution is not None:
        return f"long_side_{config.long_side_resolution}"
    return f"res_{model_spec.resolution}"


def get_window_label(window_size: int) -> str:
    if window_size == 0:
        return "window_all"
    return f"window_{window_size}"


def output_dir_for_scene(
    output_root: Path,
    model_name: str,
    window_size: int,
    resolution_label: str,
    split: str,
    scene_name: str,
) -> Path:
    return output_root / model_name / f"{get_window_label(window_size)}_{resolution_label}" / split / scene_name


def build_raw_views_for_batch(
    image_paths: list[Path],
    scene_dir: Path,
    intrinsics_by_frame: dict[str, np.ndarray],
    index_offset: int = 0,
) -> list[dict[str, Any]]:
    raw_views: list[dict[str, Any]] = []
    for local_index, image_path in enumerate(image_paths):
        frame_id = scene_relative_path(scene_dir, image_path)
        with Image.open(image_path) as image:
            rgb_image = image.convert("RGB").copy()
        raw_views.append(
            {
                "img": rgb_image,
                "intrinsics": intrinsics_by_frame[frame_id].copy(),
                "idx": index_offset + local_index,
                "instance": frame_id,
            }
        )
    return raw_views


def preprocess_batch_views(
    raw_views: list[dict[str, Any]],
    model_spec: ModelSpec,
    config: MapFreeInferenceConfig,
) -> list[dict[str, Any]]:
    preprocess_kwargs: dict[str, Any] = {
        "norm_type": model_spec.norm_type,
        "patch_size": model_spec.patch_size,
        "verbose": False,
    }
    if config.long_side_resolution is None:
        preprocess_kwargs["resolution_set"] = model_spec.resolution
    else:
        preprocess_kwargs["resize_mode"] = "longest_side"
        preprocess_kwargs["size"] = config.long_side_resolution
        preprocess_kwargs["resolution_set"] = model_spec.resolution

    return preprocess_inputs(
        raw_views,
        **preprocess_kwargs,
    )


def move_views_to_device(
    views: list[dict[str, Any]],
    device: torch.device,
) -> list[dict[str, Any]]:
    return [
        to_device(view, device, non_blocking=device.type == "cuda")
        for view in views
    ]


def run_model_inference(
    model_name: str,
    model: torch.nn.Module,
    views: list[dict[str, Any]],
    device: torch.device,
    config: MapFreeInferenceConfig,
) -> list[dict[str, torch.Tensor]]:
    model_spec = MODEL_SPECS[model_name]
    return model_spec.adapter(model, views, device, config)


def _squeeze_view_batch(
    tensor: torch.Tensor,
    key: str,
    expected_ndim_after_squeeze: int | None = None,
) -> torch.Tensor:
    if tensor.shape[0] != 1:
        raise ValueError(
            f"Expected per-view batch size 1 for '{key}', but received shape {tuple(tensor.shape)}."
        )
    squeezed = tensor[0]
    if expected_ndim_after_squeeze is not None and squeezed.ndim != expected_ndim_after_squeeze:
        raise ValueError(
            f"Unexpected shape for '{key}' after squeezing batch dimension: "
            f"{tuple(squeezed.shape)}."
        )
    return squeezed


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    tensor = tensor.detach().cpu()
    if torch.is_floating_point(tensor):
        tensor = tensor.float()
    return tensor.numpy()


def normalize_predictions_for_saving(
    predictions: list[dict[str, torch.Tensor]],
) -> dict[str, np.ndarray]:
    normalized_camera_poses: list[np.ndarray] = []

    for prediction in predictions:
        if "camera_poses" not in prediction:
            raise KeyError("Prediction is missing required key: 'camera_poses'")

        camera_poses = _squeeze_view_batch(
            prediction["camera_poses"], "camera_poses", expected_ndim_after_squeeze=2
        )
        if tuple(camera_poses.shape) != (4, 4):
            raise ValueError(
                f"Expected 'camera_poses' shape (4, 4), got {tuple(camera_poses.shape)}."
            )

        normalized_camera_poses.append(_to_numpy(camera_poses))

    return {
        "camera_poses": np.stack(normalized_camera_poses, axis=0),
    }


def save_batch_output(
    save_path: Path,
    model_name: str,
    split_name: str,
    scene_name: str,
    frame_ids: list[str],
    window_start: int,
    window_end: int,
    predictions: dict[str, np.ndarray],
) -> None:
    np.savez(
        save_path,
        model=np.array(model_name),
        date=np.array(split_name),
        drive=np.array(scene_name),
        camera=np.array(MAP_FREE_CAMERA_NAME),
        frame_ids=np.asarray(frame_ids),
        window_start=np.int64(window_start),
        window_end=np.int64(window_end),
        poses=predictions["camera_poses"],
    )


def _metadata_summary(scene_dir: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for filename in ("intrinsics.txt", "poses.txt", "poses_device.txt", "overlaps.npz"):
        path = scene_dir / filename
        metadata[filename] = {"exists": path.exists()}
    return metadata


def write_run_manifest(
    output_dir: Path,
    split_name: str,
    scene_name: str,
    config: MapFreeInferenceConfig,
    model_spec: ModelSpec,
    resolved_device: torch.device,
    scene_dir: Path,
    image_paths: list[Path],
) -> None:
    manifest_path = output_dir / "run_manifest.json"
    frame_ids = [scene_relative_path(scene_dir, path) for path in image_paths]
    manifest = {
        "model": config.model,
        "date": split_name,
        "drive": scene_name,
        "camera": MAP_FREE_CAMERA_NAME,
        "resolved_device": str(resolved_device),
        "image_list_csv": str(config.image_list_csv) if config.image_list_csv is not None else None,
        "image_list_mode": config.image_list_csv is not None,
        "num_listed_frames": len(frame_ids) if config.image_list_csv is not None else None,
        "first_frame_id": frame_ids[0] if frame_ids else None,
        "last_frame_id": frame_ids[-1] if frame_ids else None,
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
        "model_spec": {
            "resolution": model_spec.resolution,
            "norm_type": model_spec.norm_type,
            "patch_size": model_spec.patch_size,
        },
        "map_free_metadata": _metadata_summary(scene_dir),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def run_scene_job(
    model: torch.nn.Module,
    model_spec: ModelSpec,
    scene_dir: Path,
    config: MapFreeInferenceConfig,
    device: torch.device,
    image_paths: list[Path] | None = None,
) -> None:
    if image_paths is None:
        image_paths = load_scene_image_paths(scene_dir)
    else:
        image_paths = list(image_paths)
    if not image_paths:
        print(f"Warning: No images found in Map-Free scene: {scene_dir}, skipping.")
        return
    print(f"Map-Free scene {config.split}/{scene_dir.name}: {len(image_paths)} images")

    intrinsics_by_frame = parse_map_free_intrinsics(scene_dir)
    validate_intrinsics_for_images(scene_dir, image_paths, intrinsics_by_frame)

    save_dir = output_dir_for_scene(
        output_root=config.output_root,
        model_name=config.model,
        window_size=config.window_size,
        resolution_label=get_resolution_label(model_spec, config),
        split=config.split,
        scene_name=scene_dir.name,
    )
    if config.resume and should_resume_skip_job(save_dir, len(image_paths), config.window_size):
        print(f"Skipping completed Map-Free scene: {config.split}/{scene_dir.name}")
        return

    save_dir.mkdir(parents=True, exist_ok=True)
    write_run_manifest(
        output_dir=save_dir,
        split_name=config.split,
        scene_name=scene_dir.name,
        config=config,
        model_spec=model_spec,
        resolved_device=device,
        scene_dir=scene_dir,
        image_paths=image_paths,
    )

    batch_ranges = build_batch_ranges(len(image_paths), config.window_size)
    batch_filenames = build_batch_output_filenames(batch_ranges, config.window_size)
    batch_iterator = tqdm(
        zip(batch_filenames, batch_ranges, strict=True),
        total=len(batch_ranges),
        desc=f"{config.split}/{scene_dir.name}",
        leave=False,
    )
    for batch_filename, (start, end) in batch_iterator:
        save_path = save_dir / batch_filename
        if save_path.exists() and not config.overwrite:
            continue

        batch_image_paths = image_paths[start:end]
        raw_views = build_raw_views_for_batch(
            batch_image_paths,
            scene_dir=scene_dir,
            intrinsics_by_frame=intrinsics_by_frame,
            index_offset=start,
        )
        processed_views = preprocess_batch_views(raw_views, model_spec, config)
        predictions = run_model_inference(
            model_name=config.model,
            model=model,
            views=processed_views,
            device=device,
            config=config,
        )
        normalized_predictions = normalize_predictions_for_saving(predictions)
        save_batch_output(
            save_path=save_path,
            model_name=config.model,
            split_name=config.split,
            scene_name=scene_dir.name,
            frame_ids=[scene_relative_path(scene_dir, path) for path in batch_image_paths],
            window_start=start,
            window_end=end,
            predictions=normalized_predictions,
        )


def run_map_free_inference(config: MapFreeInferenceConfig) -> None:
    device = resolve_device(config.device)
    model_spec = MODEL_SPECS[config.model]
    scene_dirs = discover_scene_dirs(config.dataset_root, config.split, config.scenes)
    resolution_label = get_resolution_label(model_spec, config)
    image_paths_by_scene: dict[str, list[Path]] | None = None
    if config.image_list_csv is not None:
        loaded_image_paths_by_scene = load_image_list_csv(
            config.dataset_root,
            config.split,
            config.image_list_csv,
        )
        image_paths_by_scene = select_image_list_for_scene_dirs(
            scene_dirs,
            loaded_image_paths_by_scene,
            config.image_list_csv,
        )
        validate_image_list_intrinsics_for_scene_dirs(scene_dirs, image_paths_by_scene)

    if config.resume:
        pending_scene_dirs: list[Path] = []
        for scene_dir in scene_dirs:
            image_paths = (
                image_paths_by_scene[scene_dir.name]
                if image_paths_by_scene is not None
                else load_scene_image_paths(scene_dir)
            )
            if not image_paths:
                pending_scene_dirs.append(scene_dir)
                continue

            save_dir = output_dir_for_scene(
                output_root=config.output_root,
                model_name=config.model,
                window_size=config.window_size,
                resolution_label=resolution_label,
                split=config.split,
                scene_name=scene_dir.name,
            )
            if should_resume_skip_job(save_dir, len(image_paths), config.window_size):
                print(f"Skipping completed Map-Free scene: {config.split}/{scene_dir.name}")
                continue

            pending_scene_dirs.append(scene_dir)

        scene_dirs = pending_scene_dirs
        if not scene_dirs:
            print("All requested Map-Free scenes are already processed; nothing to do.")
            return

    model = model_spec.loader(config, device)

    failed_scenes: list[str] = []
    for scene_dir in tqdm(scene_dirs, desc="Map-Free scene jobs"):
        try:
            run_scene_job(
                model=model,
                model_spec=model_spec,
                scene_dir=scene_dir,
                config=config,
                device=device,
                image_paths=image_paths_by_scene[scene_dir.name]
                if image_paths_by_scene is not None
                else None,
            )
        except Exception as exc:
            if not config.skip_failures:
                raise
            failed_scenes.append(scene_dir.name)
            print(
                "Warning: Skipping failed Map-Free scene "
                f"{config.split}/{scene_dir.name}: {type(exc).__name__}: {exc}"
            )

    if failed_scenes:
        print(
            "Completed Map-Free inference with "
            f"{len(failed_scenes)} skipped failed scene(s): {', '.join(failed_scenes)}"
        )
