from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from mapanything.models import MapAnything, init_model_from_config
from mapanything.utils.device import to_device
from mapanything.utils.device import get_amp_dtype, get_autocast_device_type, get_device
from mapanything.utils.image import preprocess_inputs
from mapanything.utils.inference import postprocess_model_outputs_for_inference

DEFAULT_DATASET_ROOT = Path("/home/kobayashi/dataset/kitti_odom/dataset")
DEFAULT_OUTPUT_ROOT = Path("/home/kobayashi/dataset/kitti_odom/inference_outputs")
DEFAULT_MAPANYTHING_MODEL_ID = "facebook/map-anything"
DEFAULT_WINDOW_SIZE = 100
SUPPORTED_MODELS = ("mapanything", "pi3x", "da3_nested")
SUPPORTED_CAMERAS = ("image_2", "image_3")
PROJECTION_KEY_BY_CAMERA = {
    "image_2": "P2",
    "image_3": "P3",
}


@dataclass(frozen=True)
class KittiOdomInferenceConfig:
    model: str
    dataset_root: Path = DEFAULT_DATASET_ROOT
    sequences: list[str] | None = None
    cameras: list[str] | None = None
    window_size: int = DEFAULT_WINDOW_SIZE
    long_side_resolution: int | None = None
    device: str = "auto"
    machine: str = "default"
    mapanything_model_id: str = DEFAULT_MAPANYTHING_MODEL_ID
    output_root: Path = DEFAULT_OUTPUT_ROOT
    overwrite: bool = False
    use_amp: bool = True
    amp_dtype: str = "bf16"

    def __post_init__(self) -> None:
        if self.model not in SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model '{self.model}'. Expected one of {SUPPORTED_MODELS}."
            )

        if self.window_size <= 0:
            raise ValueError(
                f"window_size must be positive, but received {self.window_size}."
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

        normalized_cameras = list(self.cameras) if self.cameras is not None else ["image_2"]
        invalid_cameras = [camera for camera in normalized_cameras if camera not in SUPPORTED_CAMERAS]
        if invalid_cameras:
            raise ValueError(
                f"Unsupported camera(s): {invalid_cameras}. Expected subset of {SUPPORTED_CAMERAS}."
            )

        object.__setattr__(self, "dataset_root", Path(self.dataset_root))
        object.__setattr__(self, "output_root", Path(self.output_root))
        object.__setattr__(self, "cameras", normalized_cameras)
        if self.sequences is not None:
            object.__setattr__(self, "sequences", list(self.sequences))


@dataclass(frozen=True)
class ModelSpec:
    name: str
    resolution: int
    norm_type: str
    patch_size: int
    loader: Callable[[KittiOdomInferenceConfig, torch.device], torch.nn.Module]
    adapter: Callable[
        [torch.nn.Module, list[dict[str, Any]], torch.device, KittiOdomInferenceConfig],
        list[dict[str, torch.Tensor]],
    ]


def load_mapanything_model(
    config: KittiOdomInferenceConfig,
    device: torch.device,
) -> torch.nn.Module:
    model = MapAnything.from_pretrained(config.mapanything_model_id).to(device)
    model.eval()
    return model


def load_hydra_model(
    model_name: str,
    config: KittiOdomInferenceConfig,
    device: torch.device,
) -> torch.nn.Module:
    model = init_model_from_config(model_name, device=device, machine=config.machine)
    model.eval()
    return model


def mapanything_adapter(
    model: torch.nn.Module,
    views: list[dict[str, Any]],
    device: torch.device,
    config: KittiOdomInferenceConfig,
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
    config: KittiOdomInferenceConfig,
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
    "da3_nested": ModelSpec(
        name="da3_nested",
        resolution=504,
        norm_type="dinov2",
        patch_size=14,
        loader=lambda config, device: load_hydra_model("da3_nested", config, device),
        adapter=wrapper_adapter,
    ),
}


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return get_device()
    return torch.device(device_arg)


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


def discover_sequence_dirs(
    dataset_root: Path,
    selected_sequences: list[str] | None,
) -> list[Path]:
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
    output_root: Path,
    model_name: str,
    window_size: int,
    resolution_label: str,
    camera: str,
    sequence_name: str,
) -> Path:
    return (
        output_root
        / model_name
        / f"window_{window_size}_{resolution_label}"
        / camera
        / sequence_name
    )


def build_raw_views_for_batch(
    image_paths: list[Path],
    intrinsics: np.ndarray,
) -> list[dict[str, Any]]:
    raw_views: list[dict[str, Any]] = []
    for image_path in image_paths:
        with Image.open(image_path) as image:
            rgb_image = image.convert("RGB").copy()
        raw_views.append(
            {
                "img": rgb_image,
                "intrinsics": intrinsics.copy(),
                "idx": int(image_path.stem),
                "instance": image_path.name,
            }
        )
    return raw_views


def preprocess_batch_views(
    raw_views: list[dict[str, Any]],
    model_spec: ModelSpec,
    config: KittiOdomInferenceConfig,
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
    config: KittiOdomInferenceConfig,
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
    required_keys = ("depth_z", "depth_along_ray", "conf", "intrinsics", "camera_poses")
    normalized: dict[str, list[np.ndarray]] = {key: [] for key in required_keys}
    valid_masks: list[np.ndarray] = []

    for prediction in predictions:
        missing = [key for key in required_keys if key not in prediction]
        if missing:
            raise KeyError(f"Prediction is missing required keys: {missing}")

        depth_z = _squeeze_view_batch(prediction["depth_z"], "depth_z")
        if depth_z.ndim == 2:
            depth_z = depth_z.unsqueeze(-1)
        if depth_z.ndim != 3 or depth_z.shape[-1] != 1:
            raise ValueError(f"Expected 'depth_z' shape (H, W, 1), got {tuple(depth_z.shape)}.")

        depth_along_ray = _squeeze_view_batch(prediction["depth_along_ray"], "depth_along_ray")
        if depth_along_ray.ndim == 2:
            depth_along_ray = depth_along_ray.unsqueeze(-1)
        if depth_along_ray.ndim != 3 or depth_along_ray.shape[-1] != 1:
            raise ValueError(
                f"Expected 'depth_along_ray' shape (H, W, 1), got {tuple(depth_along_ray.shape)}."
            )

        conf = _squeeze_view_batch(prediction["conf"], "conf")
        if conf.ndim == 3 and conf.shape[-1] == 1:
            conf = conf[..., 0]
        if conf.ndim != 2:
            raise ValueError(f"Expected 'conf' shape (H, W), got {tuple(conf.shape)}.")

        intrinsics = _squeeze_view_batch(
            prediction["intrinsics"], "intrinsics", expected_ndim_after_squeeze=2
        )
        if tuple(intrinsics.shape) != (3, 3):
            raise ValueError(f"Expected 'intrinsics' shape (3, 3), got {tuple(intrinsics.shape)}.")

        camera_poses = _squeeze_view_batch(
            prediction["camera_poses"], "camera_poses", expected_ndim_after_squeeze=2
        )
        if tuple(camera_poses.shape) != (4, 4):
            raise ValueError(
                f"Expected 'camera_poses' shape (4, 4), got {tuple(camera_poses.shape)}."
            )

        normalized["depth_z"].append(_to_numpy(depth_z))
        normalized["depth_along_ray"].append(_to_numpy(depth_along_ray))
        normalized["conf"].append(_to_numpy(conf))
        normalized["intrinsics"].append(_to_numpy(intrinsics))
        normalized["camera_poses"].append(_to_numpy(camera_poses))
        valid_masks.append(_to_numpy(depth_z[..., 0] > 0))

    return {
        "depth_z": np.stack(normalized["depth_z"], axis=0),
        "depth_along_ray": np.stack(normalized["depth_along_ray"], axis=0),
        "conf": np.stack(normalized["conf"], axis=0),
        "intrinsics": np.stack(normalized["intrinsics"], axis=0),
        "camera_poses": np.stack(normalized["camera_poses"], axis=0),
        "valid_mask": np.stack(valid_masks, axis=0),
    }


def save_batch_output(
    save_path: Path,
    model_name: str,
    sequence_name: str,
    camera: str,
    frame_ids: list[str],
    window_start: int,
    window_end: int,
    predictions: dict[str, np.ndarray],
) -> None:
    np.savez(
        save_path,
        model=np.array(model_name),
        sequence=np.array(sequence_name),
        camera=np.array(camera),
        frame_ids=np.asarray(frame_ids),
        window_start=np.int64(window_start),
        window_end=np.int64(window_end),
        **predictions,
    )


def write_run_manifest(
    output_dir: Path,
    sequence_name: str,
    camera: str,
    config: KittiOdomInferenceConfig,
    model_spec: ModelSpec,
    resolved_device: torch.device,
) -> None:
    manifest_path = output_dir / "run_manifest.json"
    manifest = {
        "model": config.model,
        "sequence": sequence_name,
        "camera": camera,
        "resolved_device": str(resolved_device),
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
        "model_spec": {
            "resolution": model_spec.resolution,
            "norm_type": model_spec.norm_type,
            "patch_size": model_spec.patch_size,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def get_resolution_label(
    model_spec: ModelSpec,
    config: KittiOdomInferenceConfig,
) -> str:
    if config.long_side_resolution is not None:
        return f"long_side_{config.long_side_resolution}"
    return f"res_{model_spec.resolution}"


def run_sequence_camera_job(
    model: torch.nn.Module,
    model_spec: ModelSpec,
    sequence_dir: Path,
    camera: str,
    config: KittiOdomInferenceConfig,
    device: torch.device,
) -> None:
    camera_dir = sequence_dir / camera
    if not camera_dir.exists():
        raise FileNotFoundError(f"Missing camera directory: {camera_dir}")

    image_paths = sorted(camera_dir.glob("*.png"))
    if not image_paths:
        raise ValueError(f"No PNG images found in {camera_dir}")

    intrinsics = parse_kitti_odom_intrinsics(sequence_dir, camera)
    save_dir = output_dir_for_sequence(
        output_root=config.output_root,
        model_name=config.model,
        window_size=config.window_size,
        resolution_label=get_resolution_label(model_spec, config),
        camera=camera,
        sequence_name=sequence_dir.name,
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    write_run_manifest(
        output_dir=save_dir,
        sequence_name=sequence_dir.name,
        camera=camera,
        config=config,
        model_spec=model_spec,
        resolved_device=device,
    )

    batch_ranges = build_batch_ranges(len(image_paths), config.window_size)
    batch_iterator = tqdm(
        enumerate(batch_ranges),
        total=len(batch_ranges),
        desc=f"{sequence_dir.name}/{camera}",
        leave=False,
    )
    for batch_idx, (start, end) in batch_iterator:
        save_path = save_dir / f"{batch_idx}.npz"
        if save_path.exists() and not config.overwrite:
            continue

        batch_image_paths = image_paths[start:end]
        raw_views = build_raw_views_for_batch(batch_image_paths, intrinsics)
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
            sequence_name=sequence_dir.name,
            camera=camera,
            frame_ids=[path.stem for path in batch_image_paths],
            window_start=start,
            window_end=end,
            predictions=normalized_predictions,
        )


def run_kitti_odom_inference(config: KittiOdomInferenceConfig) -> None:
    device = resolve_device(config.device)
    model_spec = MODEL_SPECS[config.model]
    sequence_dirs = discover_sequence_dirs(config.dataset_root, config.sequences)
    jobs = [(sequence_dir, camera) for sequence_dir in sequence_dirs for camera in config.cameras]

    model = model_spec.loader(config, device)

    for sequence_dir, camera in tqdm(jobs, desc="sequence/camera jobs"):
        run_sequence_camera_job(
            model=model,
            model_spec=model_spec,
            sequence_dir=sequence_dir,
            camera=camera,
            config=config,
            device=device,
        )
