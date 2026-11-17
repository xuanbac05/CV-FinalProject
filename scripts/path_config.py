from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

# Enforce local-only / offline loading for Hugging Face libraries
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["DIFFUSERS_OFFLINE"] = "1"

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PATHS_CONFIG = REPO_ROOT / "configs" / "paths.local.json"

REQUIRED_MODEL_SUBDIRS: Dict[str, str] = {
    "unet_path": "unet",
    "scheduler_path": "scheduler",
    "vae_path": "vae",
    "tokenizer_path": "tokenizer",
    "text_encoder_path": "text_encoder",
}


@dataclass(frozen=True)
class CelebAPaths:
    images_dir: Path
    attributes_file: Path
    partition_file: Path


@dataclass(frozen=True)
class ProjectPaths:
    config_path: Path
    celeba: CelebAPaths
    sd_turbo_dir: Path
    output_dir: Path


def load_project_paths(config_path: str | Path | None = None) -> ProjectPaths:
    """Load path configuration from JSON without triggering any downloads."""
    resolved_config = (
        Path(config_path).expanduser().resolve()
        if config_path is not None
        else DEFAULT_PATHS_CONFIG
    )
    if not resolved_config.exists():
        example_path = REPO_ROOT / "configs" / "paths.example.json"
        raise FileNotFoundError(
            f"Paths configuration file not found: {resolved_config}\n"
            f"Please copy '{example_path}' to '{resolved_config}' and set your mounted Kaggle paths."
        )

    with resolved_config.open("r", encoding="utf-8") as f:
        try:
            raw = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in paths configuration {resolved_config}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ValueError(f"Expected JSON object in {resolved_config}, got {type(raw).__name__}.")

    celeba_raw = raw.get("celeba")
    if not isinstance(celeba_raw, dict):
        raise ValueError(f"Missing or invalid 'celeba' section in {resolved_config}.")

    for key in ("images_dir", "attributes_file", "partition_file"):
        if not celeba_raw.get(key):
            raise ValueError(f"Missing required key 'celeba.{key}' in {resolved_config}.")

    model_raw = raw.get("model")
    if not isinstance(model_raw, dict) or not model_raw.get("sd_turbo_dir"):
        raise ValueError(f"Missing required key 'model.sd_turbo_dir' in {resolved_config}.")

    output_dir_raw = raw.get("output_dir")
    if not output_dir_raw:
        raise ValueError(f"Missing required key 'output_dir' in {resolved_config}.")

    return ProjectPaths(
        config_path=resolved_config,
        celeba=CelebAPaths(
            images_dir=Path(celeba_raw["images_dir"]).expanduser(),
            attributes_file=Path(celeba_raw["attributes_file"]).expanduser(),
            partition_file=Path(celeba_raw["partition_file"]).expanduser(),
        ),
        sd_turbo_dir=Path(model_raw["sd_turbo_dir"]).expanduser(),
        output_dir=Path(output_dir_raw).expanduser(),
    )


def validate_celeba_paths(paths: ProjectPaths) -> None:
    """Verify CelebA dataset directories and annotation files exist locally."""
    missing = []
    if not paths.celeba.images_dir.exists():
        missing.append(f"celeba.images_dir (not found): {paths.celeba.images_dir}")
    elif not paths.celeba.images_dir.is_dir():
        missing.append(f"celeba.images_dir (not a directory): {paths.celeba.images_dir}")

    if not paths.celeba.attributes_file.exists():
        missing.append(f"celeba.attributes_file (not found): {paths.celeba.attributes_file}")
    elif not paths.celeba.attributes_file.is_file():
        missing.append(f"celeba.attributes_file (not a file): {paths.celeba.attributes_file}")

    if not paths.celeba.partition_file.exists():
        missing.append(f"celeba.partition_file (not found): {paths.celeba.partition_file}")
    elif not paths.celeba.partition_file.is_file():
        missing.append(f"celeba.partition_file (not a file): {paths.celeba.partition_file}")

    if missing:
        details = "\n  - ".join(missing)
        raise FileNotFoundError(
            f"CelebA path validation failed (configured in {paths.config_path}):\n  - {details}\n"
            "Ensure the CelebA Kaggle Dataset is attached and paths in configs/paths.local.json match '/kaggle/input/...'."
        )


def validate_sd_turbo_paths(paths: ProjectPaths) -> Dict[str, str]:
    """Verify SD-Turbo model root and all five required component subdirectories exist locally."""
    sd_root = paths.sd_turbo_dir
    if "ACTUAL_KAGGLE_MODEL_PATH" in str(sd_root):
        raise FileNotFoundError(
            f"model.sd_turbo_dir in {paths.config_path} is still set to placeholder '{sd_root}'.\n"
            "Please update it to the actual mounted Kaggle Model path under /kaggle/input/."
        )

    if not sd_root.exists():
        raise FileNotFoundError(
            f"SD-Turbo model directory not found: {sd_root}\n"
            f"(configured in {paths.config_path}). Do not download weights; attach the SD-Turbo Kaggle Model and update 'model.sd_turbo_dir'."
        )
    if not sd_root.is_dir():
        raise NotADirectoryError(f"model.sd_turbo_dir is not a directory: {sd_root}")

    component_paths: Dict[str, str] = {}
    missing_components = []
    for key, subdir in REQUIRED_MODEL_SUBDIRS.items():
        comp_dir = sd_root / subdir
        if not comp_dir.exists() or not comp_dir.is_dir():
            missing_components.append(f"{subdir}/ (expected at {comp_dir})")
        else:
            component_paths[key] = str(comp_dir.resolve())

    if missing_components:
        details = "\n  - ".join(missing_components)
        raise FileNotFoundError(
            f"SD-Turbo directory '{sd_root}' is missing required component subdirectories:\n  - {details}\n"
            "Expected subdirectories: unet/, scheduler/, vae/, tokenizer/, text_encoder/."
        )

    return component_paths

