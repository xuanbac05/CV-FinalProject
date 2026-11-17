from __future__ import annotations

import argparse
import json
import logging
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import path_config first so offline env vars are set before diffusers/transformers import
from path_config import (
    DEFAULT_PATHS_CONFIG,
    load_project_paths,
    validate_sd_turbo_paths,
)

import diffusers
import numpy as np
import torch
import transformers
from PIL import Image

from pipeline_chord import ChordEditPipeline

LOGGER = logging.getLogger("run_baseline_celeba")

BASE_CONFIG: Dict[str, Any] = {
    "noise_samples": 1,
    "n_steps": 1,
    "t_start": 0.90,
    "t_end": 0.30,
    "t_delta": 0.15,
    "step_scale": 1.0,
    "cleanup": True,
}

REFINEMENT_CONDITIONS: Tuple[Tuple[str, bool], ...] = (
    ("no_refinement", False),
    ("refinement", True),
)

DEFAULT_SEED = 42
DEFAULT_IMAGE_SIZE = 512


def get_git_commit() -> Optional[str]:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return res.stdout.strip() or None
    except Exception:
        return None


def collect_environment_metadata(sd_turbo_dir: Path, device_str: str) -> Dict[str, Any]:
    gpu_name: Optional[str] = None
    if torch.cuda.is_available() and device_str.startswith("cuda"):
        try:
            idx = torch.device(device_str).index or 0
            gpu_name = torch.cuda.get_device_name(idx)
        except Exception:
            gpu_name = torch.cuda.get_device_name(0)

    # Check if model_index.json or unet/config.json has version metadata
    checkpoint_info: Dict[str, Any] = {"sd_turbo_dir": str(sd_turbo_dir)}
    unet_cfg_path = sd_turbo_dir / "unet" / "config.json"
    if unet_cfg_path.is_file():
        try:
            with unet_cfg_path.open("r", encoding="utf-8") as f:
                unet_cfg = json.load(f)
            if "_diffusers_version" in unet_cfg:
                checkpoint_info["unet_diffusers_version"] = unet_cfg["_diffusers_version"]
            if "_name_or_path" in unet_cfg:
                checkpoint_info["unet_name_or_path"] = unet_cfg["_name_or_path"]
        except Exception:
            pass

    return {
        "git_commit": get_git_commit(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "diffusers_version": diffusers.__version__,
        "transformers_version": transformers.__version__,
        "device": device_str,
        "gpu_name": gpu_name,
        "precision": "fp32",
        "checkpoint": checkpoint_info,
    }


def reset_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_manifest(manifest_path: Path) -> List[Dict[str, Any]]:
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Manifest file not found: {manifest_path}\n"
            "Run 'python scripts/prepare_celeba_manifests.py' first to generate the validation and test manifests."
        )
    records: List[Dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line_num, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_num} in {manifest_path}: {exc}") from exc
            for req_key in (
                "sample_id",
                "split",
                "filename",
                "image_path",
                "attribute",
                "direction",
                "source_prompt",
                "target_prompt",
            ):
                if req_key not in rec:
                    raise ValueError(f"Manifest record on line {line_num} missing key '{req_key}'.")
            records.append(rec)
    if not records:
        raise ValueError(f"Manifest {manifest_path} contains no records.")
    return records


def load_existing_results(results_path: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    existing: Dict[Tuple[str, str], Dict[str, Any]] = {}
    if not results_path.is_file():
        return existing
    with results_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                key = (rec["sample_id"], rec["refinement_condition"])
                existing[key] = rec
            except Exception:
                continue
    return existing


def write_all_results(results_path: Path, results_map: Dict[Tuple[str, str], Dict[str, Any]]) -> None:
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", encoding="utf-8") as f:
        for rec in results_map.values():
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Original ChordEdit baseline on CelebA with proximal refinement off and on."
    )
    parser.add_argument(
        "--paths-config",
        type=str,
        default=str(DEFAULT_PATHS_CONFIG),
        help="Path to JSON configuration file (default: configs/paths.local.json).",
    )
    parser.add_argument(
        "--split",
        choices=["validation", "test"],
        default="validation",
        help="Dataset split to run if --manifest is not explicitly provided (default: validation).",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Explicit path to a JSONL manifest file. Overrides --split default manifest location.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N requests from the manifest (useful for sanity checks).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for ChordEdit noise sampling (default: 42).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device override (e.g. 'cuda' or 'cpu'). Defaults to cuda if available.",
    )
    parser.add_argument(
        "--overwrite",
        "--fresh-run",
        dest="overwrite",
        action="store_true",
        help="Overwrite existing edited images and metadata instead of resuming.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = parse_args()

    paths = load_project_paths(args.paths_config)
    component_paths = validate_sd_turbo_paths(paths)

    if args.manifest is not None:
        manifest_path = Path(args.manifest).expanduser().resolve()
    else:
        manifest_path = (paths.output_dir / "manifests" / f"{args.split}.jsonl").resolve()

    records = load_manifest(manifest_path)
    split_name = records[0]["split"]

    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError(f"--limit must be a positive integer, got {args.limit}.")
        records = records[: args.limit]

    # Validate that all source images in the selected slice exist before loading the model
    missing_images = [r["image_path"] for r in records if not Path(r["image_path"]).is_file()]
    if missing_images:
        preview = "\n  - ".join(missing_images[:5])
        raise FileNotFoundError(
            f"{len(missing_images)} source image(s) referenced in {manifest_path} do not exist on disk:\n"
            f"  - {preview}"
        )

    device_str = args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device_str.startswith("cuda") and torch.cuda.is_available()

    split_output_dir = (paths.output_dir / "baseline" / split_name).resolve()
    no_ref_dir = split_output_dir / "no_refinement"
    ref_dir = split_output_dir / "refinement"
    results_path = split_output_dir / "results.jsonl"

    no_ref_dir.mkdir(parents=True, exist_ok=True)
    ref_dir.mkdir(parents=True, exist_ok=True)

    results_map: Dict[Tuple[str, str], Dict[str, Any]] = (
        {} if args.overwrite else load_existing_results(results_path)
    )

    env_meta = collect_environment_metadata(paths.sd_turbo_dir.resolve(), device_str)

    LOGGER.info(
        "Loading SD-Turbo & ChordEditPipeline from %s on device=%s (fp32)...",
        paths.sd_turbo_dir,
        device_str,
    )
    pipeline = ChordEditPipeline.from_local_weights(
        component_paths=component_paths,
        default_edit_config=dict(BASE_CONFIG),
        device=device_str,
        torch_dtype=torch.float32,
        image_size=DEFAULT_IMAGE_SIZE,
        use_center_crop=True,
        compute_dtype=torch.float32,
        use_attention_mask=False,
        use_safety_checker=False,
    )

    total_requests = len(records)
    completed_pairs = 0
    skipped_pairs = 0
    failed_pairs = 0

    for idx, item in enumerate(records, start=1):
        sample_id = item["sample_id"]
        src_img_path = Path(item["image_path"])

        try:
            with Image.open(src_img_path) as raw_img:
                source_image = raw_img.convert("RGB")
        except Exception as exc:
            LOGGER.error("[%d/%d] Failed to open source image %s: %s", idx, total_requests, src_img_path, exc)
            for cond_name, cleanup_flag in REFINEMENT_CONDITIONS:
                cond_cfg = {**BASE_CONFIG, "cleanup": cleanup_flag}
                out_path = (split_output_dir / cond_name / f"{sample_id}.png").resolve()
                results_map[(sample_id, cond_name)] = {
                    "sample_id": sample_id,
                    "split": item["split"],
                    "filename": item["filename"],
                    "source_image_path": str(src_img_path),
                    "attribute": item["attribute"],
                    "direction": item["direction"],
                    "source_prompt": item["source_prompt"],
                    "target_prompt": item["target_prompt"],
                    "refinement_condition": cond_name,
                    "cleanup": cleanup_flag,
                    "edit_config": cond_cfg,
                    "seed": int(args.seed),
                    "image_size": DEFAULT_IMAGE_SIZE,
                    "output_image_path": str(out_path),
                    "runtime_sec": None,
                    "status": "failure",
                    "error_message": f"Source image read error: {exc}",
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "environment": env_meta,
                }
                failed_pairs += 1
            write_all_results(results_path, results_map)
            continue

        for cond_name, cleanup_flag in REFINEMENT_CONDITIONS:
            out_path = (split_output_dir / cond_name / f"{sample_id}.png").resolve()
            key = (sample_id, cond_name)

            if (
                not args.overwrite
                and key in results_map
                and results_map[key].get("status") == "success"
                and out_path.is_file()
            ):
                skipped_pairs += 1
                continue

            cond_cfg = {**BASE_CONFIG, "cleanup": cleanup_flag}
            reset_all_seeds(int(args.seed))

            if use_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            try:
                output = pipeline(
                    image=source_image,
                    source_prompt=item["source_prompt"],
                    target_prompt=item["target_prompt"],
                    edit_config=cond_cfg,
                    seed=int(args.seed),
                    output_type="pil",
                )
                if use_cuda:
                    torch.cuda.synchronize()
                runtime_sec = round(time.perf_counter() - t0, 6)

                edited_img = output.images[0]
                edited_img.save(out_path, format="PNG")

                results_map[key] = {
                    "sample_id": sample_id,
                    "split": item["split"],
                    "filename": item["filename"],
                    "source_image_path": str(src_img_path),
                    "attribute": item["attribute"],
                    "direction": item["direction"],
                    "source_prompt": item["source_prompt"],
                    "target_prompt": item["target_prompt"],
                    "refinement_condition": cond_name,
                    "cleanup": cleanup_flag,
                    "edit_config": cond_cfg,
                    "seed": int(args.seed),
                    "image_size": DEFAULT_IMAGE_SIZE,
                    "output_image_path": str(out_path),
                    "runtime_sec": runtime_sec,
                    "status": "success",
                    "error_message": None,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "environment": env_meta,
                }
                completed_pairs += 1
            except Exception as exc:
                if use_cuda:
                    torch.cuda.synchronize()
                runtime_sec = round(time.perf_counter() - t0, 6)
                LOGGER.error(
                    "[%d/%d] Inference failed for %s (%s): %s",
                    idx,
                    total_requests,
                    sample_id,
                    cond_name,
                    exc,
                )
                results_map[key] = {
                    "sample_id": sample_id,
                    "split": item["split"],
                    "filename": item["filename"],
                    "source_image_path": str(src_img_path),
                    "attribute": item["attribute"],
                    "direction": item["direction"],
                    "source_prompt": item["source_prompt"],
                    "target_prompt": item["target_prompt"],
                    "refinement_condition": cond_name,
                    "cleanup": cleanup_flag,
                    "edit_config": cond_cfg,
                    "seed": int(args.seed),
                    "image_size": DEFAULT_IMAGE_SIZE,
                    "output_image_path": str(out_path),
                    "runtime_sec": runtime_sec,
                    "status": "failure",
                    "error_message": str(exc),
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "environment": env_meta,
                }
                failed_pairs += 1

            write_all_results(results_path, results_map)

        LOGGER.info(
            "Progress: %d/%d requests | saved=%d, skipped=%d, failed=%d",
            idx,
            total_requests,
            completed_pairs,
            skipped_pairs,
            failed_pairs,
        )

    LOGGER.info(
        "Baseline run finished for split='%s'. Saved=%d, Skipped=%d, Failed=%d. Metadata: %s",
        split_name,
        completed_pairs,
        skipped_pairs,
        failed_pairs,
        results_path,
    )


if __name__ == "__main__":
    main()

