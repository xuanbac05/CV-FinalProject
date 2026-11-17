from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from path_config import DEFAULT_PATHS_CONFIG, load_project_paths, validate_celeba_paths

LOGGER = logging.getLogger("prepare_celeba_manifests")

REQUIRED_ATTRIBUTES = (
    "Black_Hair",
    "Brown_Hair",
    "Blond_Hair",
    "Smiling",
    "Eyeglasses",
    "Mustache",
)

PROMPTS: Dict[Tuple[str, str], Tuple[str, str]] = {
    ("Hair_Color", "black_to_blond"): (
        "A portrait photo of a person with black hair",
        "A portrait photo of a person with blond hair",
    ),
    ("Hair_Color", "brown_to_blond"): (
        "A portrait photo of a person with brown hair",
        "A portrait photo of a person with blond hair",
    ),
    ("Smiling", "add"): (
        "A portrait photo of a person with a neutral expression",
        "A portrait photo of a person smiling",
    ),
    ("Smiling", "remove"): (
        "A portrait photo of a person smiling",
        "A portrait photo of a person with a neutral expression",
    ),
    ("Eyeglasses", "add"): (
        "A portrait photo of a person without eyeglasses",
        "A portrait photo of a person wearing eyeglasses",
    ),
    ("Eyeglasses", "remove"): (
        "A portrait photo of a person wearing eyeglasses",
        "A portrait photo of a person without eyeglasses",
    ),
    ("Mustache", "add"): (
        "A portrait photo of a person without a mustache",
        "A portrait photo of a person with a mustache",
    ),
    ("Mustache", "remove"): (
        "A portrait photo of a person with a mustache",
        "A portrait photo of a person without a mustache",
    ),
}


def parse_partition_file(partition_path: Path) -> Dict[str, int]:
    """Parse CelebA list_eval_partition.txt into {filename: split_int}."""
    partitions: Dict[str, int] = {}
    with partition_path.open("r", encoding="utf-8") as f:
        for line_num, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(
                    f"Malformed line {line_num} in partition file {partition_path}: {raw_line!r}"
                )
            filename, part_str = parts
            try:
                part_id = int(part_str)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid partition integer on line {line_num} in {partition_path}: {part_str!r}"
                ) from exc
            if part_id not in (0, 1, 2):
                raise ValueError(
                    f"Unexpected partition value {part_id} on line {line_num} in {partition_path} (expected 0, 1, or 2)."
                )
            partitions[filename] = part_id
    if not partitions:
        raise ValueError(f"Partition file {partition_path} is empty.")
    return partitions


def parse_attributes_file(attributes_path: Path) -> Dict[str, Dict[str, int]]:
    """Parse official CelebA list_attr_celeba.txt into {filename: {attr_name: -1 | 1}}."""
    attributes: Dict[str, Dict[str, int]] = {}
    with attributes_path.open("r", encoding="utf-8") as f:
        first_line = f.readline()
        if not first_line:
            raise ValueError(f"Attributes file {attributes_path} is empty.")
        first_tokens = first_line.strip().split()

        # Official list_attr_celeba.txt has row count on line 1 and header on line 2
        if len(first_tokens) == 1 and first_tokens[0].isdigit():
            header_line = f.readline()
            if not header_line:
                raise ValueError(f"Attributes file {attributes_path} missing attribute header line.")
            attr_names = header_line.strip().split()
            start_line_num = 3
        else:
            attr_names = first_tokens
            start_line_num = 2

        missing_cols = [col for col in REQUIRED_ATTRIBUTES if col not in attr_names]
        if missing_cols:
            raise ValueError(
                f"Attributes file {attributes_path} is missing required attribute columns: {missing_cols}"
            )

        req_indices = {col: attr_names.index(col) for col in REQUIRED_ATTRIBUTES}
        num_attrs = len(attr_names)

        for line_num, raw_line in enumerate(f, start=start_line_num):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != num_attrs + 1:
                raise ValueError(
                    f"Malformed attribute row at line {line_num} in {attributes_path}: "
                    f"expected {num_attrs + 1} tokens, got {len(parts)}."
                )
            filename = parts[0]
            values = parts[1:]
            row_dict: Dict[str, int] = {}
            for col, idx in req_indices.items():
                val = int(values[idx])
                if val not in (-1, 1):
                    raise ValueError(
                        f"Unexpected label {val} for {col} on {filename} (line {line_num}); expected -1 or 1."
                    )
                row_dict[col] = val
            attributes[filename] = row_dict

    if not attributes:
        raise ValueError(f"No attribute records parsed from {attributes_path}.")
    return attributes


def select_candidates(
    eligible_filenames: List[str],
    count: int,
    used_filenames: Set[str],
    rng: random.Random,
    task_desc: str,
) -> List[str]:
    """Deterministically sample `count` unused filenames without replacement."""
    available = [fn for fn in sorted(eligible_filenames) if fn not in used_filenames]
    if len(available) < count:
        raise RuntimeError(
            f"Allocation failed for '{task_desc}': requested {count} images, "
            f"but only {len(available)} eligible unused images exist."
        )
    pool = list(available)
    rng.shuffle(pool)
    selected = sorted(pool[:count])
    used_filenames.update(selected)
    return selected


def build_split_manifest(
    split_name: str,
    split_id: int,
    target_per_family: int,
    images_dir: Path,
    partitions: Dict[str, int],
    attributes: Dict[str, Dict[str, int]],
    global_used_filenames: Set[str],
    seed: int,
) -> List[Dict[str, str]]:
    """Build deterministic editing manifest for a single split (validation or test)."""
    if target_per_family % 2 != 0:
        raise ValueError(f"target_per_family must be even to balance add/remove, got {target_per_family}.")
    half_target = target_per_family // 2

    # Collect valid filenames in this partition whose image files exist on disk
    split_filenames: List[str] = []
    for filename in sorted(attributes.keys()):
        if partitions.get(filename) != split_id:
            continue
        if filename in global_used_filenames:
            continue
        img_path = images_dir / filename
        if img_path.is_file():
            split_filenames.append(filename)

    if not split_filenames:
        raise RuntimeError(
            f"No existing source images found in {images_dir} for split '{split_name}' (partition={split_id})."
        )

    rng = random.Random(seed)
    records: List[Dict[str, str]] = []

    def _add_records(selected_files: List[str], attribute: str, direction: str) -> None:
        src_prompt, tgt_prompt = PROMPTS[(attribute, direction)]
        attr_slug = attribute.lower()
        for fn in selected_files:
            stem = Path(fn).stem
            sample_id = f"{split_name}_{attr_slug}_{direction}_{stem}"
            records.append(
                {
                    "sample_id": sample_id,
                    "split": split_name,
                    "filename": fn,
                    "image_path": str((images_dir / fn).resolve()),
                    "attribute": attribute,
                    "direction": direction,
                    "source_prompt": src_prompt,
                    "target_prompt": tgt_prompt,
                }
            )

    # 1. Hair color: Black or brown hair -> blond hair (Blond_Hair == -1)
    # Sample rare attributes (like Mustache remove or Eyeglasses remove) first or in deterministic order?
    # Note: Mustache==1 and Eyeglasses==1 are rarer in CelebA (~4-6% of images), so selecting rarer
    # positive subsets before large subsets avoids any chance of starving rare classes, while remaining 100% deterministic.
    # Let's prepare the candidate lists first:
    mustache_remove_pool = [fn for fn in split_filenames if attributes[fn]["Mustache"] == 1]
    mustache_add_pool = [fn for fn in split_filenames if attributes[fn]["Mustache"] == -1]

    eyeglasses_remove_pool = [fn for fn in split_filenames if attributes[fn]["Eyeglasses"] == 1]
    eyeglasses_add_pool = [fn for fn in split_filenames if attributes[fn]["Eyeglasses"] == -1]

    smiling_remove_pool = [fn for fn in split_filenames if attributes[fn]["Smiling"] == 1]
    smiling_add_pool = [fn for fn in split_filenames if attributes[fn]["Smiling"] == -1]

    black_hair_pool = [
        fn
        for fn in split_filenames
        if attributes[fn]["Black_Hair"] == 1
        and attributes[fn]["Brown_Hair"] == -1
        and attributes[fn]["Blond_Hair"] == -1
    ]
    brown_hair_pool = [
        fn
        for fn in split_filenames
        if attributes[fn]["Brown_Hair"] == 1
        and attributes[fn]["Black_Hair"] == -1
        and attributes[fn]["Blond_Hair"] == -1
    ]

    # Sample from rarest to most common to avoid depleting small positive pools, then emit in canonical family order
    sel_mustache_rem = select_candidates(
        mustache_remove_pool, half_target, global_used_filenames, rng, f"{split_name}/Mustache/remove"
    )
    sel_eyeglasses_rem = select_candidates(
        eyeglasses_remove_pool, half_target, global_used_filenames, rng, f"{split_name}/Eyeglasses/remove"
    )
    sel_black_hair = select_candidates(
        black_hair_pool, half_target, global_used_filenames, rng, f"{split_name}/Hair_Color/black_to_blond"
    )
    sel_brown_hair = select_candidates(
        brown_hair_pool, half_target, global_used_filenames, rng, f"{split_name}/Hair_Color/brown_to_blond"
    )
    sel_smiling_rem = select_candidates(
        smiling_remove_pool, half_target, global_used_filenames, rng, f"{split_name}/Smiling/remove"
    )
    sel_smiling_add = select_candidates(
        smiling_add_pool, half_target, global_used_filenames, rng, f"{split_name}/Smiling/add"
    )
    sel_eyeglasses_add = select_candidates(
        eyeglasses_add_pool, half_target, global_used_filenames, rng, f"{split_name}/Eyeglasses/add"
    )
    sel_mustache_add = select_candidates(
        mustache_add_pool, half_target, global_used_filenames, rng, f"{split_name}/Mustache/add"
    )

    # Emit in canonical order: Hair_Color, Smiling, Eyeglasses, Mustache
    _add_records(sel_black_hair, "Hair_Color", "black_to_blond")
    _add_records(sel_brown_hair, "Hair_Color", "brown_to_blond")
    _add_records(sel_smiling_add, "Smiling", "add")
    _add_records(sel_smiling_rem, "Smiling", "remove")
    _add_records(sel_eyeglasses_add, "Eyeglasses", "add")
    _add_records(sel_eyeglasses_rem, "Eyeglasses", "remove")
    _add_records(sel_mustache_add, "Mustache", "add")
    _add_records(sel_mustache_rem, "Mustache", "remove")

    return records


def write_jsonl(records: List[Dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate deterministic CelebA validation (80) and test (120) manifests for ChordEdit."
    )
    parser.add_argument(
        "--paths-config",
        type=str,
        default=str(DEFAULT_PATHS_CONFIG),
        help="Path to JSON configuration file (default: configs/paths.local.json).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic sampling seed (default: 42).",
    )
    parser.add_argument(
        "--manifest-dir",
        type=str,
        default=None,
        help="Directory to write validation.jsonl and test.jsonl (default: <output_dir>/manifests).",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = parse_args()

    paths = load_project_paths(args.paths_config)
    validate_celeba_paths(paths)

    LOGGER.info("Reading CelebA partitions from %s", paths.celeba.partition_file)
    partitions = parse_partition_file(paths.celeba.partition_file)

    LOGGER.info("Reading CelebA attributes from %s", paths.celeba.attributes_file)
    attributes = parse_attributes_file(paths.celeba.attributes_file)

    global_used_filenames: Set[str] = set()

    # Validation: 80 requests (20 per family) from partition 1
    val_records = build_split_manifest(
        split_name="validation",
        split_id=1,
        target_per_family=20,
        images_dir=paths.celeba.images_dir,
        partitions=partitions,
        attributes=attributes,
        global_used_filenames=global_used_filenames,
        seed=args.seed,
    )

    # Test: 120 requests (30 per family) from partition 2
    test_records = build_split_manifest(
        split_name="test",
        split_id=2,
        target_per_family=30,
        images_dir=paths.celeba.images_dir,
        partitions=partitions,
        attributes=attributes,
        global_used_filenames=global_used_filenames,
        seed=args.seed + 1,
    )

    # Sanity check: no duplicates within or across manifests
    val_files = {r["filename"] for r in val_records}
    test_files = {r["filename"] for r in test_records}
    assert len(val_files) == len(val_records) == 80, "Validation manifest must contain 80 unique images."
    assert len(test_files) == len(test_records) == 120, "Test manifest must contain 120 unique images."
    assert val_files.isdisjoint(test_files), "Validation and test manifests must not overlap."

    manifest_dir = (
        Path(args.manifest_dir).expanduser().resolve()
        if args.manifest_dir
        else (paths.output_dir / "manifests").resolve()
    )
    val_manifest_path = manifest_dir / "validation.jsonl"
    test_manifest_path = manifest_dir / "test.jsonl"

    write_jsonl(val_records, val_manifest_path)
    write_jsonl(test_records, test_manifest_path)

    LOGGER.info("Wrote %d validation requests to %s", len(val_records), val_manifest_path)
    LOGGER.info("Wrote %d test requests to %s", len(test_records), test_manifest_path)


if __name__ == "__main__":
    main()

