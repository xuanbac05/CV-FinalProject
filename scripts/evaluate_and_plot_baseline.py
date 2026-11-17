from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import os
from path_config import DEFAULT_PATHS_CONFIG, load_project_paths, validate_celeba_paths
from prepare_celeba_manifests import parse_attributes_file, parse_partition_file

# Allow evaluator models (CLIP feature extractor & ArcFace) to be fetched if not cached locally
os.environ.pop("HF_HUB_OFFLINE", None)
os.environ.pop("TRANSFORMERS_OFFLINE", None)
os.environ.pop("DIFFUSERS_OFFLINE", None)

LOGGER = logging.getLogger("evaluate_baseline")

TARGET_ATTRS = ("Blond_Hair", "Smiling", "Eyeglasses", "Mustache")

DIRECTIONS_ORDER = [
    ("Hair_Color", "black_to_blond", "Blond_Hair", 1),
    ("Hair_Color", "brown_to_blond", "Blond_Hair", 1),
    ("Smiling", "add", "Smiling", 1),
    ("Smiling", "remove", "Smiling", -1),
    ("Eyeglasses", "add", "Eyeglasses", 1),
    ("Eyeglasses", "remove", "Eyeglasses", -1),
    ("Mustache", "add", "Mustache", 1),
    ("Mustache", "remove", "Mustache", -1),
]


def center_crop_512(img_path: Path) -> Image.Image:
    with Image.open(img_path) as im:
        rgb = im.convert("RGB")
    side = min(rgb.size)
    try:
        resample = Image.Resampling.LANCZOS
    except AttributeError:
        resample = Image.LANCZOS
    cropped = ImageOps.fit(rgb, (side, side), method=resample, centering=(0.5, 0.5))
    return cropped.resize((512, 512), resample=resample)


class FrozenCLIPAttributeClassifier:
    """Frozen CLIP vision backbone + 4-attribute linear head trained on CelebA train split (partition=0)
    and threshold-calibrated on validation source images."""

    def __init__(self, device: str = "cuda", clip_id: str = "openai/clip-vit-base-patch32") -> None:
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.clip_id = clip_id
        self.processor = CLIPImageProcessor.from_pretrained(clip_id)
        self.backbone = CLIPVisionModelWithProjection.from_pretrained(clip_id).to(self.device)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

        embed_dim = self.backbone.config.projection_dim
        self.head = nn.Linear(embed_dim, len(TARGET_ATTRS)).to(self.device)
        self.thresholds: Dict[str, float] = {attr: 0.5 for attr in TARGET_ATTRS}

    @torch.no_grad()
    def extract_features(self, images: List[Image.Image], batch_size: int = 64) -> torch.Tensor:
        feats = []
        for i in range(0, len(images), batch_size):
            batch_imgs = images[i : i + batch_size]
            inputs = self.processor(images=batch_imgs, return_tensors="pt").to(self.device)
            embeds = self.backbone(**inputs).image_embeds
            embeds = embeds / embeds.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            feats.append(embeds.detach())
        return torch.cat(feats, dim=0)

    def fit_or_load(
        self,
        ckpt_path: Path,
        images_dir: Path,
        partitions: Dict[str, int],
        attributes: Dict[str, Dict[str, int]],
        val_records: List[Dict[str, Any]],
        seed: int = 42,
    ) -> Dict[str, float]:
        if ckpt_path.is_file():
            LOGGER.info("Loading frozen attribute classifier checkpoint from %s", ckpt_path)
            state = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            self.head.load_state_dict(state["head_state_dict"])
            self.thresholds = state["thresholds"]
            self.head.eval()
            return state["val_agreement"]

        LOGGER.info("Training frozen-CLIP attribute classifier head on CelebA training partition (partition=0)...")
        rng = np.random.default_rng(seed)
        train_files = [fn for fn, pid in sorted(partitions.items()) if pid == 0 and (images_dir / fn).is_file()]

        # Select balanced training subset (up to 600 positive + 600 negative per attribute, deduplicated)
        selected_set = set()
        for attr in TARGET_ATTRS:
            pos = [fn for fn in train_files if attributes[fn][attr] == 1]
            neg = [fn for fn in train_files if attributes[fn][attr] == -1]
            rng.shuffle(pos)
            rng.shuffle(neg)
            selected_set.update(pos[:600])
            selected_set.update(neg[:600])

        sampled_train = sorted(selected_set)
        LOGGER.info("Extracting CLIP features for %d CelebA train images...", len(sampled_train))
        train_imgs = [center_crop_512(images_dir / fn) for fn in sampled_train]
        X_train = self.extract_features(train_imgs, batch_size=64)
        Y_train = torch.tensor(
            [[1.0 if attributes[fn][attr] == 1 else 0.0 for attr in TARGET_ATTRS] for fn in sampled_train],
            dtype=torch.float32,
            device=self.device,
        )

        # Compute positive class weights to handle class imbalance
        pos_counts = Y_train.sum(dim=0).clamp(min=1.0)
        neg_counts = (len(Y_train) - pos_counts).clamp(min=1.0)
        pos_weight = neg_counts / pos_counts

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(self.head.parameters(), lr=5e-2, weight_decay=1e-3)

        self.head.train()
        for _ in range(300):
            optimizer.zero_grad()
            logits = self.head(X_train)
            loss = criterion(logits, Y_train)
            loss.backward()
            optimizer.step()
        self.head.eval()

        # Validate agreement and calibrate thresholds on validation source images
        val_imgs = [center_crop_512(images_dir / r["filename"]) for r in val_records]
        X_val = self.extract_features(val_imgs, batch_size=32)
        with torch.no_grad():
            val_probs = torch.sigmoid(self.head(X_val)).cpu().numpy()

        val_agreement: Dict[str, float] = {}
        for idx, attr in enumerate(TARGET_ATTRS):
            y_true = np.array([1 if attributes[r["filename"]][attr] == 1 else 0 for r in val_records])
            best_thr = 0.5
            best_acc = 0.0
            for thr in np.linspace(0.30, 0.70, 41):
                preds = (val_probs[:, idx] >= thr).astype(int)
                acc = float((preds == y_true).mean())
                if acc > best_acc:
                    best_acc = acc
                    best_thr = float(round(thr, 3))
            self.thresholds[attr] = best_thr
            val_agreement[attr] = round(best_acc * 100.0, 2)
            LOGGER.info(
                "Validation source-label agreement for %s: %.2f%% (threshold=%.3f)",
                attr,
                val_agreement[attr],
                best_thr,
            )

        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "clip_id": self.clip_id,
                "head_state_dict": self.head.state_dict(),
                "thresholds": self.thresholds,
                "val_agreement": val_agreement,
            },
            ckpt_path,
        )
        return val_agreement

    @torch.no_grad()
    def predict_probs(self, images: List[Image.Image]) -> np.ndarray:
        feats = self.extract_features(images, batch_size=32)
        probs = torch.sigmoid(self.head(feats)).cpu().numpy()
        return probs


class ArcFaceEvaluator:
    """Frozen ArcFace recognition model (insightface buffalo_l w600k_r50.onnx) for source-output cosine similarity."""

    def __init__(self) -> None:
        from insightface.app import FaceAnalysis

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.app = FaceAnalysis(name="buffalo_l", providers=providers)
        self.app.prepare(ctx_id=0, det_size=(256, 256))

    def get_embedding(self, pil_img: Image.Image) -> Optional[np.ndarray]:
        rgb = np.asarray(pil_img.convert("RGB"))
        bgr = rgb[:, :, ::-1].copy()
        faces = self.app.get(bgr)
        if not faces:
            return None
        # Pick largest detected face near center
        faces_sorted = sorted(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            reverse=True,
        )
        emb = faces_sorted[0].normed_embedding
        if emb is None:
            raw_emb = faces_sorted[0].embedding
            if raw_emb is None:
                return None
            emb = raw_emb / max(np.linalg.norm(raw_emb), 1e-8)
        return np.asarray(emb, dtype=np.float32)


def generate_qualitative_figures(
    test_records: List[Dict[str, Any]],
    no_ref_dir: Path,
    ref_dir: Path,
    images_dir: Path,
    figures_dir: Path,
    samples_per_page: int = 4,
) -> List[Path]:
    """Generate 1 full-page 3-column figure per editing prompt (4 rows x 3 columns)."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: List[Path] = []

    for attr, direction, _, _ in DIRECTIONS_ORDER:
        subset = [r for r in test_records if r["attribute"] == attr and r["direction"] == direction]
        chosen = subset[:samples_per_page]
        if not chosen:
            continue

        n_rows = len(chosen)
        fig, axes = plt.subplots(n_rows, 3, figsize=(11.5, 3.6 * n_rows))
        if n_rows == 1:
            axes = [axes]

        src_prompt = chosen[0]["source_prompt"]
        tgt_prompt = chosen[0]["target_prompt"]
        fig.suptitle(
            f"Attribute: {attr} ({direction})\n"
            f"Source: \"{src_prompt}\"  →  Target: \"{tgt_prompt}\"",
            fontsize=12.5,
            fontweight="bold",
            y=0.995,
        )

        col_headers = [
            "Original",
            "ChordEdit (w/o prox)",
            "ChordEdit (w/ prox)",
        ]

        for row_idx, rec in enumerate(chosen):
            sid = rec["sample_id"]
            fn = rec["filename"]
            orig_img = center_crop_512(images_dir / fn)
            wo_prox_img = Image.open(no_ref_dir / f"{sid}.png").convert("RGB")
            w_prox_img = Image.open(ref_dir / f"{sid}.png").convert("RGB")

            row_imgs = [orig_img, wo_prox_img, w_prox_img]
            for col_idx, (ax, im) in enumerate(zip(axes[row_idx], row_imgs)):
                ax.imshow(im)
                if row_idx == 0:
                    ax.set_title(col_headers[col_idx], fontsize=12, fontweight="bold", pad=8)
                if col_idx == 0:
                    ax.set_ylabel(fn, fontsize=10, fontweight="bold", rotation=90, labelpad=8)
                ax.set_xticks([])
                ax.set_yticks([])

        plt.tight_layout(rect=[0, 0, 1, 0.965])
        out_file = figures_dir / f"figure_{attr.lower()}_{direction}.png"
        plt.savefig(out_file, dpi=180, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(out_file)
        LOGGER.info("Saved full-page qualitative figure: %s", out_file)

    return saved_paths


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    parser = argparse.ArgumentParser(description="Evaluate Baseline (Section 6.1) and generate report figures.")
    parser.add_argument("--paths-config", type=str, default=str(DEFAULT_PATHS_CONFIG))
    parser.add_argument("--samples-per-figure", type=int, default=4, help="Rows per full-page figure (default: 4).")
    args = parser.parse_args()

    paths = load_project_paths(args.paths_config)
    validate_celeba_paths(paths)

    partitions = parse_partition_file(paths.celeba.partition_file)
    attributes = parse_attributes_file(paths.celeba.attributes_file)

    val_manifest = paths.output_dir / "manifests" / "validation.jsonl"
    test_manifest = paths.output_dir / "manifests" / "test.jsonl"
    no_ref_dir = paths.output_dir / "baseline" / "test" / "no_refinement"
    ref_dir = paths.output_dir / "baseline" / "test" / "refinement"

    # Fallback: if outputs were attached as a Kaggle Input dataset instead of /kaggle/working
    input_root = Path("/kaggle/input")
    if not val_manifest.is_file() and input_root.is_dir():
        found_val = list(input_root.rglob("validation.jsonl"))
        found_test = list(input_root.rglob("test.jsonl"))
        if found_val and found_test:
            val_manifest = found_val[0]
            test_manifest = found_test[0]
            LOGGER.info("Using manifests from /kaggle/input: %s", val_manifest.parent)

    if not no_ref_dir.is_dir() and input_root.is_dir():
        found_noref = [p for p in input_root.rglob("no_refinement") if p.is_dir() and "test" in str(p)]
        found_ref = [p for p in input_root.rglob("refinement") if p.is_dir() and "test" in str(p)]
        if found_noref and found_ref:
            no_ref_dir = found_noref[0]
            ref_dir = found_ref[0]
            LOGGER.info("Using baseline test images from /kaggle/input: %s", no_ref_dir.parent)

    val_records = [json.loads(l) for l in val_manifest.read_text().splitlines() if l.strip()]
    test_records = [json.loads(l) for l in test_manifest.read_text().splitlines() if l.strip()]

    eval_dir = paths.output_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    # 1. Attribute Classifier
    attr_clf = FrozenCLIPAttributeClassifier(device="cuda" if torch.cuda.is_available() else "cpu")
    val_agreement = attr_clf.fit_or_load(
        ckpt_path=eval_dir / "celeba_attr_classifier.pt",
        images_dir=paths.celeba.images_dir,
        partitions=partitions,
        attributes=attributes,
        val_records=val_records,
    )

    # 2. ArcFace Identity Evaluator
    LOGGER.info("Initializing ArcFace (insightface buffalo_l)...")
    arcface = ArcFaceEvaluator()

    # Pre-load images and evaluate
    LOGGER.info("Evaluating 120 test requests across w/o prox and w/ prox (240 edited images)...")
    orig_imgs = [center_crop_512(paths.celeba.images_dir / r["filename"]) for r in test_records]
    wo_imgs = [Image.open(no_ref_dir / f"{r['sample_id']}.png").convert("RGB") for r in test_records]
    w_imgs = [Image.open(ref_dir / f"{r['sample_id']}.png").convert("RGB") for r in test_records]

    probs_wo = attr_clf.predict_probs(wo_imgs)
    probs_w = attr_clf.predict_probs(w_imgs)

    attr_to_idx = {a: i for i, a in enumerate(TARGET_ATTRS)}
    target_map = {(a, d): (t_attr, t_sign) for a, d, t_attr, t_sign in DIRECTIONS_ORDER}

    per_sample_eval = []
    grouped_stats: Dict[Tuple[str, str, str], Dict[str, List[Any]]] = defaultdict(lambda: defaultdict(list))

    for i, rec in enumerate(test_records):
        attr = rec["attribute"]
        direction = rec["direction"]
        t_attr, t_sign = target_map[(attr, direction)]
        t_idx = attr_to_idx[t_attr]
        thr = attr_clf.thresholds[t_attr]

        src_emb = arcface.get_embedding(orig_imgs[i])

        for cond_label, probs_arr, edit_img in [
            ("w/o prox", probs_wo[i], wo_imgs[i]),
            ("w/ prox", probs_w[i], w_imgs[i]),
        ]:
            p_pos = float(probs_arr[t_idx])
            if t_sign == 1:
                target_conf = p_pos
                attr_success = int(p_pos >= thr)
            else:
                target_conf = 1.0 - p_pos
                attr_success = int(p_pos < thr)

            out_emb = arcface.get_embedding(edit_img)
            if src_emb is not None and out_emb is not None:
                cos_sim = float(np.dot(src_emb, out_emb))
                det_fail = 0
            else:
                cos_sim = None
                det_fail = 1

            key = (attr, direction, cond_label)
            grouped_stats[key]["attr_success"].append(attr_success)
            grouped_stats[key]["target_conf"].append(target_conf)
            grouped_stats[key]["det_fail"].append(det_fail)
            if cos_sim is not None:
                grouped_stats[key]["cos_sim"].append(cos_sim)

            per_sample_eval.append(
                {
                    "sample_id": rec["sample_id"],
                    "filename": rec["filename"],
                    "attribute": attr,
                    "direction": direction,
                    "condition": cond_label,
                    "target_attribute": t_attr,
                    "target_state": t_sign,
                    "target_confidence": round(target_conf, 4),
                    "attribute_success": bool(attr_success),
                    "arcface_cosine_sim": round(cos_sim, 4) if cos_sim is not None else None,
                    "face_detection_failure": bool(det_fail),
                }
            )

    # Save per-sample evaluation records
    with (eval_dir / "test_evaluation_records.jsonl").open("w", encoding="utf-8") as f:
        for row in per_sample_eval:
            f.write(json.dumps(row) + "\n")

    # Build quantitative summary table
    md_lines = [
        "# Baseline Quantitative Results (Section 6.1 Automatic Measurements)",
        "",
        "## 1. Validation Source-Label Agreement & Fixed Classifier Thresholds",
        "| Target Attribute | Validation Source Agreement (%) | Fixed Threshold ($\\theta$) |",
        "| :--- | :---: | :---: |",
    ]
    for a in TARGET_ATTRS:
        md_lines.append(f"| `{a}` | {val_agreement[a]:.2f}% | {attr_clf.thresholds[a]:.3f} |")

    md_lines.extend(
        [
            "",
            "## 2. Held-Out Test Set Evaluation (120 Requests × 2 Conditions = 240 Outputs)",
            "| Attribute Family | Direction | N | Method | Attr. Success (%) | Mean Target Conf. | ArcFace Sim ($s_{\\text{id}}$) | Face Det. Failures |",
            "| :--- | :--- | :---: | :--- | :---: | :---: | :---: | :---: |",
        ]
    )

    summary_rows = []
    for attr, direction, _, _ in DIRECTIONS_ORDER:
        for cond_label in ("w/o prox", "w/ prox"):
            st = grouped_stats[(attr, direction, cond_label)]
            n_total = len(st["attr_success"])
            succ_rate = 100.0 * sum(st["attr_success"]) / max(1, n_total)
            mean_conf = sum(st["target_conf"]) / max(1, n_total)
            sims = st["cos_sim"]
            mean_sim = sum(sims) / len(sims) if sims else None
            det_fails = sum(st["det_fail"])
            sim_str = f"{mean_sim:.3f}" if mean_sim is not None else "N/A"
            md_lines.append(
                f"| **{attr}** | `{direction}` | {n_total} | ChordEdit ({cond_label}) | "
                f"{succ_rate:.1f}% | {mean_conf:.3f} | {sim_str} | {det_fails}/{n_total} |"
            )
            summary_rows.append(
                {
                    "attribute": attr,
                    "direction": direction,
                    "n": n_total,
                    "method": f"ChordEdit ({cond_label})",
                    "attr_success_pct": round(succ_rate, 2),
                    "mean_target_conf": round(mean_conf, 4),
                    "mean_arcface_sim": round(mean_sim, 4) if mean_sim is not None else None,
                    "det_failures": det_fails,
                }
            )

    # Overall averages for w/o prox and w/ prox
    for cond_label in ("w/o prox", "w/ prox"):
        all_succ = [x["attribute_success"] for x in per_sample_eval if x["condition"] == cond_label]
        all_conf = [x["target_confidence"] for x in per_sample_eval if x["condition"] == cond_label]
        all_sims = [
            x["arcface_cosine_sim"]
            for x in per_sample_eval
            if x["condition"] == cond_label and x["arcface_cosine_sim"] is not None
        ]
        all_fails = [x["face_detection_failure"] for x in per_sample_eval if x["condition"] == cond_label]
        n_tot = len(all_succ)
        s_pct = 100.0 * sum(all_succ) / max(1, n_tot)
        m_conf = sum(all_conf) / max(1, n_tot)
        m_sim = sum(all_sims) / len(all_sims) if all_sims else 0.0
        f_cnt = sum(all_fails)
        md_lines.append(
            f"| **All Families** | **Overall** | **{n_tot}** | **ChordEdit ({cond_label})** | "
            f"**{s_pct:.1f}%** | **{m_conf:.3f}** | **{m_sim:.3f}** | **{f_cnt}/{n_tot}** |"
        )
        summary_rows.append(
            {
                "attribute": "OVERALL",
                "direction": "all",
                "n": n_tot,
                "method": f"ChordEdit ({cond_label})",
                "attr_success_pct": round(s_pct, 2),
                "mean_target_conf": round(m_conf, 4),
                "mean_arcface_sim": round(m_sim, 4),
                "det_failures": f_cnt,
            }
        )

    table_md_path = eval_dir / "baseline_quantitative_table.md"
    table_md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    (eval_dir / "baseline_metrics_summary.json").write_text(
        json.dumps(
            {
                "val_source_agreement_pct": val_agreement,
                "fixed_thresholds": attr_clf.thresholds,
                "test_summary": summary_rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    # 3. Generate the 8 full-page qualitative figures
    figures_dir = eval_dir / "figures"
    generate_qualitative_figures(
        test_records=test_records,
        no_ref_dir=no_ref_dir,
        ref_dir=ref_dir,
        images_dir=paths.celeba.images_dir,
        figures_dir=figures_dir,
        samples_per_page=args.samples_per_figure,
    )

    print("\n" + "\n".join(md_lines))


if __name__ == "__main__":
    main()
