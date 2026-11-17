# Kaggle Workflow Summary (Tesla T4 GPU)

This document summarizes the end-to-end execution performed in the **Kaggle Notebook** ([`kaggle_baseline.ipynb`](kaggle_baseline.ipynb)) to reproduce the **Original ChordEdit Baseline (Row 1, Table 2 in the Revised Proposal)** and run the **Automatic Evaluation Pipeline (Section 6.1)**.

---

## 1. Attached Read-Only Kaggle Resources (`/kaggle/input/`)

In the Kaggle Notebook (configured with a **Tesla T4 GPU** accelerator), two read-only inputs are mounted without downloading weights from Hugging Face or duplicating the dataset into `/kaggle/working/`:

1. **CelebA Dataset:**
   - Contains the aligned facial images directory `img_align_celeba/` (`202,599` `.jpg` images).
   - Contains official facial-attribute annotations `list_attr_celeba.txt` (`40` binary attributes encoded as `-1` / `1`) and official split assignments `list_eval_partition.txt` (`0`: train, `1`: validation, `2`: test).
   - *Kaggle format handling:* Since some Kaggle CelebA datasets store annotations as `.csv` files (`list_attr_celeba.csv`, `list_eval_partition.csv`) or nest `img_align_celeba/img_align_celeba/`, the notebook automatically detects the mounted layout, converts CSV annotations to the official CelebA `.txt` format if needed, and writes the verified paths into `configs/paths.local.json`.
2. **Pretrained SD-Turbo Model:**
   - Contains the five standard `stabilityai/sd-turbo` component directories: `unet/`, `vae/`, `text_encoder/`, `tokenizer/`, and `scheduler/`.
   - Loaded directly from `/kaggle/input/` with `local_files_only=True` in **FP32** reference precision.

---

## 2. Step-by-Step Execution in the Kaggle Notebook

### Step 1: Repository Setup & Dependencies
- Clones the `ChordEdit` repository into `/kaggle/working/ChordEdit`.
- Installs required packages for inference and Section 6.1 evaluation: `diffusers`, `transformers`, `accelerate`, `open-clip-torch`, `insightface`, `onnxruntime-gpu`, and `scikit-learn`.

### Step 2: Path Auto-Detection & Validation (`configs/paths.local.json`)
- Scans `/kaggle/input` to locate `img_align_celeba`, `list_attr_celeba.txt`, `list_eval_partition.txt`, and the `sd-turbo` directory.
- Generates `configs/paths.local.json` and validates all required inputs via `scripts/path_config.py`.

### Step 3: Reproducible Manifest Generation (`scripts/prepare_celeba_manifests.py`)
- Uses deterministic sampling (`seed = 42`) on the official CelebA splits with zero image overlap across tasks or splits:
  - **`outputs/manifests/validation.jsonl` (`80` requests from `partition = 1`):** `8` editing directions $\times$ `10` images per direction.
  - **`outputs/manifests/test.jsonl` (`120` requests from `partition = 2`):** `8` editing directions $\times$ `15` images per direction.
- Covers the **4 attribute families (8 editing directions)**:
  1. `Hair_Color`: `black_to_blond`
  2. `Hair_Color`: `brown_to_blond`
  3. `Smiling`: `add`
  4. `Smiling`: `remove`
  5. `Eyeglasses`: `add`
  6. `Eyeglasses`: `remove`
  7. `Mustache`: `add`
  8. `Mustache`: `remove`

### Step 4: Paired Baseline Inference (`scripts/run_baseline_celeba.py`)
- Runs the unmodified `ChordEditPipeline` (`pipeline_chord.py`) using fixed default parameters:
  - `t_start = 0.90`, `t_delta = 0.15`, `t_end = 0.30`, `step_scale = 1.0`, `n_steps = 1`, `noise_samples = 1`, `seed = 42`, and `512 × 512` center-crop preprocessing.
- For each source request, generates two paired outputs with identical stochastic noise initialization (`seed = 42`):
  - **Condition A – `no_refinement` (`ChordEdit w/o prox`):** `cleanup = False` (~`2.82 s/image` on Tesla T4).
  - **Condition B – `refinement` (`ChordEdit w/ prox`):** `cleanup = True` at $t_c = 0.30$ (~`3.15 s/image` on Tesla T4).
- **Generated Outputs:**
  - Validation sanity check: `4 requests × 2 = 8 PNG images` (plus full 80-request validation support).
  - Full held-out test run: **`120 requests × 2 = 240 PNG images`** saved under `outputs/baseline/test/no_refinement/` and `outputs/baseline/test/refinement/`, along with per-sample runtime and configuration metadata in `outputs/baseline/test/results.jsonl` (~`12.5 minutes` total on a single T4 GPU).

### Step 5: Automatic Evaluation (Section 6.1) & Figure Generation (`scripts/evaluate_and_plot_baseline.py`)
1. **Attribute Correctness Evaluator (Frozen CLIP Linear Probe):**
   - Extracts `CLIP ViT-B/32` visual embeddings on the official CelebA **Training split (`partition = 0`)** to train binary classifiers for `Blond_Hair`, `Smiling`, `Eyeglasses`, and `Mustache`.
   - Verifies source-label agreement on the **80 Validation requests (`partition = 1`)** and fixes decision thresholds $\theta$:
     - `Blond_Hair`: **91.25%** ($\theta = 0.590$)
     - `Smiling`: **95.00%** ($\theta = 0.400$)
     - `Eyeglasses`: **98.75%** ($\theta = 0.300$)
     - `Mustache`: **95.00%** ($\theta = 0.570$)
2. **Identity Preservation Evaluator (Frozen ArcFace `buffalo_l / w600k_r50`):**
   - Detects and aligns faces on each source image $x$ and edited output $y$, computing unit-normalized cosine similarity $s_{\text{id}}(x,y) = \phi(x)^\top \phi(y)$ while tracking face detection failures (`Det. Failures`) separately.
3. **Quantitative Table & Full-Page Qualitative Figures:**
   - Exports the complete evaluation table to `report_results/quantitative_results_table.md`.
   - Generates **8 full-page qualitative comparison figures (`300 DPI`)**—one per editing prompt—arranged in **3 columns (`Original` – `ChordEdit (w/o prox)` – `ChordEdit (w/ prox)`)** with 4 test portraits per page under `report_results/figures/`.
4. **Archiving Deliverables:**
   - Packages `outputs/` into `baseline_outputs.zip` and `report_results/` into `report_evaluation_and_figures.zip` for local download and LaTeX report compilation.
