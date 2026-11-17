<div align="center">
  <h1>[CVPR 2026 Oral] ChordEdit: One-Step Low-Energy Transport for Image Editing</h1>
  <div class="authors">
    <span><a href="https://luliangsi.github.io">Liangsi Lu</a><sup>1</sup>, <a href="https://cxh.netlify.app/">Xuhang Chen</a><sup>2</sup>, <a href="#">Minzhe Guo</a><sup>1</sup>, <a href="#">Shichu Li</a><sup>3</sup>, <a href="#">Jingchao Wang</a><sup>4</sup>, <a href="https://cnshiyang.github.io">Yang Shi</a><sup>1†</sup></span><br>
    <span style="color: #666; font-size: 0.9em;"><sup>1</sup> Guangdong University of Technology, <sup>2</sup> Huizhou University, <sup>3</sup> Shenzhen University, <sup>4</sup> Peking University <br> <sup>†</sup> Corresponding author</span>
  </div>

  <a href="https://chordedit.github.io"><img src="https://img.shields.io/badge/Project-Page-2b7de9"></a>
  <a href="https://arxiv.org/pdf/2602.19083"><img src="https://img.shields.io/badge/arXiv-2602.19083-b31b1b.svg"></a>

  <img src="chord_show.gif" alt="ChordEdit demo" width="100%" />
</div>

## 1. Environment
- Python 3.12
- PyTorch 2.5.0
- This repository requires the `sd-turbo` weights: https://huggingface.co/stabilityai/sd-turbo
- Model root should contain:
  - `unet/`
  - `scheduler/`
  - `text_encoder/`
  - `tokenizer/`
  - `vae/`

## 2. Install Dependencies
```bash
pip install -r requirement.txt
```

## 3. Run the Web Demo
Launch the interactive demo:
```bash
python app.py --model-root /path/to/sd-turbo --server-port 7860
```

Running `python app.py` now launches a local Gradio web app.
- Left panel: upload the original image, set source prompt, target prompt, and tuning parameters.
- Right panel: view the edited output image.
- Bottom section: click built-in examples (image + source prompt + target prompt) to auto-fill inputs.

<img src="chord_app.png" alt="ChordEdit app" width="100%" />

## 4. Run PIE Benchmark Export
Run PIE-Bench export with:
```bash
python run_pie_bench.py --model-root /path/to/sd-turbo --pie-root /path/to/pie_bench
```
`--pie-root` should point to a PIE-Bench folder containing at least:

1. `annotation_images/` — original PIE-Bench images (subfolders keep the official naming).
2. `mapping_file.json` — the mapping metadata describing prompts, instructions, and masks.

Example layout:
```
pie_bench
|-annotation_images
|-mapping_file.json
```

For PIE-Bench data preparation and protocol details, please refer to:
https://github.com/cure-lab/PnPInversion

## 5. CelebA Baseline Execution on Kaggle

This section describes how to run the **Original ChordEdit** baseline (with proximal refinement disabled and enabled) on CelebA inside a Kaggle GPU notebook.

### 5.1 Attach the Kaggle Dataset and Kaggle Model
1. Open a new Kaggle Notebook and set **Accelerator** to **GPU T4 x2** (or **GPU P100**).
2. In the right-hand **Input** panel, click **+ Add Input**:
   - Under **Datasets**, attach your shared **CelebA** Kaggle Dataset (`jenninn/celeba`).
   - Under **Models**, attach your shared **SD-Turbo** Kaggle Model (`jenninn/sd-turbo`).

### 5.2 Clone the Repository and Install Dependencies
In a notebook cell, clone your repository into `/kaggle/working/` and ensure the required packages are installed (leaving Kaggle's pre-installed CUDA PyTorch intact):
```bash
%cd /kaggle/working
!git clone https://github.com/xuanbac05/CV-FinalProject.git ChordEdit
%cd /kaggle/working/ChordEdit
!pip install -q diffusers transformers accelerate pyyaml Pillow
```

### 5.3 Identify Mounted `/kaggle/input` Paths
Run a cell in Kaggle to inspect the exact read-only mount locations of the CelebA files and SD-Turbo components:
```bash
!find /kaggle/input -maxdepth 4 \( -name "list_attr_celeba.txt" -o -name "list_eval_partition.txt" -o -name "img_align_celeba" -o -name "unet" \)
```
- The `images_dir` path must point directly to the folder containing `000001.jpg`, `000002.jpg`, etc.
- The `sd_turbo_dir` path must point to the directory containing the five subdirectories: `unet/`, `scheduler/`, `vae/`, `tokenizer/`, and `text_encoder/`.

### 5.4 Configure `configs/paths.local.json`
Copy `configs/paths.example.json` to `configs/paths.local.json` and update it with the actual paths found in Step 5.3:
```python
import json
from pathlib import Path

paths_cfg = {
    "celeba": {
        "images_dir": "/kaggle/input/datasets/jenninn/celeba/img_align_celeba/img_align_celeba",
        "attributes_file": "/kaggle/input/datasets/jenninn/celeba/list_attr_celeba.txt",
        "partition_file": "/kaggle/input/datasets/jenninn/celeba/list_eval_partition.txt",
    },
    "model": {
        "sd_turbo_dir": "/kaggle/input/models/jenninn/sd-turbo/pytorch/diffusers-fp32/1",
    },
    "output_dir": "/kaggle/working/ChordEdit/outputs",
}

Path("configs/paths.local.json").write_text(json.dumps(paths_cfg, indent=2) + "\n")
```
*(No weights or datasets are downloaded or copied; all scripts load directly from `/kaggle/input/` in offline mode.)*

### 5.5 Generate Validation and Test Manifests
Generate the deterministic validation (`80` requests) and test (`120` requests) manifests:
```bash
!python scripts/prepare_celeba_manifests.py --paths-config configs/paths.local.json --seed 42
```
This writes:
- `/kaggle/working/ChordEdit/outputs/manifests/validation.jsonl` (80 requests: 20 per attribute family)
- `/kaggle/working/ChordEdit/outputs/manifests/test.jsonl` (120 requests: 30 per attribute family)

### 5.6 Run a Small Validation Sanity Check
Run the first 4 validation requests (producing 4 × 2 = 8 edited images) to verify that the pipeline works:
```bash
!python scripts/run_baseline_celeba.py \
    --paths-config configs/paths.local.json \
    --split validation \
    --limit 4 \
    --seed 42
```

### 5.7 Run the Full Test Baseline
After visually verifying the validation outputs, run the full 120-request test baseline (producing 120 × 2 = 240 edited images):
```bash
!python scripts/run_baseline_celeba.py \
    --paths-config configs/paths.local.json \
    --split test \
    --seed 42
```

### 5.8 Output Structure and Logs
All generated files and metadata are stored under `/kaggle/working/ChordEdit/outputs/`:
```text
outputs/
├── manifests/
│   ├── validation.jsonl
│   └── test.jsonl
└── baseline/
    ├── validation/
    │   ├── no_refinement/
    │   │   └── <sample_id>.png
    │   ├── refinement/
    │   │   └── <sample_id>.png
    │   └── results.jsonl
    └── test/
        ├── no_refinement/
        │   └── <sample_id>.png
        ├── refinement/
        │   └── <sample_id>.png
        └── results.jsonl
```
Each line in `results.jsonl` records the sample ID, source filename and path, attribute and direction, prompts, refinement condition (`no_refinement` vs. `refinement`), full inference config, seed, output path, runtime in seconds, status, error message (if any), and environment/hardware metadata.

### 5.9 Resuming an Interrupted Run
By default, `scripts/run_baseline_celeba.py` checks `results.jsonl` and existing PNG files and **automatically skips** any `(sample_id, refinement_condition)` pair that already succeeded. To resume an interrupted run, simply re-run the exact same command:
```bash
!python scripts/run_baseline_celeba.py --paths-config configs/paths.local.json --split test --seed 42
```
If you want to force a fresh run and overwrite existing outputs, pass `--overwrite`:
```bash
!python scripts/run_baseline_celeba.py --paths-config configs/paths.local.json --split validation --limit 4 --overwrite
```

# Citation
If you find our work helpful, please **star 🌟** this repo and **cite 📑** our paper. Thanks for your support!
```
@article{lu2026chordedit,
  title={ChordEdit: One-Step Low-Energy Transport for Image Editing},
  author={Lu, Liangsi and Chen, Xuhang and Guo, Minzhe and Li, Shichu and Wang, Jingchao and Shi, Yang},
  journal={arXiv preprint arXiv:2602.19083},
  year={2026}
}
```
