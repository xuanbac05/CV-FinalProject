# CS412 - Computer Vision: Baseline Reproduction Report

**Title:** Face Attribute Editing in 2D Images: Learning Adaptive Transport Control for ChordEdit  
**Instructors:** Dr. Tran Thai Son and Dr. Vo Hoai Viet  

| Student ID | Name | Email |
| :--- | :--- | :--- |
| 23125001 | Tang Xuan Bac | `txbac23@apcs.fitus.edu.vn` |
| 23125006 | Vo Gia Huy | `vghuy23@apcs.fitus.edu.vn` |
| 23125010 | Le Dai King | `ldking23@apcs.fitus.edu.vn` |

---

## 1. Quantitative Results

### 1.1 Experimental Setup and Automatic Evaluators
We evaluate the original **ChordEdit** baseline using frozen **SD-Turbo** weights (Row 1 of Table 2 in our revised proposal) across four CelebA facial attribute families:
- **Hair Color:** `black_to_blond`, `brown_to_blond`
- **Smiling:** `add`, `remove`
- **Eyeglasses:** `add`, `remove`
- **Mustache:** `add`, `remove`

All source images are center-cropped and resized to $512 \times 512$ in FP32 precision with random seed $42$ and default transport parameters $(\lambda, \tau) = (1.0, 0.75)$, where $t_{\text{start}} = 0.90$, $\delta = 0.15$, and $t_{\text{end}} = 0.30$. For each of the $120$ held-out test requests ($15$ per editing direction), we evaluate both **ChordEdit (w/o prox)** (`cleanup=False`) and **ChordEdit (w/ prox)** (`cleanup=True`), yielding $240$ edited test images.

Following Section 6.1 of the revised proposal, we report two automatic measurements:
1. **Attribute correctness:** A frozen CLIP-based facial-attribute classifier trained on the official CelebA training partition (`partition=0`) predicts the target state for `Blond_Hair`, `Smiling`, `Eyeglasses`, and `Mustache`. Its decision thresholds $\theta$ are fixed after verifying agreement with ground-truth source labels on the $80$-image validation split (Table 1).
2. **Identity preservation:** A frozen ArcFace recognition model (`buffalo_l / w600k_r50`) computes source–output cosine similarity on unit-normalized aligned-face embeddings:
   $$s_{\text{id}}(x, y) = \phi(x)^\top \phi(y)$$
   Face detection failures are tracked and reported separately rather than silently excluded.

**Table 1: Validation source-label agreement and fixed classifier thresholds on the 80 validation requests.**

| Target Attribute | Validation Agreement (%) | Fixed Threshold ($\theta$) |
| :--- | :---: | :---: |
| `Blond_Hair` | 91.25% | 0.590 |
| `Smiling` | 95.00% | 0.400 |
| `Eyeglasses` | 98.75% | 0.300 |
| `Mustache` | 95.00% | 0.570 |

### 1.2 Baseline Automatic Measurements on Held-Out Test Set

**Table 2: Automatic evaluation of the Original ChordEdit baseline on the held-out CelebA test set ($120$ requests $\times$ $2$ refinement conditions $= 240$ outputs).**

| Attribute Family | Editing Direction | $N$ | Method | Attr. Success (%) | Mean Target Conf. | ArcFace Sim ($s_{\text{id}}$) | Det. Failures |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: | :---: |
| **Hair_Color** | `black_to_blond` | 15 | ChordEdit (w/o prox) | 0.0% | 0.093 | 0.612 | 0/15 |
| **Hair_Color** | `black_to_blond` | 15 | ChordEdit (w/ prox) | 33.3% | 0.333 | 0.280 | 0/15 |
| **Hair_Color** | `brown_to_blond` | 15 | ChordEdit (w/o prox) | 46.7% | 0.496 | 0.689 | 0/15 |
| **Hair_Color** | `brown_to_blond` | 15 | ChordEdit (w/ prox) | 60.0% | 0.648 | 0.330 | 0/15 |
| **Smiling** | `add` | 15 | ChordEdit (w/o prox) | 93.3% | 0.872 | 0.515 | 0/15 |
| **Smiling** | `add` | 15 | ChordEdit (w/ prox) | 100.0% | 0.983 | 0.213 | 0/15 |
| **Smiling** | `remove` | 15 | ChordEdit (w/o prox) | 53.3% | 0.552 | 0.570 | 0/15 |
| **Smiling** | `remove` | 15 | ChordEdit (w/ prox) | 66.7% | 0.635 | 0.281 | 0/15 |
| **Eyeglasses** | `add` | 15 | ChordEdit (w/o prox) | 93.3% | 0.881 | 0.632 | 0/15 |
| **Eyeglasses** | `add` | 15 | ChordEdit (w/ prox) | 93.3% | 0.921 | 0.286 | 0/15 |
| **Eyeglasses** | `remove` | 15 | ChordEdit (w/o prox) | 0.0% | 0.008 | 0.665 | 0/15 |
| **Eyeglasses** | `remove` | 15 | ChordEdit (w/ prox) | 0.0% | 0.009 | 0.299 | 0/15 |
| **Mustache** | `add` | 15 | ChordEdit (w/o prox) | 86.7% | 0.815 | 0.741 | 0/15 |
| **Mustache** | `add` | 15 | ChordEdit (w/ prox) | 86.7% | 0.851 | 0.361 | 1/15 |
| **Mustache** | `remove` | 15 | ChordEdit (w/o prox) | 26.7% | 0.228 | 0.742 | 0/15 |
| **Mustache** | `remove` | 15 | ChordEdit (w/ prox) | 66.7% | 0.500 | 0.424 | 0/15 |
| **Overall** | **All 8 Directions** | **120** | **ChordEdit (w/o prox)** | **50.0%** | **0.493** | **0.646** | **0/120** |
| **Overall** | **All 8 Directions** | **120** | **ChordEdit (w/ prox)** | **63.3%** | **0.610** | **0.309** | **1/120** |

**Key Quantitative Observations:**
- **Effect of Proximal Refinement (RQ3):** Enabling proximal refinement (`w/ prox`) increases overall target-attribute success from **50.0%** to **63.3%** ($+13.3\%$ absolute gain, with mean target confidence rising from $0.493$ to $0.610$), with strong improvements on `Mustache/remove` ($26.7\% \to 66.7\%$) and `Hair_Color/black_to_blond` ($0.0\% \to 33.3\%$). However, this comes at a clear cost to biometric identity preservation: mean ArcFace similarity drops from **0.646** (`w/o prox`) to **0.309** (`w/ prox`), alongside a modest $+11.9\%$ runtime overhead ($2.819\,\text{s}$ vs. $3.154\,\text{s}$ per image on Tesla T4).
- **Task Asymmetry and Need for Adaptive Control (RQ1 & RQ2):** Under the fixed default $(\lambda, \tau) = (1.0, 0.75)$, attribute addition tasks (`Smiling/add`: $93.3\%\text{–}100\%$, `Eyeglasses/add`: $93.3\%$, `Mustache/add`: $86.7\%$) succeed reliably, whereas removal tasks (`Eyeglasses/remove`: $0.0\%$) and dark-to-blond hair edits struggle or suffer identity drift, motivating an input-conditioned transport controller.

---

## 2. Qualitative Results

Figures 1–8 present full-page qualitative comparisons across all eight editing prompts on the held-out test set. Each figure displays four test portraits arranged in three columns: **Original**, **ChordEdit (w/o prox)**, and **ChordEdit (w/ prox)**. Without proximal refinement (`w/o prox`), the one-step linear latent update preserves coarse facial geometry ($s_{\text{id}} = 0.646$) but introduces off-manifold color blotches around edited regions (mouth, eyes, and hairline). Enabling proximal refinement (`w/ prox`) projects the edited latent back onto the natural face manifold, eliminating high-frequency artifacts and sharpening the target attribute, though sometimes altering fine facial identity features when $\lambda = 1.0$ is overly aggressive.

### Figure 1: Hair Color (`black_to_blond`)
![Hair Color black_to_blond](report_results/figures/figure_hair_color_black_to_blond.png)

### Figure 2: Hair Color (`brown_to_blond`)
![Hair Color brown_to_blond](report_results/figures/figure_hair_color_brown_to_blond.png)

### Figure 3: Smiling (`add`)
![Smiling add](report_results/figures/figure_smiling_add.png)

### Figure 4: Smiling (`remove`)
![Smiling remove](report_results/figures/figure_smiling_remove.png)

### Figure 5: Eyeglasses (`add`)
![Eyeglasses add](report_results/figures/figure_eyeglasses_add.png)

### Figure 6: Eyeglasses (`remove`)
![Eyeglasses remove](report_results/figures/figure_eyeglasses_remove.png)

### Figure 7: Mustache (`add`)
![Mustache add](report_results/figures/figure_mustache_add.png)

### Figure 8: Mustache (`remove`)
![Mustache remove](report_results/figures/figure_mustache_remove.png)

---

## References

1. **Liangsi Lu, Xuhang Chen, Minzhe Guo, Shichu Li, Jingchao Wang, and Yang Shi.** *ChordEdit: One-step low-energy transport for image editing.* arXiv preprint arXiv:2602.19083, 2026. Version 2, March 25, 2026.
2. **Minghan Li, Jeremy Moebel, and Mengyu Wang.** *Rethinking one-step image editing through ChordEdit: Reproduction, simplification, and new insights.* arXiv preprint arXiv:2606.14042, 2026. Version 1. https://arxiv.org/abs/2606.14042v1.
3. **Yunjey Choi, Minje Choi, Munyoung Kim, Jung-Woo Ha, Sunghun Kim, and Jaegul Choo.** *StarGAN: Unified generative adversarial networks for multi-domain image-to-image translation.* In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 8789–8797, 2018.
4. **Ming Liu, Yukang Ding, Min Xia, Xiao Liu, Errui Ding, Wangmeng Zuo, and Shilei Wen.** *STGAN: A unified selective transfer network for arbitrary image attribute editing.* In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 3673–3682, 2019.
5. **Sihan Xu, Yidong Huang, Jiayi Pan, Ziqiao Ma, and Joyce Chai.** *Inversion-free image editing with language-guided diffusion models.* In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 9452–9461, 2024.
6. **Vladimir Kulikov, Matan Kleiner, Inbar Huberman-Spiegelglas, and Tomer Michaeli.** *FlowEdit: Inversion-free text-based editing using pre-trained flow models.* In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 19721–19730, 2025.
7. **ChordEdit Authors.** *ChordEdit: Official implementation.* Official GitHub repository, 2026. Accessed October 27, 2026.
8. **Stability AI.** *SD-Turbo model card.* https://huggingface.co/stabilityai/sd-turbo. Accessed October 27, 2026.
9. **Alec Radford et al.** *Learning transferable visual models from natural language supervision.* In Proceedings of the 38th International Conference on Machine Learning (ICML), volume 139 of PMLR, pages 8748–8763, 2021.
10. **Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaoou Tang.** *Deep learning face attributes in the wild.* In Proceedings of the IEEE International Conference on Computer Vision (ICCV), pages 3730–3738, 2015.
11. **Multimedia Laboratory, The Chinese University of Hong Kong.** *Large-scale CelebFaces attributes (CelebA) dataset.* Official dataset page and usage agreement. Accessed October 27, 2026.
12. **Jiankang Deng, Jia Guo, Niannan Xue, and Stefanos Zafeiriou.** *ArcFace: Additive angular margin loss for deep face recognition.* In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 4690–4699, 2019.


