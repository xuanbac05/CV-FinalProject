# Quantitative Results (Section 6.1 Automatic Measurements)

### Validation Source-Label Agreement & Fixed Thresholds
| Attribute | Validation Agreement (%) | Threshold |
| :--- | :---: | :---: |
| `Blond_Hair` | 91.25% | 0.590 |
| `Smiling` | 95.00% | 0.400 |
| `Eyeglasses` | 98.75% | 0.300 |
| `Mustache` | 95.00% | 0.570 |

### Test Set Automatic Evaluation (120 Requests × 2 Conditions = 240 Images)
| Attribute Family | Editing Direction | N | Method | Attr. Success (%) | Mean Target Conf. | ArcFace Sim ($s_{\text{id}}$) | Det. Failures |
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
