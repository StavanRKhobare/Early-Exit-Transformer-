# Green ViT: Results Comparison

*Results as of 5 November 2026.*

The new method used **about 35% less measured energy at batch 32** than the fine-tuned ViT-Tiny baseline (B0), and lost **0.24 points of test accuracy** against B0's 96.70%. These results come from one trained backbone on CIFAR-10, so they are a proof of concept.

## 1. The method

| Part | What it does |
|---|---|
| Shared training | One backbone and final head are trained across six token-merging schedules, sampling one schedule at each step |
| Patch-average early exits | Exit heads average the patch tokens, and an image stops when the head's entropy is below a threshold |
| Selection with an accuracy limit | A Pareto path of designs is tested in a fixed order on unused images, using an exact statistical bound at 90% confidence |

| Setup | Value |
|---|---|
| Model | ViT-Tiny, 12 blocks, 197 tokens at 224×224 |
| Dataset | CIFAR-10: 20,000 train, 1,000 calibration, 4,000 validation, 10,000 test |
| Selection data | 25,000 unused training images: 10,000 to build the path, 15,000 to test it |
| Hardware | GTX 1650 Mobile and i5-10300H |
| Energy meters | NVML (GPU) and RAPL (CPU package) |

**Reading the tables:** `6r50 g8@0.05 L10@0.12` means a 50% merge after block 6, a first exit after block 8 at threshold 0.05, and a second exit after block 10 at threshold 0.12. "Points" are percentage points of accuracy. **Measured** values are real energy readings, and **predicted** values come from a table of per-operation costs.

## 2. New method against the earlier frozen study

The new method kept the frozen study's batch-32 savings, while losing less accuracy from a higher baseline. Baselines and conditions differ, so this is context and not a controlled comparison.

| Measure | Frozen ViT-Tiny, exit + merge (1-point tolerance) | New method (0.5-point limit) |
|---|---|---|
| B0 test accuracy | 92.03% | **96.70%** |
| Test accuracy lost | 0.86 points | **0.24 points** |
| Measured energy saved, GPU batch 32 | 35.2% (float16), 35.5% (float32) | **36.2%** (float32) |
| Measured energy saved, CPU batch 32 | 35.7% | **34.8%** |
| How the design was selected | Validation data, no guarantee | Exact bound at 90% confidence |

## 3. Measured energy saved against B0

Each design was measured against fine-tuned B0 in 6 paired repeats, in random order, on 1,000 unused images. There were 144 measurements in total. Values are percent saved with 95% confidence interval half-widths.

| Design | Accuracy limit (points) | Test accuracy change (points) | GPU batch 32 | CPU batch 32 | GPU batch 1 | CPU batch 1 |
|---|---|---|---|---|---|---|
| `6r50 g8@0.05 L10@0.12` | 0.5 | −0.24 | **36.2 ± 0.9** | **34.8 ± 2.7** | 29.9 ± 1.9 | 26.9 ± 1.1 |
| `6r50 g6@0.03 L8@0.05` | 0.5 | −0.24 | 35.6 ± 1.7 | 34.7 ± 3.6 | **31.3 ± 1.5** | **27.4 ± 0.6** |
| `6r50 g4@0.12 L8@0.08` | 1 | −0.74 | 43.4 ± 1.1 | 40.5 ± 3.0 | 38.0 ± 1.3 | 35.1 ± 1.6 |
| `4r50 g6@0.05 L8@0.16` | 1 | −1.08 (over limit) | 47.1 ± 0.6 | 46.2 ± 2.3 | 40.6 ± 1.0 | 36.0 ± 1.1 |
| `4r50 g6@0.03 L8@0.16` | 1 | −1.06 (over limit) | 46.1 ± 0.9 | 46.3 ± 1.6 | 39.5 ± 1.5 | 35.7 ± 0.5 |

Bold marks the design selected for that setting at the 0.5-point limit. The two 1-point `4r50` designs went over their limit by less than one standard error (0.17 points).

### Energy per image

| Setting | B0 (J per image) | `6r50 g8@0.05 L10@0.12` (J per image) |
|---|---|---|
| GPU batch 32 | 0.123 | 0.078 |
| CPU batch 32 | 0.344 | 0.224 |
| GPU batch 1 | 0.148 | 0.104 |
| CPU batch 1 | 0.402 | 0.294 |

## 4. Selection rules compared

The exact-bound rule stayed within the accuracy limit, while selecting with no guarantee broke it. Both tables are at GPU batch 32 with a 0.5-point limit, and savings are predicted.

### Main run, scored on the test set

| Selection rule | Energy saved (predicted) | Test accuracy change (points) | Within 0.5-point limit? |
|---|---|---|---|
| No guarantee (naive) | 40.5% | −0.62 | **No** |
| Approximate bound (z margin) | 36.2% | −0.32 | Yes |
| **Pareto test, exact bound (ours)** | **34.0%** | **−0.24** | **Yes** |
| Pareto test, empirical Bernstein bound | 31.9% | −0.17 | Yes |
| One-shot exact bound (Bonferroni) | 18.4% | −0.11 | Yes |

### Repeated over 50 random splits of the unused images

Each split selects on 20,000 images. "Over the limit" is checked against all 25,000 unused images, and the figures in brackets use only the 5,000 held-out images, which are noisier.

| Selection rule | Mean energy saved (predicted) | SD (points) | Over the limit | Found a design |
|---|---|---|---|---|
| No guarantee (naive) | 40.1% | 1.8 | 42% (42%) | 100% |
| Approximate bound (z margin) | 36.0% | 0.5 | 0% (14%) | 100% |
| **Pareto test, exact bound (ours)** | **24.4%** | 7.9 | **0% (2%)** | 94% |

The 34.0% in the main run is above this average, so predicted savings under the exact rule are best stated as **about 24–34%**. The rule is safe but varies, because one early failed test ends the search.

## 5. Model and technique comparisons

### Shared training against dedicated training

Shared training gave six usable schedules from one run of about 1 hour 43 minutes. On the two schedules with a dedicated model, the gaps can't be told apart from zero (standard error about 0.27 points on 4,000 validation images).

| Schedule | Shared model, validation | Dedicated model, validation | Gap (points) |
|---|---|---|---|
| No merging | 96.88% | 96.98% | −0.10 |
| `6r50_8r25` | 96.62% | 96.70% | −0.08 |
| `6r50` | 96.82% | Not trained | |
| `8r50` | 96.52% | Not trained | |
| `4r25_8r25` | 96.22% | Not trained | |
| `4r50` | 96.22% | Not trained | |

### Model and schedule options, under the same selection

Savings are predicted at the 0.5-point limit, with the exact bound unless stated.

| Option | GPU batch 32 saved | CPU batch 32 saved | CPU batch 1 saved | Test accuracy change, GPU batch 32 (points) |
|---|---|---|---|---|
| **Shared model, all six schedules** | **34.0%** | **34.2%** | **30.9%** | **−0.24** |
| Shared model, no merging | 24.9% | 26.0% | 29.1% | −0.29 |
| Dedicated B0, exits only | 24.5% | 25.6% | 29.3% | −0.20 |
| Dedicated `6r50_8r25`, exact bound | None passed | None passed | None passed | |
| Dedicated `6r50_8r25`, approximate bound | 36.2% | 37.1% | 33.3% | −0.29 |

Adding merging raised predicted savings by about 8–9 points at batch 32 and about 2 points on CPU batch 1. The dedicated `6r50_8r25` result doesn't show shared training is better, because the shared model's best design uses a different schedule (`6r50`).

### Exit head type

These figures use the dedicated merged model on validation data, with savings predicted at the 0.5-point limit.

| Exit head | Block 4 accuracy | Block 6 accuracy | Block 8 accuracy | Block 10 accuracy | GPU batch 32 saved | Accuracy change (points) |
|---|---|---|---|---|---|---|
| CLS token (standard) | 69.2% | 80.1% | 86.8% | 95.3% | 35.4% | −0.79 |
| **Patch average (ours)** | **80.1%** | **87.5%** | **91.1%** | 94.5% | **39.0%** | **−0.51** |

### Predicted against measured savings

| Setting | Design | Predicted | Measured | Measured − predicted (points) |
|---|---|---|---|---|
| GPU batch 32 | `6r50 g8@0.05 L10@0.12` | 34.0% | 36.2% | +2.2 |
| CPU batch 32 | `6r50 g8@0.05 L10@0.12` | 34.2% | 34.8% | +0.6 |
| CPU batch 1 | `6r50 g6@0.03 L8@0.05` | 30.9% | 27.4% | −3.5 |
| GPU batch 1 | `6r50 g6@0.03 L8@0.05` | 15.7% | 31.3% | +15.6 |

The GPU batch 1 gap arises because the GPU cost table used float16, while the real runs used float32.

## 6. What worked and what did not

| Idea | Result | Outcome |
|---|---|---|
| Fine-tuning with merging on | B0 rose from 92.03% to 96.70%, and merging's accuracy gap fell from 0.88 to about 0.28 points | **Worked** |
| Shared training across schedules | Within about 0.1 points of dedicated models on the two schedules compared | **Worked** |
| Patch-average exit heads | Block 4 rose from 69% to 80%, and block 6 from 80% to 87.5% | **Worked** |
| Selection with an exact bound | 0% over the limit, against 42% with no guarantee | **Worked**, at a cost in savings |
| Merging at batch 32 and above | Raised predicted savings by about 8–9 points | **Worked** (not measured directly) |
| Merging at batch 1 | Added about 2 predicted points | Little benefit |
| Per-device design selection | Using another device's design cost at most 1.6 predicted points | Little benefit |
| Ordering designs by energy, not MACs | 34.0% against 33.4% predicted | Little benefit |
| Confidence-guided merging | Added about 0–4 points of savings | Did not work |
| Exit-head-guided token choice | Best gain +0.21 points, with a confidence interval including zero | Did not work |
| Reusing exit outputs in the final decision | No benefit after fine-tuning | Did not work |
| Joint training of backbone and exits | Exit heads improved, but the final head fell to 96.08%, with no net gain | Did not work |
| Learned exit gate | AUROC 0.835 against 0.834 for entropy at block 4 | Did not work |

## 7. Limits of the evidence

- **One backbone seed on one dataset.** Three exit-head seeds gave similar results (34–35% predicted, −0.20 to −0.24 on test), but this doesn't capture variation between backbones.
- **Guarantee scope.** It covers accuracy on images drawn like the selection data, at 90% confidence per selection. It doesn't cover energy or data shift.
- **Shared against dedicated.** The best schedule, `6r50`, has no dedicated model for comparison.
- **Merging's added benefit is predicted.** No exit-only design was measured in this run.
- **Measurement conditions.** Other applications were open during the energy runs, though run order was randomized within each repeat. All GPU runs reported the software thermal slowdown flag, at 70–75 °C. GPU figures exclude CPU energy of about 0.14 J per image.
- **Test set use.** Designs were locked before this test report, but earlier pilots also read the test set.
- **Novelty.** The literature check isn't finished.

### Still to test

1. At least 3 backbone seeds of the shared model.
2. A dedicated `6r50` model, for a fair comparison with shared training.
3. A measured exit-only design.
4. A harder dataset, such as CIFAR-100 or Tiny-ImageNet.
