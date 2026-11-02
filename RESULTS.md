# When does token merging save energy? Results

Frozen ViT-Tiny (`vit_tiny_patch16_224`, `augreg_in21k_ft_in1k`) on CIFAR-10, on one laptop (GTX 1650 and its Intel CPU). Measured 1 November 2026.

This file compiles every result of the study. Each table is filled from a CSV in `outputs/results/` by `scripts/make_results_doc.py`, and files named below are in that folder. "Pts" means percentage points of accuracy. "B0" is the full model with a linear head: 92.03% test accuracy (± 0.11 across 5 seeds).

## Main findings

1. **Batch size decides whether merging is worth it.** Within 1 point of B0's validation accuracy, at batch 32 and above, adding merging to early exit raised measured savings from 25–27% to 35–37%. It added 10.0–10.6 points on the CPU, GPU float16 and cooled GPU float32. At batch 1 it added 0.3–2.5 points and lowered accuracy further.
2. **Merging alone can waste energy at batch 1.** On the cooled GPU in float32, a single merge cost 9.1% *more* energy than B0 (median of 3 repeats, range −9.5% to +4.8% saved) and took 22.6% longer, while MACs claimed a 25.7% saving.
3. **Early exit is the better first choice at a tight accuracy budget.** At a 0.5-point tolerance, exit only beat merge only in all five regimes measured, and stayed within tolerance on test. At 1 point and batch 32+, merge only led exit only by 2.1–3.3 points of savings, with 0.2 points less accuracy.
4. **Where a merge happens matters more than how much it removes.** A 50% merge costs 6.70 points after block 2 and 0.13 points after block 10.
5. **MACs choose well at large batches and mislead at batch 1.** At batch 32+ they chose the same design as energy at the 1- and 2-point tolerances, with at most 1.8% regret at any tolerance. At batch 1 the regret reached 7.0%, and they overstated the saving by up to 2.7× on float16 (35.3% claimed, 13.0% measured). They also got the sign wrong in finding 2.
6. **On this card, number format matters more than either method.** Switching B0 from float16 autocast to float32 cut measured GPU energy by 51–76%, against at most 37% from exit + merge. This was measured on one GPU model, so don't generalise it without testing other cards.

**Suggested rule for the paper, on this evidence:** use merging at batch 32 or above, after block 6 or later, combined with early exit. Don't use it for one-at-a-time inference, or when the test accuracy budget is under about 1 point, since the combined design lands near −0.86.

**Contents:** [1 Measured energy](#1-measured-energy-the-main-evidence) · [2 Accuracy cost](#2-accuracy-cost) · [3 Break-even](#3-why-batch-size-matters-merge-break-even) · [4 All designs](#4-accuracy-against-energy-all-designs) · [5 Predictions and MACs](#5-can-the-cost-table-and-macs-be-trusted) · [6 Conditions](#6-measurement-conditions) · [7 Limits](#7-limits-to-state-in-the-paper) · [8 Corrections](#8-corrections-made-while-checking-these-results) · [9 Reproduce](#9-reproduce) · [Appendix: full tables](#appendix-full-tables)

## 1. Measured energy, the main evidence

![Measured savings per regime and method](outputs/results/fig_measured_savings.png)

**Figure 1.** Measured savings against B0 run in the same repeat. Bars are medians, whiskers the range over repeats, and black dashes the cost-table prediction (no dash for cooled float32, whose table was unusable). Source: `measured_comparison.csv`, `float32_cooled_summary.csv`. Raw runs: `real_runs.csv`, `real_runs_float32_cooled.csv`.

### 1-point tolerance (CPU and GPU float16)

| Regime                | Method       |   Test accuracy vs B0 (pts) | Energy saved, measured % (range)   |   Predicted % |   MACs claim % |   Time saved % |   With CPU package % |   mJ per image |
|:----------------------|:-------------|----------------------------:|:-----------------------------------|--------------:|---------------:|---------------:|---------------------:|---------------:|
| CPU batch 1           | Exit only    |                       -0.65 | 23.0 (22.0 to 28.6)                |          25.2 |           26.1 |           24.2 |                      |            290 |
| CPU batch 1           | Merge only   |                       -0.88 | 16.1 (15.1 to 16.9)                |          17   |           29.6 |           16.2 |                      |            316 |
| CPU batch 1           | Exit + merge |                       -1.19 | 25.5 (25.5 to 25.7)                |          28.7 |           32.7 |           27.4 |                      |            278 |
| CPU batch 32          | Exit only    |                       -0.65 | 25.7 (25.3 to 26.0)                |          21.2 |           26.1 |           25.4 |                      |            215 |
| CPU batch 32          | Merge only   |                       -0.88 | 28.4 (28.2 to 28.4)                |          27.8 |           29.6 |           28.1 |                      |            207 |
| CPU batch 32          | Exit + merge |                       -0.86 | 35.7 (35.6 to 36.1)                |          34.5 |           37.4 |           35.5 |                      |            186 |
| GPU float16 batch 1   | Exit only    |                       -0.65 | 12.0 (11.0 to 12.5)                |          12.4 |           26.1 |           11.1 |                  7.9 |            588 |
| GPU float16 batch 1   | Merge only   |                       -0.79 | 10.0 (9.9 to 10.3)                 |           9.3 |           25.7 |            9.9 |                 10.3 |            605 |
| GPU float16 batch 1   | Exit + merge |                       -0.89 | 13.0 (12.4 to 15.2)                |          14.5 |           35.3 |           11.8 |                 17   |            574 |
| GPU float16 batch 32  | Exit only    |                       -0.65 | 25.0 (19.5 to 25.3)                |          20.3 |           26.1 |           25   |                 24.7 |            166 |
| GPU float16 batch 32  | Merge only   |                       -0.88 | 27.7 (27.5 to 28.0)                |          27   |           29.6 |           27.9 |                 27.6 |            160 |
| GPU float16 batch 32  | Exit + merge |                       -0.86 | 35.2 (34.7 to 35.5)                |          33.7 |           37.4 |           35.8 |                 35.4 |            143 |
| GPU float16 batch 128 | Exit only    |                       -0.65 | 26.5 (25.1 to 28.7)                |          22.2 |           26.1 |           26.1 |                 25.4 |            163 |
| GPU float16 batch 128 | Merge only   |                       -0.88 | 28.6 (21.2 to 28.6)                |          28.6 |           29.6 |           29.1 |                 29.8 |            159 |
| GPU float16 batch 128 | Exit + merge |                       -0.86 | 36.6 (36.4 to 36.7)                |          35.1 |           37.4 |           36.9 |                 37   |            141 |

Designs were selected on validation data. "With CPU package" adds RAPL package energy read during the GPU run. It's noisier, because the package also serves everything else on the machine.

### 1-point tolerance (GPU float32, cool-down before every run)

|   Batch | Method       | Design                                           |   Test accuracy vs B0 (pts) | Energy saved % (range)   |   MACs claim % |   Time saved % |   With CPU package % |   mJ per image |   B0 mJ |
|--------:|:-------------|:-------------------------------------------------|----------------------------:|:-------------------------|---------------:|---------------:|---------------------:|---------------:|--------:|
|       1 | Exit only    | `m[none]e[6mlp192_10mlp192]f[lin]` τ=0.2         |                       -0.65 | 10.1 (9.9 to 16.2)       |           26.1 |            9.5 |                 14.2 |            170 |     189 |
|       1 | Merge only   | `m[6r50]e[none]f[lin]` τ=0                       |                       -0.79 | -9.1 (-9.5 to 4.8)       |           25.7 |          -22.6 |                 -5   |            200 |     189 |
|       1 | Exit + merge | `m[8r50]e[6mlp192_10lin]f[lin]` τ=0.2            |                       -1.19 | 10.4 (10.3 to 16.5)      |           32.7 |            5.3 |                 12.8 |            170 |     189 |
|      32 | Exit only    | `m[none]e[6mlp192_10mlp192]f[lin]` τ=0.2         |                       -0.65 | 25.5 (25.4 to 26.2)      |           26.1 |           25.3 |                 19.4 |             81 |     110 |
|      32 | Merge only   | `m[6r50_8r25]e[none]f[lin]` τ=0                  |                       -0.88 | 28.8 (28.2 to 29.9)      |           29.6 |           28.7 |                 28.3 |             78 |     110 |
|      32 | Exit + merge | `m[6r50_8r25]e[4mlp192_10mlp192]f[lin]` τ=0.16   |                       -0.86 | 35.5 (35.2 to 35.6)      |           37.4 |           35.1 |                 30   |             71 |     110 |
|     128 | Exit only    | `m[none]e[10mlp192]f[lin]` τ=0.5                 |                       -0.69 | 15.9 (15.7 to 17.2)      |           15.8 |           15.5 |                 13.8 |             90 |     108 |
|     128 | Exit only    | `m[none]e[6mlp192_10mlp192]f[lin]` τ=0.2 (added) |                       -0.65 | 25.3 (24.6 to 25.9)      |           26.1 |           25.7 |                 25.3 |             80 |     108 |
|     128 | Merge only   | `m[6r50_8r25]e[none]f[lin]` τ=0                  |                       -0.88 | 28.2 (28.1 to 28.9)      |           29.6 |           28.1 |                 27.8 |             77 |     108 |
|     128 | Exit + merge | `m[6r50_8r25]e[4mlp192_10mlp192]f[lin]` τ=0.16   |                       -0.86 | 35.9 (35.8 to 36.1)      |           37.4 |           35.6 |                 35.8 |             69 |     108 |

"(added)" marks a design measured after the main run; see Section 6. Each repeat is 5 seconds or more, starting at 70–71 °C.

### 0.5-point tolerance (CPU and GPU float16)

| Regime                | Method       |   Test accuracy vs B0 (pts) | Energy saved, measured % (range)   |   Predicted % |   MACs claim % |   Time saved % |   With CPU package % |   mJ per image |
|:----------------------|:-------------|----------------------------:|:-----------------------------------|--------------:|---------------:|---------------:|---------------------:|---------------:|
| CPU batch 1           | Exit only    |                       -0.41 | 20.3 (19.4 to 21.3)                |          22.4 |           23.4 |           20.1 |                      |            301 |
| CPU batch 1           | Merge only   |                       -0.45 | 8.6 (8.3 to 14.0)                  |          10   |           17.1 |            9.3 |                      |            341 |
| CPU batch 1           | Exit + merge |                       -0.78 | 23.8 (23.2 to 23.9)                |          26.4 |           31   |           24.7 |                      |            286 |
| CPU batch 32          | Exit only    |                       -0.41 | 22.9 (22.8 to 23.0)                |          19.1 |           23.4 |           23.1 |                      |            223 |
| CPU batch 32          | Merge only   |                       -0.46 | 8.3 (4.8 to 17.7)                  |          17.4 |           19   |            7   |                      |            265 |
| CPU batch 32          | Exit + merge |                       -0.86 | 28.4 (26.8 to 28.5)                |          28.6 |           30.1 |           28.3 |                      |            207 |
| GPU float16 batch 1   | Exit only    |                       -0.41 | 10.5 (8.9 to 11.1)                 |          11   |           23.4 |            9.5 |                 10.6 |            601 |
| GPU float16 batch 1   | Merge only   |                       -0.45 | 7.0 (6.2 to 8.4)                   |           5.7 |           17.1 |            6.5 |                  7.1 |            619 |
| GPU float16 batch 1   | Exit + merge |                       -0.78 | 13.7 (12.7 to 14.6)                |          13.2 |           31   |           12.1 |                 17.3 |            576 |
| GPU float16 batch 32  | Exit only    |                       -0.41 | 22.3 (19.2 to 22.6)                |          18.3 |           23.4 |           22.3 |                 21.8 |            172 |
| GPU float16 batch 32  | Merge only   |                       -0.46 | 17.8 (16.8 to 18.5)                |          17.3 |           19   |           18   |                 17.3 |            181 |
| GPU float16 batch 32  | Exit + merge |                       -0.86 | 27.5 (18.7 to 28.0)                |          27.7 |           30.1 |           28.7 |                 27.5 |            160 |
| GPU float16 batch 128 | Exit only    |                       -0.41 | 23.0 (22.4 to 25.2)                |          20   |           23.4 |           23.3 |                 23.6 |            171 |
| GPU float16 batch 128 | Merge only   |                       -0.46 | 17.6 (16.7 to 20.0)                |          18.3 |           19   |           18.8 |                 18.2 |            183 |
| GPU float16 batch 128 | Exit + merge |                       -0.86 | 29.4 (28.8 to 29.9)                |          29.2 |           30.1 |           29.8 |                 29.3 |            157 |

The CPU batch 32 merge-only value of 8.3% looks unreliable: its repeats range from 4.8% to 17.7%, and the same design measures 17.8% on the GPU.

### Latency at the 1-point tolerance (CPU and GPU float16)

| Regime                | Method       |   p95 ms per image |   B0 p95 ms |
|:----------------------|:-------------|-------------------:|------------:|
| CPU batch 1           | Exit only    |               18.5 |        19.1 |
| CPU batch 1           | Merge only   |               16.1 |        19.1 |
| CPU batch 1           | Exit + merge |               17   |        19.1 |
| CPU batch 32          | Exit only    |               11.1 |        14.4 |
| CPU batch 32          | Merge only   |               10.2 |        14.4 |
| CPU batch 32          | Exit + merge |                9.2 |        14.4 |
| GPU float16 batch 1   | Exit only    |               31.8 |        31.5 |
| GPU float16 batch 1   | Merge only   |               28.4 |        31.5 |
| GPU float16 batch 1   | Exit + merge |               29.7 |        31.5 |
| GPU float16 batch 32  | Exit only    |                8   |        10.1 |
| GPU float16 batch 32  | Merge only   |                7.3 |        10.1 |
| GPU float16 batch 32  | Exit + merge |                6.7 |        10.1 |
| GPU float16 batch 128 | Exit only    |                7.5 |         9.8 |
| GPU float16 batch 128 | Merge only   |                7   |         9.8 |
| GPU float16 batch 128 | Exit + merge |                6.3 |         9.8 |

Per-image latency at the median of repeats. At batch 32 and above this is batch time divided by batch size, so it doesn't include waiting to fill a batch.

### How often samples exit and reach a merge

| Design                                  |    τ | Method       |   Leave at first exit % |   Leave at second exit % |   Reach the first merge % |
|:----------------------------------------|-----:|:-------------|------------------------:|-------------------------:|--------------------------:|
| `m[none]e[6mlp192_10mlp192]f[lin]`      | 0.16 | Exit only    |                    32.5 |                     44.3 |                           |
| `m[8r50]e[none]f[lin]`                  | 0    | Merge only   |                     0   |                      0   |                     100   |
| `m[8r50]e[6mlp192_10mlp192]f[lin]`      | 0.16 | Exit + merge |                    32.5 |                     43.9 |                      67.5 |
| `m[none]e[6mlp192_10mlp192]f[lin]`      | 0.2  | Exit only    |                    38.5 |                     42.3 |                           |
| `m[6r50_8r25]e[none]f[lin]`             | 0    | Merge only   |                     0   |                      0   |                     100   |
| `m[8r50]e[6mlp192_10lin]f[lin]`         | 0.2  | Exit + merge |                    38.5 |                     40.6 |                      61.5 |
| `m[8r50_10r25]e[none]f[lin]`            | 0    | Merge only   |                     0   |                      0   |                     100   |
| `m[6r25_8r50]e[10mlp192]f[lin]`         | 0.25 | Exit + merge |                    80.4 |                      0   |                     100   |
| `m[6r50_8r25]e[4mlp192_10mlp192]f[lin]` | 0.16 | Exit + merge |                    13.1 |                     60.4 |                      86.9 |
| `m[6r50]e[none]f[lin]`                  | 0    | Merge only   |                     0   |                      0   |                     100   |
| `m[6r50]e[6mlp192_10mlp192]f[lin]`      | 0.12 | Exit + merge |                    25.7 |                     45.7 |                      74.3 |

Fractions are on validation, averaged across seeds. Source: `all_points.csv.gz`.

## 2. Accuracy cost

### Baselines by final head

| Final head   |   Validation % | Test % (± seeds)   |   Test F1 |   vs B0 (pts) |   ECE |
|:-------------|---------------:|:-------------------|----------:|--------------:|------:|
| lin          |          92.48 | 92.03 ± 0.11       |    0.9203 |          0    | 0.009 |
| mlp128       |          92.61 | 92.48 ± 0.05       |    0.9248 |          0.45 | 0.007 |
| mlp192       |          92.66 | 92.56 ± 0.09       |    0.9257 |          0.54 | 0.008 |

The larger final heads beat B0 without any exit or merge benefit, so all comparisons use the linear head as the reference. Source: `all_points.csv.gz`, `finalist_metrics.csv`.

### Exit head accuracy by block, no merging

| Block   | Head   |   Validation % |   Test % |   ECE before |   ECE after |   Temperature |
|:--------|:-------|---------------:|---------:|-------------:|------------:|--------------:|
| 2       | lin    |           46.4 |     45.2 |        0.073 |       0.034 |          0.85 |
| 2       | mlp128 |           51   |     50.9 |        0.039 |       0.028 |          0.96 |
| 2       | mlp192 |           51.5 |     51.2 |        0.033 |       0.026 |          0.97 |
| 4       | lin    |           61.6 |     61.5 |        0.08  |       0.027 |          0.84 |
| 4       | mlp128 |           67.1 |     66.6 |        0.03  |       0.027 |          0.99 |
| 4       | mlp192 |           67.5 |     66.9 |        0.026 |       0.029 |          1.01 |
| 6       | lin    |           71.8 |     71.6 |        0.067 |       0.019 |          0.82 |
| 6       | mlp128 |           75.7 |     76   |        0.016 |       0.009 |          0.94 |
| 6       | mlp192 |           76.1 |     76.2 |        0.015 |       0.009 |          0.96 |
| 8       | lin    |           79.8 |     79.4 |        0.053 |       0.011 |          0.82 |
| 8       | mlp128 |           82.5 |     82.7 |        0.016 |       0.009 |          0.96 |
| 8       | mlp192 |           82.9 |     82.8 |        0.014 |       0.011 |          0.97 |
| 10      | lin    |           90.4 |     89.9 |        0.022 |       0.009 |          0.82 |
| 10      | mlp128 |           91.3 |     90.9 |        0.008 |       0.007 |          1.04 |
| 10      | mlp192 |           91.4 |     91.1 |        0.01  |       0.007 |          1.05 |
| final   | lin    |           92.5 |     92   |        0.012 |       0.009 |          1.04 |
| final   | mlp128 |           92.6 |     92.5 |        0.027 |       0.007 |          1.37 |
| final   | mlp192 |           92.7 |     92.6 |        0.031 |       0.008 |          1.44 |

Means across 5 seeds. "ECE" is calibration error on test, before and after temperature scaling. Source: `slot_metrics.csv`, which also holds every merged prefix.

![Accuracy cost of one merge](outputs/results/fig_merge_accuracy.png)

**Figure 2.** Test accuracy of one merge against B0, by block and strength. Error bars are the standard deviation across 5 seeds. Dashed line: 1 point. Source: `merge_only_effect.csv`.

|   After block | Tokens removed   |   Test accuracy vs B0 (pts) |   MACs saved % |   CPU b1 % |   CPU b32 % |   GPU f16 b1 % |   GPU f16 b128 % |
|--------------:|:-----------------|----------------------------:|---------------:|-----------:|------------:|---------------:|-----------------:|
|             2 | 25%              |                       -1.42 |           22.3 |       14.7 |        21.2 |            8.1 |             22.3 |
|             2 | 50%              |                       -6.7  |           42.9 |       28.4 |        40.3 |           16.4 |             42.5 |
|             4 | 25%              |                       -0.57 |           17.8 |       11.3 |        16.8 |            6.2 |             17.8 |
|             4 | 50%              |                       -1.81 |           34.3 |       22.2 |        32.1 |           12.8 |             33.9 |
|             6 | 25%              |                       -0.1  |           13.3 |        8   |        12.4 |            4.3 |             13.2 |
|             6 | 50%              |                       -0.79 |           25.7 |       16.1 |        23.8 |            9.3 |             25.3 |
|             8 | 25%              |                       -0.24 |            8.8 |        4.6 |         8.1 |            2.4 |              8.7 |
|             8 | 50%              |                       -0.45 |           17.1 |       10   |        15.6 |            5.7 |             16.8 |
|            10 | 25%              |                       -0.13 |            4.3 |        1.2 |         3.7 |            0.5 |              4.2 |
|            10 | 50%              |                       -0.13 |            8.5 |        3.8 |         7.4 |            2.1 |              8.2 |

Savings in this table are cost-table predictions. The full 50 schedules are in `merge_only_effect.csv`; the worst loses 10.1 points.

### Accuracy of the designs compared in Section 1

| Design                                  |    τ | Method       | Test accuracy   |   vs B0 (pts) | 95% CI, seed 0   |   ECE |
|:----------------------------------------|-----:|:-------------|:----------------|--------------:|:-----------------|------:|
| `m[none]e[6mlp192_10mlp192]f[lin]`      | 0.2  | Exit only    | 91.38 ± 0.11    |         -0.65 | -0.88 to -0.30   | 0.011 |
| `m[6r50_8r25]e[none]f[lin]`             | 0    | Merge only   | 91.14 ± 0.10    |         -0.88 | -1.18 to -0.43   | 0.008 |
| `m[8r50]e[6mlp192_10lin]f[lin]`         | 0.2  | Exit + merge | 90.83 ± 0.13    |         -1.19 | -1.52 to -0.74   | 0.014 |
| `m[6r50_8r25]e[4mlp192_10mlp192]f[lin]` | 0.16 | Exit + merge | 91.17 ± 0.08    |         -0.86 | -1.14 to -0.34   | 0.01  |
| `m[6r50]e[none]f[lin]`                  | 0    | Merge only   | 91.24 ± 0.14    |         -0.79 | -1.00 to -0.31   | 0.009 |
| `m[6r50]e[6mlp192_10mlp192]f[lin]`      | 0.12 | Exit + merge | 91.14 ± 0.11    |         -0.89 | -1.18 to -0.44   | 0.013 |

Every design here is below B0, and each seed-0 interval excludes zero. Source: `finalist_metrics.csv`.

**Selection against test results.** The tolerance is applied on validation, so test results can overshoot it. The CPU batch 1 exit + merge design lands at −1.19 points against a 1-point tolerance. At the 0.5-point tolerance every exit + merge pick overshoots (−0.78 to −0.86), while the exit-only and merge-only picks stay within it (−0.41 to −0.46).

## 3. Why batch size matters: merge break-even

![Merge break-even heatmap](outputs/results/fig_merge_breakeven.png)

**Figure 3.** Blocks a merge must run before its own cost is repaid. Source: `merge_breakeven.csv`; cost tables in `energy_tables.json` and `op_costs.csv`. Float32 is left out because its table is unreliable (Section 5).

- **Batch 1:** break-even is 0.8–2.5 blocks on the CPU and 0.8–3.6 on GPU float16, with one cell above 12. There, merging from 100 to 76 tokens saves so little that it's hard to separate from noise.
- **Batch 8 and above:** break-even is 0.1–1.0 blocks.
- **MACs:** they put break-even at 0.02–0.07 blocks, so they understate merge overhead at batch 1 by roughly 20 to 200 times.
- **Merge cost:** one merge costs 0.26–0.33 of a block on the CPU and 0.35–0.62 on GPU float16 at batch 1, and 0.04–0.10 at batch 32+.

## 4. Accuracy against energy, all designs

![Accuracy against predicted energy](outputs/results/fig_fronts.png)

**Figure 4.** Best validation accuracy at each predicted energy, across 5,408 designs and up to 15 thresholds each (80,378 rows in `all_points.csv.gz`), for points within 5 points of B0. Designs where under 5% of samples reach the merge are not credited to merging. Costs are predicted, so use Section 1 for quoted savings.

### Predicted savings at all four tolerances

| Regime                |   Tolerance (pts) |   Exit only % |   Merge only % |   Exit + merge % |   Merging adds (pts) |
|:----------------------|------------------:|--------------:|---------------:|-----------------:|---------------------:|
| CPU batch 1           |              0.25 |          19.7 |           10   |             21.9 |                  2.2 |
| CPU batch 1           |              0.5  |          22.4 |           10   |             26.4 |                  3.9 |
| CPU batch 1           |              1    |          25.2 |           17   |             28.7 |                  3.5 |
| CPU batch 1           |              2    |          28.3 |           23.1 |             35   |                  6.7 |
| CPU batch 32          |              0.25 |          17.8 |           17.4 |             24.7 |                  6.9 |
| CPU batch 32          |              0.5  |          19.1 |           17.4 |             28.6 |                  9.5 |
| CPU batch 32          |              1    |          21.2 |           27.8 |             34.5 |                 13.3 |
| CPU batch 32          |              2    |          23.9 |           36   |             44.2 |                 20.3 |
| GPU float16 batch 1   |              0.25 |           9.6 |            5.7 |             10.6 |                  1   |
| GPU float16 batch 1   |              0.5  |          11   |            5.7 |             13.2 |                  2.2 |
| GPU float16 batch 1   |              1    |          12.4 |            9.3 |             14.5 |                  2.1 |
| GPU float16 batch 1   |              2    |          14   |           11.7 |             17.5 |                  3.5 |
| GPU float16 batch 32  |              0.25 |          17.1 |           17.3 |             24.2 |                  7.1 |
| GPU float16 batch 32  |              0.5  |          18.3 |           17.3 |             27.7 |                  9.5 |
| GPU float16 batch 32  |              1    |          20.3 |           27   |             33.7 |                 13.3 |
| GPU float16 batch 32  |              2    |          23   |           35.2 |             43.8 |                 20.8 |
| GPU float16 batch 128 |              0.25 |          18.6 |           18.3 |             25.1 |                  6.4 |
| GPU float16 batch 128 |              0.5  |          20   |           18.3 |             29.2 |                  9.3 |
| GPU float16 batch 128 |              1    |          22.2 |           28.6 |             35.1 |                 13   |
| GPU float16 batch 128 |              2    |          25   |           37.1 |             45.4 |                 20.4 |

Source: `merge_added_value.csv`, reported on test. The best overall design differs across regimes; picks are in `picks.csv` and `method_comparison.csv`.

## 5. Can the cost table and MACs be trusted?

![Predicted against measured energy](outputs/results/fig_pred_vs_measured.png)

**Figure 5.** Predicted against measured energy for all measured operating points. Source: `real_measured.csv`.

| Regime                                  |   Operating points |   Median absolute error % |   Largest % |
|:----------------------------------------|-------------------:|--------------------------:|------------:|
| CPU batch 1                             |                 15 |                       1.2 |         4.5 |
| CPU batch 32                            |                 15 |                      19.9 |        26.2 |
| GPU float16 batch 1                     |                 15 |                       1.4 |         2.9 |
| GPU float16 batch 32                    |                 15 |                       1.7 |         4.3 |
| GPU float16 batch 128                   |                 15 |                       1   |         6.5 |
| GPU float32 (throttled table) batch 1   |                 18 |                      11   |        29.3 |
| GPU float32 (throttled table) batch 32  |                 18 |                     175.3 |       187.8 |
| GPU float32 (throttled table) batch 128 |                 18 |                      28.6 |        45.1 |

- **Use measured values for absolute energy.** The headline median hides large regime differences. CPU batch 32 is over-predicted by about 20%; the cause is not established.
- **The float32 table is invalid.** It was taken at the 92 °C limit with clocks swinging, and over-predicts by up to 2.9×. Its rows remain in the CSVs under `cuda_laptopfp32` and are left out of Figures 3, 4 and 6.
- **Predicted savings are close but biased.** At 1 point and batch 32+, the table under-predicted exit-only savings by 4.4–4.7 points on the CPU and GPU float16, so it overstated merge only's lead over exit only.

![MAC pick against energy pick](outputs/results/fig_mac_vs_energy.png)

**Figure 6.** Predicted savings of the design each signal selects, at 1 point. Source: `mac_vs_energy_agreement.csv`.

| Regime                |   Tolerance (pts) |   Spearman | Same design   |   Energy pick saves % |   MAC pick saves % |   MACs claim % |   MAC regret % |   Token-layer regret % |
|:----------------------|------------------:|-----------:|:--------------|----------------------:|-------------------:|---------------:|---------------:|-----------------------:|
| CPU batch 1           |              0.25 |      0.945 | yes           |                  21.9 |               21.9 |           27   |           0    |                   1.32 |
| CPU batch 1           |              0.5  |      0.945 | no            |                  26.4 |               21.3 |           30.9 |           6.95 |                   0    |
| CPU batch 1           |              1    |      0.945 | no            |                  28.7 |               27.3 |           37.4 |           2.04 |                   2.04 |
| CPU batch 1           |              2    |      0.945 | no            |                  35   |               32.4 |           47   |           3.94 |                   3.94 |
| CPU batch 32          |              0.25 |      0.996 | no            |                  24.7 |               24.4 |           27   |           0.36 |                   0.91 |
| CPU batch 32          |              0.5  |      0.996 | no            |                  28.6 |               28.6 |           30.9 |           0.07 |                   1.44 |
| CPU batch 32          |              1    |      0.996 | yes           |                  34.5 |               34.5 |           37.4 |           0    |                   0    |
| CPU batch 32          |              2    |      0.996 | yes           |                  44.2 |               44.2 |           47   |           0    |                   0    |
| GPU float16 batch 1   |              0.25 |      0.943 | yes           |                  10.6 |               10.6 |           27   |           0    |                   0.76 |
| GPU float16 batch 1   |              0.5  |      0.943 | no            |                  13.2 |                9.6 |           30.9 |           4.13 |                   0    |
| GPU float16 batch 1   |              1    |      0.943 | no            |                  14.5 |               12.9 |           37.4 |           1.91 |                   1.91 |
| GPU float16 batch 1   |              2    |      0.943 | no            |                  17.5 |               16.5 |           47   |           1.21 |                   1.21 |
| GPU float16 batch 32  |              0.25 |      0.994 | no            |                  24.2 |               22.8 |           27   |           1.8  |                   0.79 |
| GPU float16 batch 32  |              0.5  |      0.994 | no            |                  27.7 |               27.5 |           30.9 |           0.34 |                   0.79 |
| GPU float16 batch 32  |              1    |      0.994 | yes           |                  33.7 |               33.7 |           37.4 |           0    |                   0    |
| GPU float16 batch 32  |              2    |      0.994 | yes           |                  43.8 |               43.8 |           47   |           0    |                   0    |
| GPU float16 batch 128 |              0.25 |      0.996 | no            |                  25.1 |               24.8 |           27   |           0.41 |                   0.52 |
| GPU float16 batch 128 |              0.5  |      0.996 | no            |                  29.2 |               29   |           30.9 |           0.31 |                   1.44 |
| GPU float16 batch 128 |              1    |      0.996 | yes           |                  35.1 |               35.1 |           37.4 |           0    |                   0    |
| GPU float16 batch 128 |              2    |      0.996 | yes           |                  45.4 |               45.4 |           47   |           0    |                   0    |

"Regret" is the extra predicted energy from following the signal instead of energy. Spearman correlation is over all non-baseline points.

## 6. Measurement conditions

| Runs                  |   Batch | Slowdown active   |   Temp °C | Clock MHz, median (range)   |   Power W |
|:----------------------|--------:|:------------------|----------:|:----------------------------|----------:|
| float16               |       1 | 100%              |        88 | 1380 (1380 to 1770)         |      20.9 |
| float16               |      32 | 96%               |        87 | 1380 (1380 to 1785)         |      22.1 |
| float16               |     128 | 96%               |        89 | 1380 (1380 to 1785)         |      22.6 |
| float32, no cool-down |       1 | 98%               |        92 | 1282 (630 to 1785)          |      24.6 |
| float32, no cool-down |      32 | 100%              |        92 | 1065 (855 to 1395)          |      28   |
| float32, no cool-down |     128 | 98%               |        92 | 1072 (675 to 1590)          |      28.4 |
| float32, cooled       |       1 | 0%                |        82 | 1785 (1710 to 1785)         |      46.6 |
| float32, cooled       |      32 | 0%                |        81 | 1628 (1545 to 1665)         |      49   |
| float32, cooled       |     128 | 0%                |        81 | 1612 (1530 to 1665)         |      49   |

- **Meters:** NVML energy counter for the GPU (board only), RAPL package for the CPU. Don't compare energy across the two meters. Comparing float16 with float32 is fine because both use NVML.
- **Idle power:** CPU 5.2 W at the start of its table and 5.0 W at the end. GPU float16 14.1 and 16.8 W. GPU float32 14.7 and 16.9 W.
- **Real runs:** 1,000 stratified, shuffled test images with seed-0 heads, at least 5 seconds per measurement, 3 repeats in random order. Real accuracy matched offline accuracy within 0.20 points wherever it was recorded. GPU float16 batch 1 did not complete a full pass inside its time limit, so accuracy is blank there.
- **Throttling:** almost all float16 and uncooled float32 GPU runs had thermal slowdown active. The fan was on Turbo. Float16 held a stable 1380 MHz; uncooled float32 did not, which is why it was re-measured with cool-downs.
- **Warm-up:** the first runs after a cool-down hold higher clocks and cost up to about 20% more per image on float16 batch 32, so reported values are medians of repeats paired with B0.

### B0 for reference

| Regime             |   Batch |   mJ per image |   Net of idle, mJ |   ms per image |   p95 latency ms |   CPU package during GPU run, mJ |   g CO₂e per 1000 images* |
|:-------------------|--------:|---------------:|------------------:|---------------:|-----------------:|---------------------------------:|--------------------------:|
| CPU (RAPL package) |       1 |            373 |               278 |          18.25 |             19.1 |                                  |                     0.041 |
| CPU (RAPL package) |      32 |            289 |               217 |          13.76 |             14.4 |                                  |                     0.032 |
| GPU float16 (NVML) |       1 |            672 |               226 |          31.63 |             31.5 |                              382 |                     0.075 |
| GPU float16 (NVML) |      32 |            221 |                79 |          10.07 |             10.1 |                               94 |                     0.025 |
| GPU float16 (NVML) |     128 |            222 |                83 |           9.83 |              9.8 |                              114 |                     0.025 |
| GPU float32 (NVML) |       1 |            163 |                65 |           6.56 |              6.4 |                               74 |                     0.018 |
| GPU float32 (NVML) |      32 |            109 |                53 |           3.94 |              5.1 |                               34 |                     0.012 |
| GPU float32 (NVML) |     128 |            106 |                52 |           3.69 |              4.2 |                               32 |                     0.012 |

\* At an assumed 0.4 kg CO₂e/kWh. The float32 rows here are the uncooled runs. Cooled B0 is in Section 1.

### Number format

Measured B0, without cool-downs:

|   Batch |   Float16 mJ |   Float32 mJ |   Energy saved % |   Float16 ms per image |   Float32 ms per image | Accuracy on 1,000 images, float16 / float32   |
|--------:|-------------:|-------------:|-----------------:|-----------------------:|-----------------------:|:----------------------------------------------|
|       1 |          672 |          163 |             75.8 |                  31.63 |                   6.56 | not recorded / 91.9%                          |
|      32 |          221 |          109 |             50.9 |                  10.07 |                   3.94 | 91.9% / 91.9%                                 |
|     128 |          222 |          106 |             52.1 |                   9.83 |                   3.69 | 91.9% / 91.9%                                 |

Timing of single operations:

|   Batch |   Embed ms, float16 |   Embed ms, float32 |   Block ms per image, float16 |   Block ms per image, float32 |
|--------:|--------------------:|--------------------:|------------------------------:|------------------------------:|
|       1 |              13.116 |               0.139 |                         1.015 |                         0.391 |
|       8 |              13.342 |               0.41  |                         0.771 |                         0.188 |
|      16 |               0.607 |               0.761 |                         0.678 |                         0.189 |
|      32 |               1.101 |               1.291 |                         0.65  |                         0.186 |
|     128 |               4.568 |               5.141 |                         0.636 |                         0.199 |

Source: `gpu_precision_check.csv`, from `scripts/gpu_precision_check.py`, using untrained weights because timing doesn't depend on them. Float16 autocast adds about 13 ms to patch embedding at batch 8 and below, and makes a block about 3.4× slower per image at large batches. At float16 batch 1, embedding is 50% of predicted B0 energy, and neither exit nor merge can remove it. That is likely why float16 batch 1 savings are low. The underlying reason for the slowdown on this card is not established.

### Footprint of the study

Metered: extraction 71.9 kJ and head training 20.5 kJ (GPU only), plus about 96 kJ of real-run windows (each on its own meter). That's about 52 Wh, or 21 g CO₂e at the assumed 0.4 kg/kWh. It's a lower bound, because cost-table benchmarks, idle time, cool-downs and the rest of the laptop aren't included.

## 7. Limits to state in the paper

- One laptop, one dataset, one frozen backbone, and a simplified ToMe merge. Only the heads are trained, so a fine-tuned model might tolerate merging differently.
- Three repeats per design. Differences under about 2 points, and all batch 1 float32 values, should be treated as indicative.
- Heads were trained on features extracted in float16, then run in float32. The real-against-offline check above covers this.
- Several analysis choices came **after** seeing data, and should be disclosed as such:
  - using medians rather than means of repeats, after seeing warm-up outliers;
  - adding the cooled float32 protocol and the float32 regime itself, after finding the float16 slowdown;
  - adding the batch 128 exit design, because the throttled table had picked a poor one. The original pick is still reported (15.9%).
- CO₂ uses an assumed grid factor, not a measured one.
- No data-centre GPU was measured. A Kaggle T4 would test whether the float16 result is specific to this card.

## 8. Corrections made while checking these results

Earlier files may show the old values.

| Issue | Effect | Fix |
|---|---|---|
| `paired_boot` compared correctness with raw class labels | Accuracy difference read about −3.6 for every design | Compares correctness with correctness, in points |
| Thermal share counted CPU rows | Reported 58%; actual 96–100% of GPU float16 runs | Reported per GPU regime |
| Means of repeats included warm-up runs | Inflated some energies by up to about 20% | Medians, paired with B0 per repeat |
| One overall prediction error | A 2.5% median hid about 20% on CPU batch 32 | Reported per regime |
| MAC table used validation costs | Disagreed slightly with other tables | Uses test costs |

## 9. Reproduce

```bash
# CPU and GPU float16: everything through the report
GV_HW=laptop jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 --output-dir outputs --output experiment_run_cpu_float16.ipynb experiment.ipynb
# GPU float32
GV_AMP=0 GV_HW=laptopfp32 GV_CPU_REGIME=0 GV_STAGES=energy,select,real,report jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 --output-dir outputs --output experiment_run_float32.ipynb experiment.ipynb
# Cooled float32, precision check, final report and this document
python scripts/float32_cooled.py
python scripts/gpu_precision_check.py
GV_STAGES=report jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 --output-dir outputs --output report_rerun.ipynb experiment.ipynb
python scripts/make_results_doc.py
```

The executed notebooks in `outputs/` predate the final report fixes. `experiment.ipynb` is the current code.

## Appendix: full tables

### Throttled GPU float32 real runs

These are the uncooled runs, kept for completeness. Clocks swung during them, so quote the cooled table in Section 1 instead. Designs were chosen using the invalid float32 table.

| Regime                |   Tolerance (pts) | Method       | Design                                         |   Test accuracy vs B0 (pts) | Energy saved % (range)   |   Time saved % |   mJ per image |
|:----------------------|------------------:|:-------------|:-----------------------------------------------|----------------------------:|:-------------------------|---------------:|---------------:|
| GPU float32 batch 1   |               0.5 | Exit only    | `m[none]e[6mlp192_10lin]f[lin]` τ=0.16         |                       -0.5  | 4.1 (1.0 to 27.1)        |            6.1 |            155 |
| GPU float32 batch 1   |               0.5 | Merge only   | `m[8r50]e[none]f[lin]` τ=0                     |                       -0.45 | -20.5 (-23.8 to 1.6)     |          -15   |            196 |
| GPU float32 batch 1   |               0.5 | Exit + merge | `m[8r50]e[6mlp192_10mlp192]f[lin]` τ=0.16      |                       -0.78 | 9.8 (-6.6 to 21.4)       |            3.5 |            145 |
| GPU float32 batch 1   |               1   | Exit only    | `m[none]e[6mlp192_10mlp192]f[lin]` τ=0.2       |                       -0.65 | 22.5 (6.4 to 27.0)       |            8.6 |            131 |
| GPU float32 batch 1   |               1   | Merge only   | `m[6r50]e[none]f[lin]` τ=0                     |                       -0.79 | -18.9 (-25.3 to 12.3)    |          -18.3 |            192 |
| GPU float32 batch 1   |               1   | Exit + merge | `m[8r50]e[6mlp192_10lin]f[lin]` τ=0.2          |                       -1.19 | 17.6 (16.7 to 24.9)      |           16.2 |            134 |
| GPU float32 batch 32  |               0.5 | Exit only    | `m[none]e[6mlp192_10mlp192]f[lin]` τ=0.16      |                       -0.41 | 21.3 (20.7 to 25.7)      |           24.1 |             84 |
| GPU float32 batch 32  |               0.5 | Merge only   | `m[8r50_10r25]e[none]f[lin]` τ=0               |                       -0.46 | 18.3 (16.8 to 19.1)      |           17.4 |             90 |
| GPU float32 batch 32  |               0.5 | Exit + merge | `m[6r25_8r50]e[10mlp192]f[lin]` τ=0.25         |                       -0.86 | 31.3 (28.8 to 32.7)      |           35.1 |             75 |
| GPU float32 batch 32  |               1   | Exit only    | `m[none]e[6mlp192_10mlp192]f[lin]` τ=0.2       |                       -0.65 | 26.9 (26.3 to 27.9)      |           26.1 |             80 |
| GPU float32 batch 32  |               1   | Merge only   | `m[6r50_8r25]e[none]f[lin]` τ=0                |                       -0.88 | 30.5 (28.7 to 30.8)      |           34.7 |             75 |
| GPU float32 batch 32  |               1   | Exit + merge | `m[6r50_8r25]e[4mlp192_10mlp192]f[lin]` τ=0.16 |                       -0.86 | 36.4 (35.4 to 37.9)      |           35.1 |             67 |
| GPU float32 batch 128 |               0.5 | Exit only    | `m[none]e[10mlp192]f[lin]` τ=0.4               |                       -0.53 | 16.2 (0.3 to 22.2)       |           17.2 |             99 |
| GPU float32 batch 128 |               0.5 | Merge only   | `m[8r50_10r25]e[none]f[lin]` τ=0               |                       -0.46 | 19.4 (19.1 to 19.5)      |           20.4 |             86 |
| GPU float32 batch 128 |               0.5 | Exit + merge | `m[6r25_8r50]e[10mlp192]f[lin]` τ=0.25         |                       -0.86 | 29.5 (28.9 to 30.4)      |           31.3 |             75 |
| GPU float32 batch 128 |               1   | Exit only    | `m[none]e[10mlp192]f[lin]` τ=0.5               |                       -0.69 | 15.5 (12.8 to 16.2)      |           17.4 |             93 |
| GPU float32 batch 128 |               1   | Merge only   | `m[6r50_8r25]e[none]f[lin]` τ=0                |                       -0.88 | 26.3 (23.0 to 28.8)      |           27.7 |             82 |
| GPU float32 batch 128 |               1   | Exit + merge | `m[6r50_8r25]e[4mlp192_10mlp192]f[lin]` τ=0.16 |                       -0.86 | 35.9 (34.4 to 43.9)      |           38.8 |             70 |

### All 50 merge schedules

`6r50_8r25` means removing 50% of tokens after block 6, then 25% after block 8. Savings are cost-table predictions. Source: `merge_only_effect.csv`.

| Schedule   |   Merges |   Test % |   vs B0 (pts) |   MACs saved % |   CPU b1 % |   CPU b32 % |   GPU f16 b1 % |   GPU f16 b32 % |   GPU f16 b128 % |
|:-----------|---------:|---------:|--------------:|---------------:|-----------:|------------:|---------------:|----------------:|-----------------:|
| 6r25       |        1 |    91.93 |         -0.1  |           13.3 |        8   |        12.4 |            4.3 |            11.5 |             13.2 |
| 10r50      |        1 |    91.9  |         -0.13 |            8.5 |        3.8 |         7.4 |            2.1 |             7.8 |              8.2 |
| 10r25      |        1 |    91.9  |         -0.13 |            4.3 |        1.2 |         3.7 |            0.5 |             3.6 |              4.2 |
| 8r25       |        1 |    91.78 |         -0.24 |            8.8 |        4.6 |         8.1 |            2.4 |             7.6 |              8.7 |
| 6r25_10r25 |        2 |    91.64 |         -0.38 |           16.4 |        8.5 |        14.9 |            4.5 |            15   |             16.2 |
| 8r25_10r50 |        2 |    91.63 |         -0.4  |           14.9 |        5.8 |        13.4 |            2.1 |            13   |             14.1 |
| 8r50       |        1 |    91.58 |         -0.45 |           17.1 |       10   |        15.6 |            5.7 |            16   |             16.8 |
| 8r25_10r25 |        2 |    91.57 |         -0.46 |           11.9 |        5.2 |        10.5 |            2.6 |            11   |             11.7 |
| 8r50_10r25 |        2 |    91.56 |         -0.46 |           19   |        9.6 |        17.4 |            4.4 |            17.3 |             18.3 |
| 8r50_10r50 |        2 |    91.54 |         -0.49 |           21   |        9.7 |        19.4 |            5.1 |            19.5 |             20.4 |
| 6r25_8r25  |        2 |    91.53 |         -0.5  |           19.5 |       11   |        17.8 |            6.1 |            18.7 |             19.4 |
| 6r25_10r50 |        2 |    91.46 |         -0.56 |           19.4 |        9.2 |        17.8 |            4.1 |            17   |             18.7 |
| 4r25       |        1 |    91.46 |         -0.57 |           17.8 |       11.3 |        16.8 |            6.2 |            15.5 |             17.8 |
| 4r25_10r25 |        2 |    91.25 |         -0.78 |           20.9 |       11.9 |        19.3 |            6.4 |            18.9 |             20.7 |
| 6r50_10r25 |        2 |    91.25 |         -0.78 |           27.6 |       15.8 |        25.7 |            8   |            25.5 |             26.9 |
| 6r50       |        1 |    91.24 |         -0.79 |           25.7 |       16.1 |        23.8 |            9.3 |            24.2 |             25.3 |
| 4r25_10r50 |        2 |    91.2  |         -0.83 |           23.9 |       12.6 |        22.2 |            6   |            20.9 |             23.2 |
| 6r25_8r50  |        2 |    91.17 |         -0.86 |           25.6 |       12.4 |        23.8 |            5.3 |            22.8 |             24.4 |
| 4r25_8r25  |        2 |    91.15 |         -0.88 |           24   |       14.3 |        22.2 |            8   |            22.7 |             23.9 |
| 6r50_8r25  |        2 |    91.14 |         -0.88 |           29.6 |       17   |        27.8 |            8.2 |            27   |             28.6 |
| 6r50_10r50 |        2 |    91.06 |         -0.97 |           29.6 |       15.9 |        27.6 |            8.6 |            27.7 |             29   |
| 4r25_8r50  |        2 |    90.9  |         -1.13 |           30   |       15.8 |        28.2 |            7.2 |            26.7 |             28.9 |
| 4r25_6r25  |        2 |    90.81 |         -1.22 |           27.1 |       16.7 |        25.2 |            9.7 |            26.5 |             27.2 |
| 6r50_8r50  |        2 |    90.67 |         -1.36 |           33.5 |       17.4 |        31.8 |            9.4 |            31.5 |             32.8 |
| 2r25       |        1 |    90.6  |         -1.42 |           22.3 |       14.7 |        21.2 |            8.1 |            19.5 |             22.3 |
| 2r25_10r25 |        2 |    90.41 |         -1.62 |           25.4 |       15.2 |        23.7 |            8.3 |            22.9 |             25.3 |
| 2r25_8r25  |        2 |    90.31 |         -1.72 |           28.5 |       17.7 |        26.6 |            9.9 |            26.7 |             28.5 |
| 4r25_6r50  |        2 |    90.28 |         -1.75 |           36.2 |       19   |        34.1 |            8.4 |            32.5 |             34.6 |
| 2r25_10r50 |        2 |    90.27 |         -1.76 |           28.4 |       15.9 |        26.6 |            7.9 |            24.9 |             27.7 |
| 4r50       |        1 |    90.21 |         -1.81 |           34.3 |       22.2 |        32.1 |           12.8 |            32.4 |             33.9 |
| 2r25_6r25  |        2 |    90.21 |         -1.81 |           31.6 |       20.1 |        29.6 |           11.6 |            30.5 |             31.7 |
| 4r50_8r25  |        2 |    90.19 |         -1.83 |           38.2 |       23.1 |        36   |           11.7 |            35.2 |             37.1 |
| 4r50_10r25 |        2 |    90.19 |         -1.84 |           36.2 |       21.9 |        33.9 |           11.6 |            33.7 |             35.4 |
| 2r25_8r50  |        2 |    90.07 |         -1.96 |           34.5 |       19.2 |        32.6 |            9.1 |            30.7 |             33.5 |
| 4r50_10r50 |        2 |    90.04 |         -1.99 |           38.2 |       22   |        35.9 |           12.2 |            35.9 |             37.6 |
| 4r50_6r25  |        2 |    89.97 |         -2.06 |           40.1 |       24.4 |        38.1 |           11.9 |            36.8 |             38.8 |
| 2r25_4r25  |        2 |    89.95 |         -2.08 |           34.7 |       22.5 |        32.5 |           13.2 |            34.2 |             34.9 |
| 4r50_8r50  |        2 |    89.93 |         -2.1  |           42.1 |       23.5 |        40   |           13   |            39.7 |             41.4 |
| 2r25_6r50  |        2 |    89.39 |         -2.64 |           40.7 |       22.4 |        38.5 |           10.3 |            36.5 |             39.2 |
| 4r50_6r50  |        2 |    88.92 |         -3.11 |           46   |       25   |        44.1 |           13.8 |            43.5 |             45.2 |
| 2r25_4r50  |        2 |    87.95 |         -4.08 |           46.8 |       25.6 |        44.5 |           11.5 |            42.3 |             44.9 |
| 2r50_10r25 |        2 |    85.38 |         -6.65 |           44.8 |       28.1 |        42.1 |           15.1 |            41.9 |             44   |
| 2r50       |        1 |    85.33 |         -6.7  |           42.9 |       28.4 |        40.3 |           16.4 |            40.6 |             42.5 |
| 2r50_10r50 |        2 |    85.29 |         -6.74 |           46.8 |       28.2 |        44.1 |           15.8 |            44.1 |             46.1 |
| 2r50_8r25  |        2 |    85.28 |         -6.75 |           46.8 |       29.3 |        44.2 |           15.3 |            43.4 |             45.7 |
| 2r50_6r25  |        2 |    85.05 |         -6.98 |           48.7 |       30.5 |        46.4 |           15.5 |            45   |             47.4 |
| 2r50_8r50  |        2 |    84.95 |         -7.08 |           50.7 |       29.6 |        48.2 |           16.6 |            47.9 |             50   |
| 2r50_4r25  |        2 |    84.42 |         -7.61 |           50.7 |       31.8 |        48.5 |           15.7 |            46.5 |             49.1 |
| 2r50_6r50  |        2 |    83.8  |         -8.23 |           54.6 |       31.1 |        52.4 |           17.4 |            51.7 |             53.8 |
| 2r50_4r50  |        2 |    81.93 |        -10.1  |           58.5 |       32.6 |        56.5 |           18.2 |            55.5 |             57.6 |

### Break-even for every merge

Source: `merge_breakeven.csv`, gross energy. Net-of-idle and time versions are in the same file.

| Device      |   Batch | Tokens    |   Merge cost, in blocks |   Block saving % |   Break-even, blocks |   MAC break-even |
|:------------|--------:|:----------|------------------------:|-----------------:|---------------------:|-----------------:|
| CPU         |       1 | 100 → 51  |                    0.33 |             14.6 |                 2.3  |            0.019 |
| CPU         |       1 | 100 → 76  |                    0.31 |             12.2 |                 2.54 |            0.038 |
| CPU         |       1 | 148 → 75  |                    0.31 |             25.3 |                 1.23 |            0.027 |
| CPU         |       1 | 148 → 112 |                    0.29 |             18.8 |                 1.52 |            0.053 |
| CPU         |       1 | 197 → 100 |                    0.29 |             37.9 |                 0.76 |            0.034 |
| CPU         |       1 | 197 → 148 |                    0.26 |             20.7 |                 1.25 |            0.065 |
| CPU         |       8 | 100 → 51  |                    0.14 |             49.4 |                 0.28 |            0.019 |
| CPU         |       8 | 100 → 76  |                    0.13 |             24.6 |                 0.53 |            0.038 |
| CPU         |       8 | 148 → 75  |                    0.11 |             51.1 |                 0.22 |            0.027 |
| CPU         |       8 | 148 → 112 |                    0.12 |             25.4 |                 0.46 |            0.053 |
| CPU         |       8 | 197 → 100 |                    0.08 |             60.5 |                 0.14 |            0.034 |
| CPU         |       8 | 197 → 148 |                    0.08 |             40.4 |                 0.2  |            0.065 |
| CPU         |      32 | 100 → 51  |                    0.08 |             51   |                 0.16 |            0.019 |
| CPU         |      32 | 100 → 76  |                    0.07 |             26.2 |                 0.27 |            0.038 |
| CPU         |      32 | 148 → 75  |                    0.09 |             49.9 |                 0.19 |            0.027 |
| CPU         |      32 | 148 → 112 |                    0.08 |             24.7 |                 0.34 |            0.053 |
| CPU         |      32 | 197 → 100 |                    0.1  |             50.3 |                 0.2  |            0.034 |
| CPU         |      32 | 197 → 148 |                    0.09 |             26.9 |                 0.33 |            0.065 |
| GPU float16 |       1 | 100 → 51  |                    0.61 |             17.2 |                 3.55 |            0.019 |
| GPU float16 |       1 | 100 → 76  |                    0.62 |              3.9 |                15.78 |            0.038 |
| GPU float16 |       1 | 148 → 75  |                    0.45 |             19   |                 2.37 |            0.027 |
| GPU float16 |       1 | 148 → 112 |                    0.45 |             25.8 |                 1.75 |            0.053 |
| GPU float16 |       1 | 197 → 100 |                    0.35 |             43.1 |                 0.81 |            0.034 |
| GPU float16 |       1 | 197 → 148 |                    0.35 |             23   |                 1.51 |            0.065 |
| GPU float16 |       8 | 100 → 51  |                    0.16 |             44.1 |                 0.37 |            0.019 |
| GPU float16 |       8 | 100 → 76  |                    0.16 |             16.6 |                 0.99 |            0.038 |
| GPU float16 |       8 | 148 → 75  |                    0.11 |             48.3 |                 0.22 |            0.027 |
| GPU float16 |       8 | 148 → 112 |                    0.11 |             36.8 |                 0.3  |            0.053 |
| GPU float16 |       8 | 197 → 100 |                    0.09 |             54.9 |                 0.16 |            0.034 |
| GPU float16 |       8 | 197 → 148 |                    0.08 |             27.9 |                 0.3  |            0.065 |
| GPU float16 |      32 | 100 → 51  |                    0.06 |             45.7 |                 0.14 |            0.019 |
| GPU float16 |      32 | 100 → 76  |                    0.06 |             18.6 |                 0.32 |            0.038 |
| GPU float16 |      32 | 148 → 75  |                    0.06 |             46.3 |                 0.13 |            0.027 |
| GPU float16 |      32 | 148 → 112 |                    0.05 |             30.1 |                 0.18 |            0.053 |
| GPU float16 |      32 | 197 → 100 |                    0.05 |             49.7 |                 0.1  |            0.034 |
| GPU float16 |      32 | 197 → 148 |                    0.05 |             24.1 |                 0.2  |            0.065 |
| GPU float16 |     128 | 100 → 51  |                    0.04 |             48.3 |                 0.09 |            0.019 |
| GPU float16 |     128 | 100 → 76  |                    0.04 |             21.4 |                 0.19 |            0.038 |
| GPU float16 |     128 | 148 → 75  |                    0.05 |             47.8 |                 0.1  |            0.027 |
| GPU float16 |     128 | 148 → 112 |                    0.05 |             27   |                 0.17 |            0.053 |
| GPU float16 |     128 | 197 → 100 |                    0.05 |             52   |                 0.09 |            0.034 |
| GPU float16 |     128 | 197 → 148 |                    0.04 |             27.5 |                 0.16 |            0.065 |

### Operation costs

Each cell is mJ / ms per **batch**, not per image. Source: `op_costs.csv`, which has every operation and net-of-idle values. The GPU float32 columns come from the invalid throttled table and are shown only for completeness.

| Operation             | CPU b1      | CPU b8        | CPU b32       | GPU float16 b1   | GPU float16 b8   | GPU float16 b32   | GPU float16 b128   | GPU float32 b1   | GPU float32 b8   | GPU float32 b32   | GPU float32 b128   |
|:----------------------|:------------|:--------------|:--------------|:-----------------|:-----------------|:------------------|:-------------------|:-----------------|:-----------------|:------------------|:-------------------|
| Patch embedding       | 9.9 / 0.40  | 56.3 / 2.39   | 212.6 / 9.49  | 327.1 / 16.72    | 353.1 / 17.05    | 59.1 / 2.08       | 261.2 / 10.23      | 4.4 / 0.17       | 30.1 / 1.41      | 274.2 / 14.48     | 410.3 / 17.13      |
| Block, 197 tokens     | 31.6 / 1.28 | 240.2 / 10.56 | 898.6 / 40.90 | 27.2 / 1.22      | 173.5 / 7.86     | 574.5 / 26.42     | 2354.4 / 104.41    | 17.0 / 0.57      | 101.5 / 4.41     | 780.5 / 39.75     | 1373.3 / 55.31     |
| Block, 148 tokens     | 25.0 / 0.97 | 143.2 / 6.34  | 657.0 / 29.67 | 20.9 / 0.91      | 125.1 / 5.59     | 436.2 / 19.61     | 1707.4 / 75.46     | 15.3 / 0.54      | 71.9 / 3.24      | 548.3 / 27.94     | 983.4 / 39.47      |
| Block, 100 tokens     | 19.6 / 0.68 | 95.0 / 4.17   | 446.2 / 20.03 | 15.5 / 0.68      | 78.2 / 3.47      | 289.0 / 13.00     | 1130.8 / 49.90     | 10.6 / 0.38      | 56.8 / 2.65      | 261.9 / 12.56     | 653.0 / 25.76      |
| Block, 76 tokens      | 17.2 / 0.55 | 71.6 / 3.16   | 329.3 / 14.76 | 14.8 / 0.65      | 65.3 / 2.91      | 235.3 / 10.67     | 889.0 / 38.99      | 9.7 / 0.37       | 34.7 / 1.53      | 169.2 / 7.75      | 537.7 / 22.46      |
| Block, 51 tokens      | 16.7 / 0.39 | 48.1 / 2.18   | 218.6 / 10.02 | 12.8 / 0.42      | 43.7 / 2.00      | 156.9 / 7.23      | 584.4 / 25.45      | 8.4 / 0.34       | 18.7 / 0.75      | 95.4 / 4.10       | 349.6 / 14.48      |
| Merge 50%, 197 tokens | 9.1 / 0.39  | 20.4 / 0.90   | 91.3 / 4.23   | 9.5 / 0.50       | 14.8 / 0.63      | 28.0 / 0.89       | 110.5 / 3.76       | 10.2 / 0.51      | 12.4 / 0.58      | 90.1 / 4.35       | 212.7 / 9.12       |
| Merge 25%, 197 tokens | 8.2 / 0.34  | 19.8 / 0.88   | 79.6 / 3.66   | 9.4 / 0.50       | 14.5 / 0.61      | 27.0 / 0.86       | 104.7 / 3.33       | 10.0 / 0.50      | 12.4 / 0.57      | 88.2 / 4.20       | 216.9 / 9.63       |
| Merge 50%, 100 tokens | 6.5 / 0.27  | 13.0 / 0.58   | 36.2 / 1.67   | 9.4 / 0.51       | 12.7 / 0.61      | 18.1 / 0.63       | 47.5 / 1.33        | 13.1 / 0.67      | 12.3 / 0.58      | 41.5 / 2.05       | 107.2 / 4.51       |
| Merge 25%, 100 tokens | 6.1 / 0.25  | 12.5 / 0.58   | 32.1 / 1.48   | 9.6 / 0.51       | 12.8 / 0.62      | 17.2 / 0.60       | 46.9 / 1.16        | 14.1 / 0.72      | 11.8 / 0.56      | 39.5 / 1.97       | 85.7 / 3.50        |
| Exit head, lin        | 0.9 / 0.05  | 1.6 / 0.06    | 1.7 / 0.07    | 4.0 / 0.22       | 4.9 / 0.26       | 6.2 / 0.33        | 4.7 / 0.24         | 4.1 / 0.21       | 4.3 / 0.22       | 4.8 / 0.29        | 5.0 / 0.25         |
| Exit head, mlp192     | 2.5 / 0.10  | 2.8 / 0.12    | 3.1 / 0.14    | 4.9 / 0.27       | 6.0 / 0.32       | 7.7 / 0.40        | 8.4 / 0.42         | 5.1 / 0.26       | 5.5 / 0.28       | 7.2 / 0.36        | 6.7 / 0.32         |
| Final head, lin       | 1.2 / 0.05  | 2.4 / 0.10    | 7.3 / 0.34    | 2.6 / 0.14       | 4.1 / 0.17       | 8.8 / 0.28        | 30.9 / 1.08        | 2.6 / 0.13       | 4.1 / 0.20       | 31.9 / 1.74       | 45.5 / 1.98        |

## Which file proves what

| Claim | File |
|---|---|
| Measured savings per regime and tolerance | `measured_comparison.csv`, `float32_cooled_summary.csv` |
| Every individual measurement, with temperature and clock | `real_runs.csv`, `real_runs_float32_cooled.csv` |
| Accuracy and confidence intervals | `finalist_metrics.csv`, `all_points.csv.gz` |
| Exit heads, temperatures and calibration | `slot_metrics.csv` |
| All designs and thresholds | `all_points.csv.gz`, `designs.json` |
| Merge schedules | `merge_only_effect.csv` |
| Operation costs and break-even | `op_costs.csv`, `energy_tables.json`, `merge_breakeven.csv` |
| MACs against energy | `mac_vs_energy_agreement.csv`, `picks.csv` |
| Prediction accuracy | `real_measured.csv` |
| Number format | `gpu_precision_check.csv` |
| Data splits and seeds | `splits.json`, `ranges.json` |
| Run log | `progress.log` |
