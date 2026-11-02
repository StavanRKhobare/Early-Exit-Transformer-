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

![Measured savings per regime and method](fig_measured_savings.png)

**Figure 1.** Measured savings against B0 run in the same repeat. Bars are medians, whiskers the range over repeats, and black dashes the cost-table prediction (no dash for cooled float32, whose table was unusable). Source: `measured_comparison.csv`, `float32_cooled_summary.csv`. Raw runs: `real_runs.csv`, `real_runs_float32_cooled.csv`.

### 1-point tolerance (CPU and GPU float16)

{{MEASURED1}}

Designs were selected on validation data. "With CPU package" adds RAPL package energy read during the GPU run. It's noisier, because the package also serves everything else on the machine.

### 1-point tolerance (GPU float32, cool-down before every run)

{{COOLED}}

"(added)" marks a design measured after the main run; see Section 6. Each repeat is 5 seconds or more, starting at 70–71 °C.

### 0.5-point tolerance (CPU and GPU float16)

{{MEASURED05}}

The CPU batch 32 merge-only value of 8.3% looks unreliable: its repeats range from 4.8% to 17.7%, and the same design measures 17.8% on the GPU.

### Latency at the 1-point tolerance (CPU and GPU float16)

{{LATENCY}}

Per-image latency at the median of repeats. At batch 32 and above this is batch time divided by batch size, so it doesn't include waiting to fill a batch.

### How often samples exit and reach a merge

{{PICKS}}

Fractions are on validation, averaged across seeds. Source: `all_points.csv.gz`.

## 2. Accuracy cost

### Baselines by final head

{{HEADS}}

The larger final heads beat B0 without any exit or merge benefit, so all comparisons use the linear head as the reference. Source: `all_points.csv.gz`, `finalist_metrics.csv`.

### Exit head accuracy by block, no merging

{{EXITHEADS}}

Means across 5 seeds. "ECE" is calibration error on test, before and after temperature scaling. Source: `slot_metrics.csv`, which also holds every merged prefix.

![Accuracy cost of one merge](fig_merge_accuracy.png)

**Figure 2.** Test accuracy of one merge against B0, by block and strength. Error bars are the standard deviation across 5 seeds. Dashed line: 1 point. Source: `merge_only_effect.csv`.

{{MERGES}}

Savings in this table are cost-table predictions. The full 50 schedules are in `merge_only_effect.csv`; the worst loses 10.1 points.

### Accuracy of the designs compared in Section 1

{{DESIGNS}}

Every design here is below B0, and each seed-0 interval excludes zero. Source: `finalist_metrics.csv`.

**Selection against test results.** The tolerance is applied on validation, so test results can overshoot it. The CPU batch 1 exit + merge design lands at −1.19 points against a 1-point tolerance. At the 0.5-point tolerance every exit + merge pick overshoots (−0.78 to −0.86), while the exit-only and merge-only picks stay within it (−0.41 to −0.46).

## 3. Why batch size matters: merge break-even

![Merge break-even heatmap](fig_merge_breakeven.png)

**Figure 3.** Blocks a merge must run before its own cost is repaid. Source: `merge_breakeven.csv`; cost tables in `energy_tables.json` and `op_costs.csv`. Float32 is left out because its table is unreliable (Section 5).

- **Batch 1:** break-even is 0.8–2.5 blocks on the CPU and 0.8–3.6 on GPU float16, with one cell above 12. There, merging from 100 to 76 tokens saves so little that it's hard to separate from noise.
- **Batch 8 and above:** break-even is 0.1–1.0 blocks.
- **MACs:** they put break-even at 0.02–0.07 blocks, so they understate merge overhead at batch 1 by roughly 20 to 200 times.
- **Merge cost:** one merge costs 0.26–0.33 of a block on the CPU and 0.35–0.62 on GPU float16 at batch 1, and 0.04–0.10 at batch 32+.

## 4. Accuracy against energy, all designs

![Accuracy against predicted energy](fig_fronts.png)

**Figure 4.** Best validation accuracy at each predicted energy, across 5,408 designs and up to 15 thresholds each (80,378 rows in `all_points.csv.gz`), for points within 5 points of B0. Designs where under 5% of samples reach the merge are not credited to merging. Costs are predicted, so use Section 1 for quoted savings.

### Predicted savings at all four tolerances

{{ADDED}}

Source: `merge_added_value.csv`, reported on test. The best overall design differs across regimes; picks are in `picks.csv` and `method_comparison.csv`.

## 5. Can the cost table and MACs be trusted?

![Predicted against measured energy](fig_pred_vs_measured.png)

**Figure 5.** Predicted against measured energy for all measured operating points. Source: `real_measured.csv`.

{{ERRORS}}

- **Use measured values for absolute energy.** The headline median hides large regime differences. CPU batch 32 is over-predicted by about 20%; the cause is not established.
- **The float32 table is invalid.** It was taken at the 92 °C limit with clocks swinging, and over-predicts by up to 2.9×. Its rows remain in the CSVs under `cuda_laptopfp32` and are left out of Figures 3, 4 and 6.
- **Predicted savings are close but biased.** At 1 point and batch 32+, the table under-predicted exit-only savings by 4.4–4.7 points on the CPU and GPU float16, so it overstated merge only's lead over exit only.

![MAC pick against energy pick](fig_mac_vs_energy.png)

**Figure 6.** Predicted savings of the design each signal selects, at 1 point. Source: `mac_vs_energy_agreement.csv`.

{{MACS}}

"Regret" is the extra predicted energy from following the signal instead of energy. Spearman correlation is over all non-baseline points.

## 6. Measurement conditions

{{THERMAL}}

- **Meters:** NVML energy counter for the GPU (board only), RAPL package for the CPU. Don't compare energy across the two meters. Comparing float16 with float32 is fine because both use NVML.
- **Idle power:** CPU 5.2 W at the start of its table and 5.0 W at the end. GPU float16 14.1 and 16.8 W. GPU float32 14.7 and 16.9 W.
- **Real runs:** 1,000 stratified, shuffled test images with seed-0 heads, at least 5 seconds per measurement, 3 repeats in random order. Real accuracy matched offline accuracy within 0.20 points wherever it was recorded. GPU float16 batch 1 did not complete a full pass inside its time limit, so accuracy is blank there.
- **Throttling:** almost all float16 and uncooled float32 GPU runs had thermal slowdown active. The fan was on Turbo. Float16 held a stable 1380 MHz; uncooled float32 did not, which is why it was re-measured with cool-downs.
- **Warm-up:** the first runs after a cool-down hold higher clocks and cost up to about 20% more per image on float16 batch 32, so reported values are medians of repeats paired with B0.

### B0 for reference

{{B0S}}

\* At an assumed 0.4 kg CO₂e/kWh. The float32 rows here are the uncooled runs. Cooled B0 is in Section 1.

### Number format

Measured B0, without cool-downs:

{{PRECENERGY}}

Timing of single operations:

{{PRECISION}}

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

{{THROTTLED}}

### All 50 merge schedules

`6r50_8r25` means removing 50% of tokens after block 6, then 25% after block 8. Savings are cost-table predictions. Source: `merge_only_effect.csv`.

{{ALLMERGES}}

### Break-even for every merge

Source: `merge_breakeven.csv`, gross energy. Net-of-idle and time versions are in the same file.

{{BREAKEVEN}}

### Operation costs

Each cell is mJ / ms per **batch**, not per image. Source: `op_costs.csv`, which has every operation and net-of-idle values. The GPU float32 columns come from the invalid throttled table and are shown only for completeness.

{{OPS}}

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
