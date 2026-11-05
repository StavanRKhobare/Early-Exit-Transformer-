# Green ViT: what we do next

*Written 2026-11-04. Results so far are in the doc "Green ViT: Complete Results Record" and in `RESULTS.md`.*

## Live status (6-hour run, started 2026-11-04 22:57)

| Block | Work | Status | Result |
|---|---|---|---|
| 1 | Shared-backbone training | **Passed** at 00:50 (1 seed) | Validation, shared vs dedicated: no merge 96.88 vs 96.98 (−0.10); `6r50_8r25` 96.62 vs 96.70 (−0.08). Other shared schedules, no dedicated baseline: `8r50` 96.52, `6r50` 96.82, `4r25_8r25` 96.22, `4r50` 96.22. With 4,000 images the standard error is about 0.27 pts, so these gaps cannot be told apart from zero |
| 2 | Calibration and head outputs | Done at 02:19 | Results in `outputs/results/shared_backbone/` |
| 3 | Certified selection | Done at 02:33, test report at 03:44 | Results in `outputs/results/certified/` |
| 4 | Real energy measurements | Done at 03:42, 144 measurements | Results in `outputs/results/real_energy/` |

## Overnight results (finished 2026-11-05 03:44)

**Result:** one Vision Transformer, trained once across six merge schedules, used 34.8–36.2% less measured energy at batch 32 than the fine-tuned B0, and lost 0.24 points of test accuracy (B0 96.70%, standard error 0.14). The chosen design was selected under a 0.5-point limit with an exact statistical bound. This is one trained backbone on CIFAR-10, so it is a proof of concept, not a paper-ready result.

### Method
- **Training:** each step samples one of six merge schedules, with one shared backbone and one final head.
- **Exit heads:** patch-average MLP heads trained after the backbone, with an entropy threshold at each exit.
- **Selection:** a Pareto path is built on 10,000 unused images and tested in fixed order on 15,000 more. The bound is Clopper-Pearson on both discordant rates, joined by a union bound, at delta 0.1. Test images were not used for selection.
- **Baseline:** dedicated fine-tuned B0 with its own head.

### Shared training gate (validation, 4,000 images)
| Schedule | Shared | Dedicated | Gap |
|---|---|---|---|
| No merging | 96.88% | 96.98% | −0.10 |
| `6r50_8r25` | 96.62% | 96.70% | −0.08 |
| `6r50` | 96.82% | not trained | |
| `8r50` | 96.52% | not trained | |
| `4r25_8r25` | 96.22% | not trained | |
| `4r50` | 96.22% | not trained | |

The standard error of one accuracy on 4,000 images is about 0.27 points, so these gaps cannot be told apart from zero.

### Measured energy saved against B0
Six paired repeats on 1,000 unused images. Values are percent saved with 95% CI half-widths, using gross energy.

| Design | Limit it was chosen under | Test accuracy change | GPU batch 32 | CPU batch 32 | GPU batch 1 | CPU batch 1 |
|---|---|---|---|---|---|---|
| `6r50 g8@0.05 L10@0.12` | 0.5 | −0.24 | 36.2 ± 0.9 | 34.8 ± 2.7 | 29.9 ± 1.9 | 26.9 ± 1.1 |
| `6r50 g6@0.03 L8@0.05` | 0.5 | −0.24 | 35.6 ± 1.7 | 34.7 ± 3.6 | 31.3 ± 1.5 | 27.4 ± 0.6 |
| `6r50 g4@0.12 L8@0.08` | 1 | −0.74 | 43.4 ± 1.1 | 40.5 ± 3.0 | 38.0 ± 1.3 | 35.1 ± 1.6 |
| `4r50 g6@0.05 L8@0.16` | 1 | **−1.08, over the limit** | 47.1 ± 0.6 | 46.2 ± 2.3 | 40.6 ± 1.0 | 36.0 ± 1.1 |
| `4r50 g6@0.03 L8@0.16` | 1 | **−1.06, over the limit** | 46.1 ± 0.9 | 46.3 ± 1.6 | 39.5 ± 1.5 | 35.7 ± 0.5 |

### Predicted comparison on test (GPU batch 32, 0.5-point limit)
| Selection or model | Predicted energy saved | Test accuracy change |
|---|---|---|
| Shared model, no guarantee (naive) | 40.5% | −0.62, **over the limit** |
| Shared model, z margin (approximate) | 36.2% | −0.32 |
| Shared model, Pareto test, exact bound | 34.0% | −0.24 |
| Shared model, one-shot exact Bonferroni | 18.4% | −0.11 |
| Shared backbone, no merging, exact bound | 24.9% | −0.29 |
| Dedicated B0, exits only, exact bound | 24.5% | −0.20 |
| Dedicated `6r50_8r25`, exact bound | Nothing passed | |
| Dedicated `6r50_8r25`, z rule | 36.2% | −0.29 |

Under the exact bound, adding the merge option to the shared model raised predicted savings by about 9 points at batch 32, and by only 1.8 points on CPU batch 1.

### Repeatability (50 resplits of the unused images, GPU batch 32, 0.5-point limit)
Each resplit selects on 20,000 images and reports on 5,000. "Over the limit" is measured against all 25,000 unused images.

| Rule | Mean predicted saved | SD | Over the limit | Found a design |
|---|---|---|---|---|
| Naive | 40.1% | 1.8 | 42% | 100% |
| z margin | 36.0% | 0.5 | 0% (14% on the 5,000 report images) | 100% |
| Pareto test, exact bound | 24.4% | 7.9 | 0% (2% on the 5,000 report images) | 94% |

**The 34% in the main run is above the typical exact-rule result.** It uses more selection data than the resplits (25,000 against 20,000), but one early test failure ends the path, which explains the wide spread. Report about 24–34% for predicted exact-rule savings.

### What did not work or was not shown
- **Per-device selection:** it barely mattered. Running another device's design cost at most 1.6 predicted points.
- **Energy ordering against MAC ordering:** nearly the same (34.0% against 33.4%).
- **Shared against dedicated training:** not shown better. The best schedule, `6r50`, has no dedicated model to compare with. At `6r50_8r25` the two are close. Shared training's supported benefit is six usable schedules from one training run.
- **Exit-only design:** not measured, because the five-design cap dropped it, so the merging comparison above is predicted only.
- **Over-limit designs:** both 1-point `4r50` picks exceeded the limit on test by about 0.07 points, within one standard error (0.17). The bound allows failure with probability 0.1 per selection.
- **Predicted against measured:** close at batch 32 (GPU 34.0 against 36.2, CPU 34.2 against 34.8). CPU batch 1 was over-predicted (30.9 against 27.4). GPU batch 1 was far off (15.7 against 31.3), because that cost table used float16 and these runs used float32.

### Limits of this evidence
- One trained backbone on one dataset. Three exit-head seeds give similar results (predicted 33.7–34.8%, test −0.20 to −0.24), but that does not capture backbone variation.
- The guarantee covers images drawn like the unused CIFAR-10 training pool, and accuracy only. It says nothing about energy or data shift.
- The test set was also read in earlier pilots, so this is not a first look at it. Designs were locked before this test report.
- VS Code and Chrome were both open during the energy runs. Design order was randomized within each repeat, so idle load affects B0 and the designs alike, but gross energy includes it.
- All GPU runs reported the software thermal slowdown flag, at 70–75 °C. CPU energy during GPU runs (about 0.14 J per image) is outside the GPU figures.

### Next steps for a paper
1. Train at least 3 backbone seeds of the shared model.
2. Train a dedicated `6r50` model, for a fair shared-against-dedicated comparison.
3. Measure an exit-only design and rebuild the cooled float32 cost table.
4. Repeat on a harder dataset such as CIFAR-100 or Tiny-ImageNet.
5. Finish the literature check, as the method's novelty is not yet established.

**Comparison rule for Block 1:** I now compare each model's own final head, trained during fine-tuning: dedicated B0 96.97% and dedicated merged 96.70% on validation. This differs from the 96.58% and 96.46% below, which used heads retrained afterwards. The B0 and merged comparison is stricter this way, and the pass thresholds stay the same.

## The plan in short

**Target:** a credible proof of concept, then IEEE Access-level or lower if it holds up. A TNNLS-level method paper is not realistic on one laptop and CIFAR.

**Working claim, not yet proven:** *Train one ViT that runs under many token-merge schedules. Then, for each device and batch size, pick its schedule, exits and threshold with a statistical accuracy guarantee and measured energy, with no retraining.*

**Why this claim:**

- It is a new training approach plus a selection method, which fits your teacher's request to combine ideas across fields.
- It targets a real limit in your data: the best design changes with batch size and hardware, but a fine-tuned backbone is tied to one schedule.
- I found no paper with this exact combination, though these were only a few searches. The closest ones are listed below.

**Fallback if the shared backbone fails:** lock the current two-stage + patch-average design, and add certified selection and real energy measurements. If novelty stays weak, write the empirical paper: *when does token reduction actually save energy?*

---

## What the review and your friend got right and wrong

I tested the main claims on your saved data on 2026-11-04, using CPU only. Script: `scripts/offline/poc_feasibility.py`. Results: `outputs/results/offline/poc_certified.csv`, `poc_gate_auroc.csv`.

| Claim | Verdict | Evidence |
|---|---|---|
| Current results are solid but not a new method | **Agree** | Everything that worked is already published; the new ideas failed |
| Reframe around "fewer operations is not less energy" and unreliable accuracy limits | **Agree** | MACs claimed 35.3% where 13.0% was measured; selection overshoots its limit |
| Friend: start with sample-adaptive merging (Direction B) | **Reject** | You already tested this. Merging's damage is spread across difficulty, confidence routing added about 0–4 points, and guided token choice never beat ToMe |
| Friend: use a task-aware merge criterion | **Reject for now** | Evidence and logit policies did not beat ToMe. Logit similarity was clearly worse |
| Friend: per-sample EXIT / MERGE / CONTINUE at every layer (Direction A) | **Reject as a next step** | Too large for this compute, and it breaks batches into groups of different token counts |
| Review: merging only pays at batch 32+ | **Overstated** | CPU batch 1 merge only still measured 16.1% saved. It's weak at batch 1, not useless |
| Review: exit decision is the bottleneck, with a 10-point gap to the oracle | **Out of date** | That used CLS heads. With patch-average heads, the oracle saves 41.5% at GPU float16 batch 32, against 39.0% now. CPU batch 1 still has a gap: 48.8% against 37.2% |
| Review: a small learned gate should close that gap | **Not supported** | Cross-fitted gate with extra features: AUROC 0.835 vs entropy 0.834 at block 4, and 0.912 vs 0.913 at block 8. More data may help a little, but I would not lead with this |
| Review: energy-ordered testing keeps much more than Bonferroni | **Mostly wrong** | See the table below. It roughly matched Bonferroni, and half the data goes to building the path |
| Review: certified selection is the main contribution | **Partly** | It's useful and needed, but it costs savings, and Pareto Testing (ICLR 2023) already published the core method. It should support the paper, not lead it |
| Review: naive overshoot suggests other papers' "no loss" claims are artefacts | **Too strong** | We only tested our own selection. Don't claim this about other papers without testing them |
| Review: 25k images costs only minutes of GPU | **Agree** | Your logs show 20,000 images in about 5.3 minutes, so about 7–9 minutes per model, saved in chunks |
| Review: Dib (KBS 2026) may not exist | **Wrong** | A ScienceDirect listing gives Knowledge-Based Systems vol. 351, article 116702. I've only seen the abstract, so it needs reading in full |
| Review: Fusion was accepted at DSD 2026 | **Unconfirmed** | I did not verify the venue. Treat it as serious prior art either way |
| Friend: Q1 needs a distinct method, not just good numbers | **Agree** | This is the main gap |

### How much saving survives a guarantee?

This is the frozen backbone, seed 0, 908 candidates, 100 random splits, 10% target failure rate, ranked by predicted GPU float16 batch 32 cost. Each cell is predicted % saved, then the share of runs overshooting the limit on 4,000 held-out images.

| Accuracy limit | Selection images | Current rule | z margin | z + Bonferroni | Pareto + fixed sequence (z) | Same, strict finite-sample bound |
|---|---|---|---|---|---|---|
| 0.5 pts | 4,000 | 29.2%, 74% over | 23.5%, 35% | 2.8%, 0% | 5.3%, 1% | 0%, 0% |
| 0.5 pts | 10,000 | 26.8%, 75% | 24.0%, 37% | 13.4%, 6% | 13.9%, 4% | 0.1%, 0% |
| 1 pt | 4,000 | 35.1%, 60% | 30.9%, 40% | 21.4%, 0% | 18.2%, 6% | 0.5%, 0% |
| 1 pt | 10,000 | 34.4%, 50% | 31.8%, 32% | 24.9%, 4% | 24.9%, 13% | 12.5%, 0% |

**What this means:**

- A real guarantee is expensive at these sample sizes.
- The z versions are only approximate, and the strict bound needs far more data, so use the 25,000 unused images.
- This may look better on the fine-tuned model, whose merging gap is smaller, but that is untested.
- Report savings **after** the guarantee, whatever they turn out to be.

---

## The steps

### Step 0: Prepare (no GPU)

- [ ] **Free disk space.** Only 2.7 GB is free. Decide what to remove (see "What I need from you").
- [ ] **Read the closest papers in full:** Fusion, Pareto Testing, Fast yet Safe, Dib, Kubaty, Pruning All-Rounder, MergeOver.
- [ ] **Run a targeted literature check** on the working claim. If someone has already published it, change direction before training.
- [ ] **Commit the scripts.** `outputs/` is gitignored, so back up results separately.

### Step 1: Shared-backbone pilot (GPU, accuracy only, about 1–2 hours)

Train one backbone with a random merge schedule at each step, including no merging. Do this on the same 20,000 images, 5 epochs, float32, and seed 0, with heads trained afterwards. Use validation only.

**Compare with:** the dedicated model on `6r50_8r25` (96.46% validation), and fine-tuned B0 (96.58%).

**Evaluate at:** none, `8r50`, `6r50`, `6r50_8r25`, `4r25_8r25`, `4r50`.

| Result | Decision |
|---|---|
| Within about 0.3 pts of the dedicated model on `6r50_8r25`, and no-merge within about 0.3 pts of B0 | **Pass:** continue with the working claim |
| Within 0.3–0.7 pts | Try one fix once, such as sampling harder schedules less often, then decide |
| Worse than 0.7 pts | **Fail:** use the fallback |

These thresholds are set now, before the results, and one seed makes them approximate.

### Step 2: Calibration data (GPU, accuracy only, about 10–30 minutes)

Run the chosen backbones forward on the 25,000 unused images, saving 1,000-image chunks so a crash costs only one chunk. Save only head probabilities, which are small. Split them before use: about 5,000 for calibration and 20,000 for selection. Keep them separate from everything used so far.

### Step 3: Certified selection (CPU, no GPU)

Search schedule × exits × threshold for each hardware regime.

**Compare these rules:** current validation rule, z margin, Bonferroni, Pareto + fixed sequence.

**Compare these orderings:** by energy against by MACs, to test whether energy changes the choice.

**Add the energy limit:** certify a per-image energy budget alongside accuracy. Energy depends on exit rates, so test whether the bound holds on new data. This is the most novel part, but the table above shows the accuracy side alone may leave little to spare.

**Decision:**

- **Strong result:** 25% or more predicted savings at 0.5 pts with overshoot near the target, on the fine-tuned model.
- **Weak result:** report it plainly and treat the guarantee as a smaller part of the paper.

### Step 4: Energy measurement (GPU and CPU, about 1–2 hours) - CLOSE EVERYTHING

Your predictions rest on a float16 table from an anomalous mode, and the float32 table is invalid.

- Rebuild the float32 cost table with cool-downs.
- Measure real energy for the final 2–3 designs and B0, on CPU and cooled float32, at batch 1 and 32.
- Use 5 or more repeats, randomized order, and paired B0 runs.
- Run the final designs on the CIFAR-10 test set once, and disclose that earlier method comparisons used it.

### Step 5: Make it paper-grade (GPU, accuracy runs overnight, energy runs close apps)

- [ ] At least 2 more backbone seeds, with a checkpoint every epoch.
- [ ] CIFAR-100, with settings fixed in advance. This is your clean test, since CIFAR-10's test set has already influenced decisions.
- [ ] Baselines: entropy, EEFP (Kubaty's code is public), and a Fusion-style comparison where feasible.
- [ ] Optional: corruption test using simple noise and blur generated in memory. The official CIFAR-10-C download may not fit on your disk.
- [ ] Optional: short Kaggle T4 energy runs for a second GPU, in chunks under 1 hour.

---

## When to close other apps

**Close apps only where this table says so.** I will message you before starting any run that needs it, and wait for your reply.

| Step | Apps open? | Why |
|---|---|---|
| Step 0: reading and checks | **Fine** | No measurements |
| Step 1: shared-backbone pilot | **Fine, but closing speeds it up** | Accuracy doesn't depend on other apps, but heat slows training |
| Step 2: calibration data | **Fine, but closing speeds it up** | Same reason |
| Step 3: certified selection | **Fine** | CPU analysis of saved files, no measurements |
| **Step 4: energy tables and real runs** | **Close everything** | Other programs distort the readings |
| Step 5: seeds and CIFAR-100 training | **Fine, but closing speeds it up** | Accuracy only |
| **Step 5: any new energy runs** | **Close everything** | Same as Step 4 |

### Checklist before an energy run

I'll send you this at the time:

1. Plug in the charger.
2. Set the fan to Turbo (Fn+F5) and raise the back of the laptop.
3. Close browsers, IDEs, chat apps, sync clients and anything else you can.
4. Allow CPU energy reading: `sudo chmod a+r /sys/class/powercap/intel-rapl:0/energy_uj`
5. Don't use the laptop until I say the run is finished.

---

## Papers: what to read and why

| Priority | Paper | Why | Status |
|---|---|---|---|
| 1 | [Fusion](https://arxiv.org/html/2607.02612v1) | Closest overall: merging, exits, pruning, joint training, measured energy | Partly read |
| 1 | [Pareto Testing (ICLR 2023)](https://arxiv.org/abs/2210.07913) | Core method for testing a Pareto path against several risks, including Transformer exit and token settings | **New to our list, not read in full** |
| 1 | Dib, "When to stop", Knowledge-Based Systems 351, 116702 | Learned routing plus risk control, in a journal | **Please send the PDF** |
| 2 | [Fast yet Safe (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/hash/ea5a63f7ddb82e58623693fd1f4933f7-Abstract.html) | Risk control for exits | Not read in full |
| 2 | [Kubaty et al. (ICML 2026)](https://proceedings.mlr.press/v306/kubaty26a.html) | Early-exit failure prediction, with public code to use as a baseline | Not read in full |
| 2 | [Pruning All-Rounder](https://pith.science/paper/2412.06458) | One trained router across many budgets, on vision-language models. Closest to the shared-backbone idea | Summary only |
| 2 | [MergeOver (August 2026)](https://www.alphaxiv.org/abs/2608.13141) | Merging intensity by depth, with results varying by platform and batch size | Listing only |
| 3 | [SAFE-KD (arXiv 2026)](https://arxiv.org/abs/2602.03043) | Conformal risk control for exits on vision backbones | Summary only |
| 3 | [Byun et al. (MIDL 2026)](https://proceedings.mlr.press/v315/byun26b.html) | Token reduction + exits per dataset. Savings reported in FLOPs | Summary only |
| 3 | [OFAv2](https://arxiv.org/abs/2302.01888) | Once-For-All with early exits, on CNNs | Summary only |

Skip for now: TRAM, Ishibashi and Meng, and the TPAMI survey. They don't affect the next decision.

---

## Limits of this plan

- **No guarantee:** the shared backbone may fail, and the guarantee may leave little saving.
- **Novelty is unproven:** a full literature check could still find this idea.
- **Evidence so far is thin:** the fine-tuned results rest on one backbone seed and predicted savings.
- **No Q1 promise:** the realistic aim is a defensible proof of concept. Check any target journal's quartile on Scimago.

## What I need from you

1. **Go-ahead for Steps 0 and 1.** You can keep apps open.
2. **The Dib PDF.** It's the most important paper I can't read. Please also send any others you have from the table.
3. **Disk space.** Only 2.7 GB is free. Tell me which of these I may remove, or free space yourself:
   - `outputs/scratch/features/` (1.5 GB): only needed to re-run frozen-model analyses. Keep `sigs/` and `heads/`
   - `outputs/green_vit_results.zip`: an old bundle of results, so make sure you have a backup first
4. **Your teacher's deadline.** It decides whether Step 5 is in scope.
5. **What your teacher would count as results:** would a proof of concept with a shared backbone and certified selection be acceptable?
