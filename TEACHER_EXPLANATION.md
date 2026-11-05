# Green ViT: what to explain to my teacher

This file explains the whole project in plain language: the problem, what we tried, what worked, what failed, and what we can and can't claim. Read sections 1 to 6 before the meeting. Keep section 10 open for questions.

## Contents

1. [The 30-second version](#1-the-30-second-version)
2. [A 2-minute script](#2-a-2-minute-script)
3. [Background you need to know](#3-background-you-need-to-know)
4. [How the project got here](#4-how-the-project-got-here)
5. [The new method](#5-the-new-method)
6. [The main results](#6-the-main-results)
7. [What failed](#7-what-failed)
8. [What we can and can't claim](#8-what-we-can-and-cant-claim)
9. [What is needed for a paper](#9-what-is-needed-for-a-paper)
10. [Questions the teacher may ask](#10-questions-the-teacher-may-ask)
11. [Where everything is](#11-where-everything-is)

---

## 1. The 30-second version

> We built a way to run a Vision Transformer on less energy with very little accuracy loss. One model is trained once so that it can run with several token-merging settings. We add early exits so easy images can stop sooner, then pick the setting using a statistical limit on accuracy loss.
>
> On CIFAR-10, the chosen design used **about 35% less measured energy at batch 32** and lost **0.24 points of test accuracy**, against a baseline of 96.70%. This is a proof of concept from one trained model, so the next step is repeating it with more seeds and a harder dataset.

---

## 2. A 2-minute script

**The problem.** "Vision Transformers process every image the same way, through all 12 blocks and all of their tokens. Many images are easy, so a lot of that energy is wasted. Two known fixes are early exit, where easy images stop sooner, and token merging, where similar parts of the image are combined so later blocks do less work."

**What we found first.** "We measured real energy on a frozen ViT-Tiny. Exit and merging together saved about 35–37% at batch 32 and above, but cost 0.86 points of accuracy, and the MAC count was a poor guide at batch 1. Fine-tuning then raised the baseline from 92.03% to 96.70% and made merging much less damaging."

**The problem with fine-tuning.** "A fine-tuned model is normally tied to one merge setting, so trying many settings means training many models. On this laptop, one model takes over an hour."

**What is new.** "We combined three things. First, one model is trained across six merge settings by choosing a random setting at each step. Second, the exit heads use the average of the patch tokens, which is much more accurate at early blocks. Third, we choose the final design using a statistical test that limits accuracy loss, because simply choosing on validation data often goes over the limit."

**The result.** "Under a 0.5-point limit, the chosen design lost 0.24 points on test. Real energy measurements showed 36.2% saved on GPU and 34.8% on CPU at batch 32, and 26.9–29.9% at batch 1."

**Being honest about it.** "This is one trained model on CIFAR-10. Across repeated data splits, predicted savings under the guarantee averaged about 24%, compared with 34% in the main run. Choosing a separate design per device barely helped, and I still need to compare against a dedicated model and finish the literature check. I'd call this a proof of concept that is worth scaling up."

---

## 3. Background you need to know

### The model and data

- **ViT-Tiny:** a small Vision Transformer with 12 blocks. A 224×224 image becomes 196 patch tokens plus 1 CLS token, so 197 tokens in total.
- **CIFAR-10:** 60,000 small images in 10 classes, resized to 224×224 here.
- **B0:** our baseline, the full model with no exits or merging. It is the number everything is compared against.
- **Frozen or fine-tuned:** a frozen model keeps its pretrained weights, and only the heads are trained. A fine-tuned model also adjusts the backbone weights for CIFAR-10.

### The energy-saving techniques

- **Token merging:** similar tokens are combined, so later blocks have fewer tokens to process. We use a simplified version of ToMe (Token Merging). Merging 50% after block 6 reduces the tokens from 197 to 100.
- **Early exit:** a small classifier, called an exit head, sits partway through the model. If it is confident, the image stops there and skips the remaining blocks.
- **Entropy threshold:** entropy measures how unsure a head is. An image exits when its entropy is below the threshold. A higher threshold means more exits, less energy and usually lower accuracy.
- **Calibration (temperature scaling):** this adjusts a head's confidence so it better matches how often the head is right. It runs on a separate set of 1,000 images.

### The measurements

- **MACs:** a count of arithmetic operations. Many papers use this to stand in for energy, but it can mislead, as section 4 shows.
- **Measured energy:** real joules from the hardware. NVML reads the GPU and RAPL reads the CPU package.
- **Predicted energy:** an estimate built from a table of measured costs per operation. It is useful for comparing thousands of designs, but it isn't a measurement.
- **Batch size:** how many images are processed together. Batch 1 means one at a time. Batch size matters because a merge step has a cost of its own, and that cost is easier to recover at larger batches.
- **Points:** percentage points of accuracy. Going from 96.70% to 96.46% is a loss of 0.24 points.

### The hardware

Everything ran on one laptop with a GTX 1650 Mobile (4 GB), an i5-10300H and 14 GB of RAM.

### How to read a design name

`6r50 g8@0.05 L10@0.12` means:

| Part | Meaning |
|---|---|
| `6r50` | Merge at a ratio of 50% after block 6 |
| `g8@0.05` | First exit after block 8, threshold 0.05 |
| `L10@0.12` | Second exit after block 10, threshold 0.12 |

The other schedules follow the same pattern. `6r50_8r25` means 50% after block 6 and 25% after block 8, and `none` means no merging.

---

## 4. How the project got here

### Stage 1: The original plan

The base paper is **NACHOS** (IEEE TNNLS 2025), which automatically designs early-exit CNNs under a MAC budget. Our plan was to do something similar on a Vision Transformer using measured energy, with token merging included. Our earliest results were thrown out because of a normalisation bug.

### Stage 2: The frozen study, 1 November 2026

We tested 5,408 designs with a frozen backbone. **Because the backbone was frozen, we could check every design directly, so we didn't need an AutoML search algorithm.** We then measured the best designs for real.

| Finding | Evidence |
|---|---|
| Batch size decides whether merging is worth it | At batch 32 and above, merging raised savings from 25–27% to 35–37%. At batch 1 it added only 0.3–2.5 points |
| Merging alone can waste energy at batch 1 | On cooled float32 GPU, one merge used 9.1% more energy than B0, while MACs claimed a 25.7% saving. The repeats were noisy (−9.5% to +4.8% saved) |
| Where a merge happens matters | A 50% merge costs 6.70 points after block 2, but only 0.13 after block 10 |
| MACs can mislead | At batch 1 on float16, MACs claimed 35.3% where 13.0% was measured |
| Number format matters on this card | Float16 used far more energy than float32 on this GPU |

The accuracy was the weak part: exit and merging together lost about 0.86 points from a baseline of 92.03%. The teacher also asked for something new, not only a comparison.

### Stage 3: Trying new ideas, 2–4 November

| Idea | Result | Decision |
|---|---|---|
| Confidence-guided merging | Added only about 0–4 points of savings | Rejected |
| Reusing exit outputs in the final decision | Helped when frozen, but stopped helping after fine-tuning | Dropped |
| Fine-tuning with merging on | Baseline rose to 96.70%, and merging's accuracy gap fell from 0.88 to about 0.28 points | **Kept** |
| Choosing tokens with exit-head guidance | Best gain was +0.21 points, and the confidence interval included zero | Rejected |
| Patch-average exit heads | Block 4 went from 69% to 80%, block 6 from 80% to 87.5% | **Kept** |
| Joint training of backbone and exits | Exit heads improved, but the final head fell to 96.08%, with no net gain | Rejected |
| A learned exit gate | Same quality as plain entropy (AUROC 0.835 against 0.834 at block 4) | Rejected |

We also found that **choosing thresholds on validation data alone often breaks the accuracy limit**. It went over 51–93% of the time, which is why the guarantee became part of the method.

### Stage 4: The overnight run, 4–5 November

This run put together the parts that survived. It took 4 hours 47 minutes in total and produced the results in section 6.

---

## 5. The new method

No single part is new by itself. The contribution we are testing is the **combination**, which fits what the teacher asked for: a new technique or a mix of existing ones.

### Part 1: One model trained across six merge schedules

At each training step, the model randomly uses one of six schedules: `none`, `8r50`, `6r50`, `6r50_8r25`, `4r25_8r25` or `4r50`. One backbone and one final head are shared across all of them.

**Why it matters:** dedicated training would need one model per schedule. This run took about 1 hour 43 minutes of training on the laptop and produced six usable schedules.

**Did it hurt accuracy?** Not measurably on the two schedules we can compare (validation data):

| Schedule | Shared model | Dedicated model | Gap |
|---|---|---|---|
| No merging | 96.88% | 96.98% | −0.10 |
| `6r50_8r25` | 96.62% | 96.70% | −0.08 |
| `6r50` | 96.82% | Not trained | |
| `8r50` | 96.52% | Not trained | |
| `4r25_8r25` | 96.22% | Not trained | |
| `4r50` | 96.22% | Not trained | |

With 4,000 validation images, the standard error of one accuracy is about 0.27 points, so these gaps can't be told apart from zero.

### Part 2: Patch-average exit heads

Normal exit heads read only the CLS token, which is weak early in the network. Ours average all of the patch tokens, then use a small two-layer classifier. We train them after the backbone, so they can't lower the backbone's accuracy.

### Part 3: Selection with a statistical limit

The goal is to find the cheapest design that is very unlikely to lose more accuracy than the limit.

1. **Split the data:** we use 25,000 CIFAR-10 training images that were never used for training: 10,000 to build a candidate list and 15,000 to test it.
2. **Build the list:** we keep only the best trade-offs between energy and accuracy (a Pareto path) and sort them from most expensive to cheapest.
3. **Test in order:** each candidate must pass a statistical test on the 15,000 images. We stop at the first failure and take the last design that passed. This is fixed-sequence testing, from Pareto Testing (ICLR 2023).
4. **Use an exact bound:** we compare the design and B0 image by image, counting where each is right and the other is wrong. We then apply Clopper-Pearson bounds to both counts, which needs no normal approximation.

**What the guarantee says:** with 90% confidence, the chosen design's true accuracy is no more than 0.5 points below B0's, for images drawn like those in the selection data. It covers accuracy only, not energy, and it doesn't cover images from a different distribution.

The test set wasn't used to pick anything in this step. Designs were locked before the test report ran.

---

## 6. The main results

### The headline

| | Frozen study (best at 1-point tolerance) | New method (0.5-point limit) |
|---|---|---|
| B0 test accuracy | 92.03% | **96.70%** |
| Accuracy lost on test | 0.86 points | **0.24 points** |
| Measured energy saved, batch 32 | 35.2–35.7% | **34.8–36.2%** |

So the savings are about the same at batch 32, the accuracy loss is much smaller, and the baseline is higher. The two studies have different baselines and conditions, so treat this as context rather than a controlled comparison.

### Measured energy, against B0

We ran 144 real measurements: 6 paired repeats of each model in each setting, in random order, using 1,000 unused images. Values are percent saved with a 95% confidence interval.

| Design | Limit | Test accuracy change | GPU batch 32 | CPU batch 32 | GPU batch 1 | CPU batch 1 |
|---|---|---|---|---|---|---|
| `6r50 g8@0.05 L10@0.12` | 0.5 | −0.24 | 36.2 ± 0.9 | 34.8 ± 2.7 | 29.9 ± 1.9 | 26.9 ± 1.1 |
| `6r50 g6@0.03 L8@0.05` | 0.5 | −0.24 | 35.6 ± 1.7 | 34.7 ± 3.6 | 31.3 ± 1.5 | 27.4 ± 0.6 |
| `6r50 g4@0.12 L8@0.08` | 1 | −0.74 | 43.4 ± 1.1 | 40.5 ± 3.0 | 38.0 ± 1.3 | 35.1 ± 1.6 |
| `4r50 g6@0.05 L8@0.16` | 1 | −1.08 (over) | 47.1 ± 0.6 | 46.2 ± 2.3 | 40.6 ± 1.0 | 36.0 ± 1.1 |
| `4r50 g6@0.03 L8@0.16` | 1 | −1.06 (over) | 46.1 ± 0.9 | 46.3 ± 1.6 | 39.5 ± 1.5 | 35.7 ± 0.5 |

Energy in joules per image, for the first design:

| Setting | B0 | Design |
|---|---|---|
| GPU batch 32 | 0.123 | 0.078 |
| CPU batch 32 | 0.344 | 0.224 |
| GPU batch 1 | 0.148 | 0.104 |
| CPU batch 1 | 0.402 | 0.294 |

### Why the guarantee is useful

These are predicted savings at GPU batch 32 under the 0.5-point limit, with accuracy from the test set.

| How the design was chosen | Predicted energy saved | Test accuracy change |
|---|---|---|
| No guarantee (naive) | 40.5% | −0.62, **broke the limit** |
| Approximate guarantee (z margin) | 36.2% | −0.32 |
| **Our method (Pareto test, exact bound)** | **34.0%** | **−0.24** |
| Exact bound, no Pareto path (Bonferroni) | 18.4% | −0.11 |

We then repeated selection on 50 random splits of the unused images:

| Rule | Mean predicted saved | Spread (SD) | Broke the limit | Found a design |
|---|---|---|---|---|
| No guarantee | 40.1% | 1.8 | **42%** | 100% |
| Approximate | 36.0% | 0.5 | 0% | 100% |
| **Exact (ours)** | **24.4%** | 7.9 | 0% | 94% |

"Broke the limit" is measured against all 25,000 unused images. On only the 5,000 images held out in each split, the figures are 14% for approximate and 2% for exact, because small samples are noisier.

**Be careful with this table: the 34% from the main run is above the typical result.** The exact rule is safe but unstable, because one early failed test ends the search. Quote **about 24–34%** for predicted savings under the guarantee.

### What merging added

These are predicted at the 0.5-point limit, not measured.

| Setting | With merging | Without merging | Gain |
|---|---|---|---|
| GPU batch 32 | 34.0% | 24.9% | About 9 points |
| CPU batch 32 | 34.2% | 26.0% | About 8 points |
| CPU batch 1 | 30.9% | 29.1% | About 2 points |

This agrees with the frozen study: merging helps mostly at larger batches.

---

## 7. What failed

Say these before you're asked. It shows the work is careful.

- **Choosing a design per device barely helped.** Using another device's design cost at most 1.6 predicted points. We expected more from this.
- **Ordering by energy or MACs made little difference** here (34.0% against 33.4% predicted). At large batches, MACs were a fair guide.
- **Shared training isn't shown to beat dedicated training.** The winner, `6r50`, has no dedicated model to compare with. The supported benefit is several usable schedules from one training run.
- **The dedicated `6r50_8r25` model found nothing under the exact rule.** That doesn't prove shared training is better, because the shared model won through a different schedule. An approximate rule found 36.2% at −0.29 on that dedicated model.
- **Two 1-point designs went over their limit on test,** at −1.08 and −1.06. That's within one standard error (0.17), and the guarantee allows a 10% chance of failure, but it should be reported.
- **Cost predictions weren't always right.** They were close at batch 32 (34.0 against 36.2 measured on GPU, 34.2 against 34.8 on CPU). CPU batch 1 was over-predicted (30.9 against 27.4), and GPU batch 1 was far off (15.7 against 31.3), because that table used float16 and the real runs used float32.
- **The other ideas in stage 3** that were rejected.

---

## 8. What we can and can't claim

### We can say

- One ViT-Tiny, trained once across six merge schedules, stayed within about 0.1 points of dedicated models on the two schedules we could compare.
- The chosen design used about 35% less measured energy at batch 32 and 27–31% less at batch 1, with 0.24 points lost on test.
- Selecting without a guarantee broke the accuracy limit in 42% of splits. The exact rule broke it in none, at a cost in savings.

### We can't say

| Don't say | Why | Say instead |
|---|---|---|
| "This is a Q1-ready result" | One backbone seed on one dataset | "It is a proof of concept" |
| "This is a novel method" | The literature check isn't finished | "I haven't found this exact combination so far" |
| "Shared training beats dedicated training" | No fair comparison exists yet | "It gives six schedules from one training run" |
| "We always get 34% with the guarantee" | Resplits averaged 24% | "Predicted savings are about 24–34%" |
| "Per-device selection is the key" | It barely helped here | "It mattered little in this test" |
| "Merging's extra savings are measured" | No exit-only design was measured | "That part is predicted" |
| "The guarantee covers energy or new data" | It covers accuracy on similar images | State the scope exactly |

### Measurement issues to disclose

- Chrome and VS Code were open during the energy runs. Random run order should spread that load evenly across designs, but it's included in the figures.
- All GPU runs reported the software thermal slowdown flag, even at 70–75 °C.
- The GPU figures exclude CPU energy used during GPU runs, which was about 0.14 J per image.
- The energy runs used 1,000 unused images. Test accuracy came from the separate 10,000-image test set.
- The test set was also read in earlier pilots, so this isn't its first use.
- The 90% confidence applies to each selection separately, not to all selections at once.

---

## 9. What is needed for a paper

In order of importance:

1. **At least 3 backbone seeds of the shared model.** This is the largest gap.
2. **A dedicated `6r50` model,** for a fair comparison with shared training.
3. **A measured exit-only design,** to show directly what merging adds.
4. **A harder dataset,** such as CIFAR-100 or Tiny-ImageNet.
5. **A proper literature check,** including reading the closest papers in full.
6. **A cooled float32 cost table,** so GPU predictions match the real runs.
7. **Cleaner energy runs,** with other apps closed.

**Is Q1 realistic?** It is possible, but this evidence alone doesn't support it. Items 1 to 4 will show whether the method holds up, and the results could get weaker.

---

## 10. Questions the teacher may ask

**What is actually new here?**
The combination: one ViT trained across merge schedules, patch-average exits, and selection with an exact accuracy bound, checked against real energy. Each part has earlier work. So far I haven't found this exact combination, but the search isn't complete.

**Isn't this just Fusion or Pareto Testing?**
Fusion also combines merging and exits and measures energy, but it doesn't offer a statistical accuracy guarantee. Pareto Testing is the selection method we use, and it's a general method. It doesn't cover training one ViT across merge schedules. I still need to read the closest papers fully. (Fusion's venue, listed as DSD 2026, isn't confirmed either.)

**How does this relate to NACHOS, the base paper?**
NACHOS designs early-exit CNNs under a MAC budget, and takes about 2 GPU-days per search on an A40. We apply a train-once idea to a Vision Transformer, using merge schedules instead of CNN widths and depths, and check designs with measured energy. Our run was on different hardware and data, so don't compare the times directly.

**What happened to AutoML?**
With a frozen backbone, all 5,408 designs could be checked directly, so a search algorithm wasn't needed. The old code was also only a mutation loop, not real NSGA-II. In the new method, selection is automatic, but it's a statistical test over a fixed list of designs, not a search algorithm.

**Why fine-tune at all?**
It raised B0 from 92.03% to 96.70% and made merging less damaging. Our savings are measured against the fine-tuned B0, so they aren't inflated by a weak baseline.

**Why not train one dedicated model per schedule?**
It costs one training run per schedule, which is over an hour each on this laptop, and we would want to search many. Whether dedicated models are more accurate is still an open question.

**Why CIFAR-10, and why resize it to 224?**
It's small enough for this hardware. Resizing matches the input size the pretrained ViT expects. It's also an easy dataset, so a harder one is on the plan.

**Why does exit energy still matter if the accuracy is so close?**
Most images are easy. If they stop at block 8 or 10, they skip the remaining blocks, while harder images still use the whole model.

**Why does batch 1 save less?**
Merging has its own cost, which is harder to recover at batch 1. The frozen study showed the same pattern, and here the gap is roughly 35% at batch 32 against 27–31% at batch 1.

**Why trust a guarantee from one data split?**
The limit held in the 50 resplits when checked against all unused images. The test set result also stayed within the 0.5-point limit. It's still one trained model, though, and that doesn't address variation between training runs.

**Why is the guarantee at 90% and not 95%?**
It was the level fixed in advance. At 95%, the chosen designs still had test losses of about 0.18–0.20 points, with predicted savings of 32.5% at GPU batch 32 and 14.1–32.7% across the settings.

**Did you look at the test set while tuning?**
Not in this selection. Designs were locked before the test report, and the selection used only unused training images. However, earlier pilots did read the test set, and I disclose that.

**How reliable is the energy measurement?**
It uses hardware counters, with 6 randomized paired repeats, and the intervals are mostly 1–3 points wide. Limits are in section 8. Apps were left open, and it's one laptop, so the numbers may not carry over to other machines.

**How long did this take to run?**
The overnight run took 4 hours 47 minutes, including 1 hour 43 minutes of shared training, about 1.5 hours of feature and head work, and about 1 hour of energy measurement.

---

## 11. Where everything is

| What | Where |
|---|---|
| Overnight results, in full | `NEXT_STEPS.md` |
| Frozen study results | `RESULTS.md` |
| Project history and decisions | `PROJECT_CONTEXT.md` |
| All results in one document | [Green ViT: Complete Results Record](https://claude.ai/artifact/Sv3BYCgBfXJ62mYvkJxGCR) |
| Shared training | `scripts/shared_backbone.py` |
| Exit heads | `scripts/extract_heads.py` |
| Selection | `scripts/certified_selection.py` |
| Energy measurement | `scripts/real_energy.py` |
| Shared model results | `outputs/results/shared_backbone/` |
| Selection results | `outputs/results/certified/` (test results in `selection_test.csv`) |
| Energy results | `outputs/results/real_energy/` (`summary.csv` and `runs.csv`) |
