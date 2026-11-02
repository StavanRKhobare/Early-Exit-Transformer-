# Green AI Capstone: Full Option Analysis & Implementation Plan

## Selection Constraints (stated by the team)
- **No regressions allowed.** Any chosen method must be *more energy/compute-efficient* than the baseline, but must **not** perform worse than the baseline on task accuracy — this is a hard constraint, not a soft preference.
- **Limited but non-zero experimentation time.** Prefer techniques with established, low-risk implementations. One controlled "stretch" experiment is acceptable if the core deliverable doesn't depend on it succeeding.
- **Green AI is the paper's central theme** — every method needs a clear, measurable energy/carbon story, not just a latency story.

---

## Part 1: Evaluation Metrics (for the paper)

A comparison is only as useful as the metrics behind it. Beyond accuracy and CO₂, here's a fuller metric set reviewers will expect from a Green AI paper:

**Task performance**
- Accuracy / F1 (your primary classification metric)
- Calibration error (ECE) — important if you're using confidence thresholds to gate decisions
- Performance on a held-out "hard" or adversarial subset (robustness, not just average-case accuracy)

**Compute / efficiency**
- FLOPs per inference (theoretical, model-computed)
- Wall-clock latency per inference (ms), mean and p95/p99 (tail latency matters more than mean in practice)
- Average exit layer / average tokens retained (for depth- and width-adaptive methods — shows *how much* of the model is actually skipped, not just that skipping happens)
- Peak memory / VRAM footprint
- Model size (parameter count) if any method changes it (e.g., distillation)

**Energy / carbon (the core Green AI metrics)**
- Energy per inference (Joules) — CodeCarbon or RAPL
- CO₂-equivalent emissions (gCO₂eq) per inference and for full test-set evaluation
- Total training energy (kWh) and training CO₂eq — often forgotten, but reviewers will ask "what did it cost to get here"
- **Compute-normalized accuracy**: accuracy per Joule, or accuracy per gCO₂ — this is the single metric most likely to make your comparisons legible to a reviewer at a glance
- Power draw profile over time (watts vs. time) — useful for visualizing *when* energy is spent (e.g., prefill vs. decode, if you go the generation route)

**Reproducibility / variance**
- Standard deviation across repeated runs for energy measurements (CodeCarbon readings are noisy, especially on shared Colab hardware — report mean ± std over ≥3 runs, not a single number)
- Hardware/runtime disclosure (GPU type, Colab tier, CPU-only vs. GPU) since energy numbers aren't portable across hardware

---

## Part 2: Inference-Time Options — Pros & Cons

| # | Option | Pros | Cons | Regression Risk |
|---|---|---|---|---|
| 1 | **Joint Depth + Width** (early-exit + token merging) | Reuses your existing pipeline almost entirely; strong, clearly measurable novelty claim (combined savings vs. either alone); both sub-techniques are individually well-validated | Interaction effects (merging may shift entropy calibration) need tuning; slightly more moving parts to debug | **Low-Medium** — each technique alone is safe; the combination needs a short calibration pass |
| 2 | **Cost-Aware Exit Placement** | Replaces a guessed hyperparameter (layer 3) with a measured, defensible choice; cheap to add on top of Option 1; strengthens the "Green AI" framing (energy-driven design, not just energy-measured) | Needs careful, low-noise energy profiling; adds a modeling step | **Low** — it's a placement search, not a new inference mechanism, so it can't make the model itself worse |
| 3 | **Adaptive Generation Length** ("anti-babbling") | Addresses a clearly documented literature gap; conceptually elegant extension of your entropy logic | Requires switching model family (decoder, not DistilBERT) and task type (generation, not classification) — largest scope change of all 8 options | **Medium-High** — "answer completeness" is a fuzzier signal than classification confidence; stopping too early can visibly hurt output quality |
| 4 | **Confidence-Threshold Robustness Study** | Lowest engineering lift; safest scope; still original (most early-exit papers don't rigorously map failure modes) | Weakest novelty claim of the eight; more "analysis" than "new method" | **Low** — it's a study of your existing method, not a new risky mechanism |
| 5 | **Self-Speculative Decoding via Layer-Skip Drafting** | Reuses your early-exit layers directly (no separate draft model needed); active, citable 2024–2025 research thread | Requires a decoder/generation setup, not classification; verification-step logic adds real implementation complexity | **Medium** — speculative decoding is *lossless by design* (rejected drafts fall back to the full model), so accuracy risk is structurally low, but implementation bugs are easy to introduce under time pressure |
| 6 | **Cascade / Big-Little Model Routing** | Conceptually simple; strong, easy-to-explain baseline to compare against | A poorly trained gate can send hard inputs to the small model — direct accuracy risk; needs a second full model (BERT-base), more compute overhead in a "green" project (ironic tension) | **Medium** — depends entirely on gate quality, which needs its own validation |
| 7 | **Confidence-Conditioned Quantization** | Genuinely unexplored combination (quantization and early-exit are largely separate literatures); ties directly into your existing confidence signal | Aggressive quantization (int4) can cause real, sometimes unpredictable accuracy loss; more experimentation needed to find a safe precision floor | **Medium-High** — this is the option most likely to visibly hurt accuracy if not carefully tuned |
| 8 | **Train-Once, Search-at-Deploy (Length/Depth-Adaptive)** | Elegant "train once" story; strong efficiency-budget framing for a paper | Requires implementing a search procedure at deployment time; more infrastructure than the others for the payoff it gives at your project's scale | **Low-Medium** — the search itself doesn't touch model weights, so it's unlikely to make things worse, but is more engineering for a project on a deadline |

---

## Part 3: Training-Time Options — Pros & Cons

Training-time efficiency is a well-established (and often-overlooked) part of Green AI — algorithmic efficiency work groups mixed-precision training, data-centric methods, and distillation together as the main training-side levers.

| # | Option | Pros | Cons | Regression Risk |
|---|---|---|---|---|
| T1 | **Mixed-Precision Training** (fp16/bf16 autocast) | Essentially a one-line change (`torch.cuda.amp`); standard, extremely well-validated; near-zero accuracy risk; directly measurable training-energy reduction via CodeCarbon | Minimal novelty on its own — it's a "hygiene" technique, not a contribution | **Very Low** — this is closer to a free win than an experimental method |
| T2 | **Self-Distillation for Off-Ramp Training** | Train your off-ramp classifiers on the *final layer's soft predictions* instead of (or alongside) hard ground-truth labels — a well-established technique in early-exit literature (used in FastBERT-style self-distillation) that typically **improves** off-ramp accuracy and calibration, not hurts it | Adds one extra forward pass per training step (teacher logits) — a small training-time compute cost, though it's a one-time training cost, not a recurring inference cost | **Very Low** — this is a known-safe, often accuracy-*improving* change |
| T3 | **Data-Efficient Training / Informative Subset Selection** | Directly reduces training energy/CO₂, which strengthens your training-side Green AI story; recent adaptive methods (e.g., "Play it Straight") are explicitly designed to control the accuracy/efficiency tradeoff rather than trade blindly | Naive subset selection can measurably hurt accuracy if not done carefully; adds a selection algorithm to implement and validate — the most "experimental" of the training options | **Medium** — this is the one training-time option that could plausibly violate your no-regression constraint if rushed |

---

## Part 4: Final Recommended Combination

Given your constraints — **no regressions, limited-but-nonzero experimentation time, Joint Depth+Width as your anchor** — here's the risk-weighted recommendation:

### Core combination (low risk, ships the paper)
1. **Joint Depth + Width** (Option 1) — your primary inference-time contribution and novelty claim
2. **Cost-Aware Exit Placement** (Option 2) — cheap add-on that upgrades "we picked layer 3" into "we measured and chose the best layer," strengthening the Green AI framing for almost no extra risk
3. **Mixed-Precision Training** (Option T1) — near-free training-energy win, rounds out the paper with a training-side number, essentially zero risk
4. **Self-Distillation for Off-Ramp Training** (Option T2) — safe, often *improves* your off-ramp accuracy, and is standard practice in this literature so reviewers will expect to see it

**Why this combination:** every piece individually carries low regression risk, they compose naturally (placement search informs where you add off-ramps; distillation improves what those off-ramps predict; mixed precision is orthogonal and free), and together they give you both an inference-side *and* training-side Green AI story — which is more complete than most of the papers you're citing.

### Optional stretch experiment (if time allows)
Pick **one** of these as a bounded, time-boxed experiment — not a dependency for the main paper:
- **Confidence-Threshold Robustness Study** (Option 4) — lowest-risk stretch, purely additive analysis on what you've already built
- **Self-Speculative Decoding via Layer-Skip Drafting** (Option 5) — higher payoff/novelty, structurally lossless by design, but a bigger implementation lift

Avoid Options 3, 6, 7, 8, and T3 as core commitments given your timeline — not because they're bad ideas, but because each has a meaningfully higher chance of costing you time you don't have, or of visibly hurting accuracy if rushed.

---

## Part 5: Implementation Plan

**Assumes:** free Colab/Kaggle GPU, DistilBERT + SST-2 baseline already built, 2–3 person team.

### Phase 0 — Baseline Lock-In (few days)
- Freeze your current early-exit DistilBERT as the official baseline
- Run and record baseline metrics (Part 1 metric list) with ≥3 repeated energy measurements for variance
- This baseline is what every later number gets compared against — don't skip logging it properly

### Phase 1 — Training-Time Wins (low risk, do first)
- Add mixed-precision training (`torch.cuda.amp`) to your existing two-stage training loop
- Modify Stage 2 (off-ramp tuning) to use self-distillation: off-ramp loss = weighted combination of cross-entropy vs. ground truth **and** KL-divergence vs. final-layer soft predictions
- Re-measure training energy/CO₂ and confirm accuracy is at or above baseline before moving on
- *Why first:* these are your safest wins — validating them early de-risks the rest of the timeline

### Phase 2 — Cost-Aware Exit Placement
- Profile per-layer/per-component energy on the frozen backbone (CodeCarbon or RAPL, averaged over several runs)
- Fit a simple cost model (fixed overhead + FLOP-dependent term per layer)
- Search candidate exit layers against the cost model; compare against your current fixed layer-3 placement
- Lock in the chosen placement(s) before Phase 3

### Phase 3 — Token Merging Integration (the core novelty)
- Implement a ToMe-style token-merge step at the layer(s) preceding your chosen off-ramp(s)
- Start with a conservative merge ratio; validate accuracy doesn't regress before increasing aggressiveness
- Sweep merge ratio × entropy threshold jointly; this is your main experiment grid
- Log whether merging shifts the entropy distribution (does your old threshold still mean the same thing?) — this is the interaction effect that makes the combination genuinely novel, so it needs to be measured, not assumed

### Phase 4 — Full Evaluation
- Run the complete metric set from Part 1 on: baseline, training-only improvements, +cost-aware placement, +token merging (full combination)
- Report accuracy-per-Joule and accuracy-per-gCO₂ as your headline comparison metric
- Include variance (mean ± std over ≥3 runs) for every energy/CO₂ number

### Phase 5 (optional, time-boxed) — Stretch Experiment
- If Phases 1–4 finish with time to spare, attempt the robustness study or self-speculative decoding drafting as a bounded, clearly-labeled additional contribution
- Time-box this explicitly (e.g., "we'll attempt this for one week; if it's not working by then, it's future work") so it can't threaten your core deliverable

---

## References
- Xin, J. et al. "DeeBERT: Dynamic Early Exiting for Accelerating BERT Inference." ACL 2020.
- Zhou, W. et al. "BERT Loses Patience: Fast and Robust Inference with Early Exit." NeurIPS 2020.
- Liu, W. et al. "FastBERT: a Self-distilling BERT with Adaptive Inference Time." ACL 2020.
- Bolya, D. et al. "Token Merging: Your ViT But Faster." ICLR 2023.
- Kim, S. et al. "Infor-Coef: Information Bottleneck-based Dynamic Token Downsampling." arXiv:2305.12458, 2023.
- "Dissecting Transformers: A CLEAR Perspective towards Green AI." arXiv:2510.02810, 2025.
- Solovyeva, L. & Castor, F. "Towards Green AI: Decoding the Energy of LLM Inference in Software Development." 2026.
- Schuster, T. et al. "CALM: Confident Adaptive Language Modeling." NeurIPS 2022.
- Kim, S. et al. "Big Little Transformer Decoder." NeurIPS 2023.
- Kim, G. & Cho, K. "Length-Adaptive Transformer: Train Once with Length Drop, Use Anytime with Search." arXiv:2010.07003, 2020.
- Liu, Y. et al. "Accelerating Inference in Large Language Models with a Unified Layer Skipping Strategy." arXiv:2404.06954, 2024.
- Zhang, J. et al. "Draft & Verify: Lossless Large Language Model Acceleration via Self-Speculative Decoding." 2024.
- "An Efficient Model Training Framework for Green AI" (Play it Straight / Re-Play it Straight). Machine Learning, Springer, 2025.
- "Green AI Techniques for Reducing Energy Consumption in AI Systems." ScienceDirect systematic review, 2025.
