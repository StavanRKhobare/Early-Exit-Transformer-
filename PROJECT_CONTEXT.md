# Cost-Aware AutoML Early-Exit Architecture Search for Green Vision Transformers

*Project context document: everything decided, rejected and learned so far.*

**Status (2026-11-01):** All stages finished on the GTX 1650 and CPU, plus a float32 GPU regime and cooled float32 runs. Results, figures, conditions and limits: `RESULTS.md`. Merging adds about 10 points of measured savings over exit alone at batch 32+ and 0.3–2.5 at batch 1, and merge alone cost more than B0 on cooled float32 at batch 1. Float16 autocast is about 3.4x slower than float32 on this card. Older sections of this file describe the plan and earlier runs, whose results are invalid (Section 14).

---

## 1. One-paragraph pitch

NACHOS (IEEE TNNLS 2025) automates early-exit network design under a **MAC** budget, on small CNNs. We automate it under a **measured-energy** budget on a **Vision Transformer**, extend the search space to **token merging** and **head architecture**, calibrate the exit confidences, enforce a hard **no-regression** accuracy constraint, and quantify **where the energy/accuracy trade-off stops paying off** (energy saved per accuracy point lost, with a knee point). The testable core claim: *MAC-driven design does not give the energy-optimal early-exit/merge configuration, so measured energy should be the cost signal.*

---

## 2. Constraints and context

| Item | Decision |
|---|---|
| Who | Team project, separate from the blockchain+DL capstone |
| Teacher's requirement | Baseline paper must be from a **Q1 journal (Scimago)**; extract its limitations; build something that solves one; "aim for Q1, land at Q4 at worst" |
| Base paper | **NACHOS**, Gambella, Pomponi, Scardapane, Roveri, *IEEE TNNLS* vol. 36 no. 10, 2025, doi 10.1109/TNNLS.2025.3588558 |
| Compute | Free Kaggle T4 (the notebook machine shows 2 GPUs; we use GPU 0 only) |
| Hard rule | Accuracy must not be meaningfully worse than baseline (`EPS = 1 pp`) |
| Time | Limited experimentation time, some room to explore |
| Out of scope | Hardware accelerator / compiler design, supernet NAS (needs A40-days), text datasets |
| Target | Seriously publishable (IEEE Access-level or better); verify the target journal's quartile on scimagojr.com |
| Format preferences | Markdown for team docs, self-contained notebooks, LaTeX for the eventual paper |

---

## 3. Decision log

### Accepted
- **Domain:** images, CIFAR-10 (CINIC-10 was considered and is not in the current code; add later if time allows).
- **Backbone:** ImageNet-pretrained **ViT-Tiny** (12 blocks, 196 patch tokens + CLS at 224x224), **frozen**, heads trained on top.
- **Mechanism:** joint depth + width = entropy-gated early exit + token merging (ToMe-style bipartite soft matching).
- **AutoML:** a small **NSGA-II-style search** (no supernet) over exit placement, per-exit head architecture and merge ratio, scored over a whole entropy-threshold (tau) sweep.
- **Cost signals compared:** measured energy (NVML, ours) vs analytic MACs (NACHOS-style) vs token-layers.
- **Calibration:** temperature scaling per head (answers NACHOS's own stated calibration future work).
- **Analysis:** marginal efficiency (energy saved per accuracy point lost) and a knee point to decide where to stop optimizing.

### Rejected along the way
| Idea | Why rejected |
|---|---|
| Self-speculative decoding (layer-skip drafting) | Generation-only technique; this is classification |
| Train-once / search-at-deploy (Length-Adaptive) | Dropped by the team; overlaps with token merging; LengthDrop would touch backbone training |
| OFA supernet / backbone NAS (NACHOS's full pipeline) | Needs about 2 A40-GPU-days per search |
| Hardware accelerator / compiler-aware design | Different field, not wanted |
| Plain hyperparameter tuning as the whole novelty | Too thin; upgraded to exit-architecture search |
| Text datasets (SST-2, MRPC, AG News) and DistilBERT | Dropped when the project moved to images |
| CNN backbone | Token merging needs patch tokens |
| Beating NACHOS head-to-head on its own CNN/MAC benchmark | Unrealistic on free compute; instead reimplement its cost signal inside our own pipeline |
| Early plan options (cascades, generation-length stopping, confidence-conditioned quantization, data pruning) | Higher regression risk or scope change |

---

## 4. Base paper analysis: NACHOS

**What it does:** NAS for early-exit networks. Searches sub-networks of a MobileNetV3 Once-For-All supernet, places exits, trains them with two regularizers (L_cost for the MAC constraint, L_peak to align confidence with accuracy), and sets thresholds by grid search plus a greedy reduction. Cost = MACs; constraints on accuracy and MACs; CIFAR-10, CINIC-10, SVHN, Imagenette; 2 GPU-days on an A40; 5 seeds, Friedman test.

**Limitations stated by the authors**
1. Focus on mobile-scale networks (millions of parameters/MACs); hardware accelerators and compiler mappings suggested as future work.
2. Calibration is expected to matter more for deeper/higher-compute networks; left to future work (ECE is only reported for CIFAR-10, Table III).
3. Future work: other OFA supernets, other early-exit schemes, other AutoML techniques.

**Limitations we derived (our reading of their design and tables, verify before citing)**
1. "Hardware-constrained" means MACs only; no measured energy, latency or memory.
2. The constraint is an average, thresholds are tuned once by a greedy exit-by-exit procedure and then fixed, so exit rates, budget and accuracy can drift under distribution shift.
3. Early-exit accuracy can fall well below the backbone's (Table I, SVHN: about 80% vs about 88.6%).
4. Some selected exits show about 0% utilization (Table II) yet still cost compute for everything passing them.
5. CNN/vision only, tiny resolutions; no transformers.
6. Heavy search cost, repeated per dataset and constraint.
7. No batching analysis (per-sample exits vs batched throughput).

**What we address:** (1) via measured energy, (2) partly via a tau sweep and calibration, (3) via the explicit no-regression rule, (5) via ViT, plus calibration and the marginal-efficiency analysis. We do **not** address hardware accelerators, supernets or batching.

---

## 5. Methodology (v2)

**Model.** Frozen ViT-Tiny. Optional exit heads after chosen blocks, each `LayerNorm -> Linear` or `LayerNorm -> MLP(128|192)`. After an exit's head, tokens may be merged (ratio 0 / 0.25 / 0.5 = fraction of patch tokens removed; bipartite matching caps this near 0.5) before continuing. Exit rule: exit at the first head whose calibrated normalized entropy is below a shared threshold tau.

**Search.** Genome = (exit layers from {2,4,6,8,10}, at most 2; head architecture per exit; merge ratio per exit). Each candidate trains its heads briefly, is calibrated on a small split, and is scored over a 15-value tau grid. Every (config, tau) is a point with accuracy plus three costs (energy, MACs, token-layers). Elites are configs on either the energy front or the MAC front; mutations: add, remove, move exit, change head, change merge.

**Cost models**
- *Energy (ours):* one-off micro-benchmarks on GPU 0 with the NVML cumulative energy counter: each block at every reachable token count, merge ops, each head (including the exit-decision sync), embed, final. A sample's cost is the sum along its path, so a whole tau sweep is cheap. Verified later against real single-sample runs.
- *MACs (NACHOS-style):* analytic, including attention matmuls (fvcore skips those for fused attention, so the first-version MAC numbers were an undercount).
- *Token-layers:* tokens pushed through blocks.

**Calibration.** Grid search over temperature T minimizing NLL on a held-out split (replaced LBFGS, which collapsed). ECE reported on the validation split before/after.

**Operating points (chosen on validation, reported on test)**
- *Conservative:* cheapest point within 0.25 x EPS of B0.
- *Knee:* knee of the constraint-feasible Pareto front (max distance above the normalized chord; a simpler cousin of Kneedle).
- *Aggressive:* cheapest point within 2 x EPS (deliberately beyond tolerance; the no-regression check will flag it, by design).
- Two pipelines produce them: **E_*** (energy cost) and **M_*** (MAC cost). Baselines: **B0** full model, **B1** fixed exit at layer 6 with tau chosen by the constraint rule.

**Marginal efficiency.** Along the Pareto front, cost saved per accuracy point lost; the knee marks where it collapses.

**Evaluation (`evaluate.ipynb`).** Test accuracy, macro-F1, ECE, paired-bootstrap 95% CI of the accuracy difference vs B0, exit distribution, MACs / token-layers / predicted energy, **measured** energy and latency (single sample, true early exit, interleaved repeats), CO2 (assumed grid intensity constant), plots, no-regression check.

---

## 6. Files and how to run

See `README.md`. `experiment.ipynb` is the whole study; older notebooks are in `archive/v2_notebooks_fixed/`. Section 5 describes the older pipeline; Section 14 lists what changed.

---

## 7. What the first pilot run (v1) showed

Reduced budget (500 images/class, 3 epochs) because Kaggle hung in Phase 4.

- **Heads were undertrained:** proxy baseline accuracy about 71%; exits at every depth scored about 71-73%, so the depth comparison was meaningless and every front point "beat" the baseline (73.9% vs 71.1%).
- **Calibration:** worked where it converged (e.g. ECE 0.249 to 0.038, 0.303 to 0.037, 0.205 to 0.038) but LBFGS collapsed to its floor on 3 heads and made ECE worse (0.216 to 0.462, 0.115 to 0.444, 0.308 to 0.368). Uncalibrated exit heads sat at ECE 0.15-0.30 while final heads were fine (0.02-0.03).
- **Operating points collapsed:** conservative = knee, because tau was hard-coded at 0.5 and all front points had negative accuracy loss.
- **Energy was unreliable:** about 7-11 microkWh per 100 images, sub-15 s windows, two GPUs counted.
- **Mixed checkpoint folders:** knee/aggressive timestamps came from a different run than B0-conservative.
- **Token merge ratio bug:** the code removed half the tokens the ratio implied.

All of these are addressed in v2 (see the changelog at the top of `train.ipynb`).

---

## 8. Expectations and risks for the v2 run

- **Merging may not save energy.** At batch size 1 on a T4, a ViT-Tiny block is launch-bound; merging adds kernels. Measured energy may not follow MACs. That is a legitimate finding (MAC-vs-energy mismatch), not necessarily a bug. A batched, merge-only measurement could be added to show where merging does help.
- **Exit-aware batching** remains unaddressed (per-sample exits only).
- **GPU path untested** by me: fp16 autocast, NVML counter, real speed.
- **B0 accuracy gate:** if the frozen-backbone linear probe is far below about 90%, fix that before anything else.
- **Aggressive points will fail the tolerance by design.**

---

## 9. Stated simplifications (put these in the paper)

- Frozen pretrained backbone; heads trained linear-probe style, not full fine-tuning.
- One shared tau across exits (per-exit thresholds are a natural extension).
- Simplified single-merge-point ToMe, similarity on token features, no cross-layer size tracking.
- Search is a small NSGA-II-style loop over a subsampled proxy, not a supernet NAS; population and generations are scaled down from NACHOS (40 / 20 on an A40).
- Energy = GPU-0 NVML counter including idle draw during the measurement window; CO2 uses an assumed grid intensity (`CARBON_INTENSITY_KG_PER_KWH`).
- B2 is NACHOS's *cost signal* (MACs) inside our pipeline, not a reimplementation of their NAS, L_cost or L_peak.

---

## 10. What reviewers will attack, and the defense

| Attack | Defense |
|---|---|
| Novelty vs adjacent work (risk-controlled exits, adaptive thresholds, merge + exit combos) | Claim is the *measured-energy* design signal, ViT, merge-aware search, and the MAC-vs-energy test; do a literature check on risk-controlled exiting in Q1 journals before submitting |
| Baseline fairness | B2 uses the same archive, same constraint, same threshold rule, different cost signal |
| Energy noise | NVML counter, GPU 0 only, interleaved repeats, mean +/- std, cost-model validation plot |
| Statistics | Paired bootstrap CIs; add multiple seeds before submission |
| Scale | Add CINIC-10 and/or a second backbone if time allows |
| Frozen backbone | Disclose; optionally add one fully fine-tuned B0 as a sanity row |

---

## 11. References and verification status

*Verified = page/PDF seen during research. "From memory" = confirm before citing. Preprints are related work, not validation. Quartiles must be checked on Scimago.*

**Base / placement**
- Gambella et al., NACHOS, IEEE TNNLS 2025 (full PDF reviewed). Verified.
- Gambella & Roveri, EDANAS, IJCNN 2023 (cited inside NACHOS). Verified via NACHOS.
- Bouzidi et al., HADAS, DATE 2023 (cited inside NACHOS). Verified via NACHOS.
- AEBNAS, arXiv Dec 2025 (preprint).
- DistrEE, IEEE GLOBECOM 2024, doi 10.1109/GLOBECOM52923.2024.10901763.
- Early-Exit DNN Inference on HMPSoCs, IEEE EDGE 2025, doi 10.1109/EDGE67623.2025.00017.
- Pruning One More Token is Enough, WACV 2025, doi 10.1109/WACV61041.2025.00695.

**Early exit foundations / surveys**
- DeeBERT, ACL 2020, aclanthology.org/2020.acl-main.204. Verified.
- PABEE (BERT Loses Patience), NeurIPS 2020, arXiv:2006.04152. **From memory.**
- Rahmath et al., Early-Exit DNN survey, ACM Computing Surveys 2024, doi 10.1145/3698767.
- GEEP, Accelerating BERT inference with GPU-efficient exit prediction, Frontiers of Computer Science 2024, doi 10.1007/s11704-022-2341-9 (quartile unverified). Source of the batch-size-shrinkage limitation.
- DART (adaptive thresholds), arXiv 2026 (preprint). Static/overconfident thresholds: arXiv preprints; "Fast yet Safe: Early-Exiting with Risk Control", NeurIPS 2024 (**from memory**).

**Token reduction + exit**
- Token Merging (ToMe), ICLR 2023, arXiv:2210.09461.
- PoWER-BERT, ICML 2020 (PMLR). Learned Token Pruning, KDD 2022 (doi 10.1145/3534678.3539260).
- Magic Pyramid (early exit + token pruning, BERT), arXiv 2021, venue unverified. **Closest NLP prior art.**
- STEP (merging + early exit, ViT segmentation), SN Computer Science 2026.
- Fusion (unified token processing; interference between pruning and exit), arXiv July 2026 (preprint). Motivates the exit-before-merge ordering.
- Token Fusion, WACV 2024. AToM, IEEE Trans. on Computers 2025, doi 10.1109/TC.2025.3540638.
- DynaBERT, NeurIPS 2020, arXiv:2004.04037 (adaptive width/depth). From memory.

**Energy / Green AI / methods**
- "Seeing is Free, Speaking is Not" (edge VLM energy), arXiv July 2026 (preprint): token-reduction papers rarely report measured energy.
- Schwartz et al., "Green AI", CACM 2020; Strubell et al., ACL 2019. **From memory.**
- Guo et al., On Calibration of Modern Neural Networks, ICML 2017 (temperature scaling).
- Deb et al., NSGA-II, IEEE Trans. Evol. Comput. 2002. Satopaa et al., Kneedle, 2011 (related; we use a simpler chord-distance knee).
- Darlow et al., CINIC-10, 2018 (if added).

---

## 12. Open items

- [ ] Run `experiment.ipynb` at full budget; confirm B0 is about 0.93; review `SUMMARY.md`.
- [ ] Read `mac_vs_energy_agreement.csv` and `merge_breakeven.csv` honestly, whichever way they point.
- [ ] Decide on per-exit thresholds, a second dataset (CINIC-10) and extra seeds.
- [ ] Targeted literature check for risk-controlled / calibrated early exit in Q1 journals.
- [ ] Verify the Scimago quartile of every cited journal and of the target venue.
- [ ] Optional: fully fine-tuned B0 sanity row.
- [ ] Write-up: experimental setup (Section 9 simplifications), related work (Section 11), threats to validity (Section 10).


---

## 13. Glossary

- **Early exit (EE):** attach small classifiers (heads) at intermediate layers; stop computing when a head is confident.
- **Entropy gate:** normalized entropy of a head's softmax; a *derived number*, not a separate network. The head itself is a small neural net.
- **Token merging (ToMe):** merge similar patch tokens so later layers process fewer tokens.
- **MAC:** multiply-accumulate (`a*b + c`); FLOPs are about 2 x MACs. A hardware-independent compute proxy; not the same as energy.
- **NAS / AutoML:** automated design (architecture search, hyperparameter search, compression search). Ours searches only the parts attached to a fixed backbone.
- **OFA supernet:** one large network trained once whose sub-networks share weights, so many architectures can be evaluated without retraining. Out of scope here.
- **Calibration / ECE:** whether confidence matches accuracy; expected calibration error measures the gap. Temperature scaling divides logits by T to fix it.
- **Pareto front:** configurations where no other is both more accurate and cheaper.
- **Knee point:** where the accuracy gain per extra cost collapses; the sensible place to stop optimizing.
- **Marginal efficiency:** cost saved per accuracy point given up along the front.
- **No-regression rule:** accuracy >= baseline - EPS (1 pp), checked with a paired bootstrap.

---

## 14. Audit (2026-11-01)

### Results invalidated
`train` used ImageNet mean/std, but the `augreg_in21k_ft_in1k` ViT-Tiny weights expect mean/std 0.5. Reproduced on the same splits and head recipe (validation, 2,000 images):

| Input normalisation | B0 accuracy, 3 seeds |
|---|---|
| ImageNet stats (old) | 78.35%, 79.10%, 78.95% |
| 0.5 / 0.5 (fixed) | 92.55%, 93.00%, 92.85% |

The old 85% check only printed a warning. It now stops the run below 90%. The v1 and v2 accuracies, fronts and operating points are all void. This also explains the old oddity of every exit "beating" the baseline.

Accuracy of a linear head by block, with the fix: L4 61%, L6 72%, L8 78%, L9 87%, L10 91%, L11 93%, L12 93%.

### Other defects fixed
- Empty marginal-efficiency CSV crashed `train` just before packaging when a front had one point.
- Measurement windows contradicted Section 7's lesson (1.5 s benchmarks, about 2.5 s real-energy passes). Now 10 s benchmarks, and 2,000 samples x 5 repeats.

### Findings that affect the plan (not yet implemented)
Numbers below combine fixed-normalisation accuracy with the 2026-10-21 T4 cost table, so they are **predicted, not measured**, and cover linear heads with no merging.

1. **The core claim is unsupported without merging.** Across all 55 no-merge designs (1-2 exits, blocks 2-11), energy and MACs chose the same design at every tolerance (0.25, 0.5, 1, 2 pp). Reported on a separate 1,000 images (standard error about 0.8 pp): 16-22% predicted energy saved, 22-27% MACs saved, at -0.9 to -2.1 pp.
2. **Merging probably cannot save energy in the current regime.** In the T4 single-sample table, block energy is flat from 51 to 197 tokens (17.3-18.4 mJ, r = 0.25) while MACs fall by more than half, and one merge costs about one block (17 mJ). The table appears dominated by per-call overhead, not compute. Any "MACs mislead" result here could be dismissed as an artefact of batch size 1 on a data-centre GPU.
3. **Noise is the same size as the tolerance.** Accuracy standard error is 1.1 pp on the search split and 0.9 pp on the final validation split, against EPS = 1 pp. Identical no-merge final heads ranged 74.5-76.7% across seeds in the search. One seed cannot rank these designs.
4. **The search is unnecessary at this size.** With a frozen backbone, each head depends only on the token stream before it. Heads can be trained once on cached features (about 2-4 s each on CPU, vs about 89 s per model on the T4), and the 855-config space evaluated exhaustively. It is also a mutation loop, not NSGA-II: no crossover, ranking or crowding.
5. **Green accounting is missing.** Search and training energy are not measured. As a rough scale: 20 minutes at up to 70 W is about 84 kJ, so at about 0.04 J saved per inference it takes about 2 million inferences to break even.

### Recommended changes (all implemented in `experiment.ipynb`, see below)
- **P1, measure across regimes (highest value).** Measure T4 at batch 1 and at batch 32-256, plus CPU via RAPL. The contribution becomes: *when do MACs rank designs correctly, and when not?* This is a stronger and more defensible claim than a single-regime result.
- **P2, cache features and enumerate.** Evaluate the full space exactly, and use the saved compute for P3. If AutoML stays in the pitch, widen the space (per-exit thresholds, 3 exits, all blocks) so that search is justified, and benchmark it against exhaustive truth on the small space.
- **P3, at least 5 seeds,** with larger selection splits. Train the final models on 45k images.
- **P4, fair baseline.** Compare head architectures like for like (B0 with an MLP head too), so head capacity is not mistaken for an exit benefit.
- **P5, report total energy:** search + training + inference, with the break-even count. Also report energy net of idle draw, and state that GPU-only NVML excludes CPU energy.

### Study design as implemented (2026-11-01)
- **Methods:** B0 (linear and MLP final heads), exit only, merge only (new), exit + merge. 5,408 designs: 51 merge schedules x 106 exit configurations, plus 2 extra baselines, each over 15 thresholds and 5 seeds. No search: the frozen backbone allows exhaustive evaluation.
- **Why it is cheap:** one tree-shaped pass caches CLS features for all 116 feature slots; 348 head groups train on those, with seeds stacked.
- **Regimes:** GPU at batch 1/32/128, CPU at batch 1/32. Each machine gets a cost table per token count and batch size. Exits shrink the batch, and the model accounts for that.
- **Meters:** NVML counter, then sampled NVML power, then wall clock (labelled as time, never mixed with joules). RAPL for CPU. Gross, net-of-idle and time are stored.
- **Selection:** validation only; test only for reporting. Finalists at 0.5 and 1 point tolerance get real runs.
- **Dead merges:** an exit + merge design only counts as such if at least 5% of samples reach the merge (`merge_exec_frac`). Others are labelled `exit+merge_inactive`.

**Checks passed (CPU):** full notebook execution; RAPL and NVML logic against simulated hardware (not real meters); real versus offline accuracy gap 0.00 points on real images.

**Known limits:** cached features omit flip augmentation. Predictions ran 10-20% below real runs in a small CPU test (rank correlation 0.90-0.95), so quote measured numbers. Train budget stays 20k by default to control compute.

**Small CPU test only, not results (1,000 train images, 2 seeds, time not energy):** one merge from 197 to 100 tokens cut block time 45-54%, cost 0.1-0.26 of a block, and repaid itself within 0.2-0.6 blocks. Accuracy cost was about 0 for merges at block 8-10, but about 9.5 points for removing 50% at block 2. This contrasts with the flat T4 batch-1 table, which is exactly the regime question the full study will test.
