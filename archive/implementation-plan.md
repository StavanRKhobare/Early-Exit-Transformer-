# Detailed Implementation Plan
### Chosen combination: Joint Depth+Width + Cost-Aware Exit Placement + Mixed-Precision Training + Self-Distillation

---

## 1. Model Choice: Yes, stick with DistilBERT-base-uncased

Reasons to keep it rather than switch:
- **Continuity** — it's what your existing whitepaper/prototype already uses; switching models would waste the Internal Classifier and entropy-threshold work you've already validated
- **Direct comparability** — DeeBERT, PABEE, and FastBERT all report DistilBERT/BERT results, so your numbers slot directly next to theirs in a results table
- **6 layers is a sweet spot** for early-exit granularity — enough depth to have a meaningful "off-ramp" decision, but few enough layers that Cost-Aware Exit Placement (Phase 2) has a tractable search space (realistically 2–3 candidate exit points, not 12+)
- **`output_hidden_states=True` support** — already built into your custom `EarlyExitDistilBERT` class

Only reconsider the model if Phase 4 evaluation shows you're compute-starved across 3 datasets — the lever there is subsampling AG News (below), not switching architectures.

---

## 2. Dataset Selection

You need 2–3 tasks that (a) fit free Colab/Kaggle compute, (b) let you compare directly against DeeBERT/PABEE-style literature, and (c) demonstrate generalization beyond a single sentence-classification format.

| Dataset | Task type | Size (train/dev) | Why it's here |
|---|---|---|---|
| **SST-2** (GLUE) | Single-sentence, binary sentiment | ~67k / 872 | Already your baseline — keep it for continuity and direct before/after comparison |
| **MRPC** (GLUE) | Sentence-**pair**, paraphrase detection | ~3.7k / 408 | Small and fast (good for the experimentation-time budget), and it's the flagship dataset DeeBERT/PABEE report — lets you cite their numbers as a direct reference point. Also tests your off-ramps on a structurally different input (two sentences, not one) |
| **AG News** | Single-sentence, 4-class topic classification | 120k / 7.6k | Non-GLUE, multi-class (not binary) — this is your generalization claim: "works beyond binary GLUE-style sentiment tasks." Subsample train to ~20–25k examples to keep training time reasonable; this is a scope decision, not a quality compromise, and should be stated explicitly in the paper |

**Loading (HuggingFace `datasets`):**
```python
from datasets import load_dataset
sst2   = load_dataset("glue", "sst2")
mrpc   = load_dataset("glue", "mrpc")
agnews = load_dataset("ag_news")
```

**Consideration:** MRPC is a *pair* task — your tokenizer call needs `tokenizer(sentence1, sentence2, ...)` and your off-ramp/final classifiers need `num_labels` set per-dataset (2 for SST-2/MRPC, 4 for AG News). Don't hardcode `num_labels=2` anywhere in the model class — parametrize it.

**Consideration:** AG News subsampling should use stratified sampling (equal per class) to avoid skewing baseline accuracy, and the exact subsample + seed should be logged/fixed so all method variants train on the *identical* subset — otherwise your efficiency comparison is confounded by data differences.

---

## 3. Repository / File Structure

Given you're going with self-contained notebooks (not a modular `.py`-heavy repo), keep external files minimal but don't skip the ones that prevent silent inconsistency between notebooks:

```
green-ai-early-exit/
├── README.md              # setup instructions, how to run each notebook, results summary
├── requirements.txt       # torch, transformers, datasets, codecarbon, scipy, scikit-learn
├── config.py              # SHARED constants both notebooks import — see below, this is the one
│                             non-negotiable shared file
├── train.ipynb            # training notebook (Section 4)
├── evaluate_compare.ipynb # evaluation/comparison notebook (Section 5)
├── checkpoints/           # saved model weights per config × dataset (gitignored, large files)
├── results/               # metrics CSVs, comparison tables, plots (tracked in git — small files)
├── emissions/             # raw CodeCarbon logs per run (gitignored, but summarized into results/)
└── .gitignore             # checkpoints/, emissions/, __pycache__/, venv/, *.pt
```

**Why `config.py` despite preferring self-contained notebooks:** if exit-layer candidates, dataset names, seeds, or hyperparameter grids live independently inside each notebook, the two notebooks *will* drift out of sync over a multi-week project — e.g., the eval notebook silently testing a different merge-ratio grid than what training actually swept. One small shared file (under 50 lines) prevents this without meaningfully breaking the "mostly self-contained" preference. Example contents:

```python
# config.py
MODEL_NAME = "distilbert-base-uncased"
DATASETS = {
    "sst2":   {"num_labels": 2, "hf_path": ("glue", "sst2")},
    "mrpc":   {"num_labels": 2, "hf_path": ("glue", "mrpc")},
    "agnews": {"num_labels": 4, "hf_path": ("ag_news", None), "train_subsample": 24000},
}
SEED = 42
EXIT_LAYER_CANDIDATES = [2, 3, 4]        # searched in Cost-Aware Placement
MERGE_RATIO_GRID = [0.0, 0.25, 0.5]      # 0.0 = no merging, for ablation
ENTROPY_THRESHOLD_GRID = [0.2, 0.3, 0.4, 0.5]
N_ENERGY_REPEATS = 3                     # repeated runs for energy variance reporting
CHECKPOINT_DIR = "checkpoints/"
RESULTS_DIR = "results/"
```

---

## 4. Training Notebook (`train.ipynb`) — Structure

**Purpose:** produce trained checkpoints for every configuration you need to compare, per dataset. One run through this notebook (parametrized by dataset) should produce a full set of comparable checkpoints.

**Cell-by-cell outline:**

1. **Setup** — installs (`pip install transformers datasets codecarbon --quiet`), imports, `from config import *`, set seeds for reproducibility, mount/select dataset via a notebook parameter cell (e.g., `DATASET_KEY = "sst2"` at the top, so the same notebook is re-run per dataset rather than duplicated 3×)
2. **Data loading & tokenization** — load via `config.DATASETS[DATASET_KEY]`, handle single-sentence vs. pair tokenization based on dataset, apply AG News subsampling with fixed seed if applicable
3. **Model definition** — `EarlyExitDistilBERT` class: backbone + off-ramp classifier(s) at `EXIT_LAYER_CANDIDATES` positions + final classifier + token-merge module inserted before the off-ramp(s); `num_labels` parametrized from config, not hardcoded
4. **Stage 1 — Baseline convergence** — train the full backbone + final classifier only (no off-ramps active) with standard cross-entropy; wrap this training loop with `torch.cuda.amp.autocast()` + `GradScaler` for mixed precision; wrap the whole stage with `EmissionsTracker` (CodeCarbon) to log training energy
5. **Stage 2a — Cost-Aware Exit Placement (profiling)** — with the frozen Stage-1 backbone, profile per-layer energy (CodeCarbon or `torch.profiler` for FLOPs as a cheaper proxy if energy profiling is too noisy at this granularity) across `EXIT_LAYER_CANDIDATES`; fit/apply the simple cost model; select and log the chosen exit layer(s) — **this cell's output (chosen layer) should be saved to `results/`, not just printed, so the eval notebook can read it back**
6. **Stage 2b — Off-ramp + token-merge training (self-distillation)** — freeze backbone, attach off-ramp classifier(s) at the chosen layer(s) with the token-merge module preceding them, train using the combined loss: `α · CE(y, ŷ_l) + (1-α) · KL(ŷ_final || ŷ_l)` (self-distillation from Stage-1's final classifier as teacher); sweep `MERGE_RATIO_GRID × ENTROPY_THRESHOLD_GRID` from config; still wrapped in `EmissionsTracker` + mixed precision
7. **Checkpoint export** — save each trained configuration (baseline-only, +mixed-precision+distillation, +cost-aware placement, full combination at each merge-ratio/threshold point) to `checkpoints/{dataset}_{config_name}.pt` with a clear, consistent naming scheme the eval notebook can enumerate programmatically
8. **Training summary cell** — print/save a small table: config name → final training loss → training energy (J) → training CO₂eq — this is your "what did it cost to build this" number for the paper

**Consideration:** run Stage 1 *once* per dataset (it's the shared frozen backbone), then branch Stages 2a/2b for every method variant — don't retrain the backbone per variant, that's wasted compute directly contradicting your Green AI theme.

---

## 5. Comparison / Evaluation Notebook (`evaluate_compare.ipynb`) — Structure

**Purpose:** load every checkpoint produced by `train.ipynb` across all 3 datasets and produce the final comparison tables/plots for the paper.

**Cell-by-cell outline:**

1. **Setup** — imports, `from config import *`, load the exit-layer choices saved by the training notebook (don't re-derive them here)
2. **Checkpoint enumeration** — programmatically list all `checkpoints/*.pt` files and parse dataset/config name from the filename convention, so adding a new checkpoint later doesn't require editing this notebook
3. **Evaluation loop (per checkpoint × per dataset's test/dev split)** — for each config: run inference over the full eval set, recording per-example: predicted label, true label, exit layer used (or "full depth" if no exit triggered), latency (ms)
4. **Task-performance metrics** — accuracy, F1 (macro, since AG News is multi-class), ECE (calibration) computed per config
5. **Efficiency metrics** — average exit layer / average tokens retained, FLOPs estimate (via `torch.profiler` or a manual FLOP counter, since it's deterministic and doesn't need repeated runs), latency mean + p95/p99
6. **Energy/CO₂ metrics (the noisy ones — repeat these)** — wrap each config's full eval-set pass in `EmissionsTracker`, repeated `N_ENERGY_REPEATS` times (config.py), report mean ± std energy (J) and CO₂eq (g) per inference and for the full eval pass
7. **Robustness subset** — construct or reuse a "hard" subset (e.g., examples near the decision boundary from the Stage-1 baseline, or a small hand-picked adversarial set for MRPC/SST-2) and re-report accuracy restricted to that subset per config
8. **Compute-normalized headline metric** — accuracy-per-Joule and accuracy-per-gCO₂, computed from the metrics above, as the primary comparison column
9. **Results table assembly** — one master DataFrame: rows = (dataset × config), columns = all metrics above; export to `results/full_comparison.csv`
10. **Plots** — at minimum: (a) accuracy vs. energy scatter (one point per config, per dataset — this is the plot reviewers expect in an efficiency paper), (b) exit-layer distribution histogram per config, (c) merge-ratio × threshold heatmap of accuracy-per-Joule for the Phase 3 sweep
11. **Baseline comparison assertion** — a final sanity-check cell that explicitly asserts every non-baseline config's accuracy is ≥ baseline − some small tolerance (e.g., 1 point), flagging (not silently passing) any regression — this directly encodes your "no regressions" constraint as a checkable test, not just a hope

**Consideration:** keep the evaluation notebook read-only with respect to `checkpoints/` — it should never retrain anything. If a config's results look wrong, that's a bug to fix in `train.ipynb`, not a reason to patch numbers in the eval notebook.

---

## 6. Other Files to Consider

- **`README.md`** — setup steps (`pip install -r requirements.txt`), how to run `train.ipynb` per dataset, how to run `evaluate_compare.ipynb`, where results land, and a short results summary table once you have numbers (useful for both your team and anyone evaluating the repo)
- **`requirements.txt`** — pin versions (`torch`, `transformers`, `datasets`, `codecarbon`, `scikit-learn`, `scipy`, `matplotlib`) so energy numbers stay reproducible across team members' Colab sessions — package versions can shift underlying kernel implementations and slightly change energy readings
- **`.gitignore`** — exclude `checkpoints/` (large binary files), `emissions/` (verbose raw logs), `__pycache__/`, `venv/`
- **`results/README.md`** (optional, small) — one paragraph explaining what each CSV/plot in `results/` represents, since this folder is what you'll actually pull figures from when writing the paper

---

## 7. Cross-Cutting Considerations

- **Energy measurement noise:** Colab's shared, virtualized hardware makes CodeCarbon/RAPL readings noisier than a dedicated machine. Always report mean ± std over `N_ENERGY_REPEATS` runs, and run energy-measurement passes back-to-back within a single Colab session where possible (avoid comparing a run from a fresh session against one from hours into a session — thermal/throttling state can drift)
- **Fixed seeds everywhere** — data subsampling, model init, training shuffling — so any A/B difference you report is attributable to the method, not to run-to-run noise
- **Sentence-pair handling** — MRPC will break any code that assumes single-sentence input; test the tokenization path on MRPC early (Phase 0/1), not after everything else is built
- **Don't retrain the shared backbone per method variant** — Stage 1 is shared; only Stage 2 varies. This is both a compute-saving practice and a fairness requirement (all variants must start from the identical frozen backbone to be comparable)
- **Log the exit-layer decision as data, not just a printed number** — the eval notebook needs to read back what Cost-Aware Placement chose; don't make this a manual copy-paste step between notebooks

---

## References
(carried over from the option-analysis doc, plus dataset sources)
- Xin, J. et al. "DeeBERT: Dynamic Early Exiting for Accelerating BERT Inference." ACL 2020.
- Zhou, W. et al. "BERT Loses Patience: Fast and Robust Inference with Early Exit." NeurIPS 2020.
- Liu, W. et al. "FastBERT: a Self-distilling BERT with Adaptive Inference Time." ACL 2020.
- Bolya, D. et al. "Token Merging: Your ViT But Faster." ICLR 2023.
- Wang, A. et al. "GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding." 2018. (source of SST-2, MRPC)
- Zhang, X. et al. "Character-level Convolutional Networks for Text Classification." NeurIPS 2015. (source of the AG News dataset)
- "Dissecting Transformers: A CLEAR Perspective towards Green AI." arXiv:2510.02810, 2025.
