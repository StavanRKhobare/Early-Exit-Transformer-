# When does token merging help? Early exit and token merging on a frozen ViT-Tiny

Compares four methods on CIFAR-10 across hardware regimes: full model, exit only, merge only, exit + merge. Metrics for all of them are saved whichever method wins. Background and audit: `PROJECT_CONTEXT.md`.

**Results:** [`RESULTS.md`](RESULTS.md)

## Files
- `scripts/`: cooled float32 runs, the precision check, and the generator for `RESULTS.md`
- `experiment.ipynb`: the whole study, resumable, runs on Kaggle, Colab or locally
- `data/cifar-10/`: official CIFAR-10 (not needed on Kaggle/Colab, the notebook downloads it)
- `archive/`: earlier runs and notebooks. Their results used the wrong normalisation, so do not cite them.

## Run on this laptop
```bash
# Once: fix the GPU driver, then reboot
sudo apt install linux-modules-nvidia-595-open-$(uname -r)
# Once: environment (about 6 GB)
uv venv .venv --python 3.12 && uv pip install --python .venv/bin/python -r requirements.txt
# After every reboot: allow reading CPU energy
sudo chmod a+r /sys/class/powercap/intel-rapl:0/energy_uj

source .venv/bin/activate
# 1. A few minutes: proves the GPU path works. Must end with "largest gap: 0.00 points" or close to it.
SMOKE_TEST=1 GV_WORK=outputs_smoke MPLBACKEND=Agg jupyter nbconvert --to notebook --execute --output /tmp/smoke_run.ipynb experiment.ipynb
# 2. The real study: resumes if interrupted. Plug in, close other apps.
GV_HW=laptop MPLBACKEND=Agg nohup jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 --output experiment_run.ipynb experiment.ipynb > run.log 2>&1 &
tail -f outputs/results/progress.log   # watch progress
```
Needs about 2 GB of disk for features. Delete `outputs/scratch/` afterwards; `outputs/results/` is what matters.

**Heat:** this laptop's GPU hit 91 C with thermal slowdown active under sustained load. Raise the rear of the laptop and set the fan profile to Turbo (Fn+F5 on most TUF models). The notebook waits for 75 C before measuring and records temperature, clock and throttling. Disclose these in the paper.

## Run on Kaggle
Upload `experiment.ipynb`, turn GPU and Internet on, then use **Save Version > Save & Run All**. Download `green_vit_results.zip` from the Output tab, or: `kaggle kernels output <user>/<notebook-slug> -p ./outputs`

To add this laptop's measurements to a Kaggle run, unzip into `outputs/results/` here and run with `GV_STAGES=energy,real,report`.

## Settings (environment variables)
`GV_TRAIN_PC` train images per class (default 2000), `GV_SEEDS` (5), `GV_EXTRACT_BATCH` (128, lower if GPU memory runs out), `GV_WINDOW` seconds per measurement (5), `GV_REPEATS` (3), `GV_CPU_REGIME` (auto/1/0), `GV_STAGES` (extract,train,points,energy,select,real,report). `GV_AMP=0` runs the GPU in float32; give it its own `GV_HW` label ending in `fp32` so both stay comparable.

On a GTX 1650, float16 autocast is about 3.4x slower than float32. Check this on your own GPU with `scripts/gpu_precision_check.py` before measuring.

`RESULTS.md` in the project folder compiles all of these. Start there.

## Results (`outputs/results/`, or inside the zip)
| File | Answers |
|---|---|
| `SUMMARY.md` | Short automatic summary |
| `measured_comparison.csv` | Measured savings per regime, method and tolerance |
| `float32_cooled_summary.csv`, `real_runs_float32_cooled.csv` | GPU float32 with a cool-down before each run |
| `gpu_precision_check.csv` | Float16 against float32 timing |
| `op_costs.csv` | Cost of each operation per batch size |
| `all_points.csv.gz` | Every design at every threshold: accuracy, F1, MACs, token-layers, and predicted gross energy, net energy and time per regime |
| `method_comparison.csv` | Best of each method per regime and tolerance, reported on test |
| `merge_added_value.csv` | What merging adds beyond exits |
| `merge_breakeven.csv` | How many blocks a merge needs to repay itself, per token count and batch size |
| `merge_only_effect.csv` | Accuracy cost and savings of every merge schedule |
| `mac_vs_energy_agreement.csv` | Do MACs and energy pick the same design, and what a wrong pick costs |
| `real_measured.csv`, `real_runs.csv` | Real measurements: energy, net energy, latency p50/p95/p99, prediction error |
| `finalist_metrics.csv` | Test accuracy, ECE, bootstrap CI against B0 |
| `slot_metrics.csv` | Accuracy and calibration of every head |
| `phase_costs.json` | The study's own energy and time |
| `energy_tables.json` | Raw cost tables, and which meter produced them |
| `fig_*.png` | Figures |

Quote measured numbers in the paper. Predictions ran 10-20% low in testing but ranked designs well. Select only on validation columns; test columns are for reporting.
