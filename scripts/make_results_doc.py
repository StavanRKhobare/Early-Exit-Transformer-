# Writes RESULTS.md in the project folder. Tables come straight from the CSVs, so they cannot drift from the data.
import json, os
import numpy as np, pandas as pd
R = "outputs/results"; rd = lambda f: pd.read_csv(os.path.join(R, f))
LAB = {"exit_only": "Exit only", "merge_only": "Merge only", "exit+merge": "Exit + merge"}
m, cool, ag, add, be, me, fm, rv, runs = (rd(f) for f in ("measured_comparison.csv", "float32_cooled_summary.csv", "mac_vs_energy_agreement.csv", "merge_added_value.csv", "merge_breakeven.csv", "merge_only_effect.csv", "finalist_metrics.csv", "real_measured.csv", "real_runs.csv"))
prec = rd("gpu_precision_check.csv"); T = json.load(open(os.path.join(R, "energy_tables.json")))
stable = ~m.hw.str.endswith("fp32")
def md(d): return d.to_markdown(index=False)
def rng(x, a="measured_saved_pct", lo="measured_saved_min", hi="measured_saved_max"): return f"{x[a]:.1f} ({x[lo]:.1f} to {x[hi]:.1f})"

def measured(eps):
    d = m[stable & np.isclose(m.eps_pp, eps) & m.measured]
    rows = [{"Regime": r.regime, "Method": LAB[r.family], "Test accuracy vs B0 (pts)": f"{r.acc_diff_pp:+.2f}", "Energy saved, measured % (range)": rng(r._asdict()), "Predicted %": f"{r.predicted_saved_pct:.1f}",
             "MACs claim %": f"{r.macs_saved_pct:.1f}", "Time saved %": f"{r.time_saved_pct:.1f}", "With CPU package %": "" if pd.isna(r.with_cpu_saved_pct) else f"{r.with_cpu_saved_pct:.1f}", "mJ per image": f"{r.mj_per_sample:.0f}"} for r in d.itertuples()]
    return md(pd.DataFrame(rows))

def cooled():
    d = cool.copy(); order = {"exit_only": 0, "merge_only": 1, "exit+merge": 2}; d["o"] = d.family.map(order); d = d.sort_values(["batch", "o", "kind"], ascending=[True, True, False])
    return md(pd.DataFrame([{"Batch": r.batch, "Method": LAB[r.family], "Design": f"`{r.design}` τ={r.tau:g}" + (" (added)" if r.kind == "added" else ""), "Test accuracy vs B0 (pts)": f"{r.acc_diff_pp:+.2f}", "Energy saved % (range)": rng(r._asdict()),
                             "MACs claim %": f"{r.macs_saved_pct:.1f}", "Time saved %": f"{r.time_saved_pct:.1f}", "With CPU package %": f"{r.with_cpu_saved_pct:.1f}", "mJ per image": f"{r.mj_per_sample:.0f}", "B0 mJ": f"{r.b0_mj_per_sample:.0f}"} for r in d.itertuples()]))

def designs():
    d = m[np.isclose(m.eps_pp, 1.0) & stable].drop_duplicates(["design", "tau"])
    f = fm.set_index(["design", "tau"])
    return md(pd.DataFrame([{"Design": f"`{r.design}`", "τ": f"{r.tau:g}", "Method": LAB[r.family], "Test accuracy": f"{100*f.at[(r.design, r.tau), 'test_acc']:.2f} ± {100*f.at[(r.design, r.tau), 'test_acc_std']:.2f}", "vs B0 (pts)": f"{f.at[(r.design, r.tau), 'acc_diff_pp']:+.2f}",
                             "95% CI, seed 0": f"{f.at[(r.design, r.tau), 'diff_ci_lo_pp_seed0']:+.2f} to {f.at[(r.design, r.tau), 'diff_ci_hi_pp_seed0']:+.2f}", "ECE": f"{f.at[(r.design, r.tau), 'test_ece']:.3f}"} for r in d.itertuples()]))

def merges():
    d = me[me.n_merges == 1].copy(); d["l"] = d.merge_schedule.str.extract(r"^(\d+)r").astype(int); d["r"] = d.merge_schedule.str.extract(r"r(\d+)$").astype(int); d = d.sort_values(["l", "r"])
    return md(pd.DataFrame([{"After block": r.l, "Tokens removed": f"{r.r}%", "Test accuracy vs B0 (pts)": f"{r.acc_diff_pp:+.2f}", "MACs saved %": f"{r.macs_saved_pct:.1f}", "CPU b1 %": f"{r.gross_saved_pct__cpu_laptop__b1:.1f}", "CPU b32 %": f"{r.gross_saved_pct__cpu_laptop__b32:.1f}",
                             "GPU f16 b1 %": f"{r.gross_saved_pct__cuda_laptop__b1:.1f}", "GPU f16 b128 %": f"{r.gross_saved_pct__cuda_laptop__b128:.1f}"} for r in d.itertuples()]))

def macs():
    d = ag[~ag.hw.str.endswith("fp32")]; name = {"cpu_laptop": "CPU", "cuda_laptop": "GPU float16"}
    return md(pd.DataFrame([{"Regime": f"{name[r.hw]} batch {r.batch}", "Tolerance (pts)": f"{r.eps_pp:g}", "Spearman": f"{r.spearman_energy_vs_macs:.3f}", "Same design": "yes" if r.same_design else "no", "Energy pick saves %": f"{r.energy_pick_saved_pct:.1f}", "MAC pick saves %": f"{r.mac_pick_saved_pct:.1f}", "MACs claim %": f"{r.macs_claimed_saved_pct:.1f}", "MAC regret %": f"{r.mac_regret_pct:.2f}", "Token-layer regret %": f"{r.token_pick_regret_pct:.2f}"} for r in d.itertuples()]))

def errors():
    d = rv.assign(a=rv.pred_error_pct.abs()).groupby(["hw", "batch"], sort=False).agg(m=("a", "median"), x=("a", "max"), n=("a", "size")).reset_index(); name = {"cpu_laptop": "CPU", "cuda_laptop": "GPU float16", "cuda_laptopfp32": "GPU float32 (throttled table)"}
    return md(pd.DataFrame([{"Regime": f"{name[r.hw]} batch {r.batch}", "Operating points": r.n, "Median absolute error %": f"{r.m:.1f}", "Largest %": f"{r.x:.1f}"} for r in d.itertuples()]))

def thermal():
    g = runs[runs.hw.str.startswith("cuda")].copy(); g["s"] = g.gpu_sw_thermal_slowdown.astype(bool)
    rows = [{"Runs": ("float16" if not r[0].endswith("fp32") else "float32, no cool-down"), "Batch": r[1], "Slowdown active": f"{100*x.s.mean():.0f}%", "Temp °C": f"{x.gpu_temp_c.median():.0f}", "Clock MHz, median (range)": f"{x.gpu_clock_mhz.median():.0f} ({x.gpu_clock_mhz.min():.0f} to {x.gpu_clock_mhz.max():.0f})", "Power W": f"{x.gpu_power_w.median():.1f}"} for r, x in g.groupby(["hw", "batch"])]
    c = rd("real_runs_float32_cooled.csv")
    rows += [{"Runs": "float32, cooled", "Batch": b, "Slowdown active": f"{100*x.gpu_sw_thermal_slowdown.astype(bool).mean():.0f}%", "Temp °C": f"{x.gpu_temp_c.median():.0f}", "Clock MHz, median (range)": f"{x.gpu_clock_mhz.median():.0f} ({x.gpu_clock_mhz.min():.0f} to {x.gpu_clock_mhz.max():.0f})", "Power W": f"{x.gpu_power_w.median():.1f}"} for b, x in c.groupby("batch")]
    return md(pd.DataFrame(rows))

def b0s():
    d = rv[rv.design == "m[none]e[none]f[lin]"]; c = {"cpu_laptop": "CPU (RAPL package)", "cuda_laptop": "GPU float16 (NVML)", "cuda_laptopfp32": "GPU float32 (NVML)"}
    return md(pd.DataFrame([{"Regime": c[r.hw], "Batch": r.batch, "mJ per image": f"{1000*r.gross:.0f}", "Net of idle, mJ": f"{1000*r.net:.0f}", "ms per image": f"{1000*r.sec:.2f}", "p95 latency ms": f"{r.lat_p95_ms:.1f}", "CPU package during GPU run, mJ": "" if pd.isna(r.aux_cpu_j) else f"{1000*r.aux_cpu_j:.0f}", "g CO₂e per 1000 images*": f"{r.co2_g_per_1k:.3f}"} for r in d.itertuples()]))

def precision():
    a = prec[prec.batch.isin([1, 8, 16, 32, 128])].pivot(index="batch", columns="precision", values=["embed_ms", "block_ms_per_image"]).reset_index()
    return md(pd.DataFrame({"Batch": a["batch"], "Embed ms, float16": a[("embed_ms", "float16 autocast")], "Embed ms, float32": a[("embed_ms", "float32")], "Block ms per image, float16": a[("block_ms_per_image", "float16 autocast")], "Block ms per image, float32": a[("block_ms_per_image", "float32")]}))

def added():
    d = add[~add.hw.str.endswith("fp32")]; name = {"cpu_laptop": "CPU", "cuda_laptop": "GPU float16"}
    return md(pd.DataFrame([{"Regime": f"{name[r.hw]} batch {r.batch}", "Tolerance (pts)": f"{r.eps_pp:g}", "Exit only %": f"{r.exit_only_saved_pct:.1f}", "Merge only %": f"{r.merge_only_saved_pct:.1f}", "Exit + merge %": f"{r.exit_merge_saved_pct:.1f}", "Merging adds (pts)": f"{r.merge_adds_pp:.1f}"} for r in d.itertuples()]))

pts = pd.read_csv(os.path.join(R, "all_points.csv.gz"), usecols=["design", "tau", "family", "final_head", "val_acc", "test_acc", "test_acc_std", "test_f1", "macs", "merge_exec_frac", "exit1_frac", "exit2_frac"])
slots = rd("slot_metrics.csv"); ops = rd("op_costs.csv")

def heads():
    d = pts[pts.family == "baseline"].sort_values("test_acc"); f = fm.set_index(["design", "tau"])
    return md(pd.DataFrame([{"Final head": r.final_head, "Validation %": f"{100*r.val_acc:.2f}", "Test % (± seeds)": f"{100*r.test_acc:.2f} ± {100*r.test_acc_std:.2f}", "Test F1": f"{r.test_f1:.4f}", "vs B0 (pts)": f"{f.at[(r.design, 0.0), 'acc_diff_pp']:+.2f}", "ECE": f"{f.at[(r.design, 0.0), 'test_ece']:.3f}"} for r in d.itertuples()]))

def exit_heads():
    d = slots[slots.prefix == "none"].groupby(["layer", "arch"], as_index=False)[["val_acc", "test_acc", "test_ece_before", "test_ece_after", "temperature"]].mean()
    d["o"] = pd.to_numeric(d.layer, errors="coerce").fillna(99); d = d.sort_values(["o", "arch"])
    return md(pd.DataFrame([{"Block": r.layer, "Head": r.arch, "Validation %": f"{100*r.val_acc:.1f}", "Test %": f"{100*r.test_acc:.1f}", "ECE before": f"{r.test_ece_before:.3f}", "ECE after": f"{r.test_ece_after:.3f}", "Temperature": f"{r.temperature:.2f}"} for r in d.itertuples()]))

def throttled():
    d = m[m.hw.str.endswith("fp32") & m.eps_pp.isin([0.5, 1.0]) & m.measured]
    return md(pd.DataFrame([{"Regime": r.regime, "Tolerance (pts)": f"{r.eps_pp:g}", "Method": LAB[r.family], "Design": f"`{r.design}` τ={r.tau:g}", "Test accuracy vs B0 (pts)": f"{r.acc_diff_pp:+.2f}", "Energy saved % (range)": rng(r._asdict()), "Time saved %": f"{r.time_saved_pct:.1f}", "mJ per image": f"{r.mj_per_sample:.0f}"} for r in d.itertuples()]))

def picks():
    d = m[stable & m.eps_pp.isin([0.5, 1.0])].drop_duplicates(["design", "tau"]).merge(pts, on=["design", "tau"], suffixes=("", "_p"))
    return md(pd.DataFrame([{"Design": f"`{r.design}`", "τ": f"{r.tau:g}", "Method": LAB[r.family], "Leave at first exit %": f"{100*r.exit1_frac:.1f}", "Leave at second exit %": f"{100*r.exit2_frac:.1f}", "Reach the first merge %": "" if r.family == "exit_only" else f"{100*r.merge_exec_frac:.1f}"} for r in d.itertuples()]))

def latency():
    d = m[stable & np.isclose(m.eps_pp, 1.0) & m.measured]; base = rv[rv.design == "m[none]e[none]f[lin]"].set_index(["hw", "batch"]).lat_p95_ms
    return md(pd.DataFrame([{"Regime": r.regime, "Method": LAB[r.family], "p95 ms per image": f"{r.latency_p95_ms:.1f}", "B0 p95 ms": f"{base[(r.hw, r.batch)]:.1f}"} for r in d.itertuples()]))

def all_merges():
    return md(pd.DataFrame([{"Schedule": r.merge_schedule, "Merges": r.n_merges, "Test %": f"{100*r.test_acc:.2f}", "vs B0 (pts)": f"{r.acc_diff_pp:+.2f}", "MACs saved %": f"{r.macs_saved_pct:.1f}", "CPU b1 %": f"{r.gross_saved_pct__cpu_laptop__b1:.1f}", "CPU b32 %": f"{r.gross_saved_pct__cpu_laptop__b32:.1f}", "GPU f16 b1 %": f"{r.gross_saved_pct__cuda_laptop__b1:.1f}", "GPU f16 b32 %": f"{r.gross_saved_pct__cuda_laptop__b32:.1f}", "GPU f16 b128 %": f"{r.gross_saved_pct__cuda_laptop__b128:.1f}"} for r in me.sort_values("acc_diff_pp", ascending=False).itertuples()]))

def breakeven():
    d = be[(be.metric == "gross") & ~be.hw.str.endswith("fp32")]; name = {"cpu_laptop": "CPU", "cuda_laptop": "GPU float16"}
    return md(pd.DataFrame([{"Device": name[r.hw], "Batch": r.batch, "Tokens": f"{r.tokens_before} → {r.tokens_after}", "Merge cost, in blocks": f"{r.merge_cost_in_blocks:.2f}", "Block saving %": f"{r.block_saving_pct:.1f}", "Break-even, blocks": f"{r.breakeven_blocks:.2f}", "MAC break-even": f"{r.mac_breakeven_blocks:.3f}"} for r in d.sort_values(["hw", "batch", "tokens_before", "tokens_after"]).itertuples()]))

def op_table():
    keep = ["embed", "block|197", "block|148", "block|100", "block|76", "block|51", "merge|197|0.5", "merge|197|0.25", "merge|100|0.5", "merge|100|0.25", "head|lin", "head|mlp192", "final|lin"]
    d = ops[ops.op.isin(keep)].copy(); d["v"] = [f"{1000*g:.1f} / {1000*t:.2f}" for g, t in zip(d.gross, d.sec)]; d["c"] = d.regime + " b" + d.batch.astype(str)
    p = d.pivot(index="op", columns="c", values="v").reindex(keep).reset_index().rename(columns={"op": "Operation"}).fillna("")
    def nice(o):
        q = o.split("|")
        if q[0] == "block": return f"Block, {q[1]} tokens"
        if q[0] == "merge": return f"Merge {100*float(q[2]):.0f}%, {q[1]} tokens"
        if q[0] == "head": return f"Exit head, {q[1]}"
        if q[0] == "final": return f"Final head, {q[1]}"
        return "Patch embedding"
    p["Operation"] = p.Operation.map(nice)
    cols = ["Operation"] + [f"{a} b{b}" for a, bs in (("CPU", (1, 8, 32)), ("GPU float16", (1, 8, 32, 128)), ("GPU float32", (1, 8, 32, 128))) for b in bs]
    return md(p[cols])

def precision_energy():
    d = rv[rv.design == "m[none]e[none]f[lin]"].set_index(["hw", "batch"]); rows = []
    for b in (1, 32, 128):
        a, c = d.loc[("cuda_laptop", b)], d.loc[("cuda_laptopfp32", b)]
        rows.append({"Batch": b, "Float16 mJ": f"{1000*a.gross:.0f}", "Float32 mJ": f"{1000*c.gross:.0f}", "Energy saved %": f"{100*(1-c.gross/a.gross):.1f}", "Float16 ms per image": f"{1000*a.sec:.2f}", "Float32 ms per image": f"{1000*c.sec:.2f}", "Accuracy on 1,000 images, float16 / float32": ("not recorded" if pd.isna(a.real_acc) else f"{100*a.real_acc:.1f}%") + f" / {100*c.real_acc:.1f}%"})
    return md(pd.DataFrame(rows))

doc = open("scripts/results_template.md").read()
for k, f in {"MEASURED1": lambda: measured(1.0), "MEASURED05": lambda: measured(0.5), "COOLED": cooled, "DESIGNS": designs, "MERGES": merges, "MACS": macs, "ERRORS": errors, "THERMAL": thermal, "B0S": b0s, "PRECISION": precision, "ADDED": added, "HEADS": heads, "EXITHEADS": exit_heads, "THROTTLED": throttled, "PICKS": picks, "LATENCY": latency, "ALLMERGES": all_merges, "BREAKEVEN": breakeven, "OPS": op_table, "PRECENERGY": precision_energy}.items():
    assert "{{" + k + "}}" in doc, k
    doc = doc.replace("{{" + k + "}}", f())
assert "{{" not in doc
doc = doc.replace("](fig_", "](outputs/results/fig_")
open("RESULTS.md", "w").write(doc); print(len(doc), "characters")
