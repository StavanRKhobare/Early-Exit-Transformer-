# Float32 GPU real runs with a cool-down before every measurement, so clocks hold and throttling cannot pick the winner.
# Run from the project folder, after the float32 stages: python scripts/float32_cooled.py
# Output: outputs/results/real_runs_float32_cooled.csv. Rows already there are kept.
import os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
from _load_notebook import notebook_source
os.environ.update(GV_AMP="0", GV_HW="laptopfp32", GV_CPU_REGIME="0", GV_STAGES="", MPLBACKEND="Agg")
exec(compile(notebook_source(), "experiment.ipynb", "exec"))

COOL_C, TIMEOUT_S, B0 = 70, 240, "m[none]e[none]f[lin]"
EXIT = "m[none]e[6mlp192_10mlp192]f[lin]"
out = os.path.join(RESULTS, "real_runs_float32_cooled.csv")
old = pd.read_csv(out) if os.path.exists(out) else pd.DataFrame()
m = pd.read_csv(os.path.join(RESULTS, "measured_comparison.csv"))
m = m[np.isclose(m.eps_pp, 1.0) & (m.hw == "cuda_laptopfp32")]
plan = []                                                   # (batch, rep, keys): picks of each family, then one added comparison
for batch in (1, 32, 128):
    keys = [(B0, 0.0)] + [(r.design, float(r.tau)) for r in m[m.batch == batch].itertuples()]
    plan += [(batch, rep, keys, "pick") for rep in range(3)]
# The throttled table picked a late exit at batch 128. Measure the exit design that wins every other batch-32+ regime too.
plan += [(128, rep, [(B0, 0.0), (EXIT, 0.2)], "added") for rep in range(3, 6)]

F_ = load_json("finalists.json"); heads = torch.load(os.path.join(RESULTS, "finalist_heads.pt"), weights_only=True)
recs = {(r["design"], r["tau"]): r for r in F_["finalists"]}; real_idx = F_["real_idx"]
_, _, Xte, Yte = load_cifar()
meter, bb, aux = make_meter("cuda"), make_backbone("cuda"), make_rapl()
xs = torch.cat([prepare(Xte[real_idx[i:i + 64]], "cuda") for i in range(0, len(real_idx), 64)]); ys = Yte[real_idx]
rng = np.random.RandomState(7); rows = []
for batch, rep, keys, kind in plan:
    if len(old) and ((old.batch == batch) & (old.rep == rep)).any():
        continue
    for i in rng.permutation(len(keys)):
        k = keys[i]; t0 = time.time()
        while gpu_state()["temp_c"] > COOL_C and time.time() - t0 < TIMEOUT_S:
            time.sleep(3)
        start = gpu_state()["temp_c"]; time.sleep(1)
        res = real_once(RealModel(bb, recs[k], heads, "cuda"), k[1], xs, ys, batch, "cuda", meter, aux, WINDOW_S)
        st = gpu_state()
        rows.append({"batch": batch, "rep": rep, "kind": kind, "design": k[0], "tau": k[1], "family": recs[k]["family"], "temp_start_c": start, "cooled_s": time.time() - t0, **res, **{"gpu_" + a: v for a, v in st.items()}})
        pd.concat([old, pd.DataFrame(rows)]).to_csv(out, index=False)
        print(batch, rep, recs[k]["family"], round(res["gross_per_sample"], 4), start, st["temp_c"], st["clock_mhz"], st["sw_thermal_slowdown"], flush=True)
print("DONE", flush=True)
