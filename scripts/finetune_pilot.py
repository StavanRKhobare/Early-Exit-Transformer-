# Fine-tuning pilot. Stage 1: fine-tune backbone + final head (fair B0 and merged 6r50_8r25), float32. Stage 2: freeze, train exit heads on cached features.
# Stage 3: offline selection on validation, reported on test. Costs are cost-table predictions, NOT measurements. Resumable. FT_SMOKE=1 for a code check.
import os, json, time, math, subprocess
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F, timm
SMOKE = os.environ.get("FT_SMOKE", "0") == "1"
ROOT = os.path.abspath("outputs")
OUT = os.path.join(ROOT, "results", "finetune_smoke" if SMOKE else "finetune_pilot"); CK = os.path.join(ROOT, "scratch", "ft_smoke" if SMOKE else "ft_pilot")
os.makedirs(OUT, exist_ok=True); os.makedirs(CK, exist_ok=True)
EPOCHS = 1 if SMOKE else int(os.environ.get("FT_EPOCHS", 5)); BATCH = 64; SEED = 0
DEV = "cuda"; assert torch.cuda.is_available()
def log(*a):
    s = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a); print(s, flush=True); open(os.path.join(OUT, "progress.log"), "a").write(s + "\n")
def gpu():
    q = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,clocks.gr,power.draw", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()
    return q
def cool(limit=80, timeout=120):
    t0 = time.time()
    while int(gpu().split(",")[0]) > limit and time.time() - t0 < timeout: time.sleep(5)
# ---- same data, normalisation and merge as experiment.ipynb ----
d = torch.load(os.path.join(ROOT, "scratch/cache/cifar10_uint8.pt"), weights_only=True)
sp = json.load(open(os.path.join(ROOT, "results/splits.json")))
if SMOKE: sp = {"train": sp["train"][:640], "cal": sp["cal"][:200], "val": sp["val"][:400]}
nte = 400 if SMOKE else 10000
DATA = {"train": (d["Xtr"][sp["train"]], d["Ytr"][sp["train"]]), "cal": (d["Xtr"][sp["cal"]], d["Ytr"][sp["cal"]]), "val": (d["Xtr"][sp["val"]], d["Ytr"][sp["val"]]), "test": (d["Xte"][:nte], d["Yte"][:nte])}
def prepare(x):
    x = x.to(DEV).float().div_(255.0); x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    return (x - 0.5) / 0.5
def embed(bb, x):
    t = bb._pos_embed(bb.patch_embed(x)); return bb.norm_pre(t) if hasattr(bb, "norm_pre") else t
def merge_pairs(Np, ratio):
    return 0 if ratio <= 0 or Np < 4 else int(min(max(1, int(Np * ratio)), (Np + 1) // 2 - 1))
def token_merge(tokens, ratio):
    r = merge_pairs(tokens.shape[1] - 1, ratio)
    if r <= 0: return tokens
    dtype, t = tokens.dtype, tokens.float(); cls_tok, patch = t[:, :1], t[:, 1:]; a, b = patch[:, 0::2], patch[:, 1::2]
    scores = F.normalize(a, dim=-1) @ F.normalize(b, dim=-1).transpose(-1, -2)
    node_max, node_idx = scores.max(dim=-1); order = node_max.argsort(dim=-1, descending=True)
    src_idx, unm_idx = order[:, :r], order[:, r:]; dst_idx = node_idx.gather(-1, src_idx); D = a.shape[-1]
    unm = a.gather(1, unm_idx.unsqueeze(-1).expand(-1, -1, D)); src = a.gather(1, src_idx.unsqueeze(-1).expand(-1, -1, D))
    b = b.scatter_reduce(1, dst_idx.unsqueeze(-1).expand(-1, -1, D), src, reduce="mean", include_self=True)
    return torch.cat([cls_tok, unm, b], dim=1).to(dtype)
def make_bb():
    return timm.create_model("vit_tiny_patch16_224", pretrained=True, num_classes=0).to(DEV)
def forward(bb, x, merges, sink=None):
    t = embed(bb, x)
    for l, blk in enumerate(bb.blocks, 1):
        t = blk(t)
        if sink is not None and l in (4, 6, 8, 10): sink[l] = t[:, 0]
        if l in merges: t = token_merge(t, merges[l])
    return bb.norm(t)[:, 0]
def augment(x):
    n = x.shape[0]; p = F.pad(x.float(), (4, 4, 4, 4), mode="reflect")
    ox, oy = torch.randint(0, 9, (n,), device=x.device), torch.randint(0, 9, (n,), device=x.device)
    ix = (ox[:, None] + torch.arange(32, device=x.device)[None]); iy = (oy[:, None] + torch.arange(32, device=x.device)[None])
    b = torch.arange(n, device=x.device)[:, None, None, None]; c = torch.arange(3, device=x.device)[None, :, None, None]
    out = p[b, c, iy[:, None, :, None], ix[:, None, None, :]]
    flip = torch.rand(n, device=x.device) < .5; out[flip] = out[flip].flip(-1)
    return out
@torch.no_grad()
def acc_of(bb, head, merges, split):
    X, Y = DATA[split]; c = 0
    for i in range(0, len(Y), 128):
        c += (head(forward(bb, prepare(X[i:i + 128]), merges)).argmax(-1).cpu() == Y[i:i + 128]).sum().item()
    return c / len(Y)
def finetune(name, merges):
    path = os.path.join(CK, f"ft_{name}.pt")
    torch.manual_seed(SEED); np.random.seed(SEED); bb = make_bb(); head = nn.Linear(192, 10).to(DEV)
    if os.path.exists(path):
        s = torch.load(path, weights_only=True, map_location=DEV); bb.load_state_dict(s["bb"]); head.load_state_dict(s["head"]); log("loaded", name); return bb.eval(), head.eval()
    groups = [{"params": list(head.parameters()), "lr": 1e-3}, {"params": list(bb.norm.parameters()), "lr": 1e-4}]
    for l, blk in enumerate(bb.blocks, 1): groups.append({"params": list(blk.parameters()), "lr": 1e-4 * .8 ** (12 - l)})
    groups.append({"params": list(bb.patch_embed.parameters()) + [bb.cls_token, bb.pos_embed], "lr": 1e-4 * .8 ** 12})
    opt = torch.optim.AdamW(groups, weight_decay=.01)
    X, Y = DATA["train"]; steps = EPOCHS * (len(Y) // BATCH)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in groups], total_steps=steps, pct_start=.3 if SMOKE else .1)
    log(f"{name}: start, {len(Y)} images, {EPOCHS} epochs, gpu", gpu()); g = torch.Generator().manual_seed(SEED)
    for ep in range(EPOCHS):
        bb.train(); head.train(); perm = torch.randperm(len(Y), generator=g); t0 = time.time(); tot = 0
        for i in range(0, len(Y) - BATCH + 1, BATCH):
            idx = perm[i:i + BATCH]; x = prepare(augment(X[idx].to(DEV))); y = Y[idx].to(DEV)
            loss = F.cross_entropy(head(forward(bb, x, merges)), y); opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(bb.parameters(), 1.0); opt.step(); sched.step(); tot += loss.item()
        bb.eval(); head.eval(); va = acc_of(bb, head, merges, "val")
        log(f"{name}: epoch {ep + 1}/{EPOCHS} loss {tot / max(1, len(Y) // BATCH):.3f} val {100 * va:.2f}% {time.time() - t0:.0f}s | temp,clock,W {gpu()}")
        cool()
    torch.save({"bb": bb.state_dict(), "head": head.state_dict()}, path)
    return bb.eval(), head.eval()
@torch.no_grad()
def extract(bb, merges):
    out = {}
    for s, (X, Y) in DATA.items():
        rows = {k: [] for k in (4, 6, 8, 10, "final")}
        for i in range(0, len(Y), 128):
            sink = {}; f = forward(bb, prepare(X[i:i + 128]), merges, sink); sink["final"] = f
            for k in rows: rows[k].append(sink[k].cpu())
        out[s] = {k: torch.cat(v) for k, v in rows.items()}
    return out
def make_head(arch, final):
    body = nn.Linear(192, 10) if arch == "lin" else nn.Sequential(nn.Linear(192, 192), nn.GELU(), nn.Linear(192, 10))
    return nn.Sequential(nn.Identity() if final else nn.LayerNorm(192), body).to(DEV)
GRID_T = torch.logspace(-1.3, 1.0, 200)
def fit_head_probs(F_, Ys, arch, final, seed):
    torch.manual_seed(1000 + seed); h = make_head(arch, final); X = F_["train"].to(DEV); y = Ys["train"].to(DEV); ep = 2 if SMOKE else 10
    opt = torch.optim.AdamW(h.parameters(), lr=2e-3, weight_decay=1e-4); n = len(y); st = ep * math.ceil(n / 128)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=st, pct_start=.3 if SMOKE else .1)
    for _ in range(ep):
        p = torch.randperm(n, device=DEV)
        for i in range(0, n, 128):
            idx = p[i:i + 128]; l = F.cross_entropy(h(X[idx]), y[idx]); opt.zero_grad(set_to_none=True); l.backward(); opt.step(); sch.step()
    with torch.no_grad():
        lg = {s: h(F_[s].to(DEV)).cpu() for s in ("cal", "val", "test")}
    T = float(GRID_T[torch.stack([F.cross_entropy(lg["cal"] / t, Ys["cal"]) for t in GRID_T]).argmin()])
    return {s: F.softmax(lg[s] / T, -1).numpy().astype(np.float32) for s in ("val", "test")}, T
MODELS = {"b0": {}, "merged": {6: .5, 8: .25}}
PROBS = {}; Ys = {s: DATA[s][1] for s in DATA}
for name, merges in MODELS.items():
    pp = os.path.join(CK, f"probs_{name}.npz")
    if os.path.exists(pp):
        z = np.load(pp); PROBS[name] = {k: z[k] for k in z.files}; log("probs cached", name); continue
    bb, head = finetune(name, merges)
    log(name, "ft head val/test %:", round(100 * acc_of(bb, head, merges, "val"), 2), round(100 * acc_of(bb, head, merges, "test"), 2))
    feats = extract(bb, merges); P = {}; recs = []
    for slot in (4, 6, 8, 10, "final"):
        for arch in ("lin", "mlp192"):
            r = {"val": [], "test": []}
            for seed in range(3):
                pr, T = fit_head_probs({s: feats[s][slot] for s in feats}, Ys, arch, slot == "final", seed)
                for s in r: r[s].append(pr[s])
                recs.append(dict(model=name, slot=str(slot), arch=arch, seed=seed, temp=T, val=(pr["val"].argmax(-1) == Ys["val"].numpy()).mean(), test=(pr["test"].argmax(-1) == Ys["test"].numpy()).mean()))
            for s in r: P[f"{slot}|{arch}|{s}"] = np.stack(r[s])
    pd.DataFrame(recs).to_csv(os.path.join(OUT, f"heads_{name}.csv"), index=False)
    np.savez(pp, **P); PROBS[name] = P; del bb, head, feats; torch.cuda.empty_cache(); log(name, "heads done"); cool()
# ---- stage 3: selection on validation, report on test ----
yv, yt = Ys["val"].numpy(), Ys["test"].numpy(); Y = {"val": yv, "test": yt}
RES = os.path.join(ROOT, "results/"); T_ = json.load(open(RES + "energy_tables.json"))
def mc(Np, r): return Np - merge_pairs(Np, r)
def icost(tab, bs, b):
    ys = np.array([tab[str(x)]["gross"] for x in bs]); return float(np.interp(np.log(max(b, 1.)), np.log(np.array(bs, float)), ys)) if b >= 1 else b * ys[0]
def block(o, n):
    k = f"block|{n}"
    if k in o: return o[k]
    best = min((int(x.split("|")[1]) for x in o if x.startswith("block|")), key=lambda x: abs(x - n)); return o[f"block|{best}"]
def cost(tb, b, g, S, L, fin, f1, f2):
    o, bs = tb["ops"], tb["batches"]; N = 197; f = 1.; c = icost(o["embed"], bs, b)
    for l in range(1, 13):
        c += icost(block(o, N), bs, b * f)
        if l == g: c += icost(o["head|mlp192"], bs, b * f); f = 1 - f1
        if l == L: c += icost(o["head|mlp192"], bs, b * f); f = 1 - f1 - f2
        if l in S: c += icost(o[f"merge|{N}|{S[l]}"], bs, b * f); N = 1 + mc(N - 1, S[l])
    return (c + icost(o[f"final|{fin}"], bs, b * f)) / b
H = lambda p: -(p * np.log(np.clip(p, 1e-12, 1))).sum(-1) / np.log(10)
def run(m, g, L, fin, w, tau, s):
    Pm = PROBS[m]; y = Y[s]; pf = Pm[f"final|{fin}|{s}"]; a_all = Pm[f"{g}|mlp192|{s}"] if g else None; b_all = Pm[f"{L}|mlp192|{s}"] if L else None; cor = []; F1 = []; F2 = []
    for k in range(3):
        n = len(y); ex1 = np.zeros(n, bool); ex2 = np.zeros(n, bool); terms = [pf[k]]; wt = [1.]
        if g: ex1 = H(a_all[k]) < tau; terms.append(a_all[k]); wt.append(w)
        if L:
            b = (w * a_all[k] + b_all[k]) / (1 + w) if g else b_all[k]; ex2 = (~ex1) & (H(b) < tau); terms.append(b_all[k]); wt.append(w)
        c = sum(t * x for t, x in zip(wt, terms)) / sum(wt); pr = c.argmax(-1)
        if L: pr = np.where(ex2, b.argmax(-1), pr)
        if g: pr = np.where(ex1, a_all[k].argmax(-1), pr)
        cor.append(pr == y); F1.append(ex1.mean()); F2.append(ex2.mean())
    return np.stack(cor), float(np.mean(F1)), float(np.mean(F2))
TAUS = [.02, .05, .08, .12, .16, .2, .25, .3, .4, .5, .6, .7]
cfg = []
for m, S in MODELS.items():
    for fin in ("lin", "mlp192"):
        for g, L in ((4, None), (6, None), (8, None), (10, None), (4, 10), (6, 10), (8, 10), (None, None)):
            if m == "merged" and g in (8, 10): continue          # selection stays on designs whose first gate comes before the merge
            for w in ((0, .25, .5) if g else (0,)):
                for tau in (TAUS if g else [0.]):
                    c, f1, f2 = run(m, g, L, fin, w, tau, "val"); cfg.append(dict(m=m, fin=fin, g=g, L=L, w=w, tau=tau, val=c.mean(), f1=f1, f2=f2))
df = pd.DataFrame(cfg); REG = [("cpu_laptop", 1), ("cpu_laptop", 32), ("cuda_laptop", 1), ("cuda_laptop", 32), ("cuda_laptop", 128)]
nz = lambda x: None if x is None or x != x else int(x)
base = {}
for f in ("lin", "mlp192"):
    for s in ("val", "test"): base[f, s] = (PROBS["b0"][f"final|{f}|{s}"].argmax(-1) == Y[s]).mean()
log("FT B0 acc", {k: round(100 * v, 2) for k, v in base.items()})
rows = []
gap = {}
for f in ("lin", "mlp192"):
    for s in ("val", "test"):
        cb = PROBS["b0"][f"final|{f}|{s}"].argmax(-1) == Y[s]; cm = PROBS["merged"][f"final|{f}|{s}"].argmax(-1) == Y[s]; di = (cm.astype(int) - cb.astype(int)).astype(float)
        rng = np.random.default_rng(0); bo = [di[0][rng.integers(0, di.shape[1], di.shape[1])].mean() * 100 for _ in range(1000)]
        gap[f, s] = dict(final=f, split=s, merged_minus_b0_pts=100 * di.mean(), seed0_ci_low=np.percentile(bo, 2.5), seed0_ci_high=np.percentile(bo, 97.5), head_seed_std=100 * di.mean(1).std(ddof=1))
pd.DataFrame(gap.values()).to_csv(os.path.join(OUT, "merge_gap.csv"), index=False); print(pd.DataFrame(gap.values()).round(2).to_string(index=False))
VAR = {"Exit only": lambda d: (d.m == "b0") & d.g.notna() & (d.w == 0), "Merge only": lambda d: (d.m == "merged") & d.g.isna(),
       "Exit+merge": lambda d: (d.m == "merged") & d.g.notna() & (d.w == 0), "Exit+merge+reuse": lambda d: (d.m == "merged") & d.g.notna(), "Exit+reuse": lambda d: (d.m == "b0") & d.g.notna()}
for t, b in REG:
    tb = T_[t]; bc = cost(tb, b, None, {}, None, "lin", 0, 0)
    for eps in (0, .0025, .005, .01):
        for fin in ("lin", "mlp192"):
            for name, fn in VAR.items():
                d = df[fn(df) & (df.fin == fin) & (df.val >= base[fin, "val"] - eps)]
                if not len(d): rows.append(dict(reg=f"{t}_b{b}", eps=eps * 100, fin=fin, method=name)); continue
                cs = [cost(tb, b, nz(r.g), MODELS[r.m], nz(r.L), fin, r.f1, r.f2) for r in d.itertuples()]; r = d.iloc[int(np.argmin(cs))]
                c, f1, f2 = run(r.m, nz(r.g), nz(r.L), fin, r.w, r.tau, "test")
                rows.append(dict(reg=f"{t}_b{b}", eps=eps * 100, fin=fin, method=name, saved=100 * (1 - cost(tb, b, nz(r.g), MODELS[r.m], nz(r.L), fin, f1, f2) / bc), dtest=100 * (c.mean() - base[fin, "test"]), dval=100 * (r.val - base[fin, "val"]), cfg=f"g{nz(r.g)} L{nz(r.L)} w{r.w} t{r.tau} f1={f1:.2f} f2={f2:.2f}"))
o = pd.DataFrame(rows); o.to_csv(os.path.join(OUT, "pilot_selection.csv"), index=False); df.to_csv(os.path.join(OUT, "pilot_configs.csv"), index=False)
pd.set_option("display.width", 200, "display.max_rows", 500, "display.max_colwidth", 45)
print(o[o.reg.isin(["cuda_laptop_b32", "cpu_laptop_b1"])].round(2).to_string(index=False))
hd = pd.concat([pd.read_csv(os.path.join(OUT, f"heads_{n}.csv")) for n in MODELS]); print(hd.groupby(["model", "slot", "arch"])[["val", "test"]].mean().mul(100).round(2).to_string())
log("DONE")
