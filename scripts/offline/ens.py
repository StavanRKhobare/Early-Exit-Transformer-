# Offline: reuse already-computed exit outputs (ensemble), stronger final head, safer threshold rule. CPU only. Costs are cost-table predictions.
import os
import numpy as np, torch, torch.nn.functional as F, json, itertools, pandas as pd, time
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"routed.py")).read().split("# sanity")[0])
FE=ROOT+"/scratch/features/"; HD=ROOT+"/scratch/heads/"
torch.set_num_threads(2)
idx={s:R[s] for s in("val","test")}
_lg={}
def logp(prefix,layer,arch):
    k=(prefix,layer,arch)
    if k in _lg:return _lg[k]
    key=f"{prefix}__{layer}"
    X=np.load(FE+key+".npy",mmap_mode="r")
    P=torch.load(HD+f"{key}__{arch}.pt",map_location="cpu",weights_only=True)["params"]
    temps=np.load(SIG+f"{key}__{arch}.npz")["temps"]
    out={}
    for sp in("val","test"):
        a,b=idx[sp];x=torch.from_numpy(np.asarray(X[a:b]).astype(np.float32));res=[]
        for s in range(5):
            h=x
            if layer!="final":
                h=F.layer_norm(h,(192,),P["0.weight"][s].float(),P["0.bias"][s].float())
            if arch=="lin": l=h@P["1.weight"][s].float().T+P["1.bias"][s].float()
            else:
                z=F.gelu(h@P["1.0.weight"][s].float().T+P["1.0.bias"][s].float())
                l=z@P["1.2.weight"][s].float().T+P["1.2.bias"][s].float()
            res.append(F.softmax(l/float(temps[s]),-1).numpy().astype(np.float32))
        out[sp]=np.stack(res)
    _lg[k]=out;return out
# check against stored predictions
chk=logp("none","final","lin")["test"];print("logit check",(chk.argmax(-1)==sig("none__final__lin")["test_pred"]).mean())
H=lambda p:-(p*np.log(np.clip(p,1e-12,1))).sum(-1)/np.log(10)
def run(g,tau,S,L,fin,w,sp):
    L=None if (L is None or L!=L) else int(L)
    y=Ys[sp];pg=logp("none",g,"mlp192")[sp];pf=logp(ss(S),"final",fin)[sp]
    pl=logp(ss(pre(S,L)),L,"mlp192")[sp] if L else None
    corr=[];f1=[];f2=[];preds=[]
    for s in range(5):
        a=pg[s];ex1=H(a)<tau
        if L:
            b=(w*a+pl[s])/(1+w);ex2=(~ex1)&(H(b)<tau);c=(w*a+w*pl[s]+pf[s])/(1+2*w)
        else:
            b=None;ex2=np.zeros(len(y),bool);c=(w*a+pf[s])/(1+w)
        pr=c.argmax(-1)
        if L:pr=np.where(ex2,b.argmax(-1),pr)
        pr=np.where(ex1,a.argmax(-1),pr);preds.append(pr==y)
        f1.append(ex1.mean());f2.append(ex2.mean())
    return np.stack(preds),np.mean(f1),np.mean(f2)
def cost2(tb,b,g,S,L,fin,f1,f2):
    o,bs=tb["ops"],tb["batches"];c=icost(o["embed"],bs,b)+g*icost(block(o,197),bs,b)+icost(o["head|mlp192"],bs,b)
    N=197;f=1-f1;d=dict(S)
    for l in range(1,13):
        if l>g:c+=icost(block(o,N),bs,b*f)
        if l==L and l>g:
            c+=icost(o["head|mlp192"],bs,b*f);f=1-f1-f2
        if l in d:
            if l>=g:c+=icost(o[f"merge|{N}|{d[l]}"],bs,b*f)
            N=1+mc(N-1,d[l])
    c+=icost(o[f"final|{fin}"],bs,b*f)
    return c/b
scs=[S for S in scheds]
cfg=[]
t0=time.time();store={}
for g in(4,6,8,10):
  for S in scs:
    if not all(m[0]>=g for m in S):continue
    for L in((None,10) if g<10 else (None,)):
      for fin in("lin","mlp192"):
        for w in(0,.25,.5):
          for tau in TAUS:
            c,f1,f2=run(g,tau,S,L,fin,w,"val")
            cfg.append(dict(g=g,S=S,L=L,fin=fin,w=w,tau=tau,val=c.mean(),f1=f1,f2=f2))
  print(g,len(cfg),round(time.time()-t0),flush=True)
df=pd.DataFrame(cfg);df["mg"]=df.S.map(bool)
y=Ys
b0={a:(sig(f"none__final__{a}")["val_pred"]==y["val"]).mean() for a in("lin","mlp192")};b0t={a:(sig(f"none__final__{a}")["test_pred"]==y["test"]).mean() for a in("lin","mlp192")}
print("B0",b0,b0t)
REGS=[("cpu_laptop",1),("cpu_laptop",32),("cuda_laptop",1),("cuda_laptop",32),("cuda_laptop",128)]
for t,b in REGS:df[f"c_{t}_{b}"]=[cost2(T[t],b,r.g,r.S,r.L,r.fin,r.f1,r.f2) for r in df.itertuples()]
df.to_pickle("ens_cfg.pkl")
def b0cost(t,b,fin):return cost2(T[t],b,12,(),None,fin,0,0) if False else None
rows=[]
variants={"Exit only, lin":lambda d:(~d.mg)&(d.fin=="lin")&(d.w==0),
 "Exit+merge, lin":lambda d:d.mg&(d.fin=="lin")&(d.w==0),
 "Exit+merge+ens, lin":lambda d:d.mg&(d.fin=="lin"),
 "Exit+merge, mlp":lambda d:d.mg&(d.fin=="mlp192")&(d.w==0),
 "Exit+merge+ens, mlp":lambda d:d.mg&(d.fin=="mlp192"),
 "Exit only+ens, mlp":lambda d:(~d.mg)&(d.fin=="mlp192")}
for t,b in REGS:
  base=b0c(t,b,"test")
  for eps in(0,.0025,.005,.01):
    for ref in("lin","mlp192"):
      for name,fn in variants.items():
        if ref=="lin" and "mlp" in name and "ens" not in name: pass
        d=df[fn(df)&(df.val>=b0[ref]-eps)]
        if not len(d):rows.append(dict(reg=f"{t}_b{b}",eps=eps*100,ref=ref,var=name));continue
        r=d.loc[d[f"c_{t}_{b}"].idxmin()]
        c,f1,f2=run(r.g,r.tau,r.S,r.L,r.fin,r.w,"test")
        sv=100*(1-cost2(T[t],b,r.g,r.S,r.L,r.fin,f1,f2)/base)
        rows.append(dict(reg=f"{t}_b{b}",eps=eps*100,ref=ref,var=name,saved=sv,dtest=100*(c.mean()-b0t[ref]),cfg=f"g{r.g} {ss(r.S)} L{r.L} {r.fin} w{r.w} t{r.tau}"))
o=pd.DataFrame(rows);o.to_csv("ens_results.csv",index=False)
pd.set_option("display.width",220,"display.max_colwidth",50,"display.max_rows",500)
print(o[(o.reg=="cuda_laptop_b32")].round(2).to_string(index=False))
print(o[(o.reg=="cpu_laptop_b1")&(o.ref=="mlp192")].round(2).to_string(index=False))
print(o[(o.reg=="cuda_laptop_b1")&(o.ref=="mlp192")].round(2).to_string(index=False))

# ---- threshold safety: does a validation-only rule overshoot, and does a paired lower bound fix it? Repeated random splits of val+test, seed 0.
print("\nRISK CONTROL, repeated 4000/10000 splits of the 14000 pooled images")
yall=np.r_[y["val"],y["test"]]
base=(np.r_[sig("none__final__lin")["val_pred"][0],sig("none__final__lin")["test_pred"][0]]==yall)
d=df[df.mg&(df.fin=="lin")&(df.w==0)&(df.val>=b0["lin"]-.03)].sort_values("c_cuda_laptop_32").reset_index(drop=True)
C=[]
for r in d.itertuples():
    a=run(r.g,r.tau,r.S,r.L,r.fin,r.w,"val")[0][0];b=run(r.g,r.tau,r.S,r.L,r.fin,r.w,"test")[0][0];C.append(np.r_[a,b])
C=np.stack(C);print("candidates",len(C))
cost=d["c_cuda_laptop_32"].values;b0cst=b0c("cuda_laptop",32,"test")
rng=np.random.default_rng(0);res=[]
Di=C.astype(np.int8)-base.astype(np.int8)
for eps in(.0025,.005,.01):
  for rule,z in(("naive",0),("LCB z=1.64",1.645),("LCB Bonferroni",None)):
    viol=[];sv=[];gap=[]
    zz=z if z is not None else __import__("statistics").NormalDist().inv_cdf(1-.1/len(C))
    for _ in range(200):
        p=rng.permutation(14000);sel,rep=p[:4000],p[4000:]
        ds=Di[:,sel];m=ds.mean(1);se=ds.std(1)/np.sqrt(4000)
        ok=m-zz*se>=-eps
        if not ok.any():viol.append(0);sv.append(0);gap.append(0);continue
        i=np.where(ok)[0][0]
        g=Di[i,rep].mean();viol.append(g<-eps);sv.append(100*(1-cost[i]/b0cst));gap.append(100*g)
    res.append(dict(eps_pts=eps*100,rule=rule,violation_rate_pct=100*np.mean(viol),mean_saved=np.mean(sv),mean_report_diff=np.mean(gap)))
print(pd.DataFrame(res).round(2).to_string(index=False))
