# Offline test of confidence-guided merging. Read-only on project data. Costs are cost-table predictions, not measurements.
import numpy as np, pandas as pd, json, itertools, os, time
ROOT="/home/stavan-khobare/Desktop/AIML Shi/outputs"
SIG=ROOT+"/scratch/sigs/"; RES=ROOT+"/results/"
os.makedirs(RES+"offline",exist_ok=True); os.chdir(RES+"offline")
Y=np.load(ROOT+"/scratch/features/labels.npy"); R=json.load(open(RES+"ranges.json"))
Ys={s:Y[R[s][0]:R[s][1]] for s in("val","test")}
TAUS=[0.02,0.05,0.08,0.12,0.16,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
def ss(s): return "_".join(f"{l}r{int(r*100)}" for l,r in s) or "none"
_c={}
def sig(k):
    if k not in _c:
        z=np.load(SIG+k+".npz"); _c[k]={x:z[x] for x in z.files if x!="temps"}
    return _c[k]
pre=lambda S,L:tuple(m for m in S if m[0]<L)
def mc(Np,r):
    p=int(min(max(1,int(Np*r)),(Np+1)//2-1)) if r>0 and Np>=4 else 0
    return Np-p
T=json.load(open(RES+"energy_tables.json"))
def icost(tab,batches,b):
    xs=np.log(np.array(batches,float)); ys=np.array([tab[str(x)]["gross"] for x in batches])
    return float(np.interp(np.log(max(b,1.0)),xs,ys)) if b>=1 else b*ys[0]
def block(ops,n):
    k=f"block|{n}"
    if k in ops:return ops[k]
    best=min((int(x.split("|")[1]) for x in ops if x.startswith("block|")),key=lambda x:abs(x-n))
    return ops[f"block|{best}"]
def branch_cost(tb,b,g,S,L,frac,survL):
    # per-batch cost of one group that continues past the gate: frac of batch, survL = frac surviving exit L
    o,bs=tb["ops"],tb["batches"]; N=197; c=0; f=frac; ms=dict(S)
    for l in range(1,13):
        if l>g: c+=icost(block(o,N),bs,b*f)
        if l==L and l>g:
            c+=icost(o[f"head|mlp192"],bs,b*f); f=survL
        if l in ms:
            if l>=g:
                c+=icost(o[f"merge|{N}|{ms[l]}"],bs,b*f)
            N=1+mc(N-1,ms[l])
    c+=icost(o["final|lin"],bs,b*f)
    return c
def pre_cost(tb,b,g):
    o,bs=tb["ops"],tb["batches"]
    return icost(o["embed"],bs,b)+g*icost(block(o,197),bs,b)+icost(o["head|mlp192"],bs,b)
REG=[("cpu_laptop",1),("cpu_laptop",32),("cuda_laptop",1),("cuda_laptop",32),("cuda_laptop",128)]
ap=pd.read_csv(RES+"all_points.csv.gz")
B0=ap[ap.design=="m[none]e[none]f[lin]"].iloc[0]
def b0c(t,b,sp): return B0[("test_" if sp=="test" else "")+f"gross__{t}__b{b}"]
singles=[(l,r) for l in (4,6,8,10) for r in(.25,.5)]
scheds=[()]+[(s,) for s in singles]+[(a,b) for a,b in itertools.product(singles,singles) if a[0]<b[0]]
macs={S:0 for S in scheds}
for S in scheds:
    N=197;m=0;d=dict(S)
    for l in range(1,13):
        m+=N
        if l in d:N=1+mc(N-1,d[l])
    macs[S]=m
def branch(S,L,tau,sp,seed):
    fin=sig(f"{ss(S)}__final__lin")
    corr=fin[sp+"_pred"][seed]==Ys[sp]
    if L is None: return corr,np.zeros(len(corr),bool)
    h=sig(f"{ss(pre(S,L))}__{L}__mlp192")
    ex=h[sp+"_ent"][seed]<tau
    return np.where(ex,h[sp+"_pred"][seed]==Ys[sp],corr),ex
def evalcfg(g,tau,tau2,Sh,Ss,L,sp):
    accs=[];fr=[]
    for s in range(5):
        h=sig(f"none__{g}__mlp192"); e=h[sp+"_ent"][s]; y=Ys[sp]
        ch,xh=branch(Sh,L,tau,sp,s); cs,xs=branch(Ss,L,tau,sp,s)
        ex=e<tau; hard=(~ex)&(e<tau2); soft=(~ex)&(e>=tau2)
        c=np.where(ex,h[sp+"_pred"][s]==y,np.where(hard,ch,cs))
        accs.append(c.mean()); n=len(y)
        fr.append((hard.sum()/n,soft.sum()/n,(hard&~xh).sum()/n,(soft&~xs).sum()/n))
    return np.mean(accs),np.mean(fr,0)
def cost(tb,b,g,Sh,Ss,L,fr):
    c=pre_cost(tb,b,g)
    if fr[0]>0:c+=branch_cost(tb,b,g,Sh,L,fr[0],fr[2])
    if fr[1]>0:c+=branch_cost(tb,b,g,Ss,L,fr[1],fr[3])
    return c/b
# sanity: reproduce a stored static design
g,tau,S,L=4,.16,((6,.5),(8,.25)),10
a,fr=evalcfg(g,tau,9,S,S,L,"test")
row=ap[(ap.design=="m[6r50_8r25]e[4mlp192_10mlp192]f[lin]")&(ap.tau==.16)].iloc[0]
print("sanity acc",a,row.test_acc)
for t,b in REG: print(" sanity cost",t,b,round(cost(T[t],b,g,S,S,L,fr),2),round(row[f"test_gross__{t}__b{b}"],2))
# premise test: where does merging do its damage, by block-6 entropy
y=Ys["test"];rows=[]
HARD=((6,.5),(8,.25))
ents=np.stack([sig("none__6__mlp192")["test_ent"][s] for s in range(5)])
edges=[0,.05,.1,.2,.3,.4,.5,.6,.7,.8,1.01]
for lo,hi in zip(edges[:-1],edges[1:]):
    r=[];n=[];ba=[];ma=[];ga=[]
    for s in range(5):
        m=(ents[s]>=lo)&(ents[s]<hi)
        b=sig("none__final__lin")["test_pred"][s]==y; k=sig(f"{ss(HARD)}__final__lin")["test_pred"][s]==y; gg=sig("none__6__mlp192")["test_pred"][s]==y
        n.append(m.mean());ba.append(b[m].mean());ma.append(k[m].mean());ga.append(gg[m].mean())
    rows.append(dict(ent=f"{lo:.2f}-{hi:.2f}",share_pct=100*np.mean(n),head6_acc=100*np.mean(ga),full_acc=100*np.mean(ba),merged_acc=100*np.mean(ma),lost_pts_of_total=100*np.mean(n)*(np.mean(ba)-np.mean(ma))))
dmg=pd.DataFrame(rows);print(dmg.round(2).to_string(index=False));dmg.to_csv("damage_by_entropy.csv",index=False)
# oracle bound: never hurt anyone merging hurts
for s in [0]:
    b=sig("none__final__lin")["test_pred"][s]==y;k=sig(f"{ss(HARD)}__final__lin")["test_pred"][s]==y
    print("seed0 full acc",b.mean(),"merged",k.mean(),"lost",(b&~k).mean(),"gained",(~b&k).mean())
# search
t0=time.time();recs=[]
for g in (4,6,8):
    opts=[S for S in scheds if all(m[0]>=g for m in S)]
    hards=[S for S in opts if S]
    for tau in TAUS:
      for L in (None,10):
        for Sh in hards:
          softs=[S for S in opts if macs[S]>macs[Sh]]
          for Ss in softs:
            for tau2 in [t for t in TAUS if t>tau]:
                a,fr=evalcfg(g,tau,tau2,Sh,Ss,L,"val")
                if a<B0.val_acc-.0125: continue
                recs.append((g,tau,tau2,Sh,Ss,L,a,fr))
    print("gate",g,len(recs),round(time.time()-t0),flush=True)
print("feasible configs",len(recs))
out=[]
for t,b in REG:
    base=b0c(t,b,"val");tb=T[t]
    costs=np.array([cost(tb,b,r[0],r[3],r[4],r[5],r[7]) for r in recs]);accs=np.array([r[6] for r in recs])
    for eps in(0.0,.0025,.005,.01):
        m=accs>=B0.val_acc-eps
        row={"regime":f"{t} b{b}","eps_pts":eps*100}
        for fam in("exit_only","exit+merge","merge_only"):
            f=ap[(ap.family==fam)&(ap.val_acc>=B0.val_acc-eps)]
            if len(f):
                x=f.loc[f[f"gross__{t}__b{b}"].idxmin()]
                row[fam]=100*(1-x[f"test_gross__{t}__b{b}"]/b0c(t,b,"test")); row[fam+"_d"]=100*(x.test_acc-B0.test_acc); row[fam+"_id"]=x.design+f" t{x.tau}"
        if m.any():
            i=np.where(m)[0][costs[m].argmin()];r=recs[i]
            a,fr=evalcfg(r[0],r[1],r[2],r[3],r[4],r[5],"test")
            row["routed"]=100*(1-cost(tb,b,r[0],r[3],r[4],r[5],fr)/b0c(t,b,"test"));row["routed_d"]=100*(a-B0.test_acc)
            row["routed_id"]=f"g{r[0]} t{r[1]} t2{r[2]} H{ss(r[3])} S{ss(r[4])} L{r[5]} hardfrac{fr[0]:.2f} softfrac{fr[1]:.2f}"
        out.append(row)
o=pd.DataFrame(out);o.to_csv("routed_vs_static.csv",index=False)
pd.set_option("display.width",250,"display.max_colwidth",70)
print(o[["regime","eps_pts","exit_only","exit_only_d","merge_only","merge_only_d","exit+merge","exit+merge_d","routed","routed_d"]].round(2).to_string(index=False))
print(o[["regime","eps_pts","exit+merge_id","routed_id"]].to_string(index=False))
