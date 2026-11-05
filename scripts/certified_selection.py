# Block 3: choose a merge schedule, exit pair and thresholds per device, with a limit on accuracy loss against dedicated B0.
# Energy here is PREDICTED from cost tables. Nothing in this script is a measured run.
#
# Data use:
#   train, cal           exit heads and temperature (extract_heads.py)
#   val                  picks each model's final-head variant only
#   unused 25,000        selection: 10,000 build the Pareto path, 15,000 test it
#   test 10,000          reported only with --report-test, after everything here is frozen
#
# Limit: with probability at least 1-delta, true accuracy of the chosen design >= true accuracy of B0 - eps.
# Rules marked exact hold for any distribution over images drawn like the selection data.
# z rules rest on a normal approximation. Naive has no guarantee.
import os, sys, json, math, time, numpy as np, pandas as pd
from statistics import NormalDist
ROOT=os.path.abspath("outputs");IN=ROOT+"/results/shared_backbone/";OUT=ROOT+"/results/certified/";os.makedirs(OUT,exist_ok=True)
REPORT="--report-test" in sys.argv;REPEATS=int(sys.argv[sys.argv.index("--repeats")+1]) if "--repeats" in sys.argv else 100
pd.set_option("display.width",220,"display.max_rows",500,"display.max_columns",40)
def log(*a):
    s=time.strftime("%H:%M:%S ")+" ".join(str(x) for x in a);print(s,flush=True);open(OUT+"progress.log","a").write(s+"\n")
ALL={"none":{}, "8r50":{8:.5}, "6r50":{6:.5}, "6r50_8r25":{6:.5,8:.25}, "4r25_8r25":{4:.25,8:.25}, "4r50":{4:.5}}
pname=lambda M,l:"_".join(f"{k}r{int(v*100)}" for k,v in sorted(M.items()) if k<l) or "none"
PAIRS=[(4,6),(4,8),(4,10),(6,8),(6,10),(8,10)];SINGLES=[4,6,8,10]
TAU=[.005,.01,.02,.03,.05,.08,.12,.16,.2,.25,.3,.4]
SOURCES={"shared":list(ALL),"merged":["6r50_8r25"],"b0":["none"]}
FAMILIES={"shared":("shared",list(ALL)),"shared_fixed":("shared",["6r50_8r25"]),"shared_nomerge":("shared",["none"]),"merged":("merged",["6r50_8r25"]),"b0":("b0",["none"])}
REG=[("cuda_laptop",32),("cpu_laptop",32),("cpu_laptop",1),("cuda_laptop",1)]
L=np.load(IN+"labels.npz");Y={s:L[s] for s in("val","test","unused")}
PR={m:dict(np.load(IN+f"probs_{m}.npz")) for m in SOURCES if os.path.exists(IN+f"probs_{m}.npz")}
assert "b0" in PR or not REPORT
H=lambda p:-(p*np.log(np.clip(p,1e-12,1))).sum(-1)/np.log(10)
# ---------- baseline and final-head variant, chosen on validation
def final_pred(m,n,variant,split,seed):
    return PR[m][f"final|{n}|fthead|{split}"].argmax(-1) if variant=="fthead" else PR[m][f"final|{n}|{split}"][seed].argmax(-1)
FINAL={}
for m in PR:
    r={v:np.mean([(final_pred(m,n,v,"val",0)==Y["val"]).mean() for n in SOURCES[m]]) for v in("fthead","lin")};FINAL[m]=max(r,key=r.get);log("final head",m,{k:round(100*v,2) for k,v in r.items()},"->",FINAL[m])
BASEM="b0" if "b0" in PR else "shared";BASE={s:final_pred(BASEM,"none","fthead",s,0)==Y[s] for s in Y}
log("baseline",BASEM,{s:round(100*BASE[s].mean(),2) for s in Y})
# ---------- candidates
def build(m,seed):
    meta=[];cor={s:[] for s in Y if s!="val" or True};st={s:[] for s in Y};ent={}
    def E(key,s):
        k=(key,s)
        if k not in ent:p=PR[m][f"{key}|{s}"][seed];ent[k]=(H(p),p.argmax(-1))
        return ent[k]
    for n in SOURCES[m]:
        S=ALL[n];fin={s:final_pred(m,n,FINAL[m],s,seed) for s in Y}
        opts=[(None,None,None,None)]+[(g,None,t,None) for g in SINGLES for t in TAU]+[(g,l,t,u) for g,l in PAIRS for t in TAU for u in TAU]
        for g,l,t,u in opts:
            meta.append(dict(model=m,sched=n,g=g,L=l,t1=t,t2=u))
            for s in Y:
                pr=fin[s].copy();stage=np.zeros(len(pr),np.uint8)
                if l:
                    h2,p2=E(f"{l}|{pname(S,l)}",s);m2=h2<u;pr=np.where(m2,p2,pr);stage[m2]=2
                if g:
                    h1,p1=E(f"{g}|{pname(S,g)}",s);m1=h1<t;pr=np.where(m1,p1,pr);stage[m1]=1
                cor[s].append(pr==Y[s]);st[s].append(stage)
    return pd.DataFrame(meta),{s:np.stack(v) for s,v in cor.items()},{s:np.stack(v) for s,v in st.items()}
# ---------- cost model, vectorised over candidates
T_=json.load(open(ROOT+"/results/energy_tables.json"))
def mp(N,r):return 0 if r<=0 or N<4 else int(min(max(1,int(N*r)),(N+1)//2-1))
def icost(tab,bs,x):
    ys=np.array([tab[str(v)]["gross"] for v in bs]);x=np.asarray(x,float)
    return np.where(x>=1,np.interp(np.log(np.maximum(x,1.)),np.log(np.array(bs,float)),ys),x*ys[0])
def block(o,n):
    k=f"block|{n}"
    if k in o:return o[k]
    best=min((int(x.split("|")[1]) for x in o if x.startswith("block|")),key=lambda x:abs(x-n));return o[f"block|{best}"]
def cost_energy(tb,b,S,g,Lx,f1,f2):
    o,bs=tb["ops"],tb["batches"];N=197;f=np.ones_like(f1);c=icost(o["embed"],bs,float(b))*np.ones_like(f1)
    for l in range(1,13):
        c=c+icost(block(o,N),bs,b*f)
        if l==g:c=c+icost(o["head|mlp192"],bs,b*f);f=1-f1
        if l==Lx:c=c+icost(o["head|mlp192"],bs,b*f);f=1-f1-f2
        if l in S:c=c+icost(o[f"merge|{N}|{S[l]}"],bs,b*f);N=1+(N-1-mp(N-1,S[l]))
    return (c+icost(o["final|lin"],bs,b*f))/b
D=192
def cost_mac(S,g,Lx,f1,f2):
    # Counts only: embedding, attention (qkv, proj, q.k), MLP, exit head, pairwise merge similarity, final head.
    N=197;f=np.ones_like(f1);c=196*(16*16*3*D)*np.ones_like(f1)
    for l in range(1,13):
        c=c+f*(12*N*D*D+N*N*D)
        if l==g:c=c+f*(D*D+D*10);f=1-f1
        if l==Lx:c=c+f*(D*D+D*10);f=1-f1-f2
        if l in S:c=c+f*(((N-1)//2)**2*D);N=1+(N-1-mp(N-1,S[l]))
    return c+f*D*10
class Family:
    def __init__(s,meta,cor,st,keep):
        idx=np.where(meta.sched.isin(keep))[0];s.meta=meta.iloc[idx].reset_index(drop=True);s.cor={k:v[idx] for k,v in cor.items()};s.st={k:v[idx] for k,v in st.items()}
        key=[f"{a}|{b}|{c}" for a,b,c in zip(s.meta.sched,s.meta.g.fillna(-1).astype(int),s.meta.L.fillna(-1).astype(int))];s.groups=[(s.meta.iloc[i[0]],i) for i in pd.Series(range(len(key))).groupby(key).indices.values()];assert sum(len(i) for _,i in s.groups)==len(s.meta)
    def costs(s,split,sel,reg):
        st=s.st[split] if sel is None else s.st[split][:,sel];f1=(st==1).mean(1);f2=(st==2).mean(1);c=np.zeros(len(f1))
        for r,i in s.groups:
            g=None if pd.isna(r.g) else int(r.g);Lx=None if pd.isna(r.L) else int(r.L);S=ALL[r.sched]
            c[i]=cost_mac(S,g,Lx,f1[i],f2[i]) if reg=="mac" else cost_energy(T_[reg[0]],reg[1],S,g,Lx,f1[i],f2[i])
        return c
    def counts(s,split,sel):
        c=s.cor[split] if sel is None else s.cor[split][:,sel];b=BASE[split] if sel is None else BASE[split][sel]
        return (c&~b).sum(1),(b&~c).sum(1),c.shape[1]      # candidate right and B0 wrong; B0 right and candidate wrong; n
# ---------- lower confidence bounds on (candidate accuracy - B0 accuracy)
Z=lambda a:NormalDist().inv_cdf(1-a)
def lcb_z(g,w,n,dl):
    d=(g-w)/n;v=(g+w)/n-d*d;return d-Z(dl)*np.sqrt(v/n)
def lcb_eb(g,w,n,dl):
    d=(g-w)/n;v=((g+w)/n-d*d)*n/(n-1);lg=math.log(2/dl);return d-np.sqrt(2*v*lg/n)-14*lg/(3*(n-1))   # Maurer and Pontil 2009, range 2
_LF={}
def lf(n):
    if n not in _LF:_LF[n]=np.concatenate([[0],np.cumsum(np.log(np.arange(1,n+1)))])
    return _LF[n]
def binom_cdf(k,n,p):
    if k<0:return 0.
    if k>=n:return 1.
    F=lf(n);i=np.arange(k+1);return float(np.exp(F[n]-F[i]-F[n-i]+i*math.log(p)+(n-i)*math.log1p(-p)).sum())
_CP={}
def cp(k,n,a,upper):
    key=(int(k),n,a,upper)
    if key in _CP:return _CP[key]
    k=int(k);lo,hi=1e-9,1-1e-9
    if upper:
        if k>=n:return 1.
        for _ in range(45):
            m=(lo+hi)/2
            if binom_cdf(k,n,m)>a:lo=m
            else:hi=m
        v=hi
    elif k==0:v=0.
    else:
        for _ in range(45):
            m=(lo+hi)/2
            if 1-binom_cdf(k-1,n,m)<a:lo=m
            else:hi=m
        v=lo
    _CP[key]=v;return v
def lcb_cp(g,w,n,dl):
    # Exact: Clopper-Pearson on both discordant rates, split by a union bound. Tighter than empirical Bernstein for rare disagreement.
    a=dl/2;return np.array([cp(x,n,a,False)-cp(y,n,a,True) for x,y in zip(np.atleast_1d(g),np.atleast_1d(w))])
LCB={"z":lcb_z,"eb":lcb_eb,"cp":lcb_cp}
# ---------- selection rules: each returns a candidate index or None
def pareto_path(cost,d):
    path=[];best=-9.
    for i in np.argsort(cost,kind="stable"):
        if d[i]>best:best=d[i];path.append(i)
    return path[::-1]    # most expensive and most accurate first
def fst(path,g,w,n,kind,eps,dl):
    k=None
    for i in path:
        if LCB[kind](g[i],w[i],n,dl).item()>=-eps:k=i
        else:break
    return k
def cheapest(ok,cost):
    i=np.where(ok)[0];return i[np.argmin(cost[i])] if len(i) else None
def selections(fam,split,sel_all,bsel,csel,reg,eps,dl,cache):
    # sel_all is what one-shot rules see (= bsel + csel). All rules see the same images in total.
    if "all" not in cache:
        cache["all"]=fam.counts(split,sel_all);cache["bd"]=fam.counts(split,bsel);cache["ce"]=fam.counts(split,csel);cache["mac"]=fam.costs(split,bsel,"mac")
    ga,wa,na=cache["all"];gb,wb,nb=cache["bd"];gc,wc,nc=cache["ce"]
    ck=("c",reg)
    if ck not in cache:cache[ck]=(fam.costs(split,sel_all,reg),fam.costs(split,bsel,reg))
    ca,cb=cache[ck];C=len(ga);db=(gb-wb)/nb;out={}
    out["naive"]=cheapest((ga-wa)/na>=-eps,ca)
    out["z margin"]=cheapest(lcb_z(ga,wa,na,dl)>=-eps,ca)
    out["z Bonferroni"]=cheapest(lcb_z(ga,wa,na,dl/C)>=-eps,ca)
    out["exact Bonferroni (CP)"]=cheapest(lcb_cp(ga,wa,na,dl/C)>=-eps,ca)
    pk=("p",reg)
    if pk not in cache:cache[pk]=pareto_path(cb,db)
    if "pm" not in cache:cache["pm"]=pareto_path(cache["mac"],db)
    out["Pareto FST z"]=fst(cache[pk],gc,wc,nc,"z",eps,dl)
    out["Pareto FST exact (CP)"]=fst(cache[pk],gc,wc,nc,"cp",eps,dl)
    out["Pareto FST exact (EB)"]=fst(cache[pk],gc,wc,nc,"eb",eps,dl)
    out["Pareto FST exact (CP), MAC order"]=fst(cache["pm"],gc,wc,nc,"cp",eps,dl)
    return out
def design(r):
    if pd.isna(r.g):return f"{r.sched} no exit"
    return f"{r.sched} g{int(r.g)}@{r.t1}"+("" if pd.isna(r.L) else f" L{int(r.L)}@{r.t2}")
base_cost=lambda reg:float(cost_energy(T_[reg[0]],reg[1],{},None,None,np.zeros(1),np.zeros(1))[0])
# ---------- run
FAM={};t0=time.time()
for seed in(0,1,2):
    data={m:build(m,seed) for m in PR}
    FAM[seed]={f:Family(*data[m],keep) for f,(m,keep) in FAMILIES.items() if m in PR};log("seed",seed,"candidates",{f:len(v.meta) for f,v in FAM[seed].items()},f"{time.time()-t0:.0f}s")
    if seed:del data
EPS=(.0025,.005,.01)
# Coverage: unused images only, resplit many times. Test stays untouched. Costs use the same table, reported on held-out images.
rows=[];rng=np.random.default_rng(11);f0=FAM[0]
if REPEATS:
    for rep in range(REPEATS):
        p=rng.permutation(25000);bs,cs,rs=p[:8000],p[8000:20000],p[20000:];al=np.r_[bs,cs]
        for fn,fam in f0.items():
            cache={};gr,wr,nr=fam.counts("unused",rs);cr={};gp,wp,np_=fam.counts("unused",None)
            for reg in REG:
                cr[reg]=fam.costs("unused",rs,reg);bc=base_cost(reg)
                for eps in EPS:
                    for rule,i in selections(fam,"unused",al,bs,cs,reg,eps,.1,cache).items():
                        rows.append(dict(rep=rep,family=fn,reg=f"{reg[0]}_b{reg[1]}",eps=eps*100,rule=rule,found=i is not None,saved=0. if i is None else 100*(1-cr[reg][i]/bc),diff=0. if i is None else 100*(gr[i]-wr[i])/nr,over=False if i is None else (gr[i]-wr[i])/nr<-eps,over_pool=False if i is None else (gp[i]-wp[i])/np_<-eps))
        if rep%10==0:log("repeat",rep,f"{time.time()-t0:.0f}s")
    R=pd.DataFrame(rows);R.to_csv(OUT+"repeats_raw.csv.gz",index=False)
    S=R.groupby(["family","reg","eps","rule"]).agg(saved=("saved","mean"),saved_sd=("saved","std"),diff=("diff","mean"),over_pct=("over",lambda x:100*x.mean()),over_pool_pct=("over_pool",lambda x:100*x.mean()),found_pct=("found",lambda x:100*x.mean())).reset_index();S.to_csv(OUT+"repeats.csv",index=False)
    print(S[S.family=="shared"].round(2).to_string(index=False))
# Main selection: all 25,000, one fixed split, every exit-head seed.
perm=np.random.default_rng(2026).permutation(25000);bs,cs=perm[:10000],perm[10000:];al=perm;rows=[];chosen={}
for seed in(0,1,2):
    for fn,fam in FAM[seed].items():
        cache={};gs,ws,ns=fam.counts("unused",al)
        if REPORT:gt,wt,nt=fam.counts("test",None)
        for reg in REG:
            bc=base_cost(reg);cost_sel=fam.costs("unused",al,reg);cost_test=fam.costs("test",None,reg) if REPORT else None
            for eps in EPS:
                for dl in(.1,.05):
                    for rule,i in selections(fam,"unused",al,bs,cs,reg,eps,dl,cache).items():
                        row=dict(seed=seed,family=fn,reg=f"{reg[0]}_b{reg[1]}",eps=eps*100,delta=dl,rule=rule,design="none found" if i is None else design(fam.meta.iloc[i]))
                        if i is not None:
                            row["saved_select"]=100*(1-cost_sel[i]/bc);row["diff_select"]=100*(gs[i]-ws[i])/ns
                            if REPORT:
                                d=(gt[i]-wt[i])/nt;row.update(saved_test=100*(1-cost_test[i]/bc),diff_test=100*d,diff_test_se=100*math.sqrt(((gt[i]+wt[i])/nt-d*d)/nt),within=bool(d>=-eps))
                            if seed==0 and dl==.1:chosen[(fn,row["reg"],eps,rule)]=int(i)
                        rows.append(row)
    log("main seed",seed,f"{time.time()-t0:.0f}s")
M=pd.DataFrame(rows);M.to_csv(OUT+("selection_test.csv" if REPORT else "selection.csv"),index=False)
cols=[c for c in M if c not in("seed","delta","family")]
print(M[(M.seed==0)&(M.delta==.1)&(M.family=="shared")][cols].round(2).to_string(index=False))
# Does each device need its own design? Run every device's chosen design on every other device.
rule="Pareto FST exact (CP)";xrows=[];split="test" if REPORT else "unused";fam=FAM[0].get("shared")
if fam is not None:
    for eps in EPS:
        for src in REG:
            i=chosen.get(("shared",f"{src[0]}_b{src[1]}",eps,rule))
            if i is None:continue
            for dst in REG:
                own=chosen.get(("shared",f"{dst[0]}_b{dst[1]}",eps,rule));bc=base_cost(dst);c=fam.costs(split,None,dst)
                xrows.append(dict(eps=eps*100,chosen_for=f"{src[0]}_b{src[1]}",run_on=f"{dst[0]}_b{dst[1]}",design=design(fam.meta.iloc[i]),saved=100*(1-c[i]/bc),own_saved=None if own is None else 100*(1-c[own]/bc)))
    X=pd.DataFrame(xrows);X.to_csv(OUT+("cross_device_test.csv" if REPORT else "cross_device.csv"),index=False);print(X.round(2).to_string(index=False))
# Budget view: spend at most this share of B0 energy; what accuracy loss can be certified?
brows=[]
if fam is not None:
    gc,wc,nc=fam.counts("unused",cs);gb,wb,nb=fam.counts("unused",bs);db=(gb-wb)/nb
    for reg in REG:
        cb=fam.costs("unused",bs,reg);bc=base_cost(reg)
        for bud in(.5,.6,.7,.8,.9):
            ok=np.where(cb<=bud*bc)[0]
            if not len(ok):brows.append(dict(reg=f"{reg[0]}_b{reg[1]}",budget=bud*100,design="none fits"));continue
            i=ok[np.argmax(db[ok])];brows.append(dict(reg=f"{reg[0]}_b{reg[1]}",budget=bud*100,design=design(fam.meta.iloc[i]),spend_held_out=100*fam.costs("unused",cs,reg)[i]/bc,certified_diff_exact=100*lcb_cp(gc[i],wc[i],nc,.1).item(),observed_diff=100*(gc[i]-wc[i])/nc))
    B=pd.DataFrame(brows);B.to_csv(OUT+"budget.csv",index=False);print(B.round(2).to_string(index=False))
json.dump({f"{k[0]}|{k[1]}|{k[2]}|{k[3]}":design(FAM[0][k[0]].meta.iloc[v]) for k,v in chosen.items()},open(OUT+"chosen_seed0.json","w"),indent=1)
json.dump(FINAL,open(OUT+"final_variant.json","w"))
log("BLOCK3 DONE" if REPORT else "BLOCK3 DONE, test set not used",f"{time.time()-t0:.0f}s")
