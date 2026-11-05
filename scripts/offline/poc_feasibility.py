# CPU-only feasibility checks for the certified-selection plan. Cost-table predictions. Part B uses validation only.
# A: on the frozen seed-0 pool, does Pareto-path fixed-sequence testing keep savings while holding the limit?
# B: with fine-tuned patch-average heads, how large is the oracle gap, and does a cross-fitted gate beat entropy?
import os, json, numpy as np, pandas as pd, torch, torch.nn.functional as F
_src=open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"ens.py")).read().split("scs=[S for S")[0]
_file=__file__
exec(_src)
torch.set_num_threads(2)
pd.set_option("display.width",200,"display.max_rows",200)
df=pd.read_pickle("ens_cfg.pkl")
y=Ys;b0v=(sig("none__final__lin")["val_pred"]==y["val"]).mean()
yall=np.r_[y["val"],y["test"]];base=np.r_[sig("none__final__lin")["val_pred"][0],sig("none__final__lin")["test_pred"][0]]==yall
d=df[df.mg&(df.fin=="lin")&(df.w==0)&(df.val>=b0v-.03)].sort_values("c_cuda_laptop_32",ascending=False).reset_index(drop=True)
C=np.stack([np.r_[run(r.g,r.tau,r.S,r.L,r.fin,r.w,"val")[0][0],run(r.g,r.tau,r.S,r.L,r.fin,r.w,"test")[0][0]] for r in d.itertuples()])
Di=(C.astype(np.int8)-base.astype(np.int8)).astype(np.float32);cost=d["c_cuda_laptop_32"].values;b0=b0c("cuda_laptop",32,"test");true=Di.mean(1)
print("candidates",len(C),"most expensive first; best possible saved %",round(100*(1-cost.min()/b0),1))
from statistics import NormalDist
Z=lambda a:NormalDist().inv_cdf(1-a)
DELTA=.1
def lcb(x,kind,dl=DELTA):
    m=x.mean(-1);n=x.shape[-1];s=x.std(-1,ddof=1)
    if kind=="z":return m-Z(dl)*s/np.sqrt(n)
    L=np.log(2/dl)                     # empirical Bernstein (Maurer and Pontil 2009), range 2 because a paired difference lies in [-1,1]
    return m-(s*np.sqrt(2*L/n)+2*7*L/(3*(n-1)))
def pick_fst(sel,eps,kind):
    h=len(sel)//2;A,B=sel[:h],sel[h:];mA=Di[:,A].mean(1)
    # Pareto path built on A only, most expensive first, then tested in that fixed order on B.
    path=[];best=-9
    for i in range(len(cost)-1,-1,-1):
        if mA[i]>best:best=mA[i];path.append(i)
    path=[i for i in path if mA[i]>=-eps-.003][::1]        # cheapest last; candidates far beyond the limit are left off
    path=sorted(path,key=lambda i:-cost[i])
    k=None
    for i in path:
        if lcb(Di[i,B],kind)>=-eps:k=i
        else:break
    return k
rng=np.random.default_rng(3);rows=[]
for eps in(.0025,.005,.01):
    for n in(4000,10000):
        out={}
        for _ in range(100):
            p=rng.permutation(14000);rep,sel=p[:4000],p[4000:4000+n];ds=Di[:,sel]
            m=ds.mean(1);picks={}
            ok=np.where(m>=-eps)[0];picks["naive"]=ok[-1] if len(ok) else None
            ok=np.where(lcb(ds,"z")>=-eps)[0];picks["z margin"]=ok[-1] if len(ok) else None
            ok=np.where(lcb(ds,"z",DELTA/len(C))>=-eps)[0];picks["z Bonferroni"]=ok[-1] if len(ok) else None
            ok=np.where(lcb(ds,"eb",DELTA/len(C))>=-eps)[0];picks["EB Bonferroni"]=ok[-1] if len(ok) else None
            picks["Pareto FST z"]=pick_fst(sel,eps,"z");picks["Pareto FST EB"]=pick_fst(sel,eps,"eb")
            for k,i in picks.items():
                a=out.setdefault(k,[])
                a.append((0,0,0,0) if i is None else (Di[i,rep].mean()<-eps,true[i]<-eps,100*(1-cost[i]/b0),100*Di[i,rep].mean()))
        for k,v in out.items():
            v=np.array(v,float);rows.append(dict(limit=eps*100,n=n,rule=k,over_report=100*v[:,0].mean(),over_pool=100*v[:,1].mean(),saved=v[:,2].mean(),gap=v[:,3].mean()))
A=pd.DataFrame(rows);A.to_csv("poc_certified.csv",index=False);print(A.round(2).to_string(index=False))

# ---- Part B ----
ROOTR=ROOT+"/results/";d_=torch.load(ROOT+"/scratch/cache/cifar10_uint8.pt",weights_only=True);sp=json.load(open(ROOTR+"splits.json"));yv=d_["Ytr"][sp["val"]].numpy()
print("unused CIFAR-10 train images:",50000-len(set(sp["train"])|set(sp["cal"])|set(sp["val"])))
Pp=dict(np.load(ROOTR+"exit_head_screen/probs_merged.npz"));Fc=dict(np.load(ROOT+"/scratch/ft_pilot/probs_merged.npz"))["final|lin|val"];Bv=dict(np.load(ROOT+"/scratch/ft_pilot/probs_b0.npz"))["final|lin|val"]
print("saved fine-tuned arrays, pool heads:",sorted(Pp)[:3],"... CLS heads have test:", "final|lin|test" in np.load(ROOT+"/scratch/ft_pilot/probs_merged.npz").files, "pool heads have test:", any("test" in k for k in Pp))
M={6:.5,8:.25}
def icost2(tab,bs,b):
    ys=np.array([tab[str(x)]["gross"] for x in bs]);return float(np.interp(np.log(max(b,1.)),np.log(np.array(bs,float)),ys)) if b>=1 else b*ys[0]
def cost3(tb,b,g,Lx,f1,f2):
    o,bs=tb["ops"],tb["batches"];N=197;f=1.;c=icost2(o["embed"],bs,b)
    for l in range(1,13):
        c+=icost2(block(o,N),bs,b*f)
        if l==g:c+=icost2(o["head|mlp192"],bs,b*f);f=1-f1
        if l==Lx:c+=icost2(o["head|mlp192"],bs,b*f);f=1-f1-f2
        if l in M:c+=icost2(o[f"merge|{N}|{M[l]}"],bs,b*f);N=1+mc(N-1,M[l])
    return (c+icost2(o["final|lin"],bs,b*f))/b
REG=[("cuda_laptop",32),("cpu_laptop",32),("cpu_laptop",1)]
base_c={r:cost3(T[r[0]],r[1],None,None,0,0) for r in REG}
for g,Lx in((4,8),(4,10),(6,10)):
    a,b,c=Pp[f"{g}|pool"],Pp[f"{Lx}|pool"],Fc
    e1=a.argmax(-1)==c.argmax(-1);e2=(~e1)&(b.argmax(-1)==c.argmax(-1))
    print(f"oracle pool g{g} L{Lx}: leave {e1.mean():.2f}/{e2.mean():.2f}, val gap to B0 {100*((c.argmax(-1)==yv).mean()-(Bv.argmax(-1)==yv).mean()):.2f}",{f"{r[0]}_b{r[1]}":round(100*(1-cost3(T[r[0]],r[1],g,Lx,e1.mean(),e2.mean())/base_c[r]),1) for r in REG})
def auroc(s,t):
    o=np.argsort(s);r=np.empty(len(s));r[o]=np.arange(1,len(s)+1);n1=t.sum();n0=len(t)-n1;return (r[t].sum()-n1*(n1+1)/2)/(n0*n1)
def feats(p,q=None):
    ss=np.sort(p,-1);f=[H(p)[:,None],ss[:,-1:],(ss[:,-1]-ss[:,-2])[:,None],p]
    if q is not None:f+=[H(q)[:,None],(H(p)-H(q))[:,None],(p.argmax(-1)==q.argmax(-1))[:,None].astype(np.float32),(p*q).sum(-1,keepdims=True)]
    return np.concatenate(f,1).astype(np.float32)
def fit_g(X,t,Xt):
    torch.manual_seed(0);mu,sd=X.mean(0),X.std(0)+1e-6;Z_=torch.tensor((X-mu)/sd);Zt=torch.tensor((Xt-mu)/sd);tt=torch.tensor(t.astype(np.float32))
    m=torch.nn.Sequential(torch.nn.Linear(X.shape[1],32),torch.nn.GELU(),torch.nn.Linear(32,1));o=torch.optim.AdamW(m.parameters(),lr=3e-3,weight_decay=1e-1)
    for _ in range(300):
        l=F.binary_cross_entropy_with_logits(m(Z_)[:,0],tt);o.zero_grad();l.backward();o.step()
    return torch.sigmoid(m(Zt)[:,0]).detach().numpy()
K=np.random.default_rng(0).permutation(4000)%5;rows=[]
for s in range(3):
    a,b,c=Pp["4|pool"][s],Pp["8|pool"][s],Fc[s];fin=c.argmax(-1)
    for name,X,pr in(("block 4",feats(a),a),("block 8",feats(b,a),b)):
        t=pr.argmax(-1)==fin;oof=np.zeros(4000)
        for k in range(5):
            oof[K==k]=fit_g(X[K!=k],t[K!=k],X[K==k])
        rows.append(dict(seed=s,head=name,agree=t.mean(),auroc_entropy=auroc(-H(pr),t),auroc_gate=auroc(oof,t),auroc_correct_entropy=auroc(-H(pr),pr.argmax(-1)==yv),auroc_correct_gate=auroc(oof,pr.argmax(-1)==yv)))
    # savings: out-of-fold gate scores used directly, thresholds chosen on one half, scored on the other
print(pd.DataFrame(rows).groupby("head").mean(numeric_only=True).drop(columns="seed").round(4).to_string())
pd.DataFrame(rows).to_csv("poc_gate_auroc.csv",index=False)
