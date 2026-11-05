# Exit signal test on the fine-tuned merged model (6r50_8r25, gates at 6 and 10). Compares entropy, a small learned gate, and an oracle bound.
# Gate target: "this head agrees with the final head". Trained on val fold A, thresholds chosen on val fold B, reported on test. Cost-table predictions only. CPU.
import numpy as np, json, torch, torch.nn.functional as F, pandas as pd, os
ROOT=os.path.abspath("outputs"); CK=ROOT+"/scratch/ft_pilot/"; OUT=ROOT+"/results/finetune_pilot/"
d=torch.load(ROOT+"/scratch/cache/cifar10_uint8.pt",weights_only=True); sp=json.load(open(ROOT+"/results/splits.json"))
yv=d["Ytr"][sp["val"]].numpy(); yt=d["Yte"][:10000].numpy()
T_=json.load(open(ROOT+"/results/energy_tables.json"))
def mp(N,r): return 0 if r<=0 or N<4 else int(min(max(1,int(N*r)),(N+1)//2-1))
def icost(tab,bs,b):
    ys=np.array([tab[str(x)]["gross"] for x in bs]); return float(np.interp(np.log(max(b,1.)),np.log(np.array(bs,float)),ys)) if b>=1 else b*ys[0]
def block(o,n):
    k=f"block|{n}"
    if k in o:return o[k]
    best=min((int(x.split("|")[1]) for x in o if x.startswith("block|")),key=lambda x:abs(x-n));return o[f"block|{best}"]
def cost(tb,b,g,S,L,f1,f2):
    o,bs=tb["ops"],tb["batches"];N=197;f=1.;c=icost(o["embed"],bs,b)
    for l in range(1,13):
        c+=icost(block(o,N),bs,b*f)
        if l==g:c+=icost(o["head|mlp192"],bs,b*f);f=1-f1
        if l==L:c+=icost(o["head|mlp192"],bs,b*f);f=1-f1-f2
        if l in S:c+=icost(o[f"merge|{N}|{S[l]}"],bs,b*f);N=1+(N-1-mp(N-1,S[l]))
    return (c+icost(o["final|lin"],bs,b*f))/b
S={6:.5,8:.25}; REG=[("cuda_laptop",32),("cpu_laptop",32),("cpu_laptop",1),("cuda_laptop",1)]
Pm=dict(np.load(CK+"probs_merged.npz")); Pb=dict(np.load(CK+"probs_b0.npz"))
H=lambda p:-(p*np.log(np.clip(p,1e-12,1))).sum(-1)/np.log(10)
def feats(p,prev=None):
    q=np.sort(p,-1); f=[p,H(p)[...,None],q[...,-1:],(q[...,-1]-q[...,-2])[...,None]]
    if prev is not None: f+=[prev,H(prev)[...,None],(prev.argmax(-1)==p.argmax(-1))[...,None].astype(np.float32)]
    return torch.tensor(np.concatenate(f,-1),dtype=torch.float32)
def fit(X,t,seed=0):
    torch.manual_seed(seed);m=torch.nn.Sequential(torch.nn.Linear(X.shape[1],32),torch.nn.GELU(),torch.nn.Linear(32,1));o=torch.optim.AdamW(m.parameters(),lr=3e-3,weight_decay=1e-2)
    mu,sd=X.mean(0),X.std(0)+1e-6;Z=(X-mu)/sd
    for _ in range(600):
        l=F.binary_cross_entropy_with_logits(m(Z)[:,0],t);o.zero_grad();l.backward();o.step()
    return lambda A:torch.sigmoid(m((A-mu)/sd)[:,0]).detach().numpy()
rows=[];rng=np.random.default_rng(0);perm=rng.permutation(4000);A,Bf=perm[:2000],perm[2000:]
QS=[.5,.6,.7,.8,.85,.9,.93,.95,.97,.98,.99,.995,.999]
ETAU=[.02,.05,.08,.12,.16,.2,.25,.3,.4,.5,.6,.7]
for k in range(3):
    a,b,c={},{},{}
    for s in("val","test"):a[s]=Pm[f"6|mlp192|{s}"][k];b[s]=Pm[f"10|mlp192|{s}"][k];c[s]=Pm[f"final|lin|{s}"][k]
    Y={"val":yv,"test":yt}; base={s:(Pb[f"final|lin|{s}"][k].argmax(-1)==Y[s]).mean() for s in Y}
    # oracle: exit at the first head that already agrees with the final model
    for s in("test",):
        e1=a[s].argmax(-1)==c[s].argmax(-1);e2=(~e1)&(b[s].argmax(-1)==c[s].argmax(-1))
        rows.append(dict(seed=k,rule="oracle agree",f1=e1.mean(),f2=e2.mean(),dtest=100*((c[s].argmax(-1)==Y[s]).mean()-base[s]),eps=np.nan,**{f"{t}_{bb}":100*(1-cost(T_[t],bb,6,S,10,e1.mean(),e2.mean())/cost(T_[t],bb,None,{},None,0,0)) for t,bb in REG}))
    g1=fit(feats(a["val"])[A],torch.tensor((a["val"][A].argmax(-1)==c["val"][A].argmax(-1)).astype(np.float32)))
    g2=fit(feats(b["val"],a["val"])[A],torch.tensor((b["val"][A].argmax(-1)==c["val"][A].argmax(-1)).astype(np.float32)))
    sc={s:(g1(feats(a[s])),g2(feats(b[s],a[s]))) for s in Y}
    def out(r1,r2,s):
        pr=c[s].argmax(-1);e2=(~r1)&r2;pr=np.where(e2,b[s].argmax(-1),pr);pr=np.where(r1,a[s].argmax(-1),pr);return pr==Y[s],r1.mean(),e2.mean()
    cands={"entropy":[(lambda s,tt=t:(H(a[s])<tt,H(b[s])<tt)) for t in ETAU],"learned gate":[(lambda s,q1=q1,q2=q2:(sc[s][0]>q1,sc[s][1]>q2)) for q1 in QS for q2 in QS]}
    for rule,cs in cands.items():
        for eps in(.0025,.005,.01):
            best=None
            for f in cs:
                r1,r2=f("val");cr,f1,f2=out(r1[Bf],r2[Bf],"val") if False else (None,None,None)
                pr=c["val"].argmax(-1);e2=(~r1)&r2;pr=np.where(e2,b["val"].argmax(-1),pr);pr=np.where(r1,a["val"].argmax(-1),pr)
                acc=(pr==yv)[Bf].mean();bv=(Pb["final|lin|val"][k].argmax(-1)==yv)[Bf].mean()
                if acc<bv-eps:continue
                cc=cost(T_["cuda_laptop"],32,6,S,10,r1[Bf].mean(),e2[Bf].mean())
                if best is None or cc<best[0]:best=(cc,f)
            if best is None:continue
            cr,f1,f2=out(*best[1]("test"),"test")
            rows.append(dict(seed=k,rule=rule,eps=eps*100,f1=f1,f2=f2,dtest=100*(cr.mean()-base["test"]),**{f"{t}_{bb}":100*(1-cost(T_[t],bb,6,S,10,f1,f2)/cost(T_[t],bb,None,{},None,0,0)) for t,bb in REG}))
o=pd.DataFrame(rows);o.to_csv(OUT+"gate_test.csv",index=False)
pd.set_option("display.width",200)
print(o.groupby(["rule","eps"],dropna=False).mean(numeric_only=True).drop(columns="seed").round(2).to_string())
print(o[o.eps==.5][["seed","rule","dtest","cuda_laptop_32"]].round(2).to_string(index=False))
