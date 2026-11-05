# Do patch-pooled exit heads beat CLS exit heads on the fine-tuned models? VALIDATION ONLY: settings chosen on one half, scored on the other, both ways round. Cost-table predictions.
import os, json, time, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F, timm, itertools
ROOT=os.path.abspath("outputs"); OUT=ROOT+"/results/exit_head_screen"; os.makedirs(OUT,exist_ok=True); CK=ROOT+"/scratch/ft_pilot/"; DEV="cuda"
def log(*a):
    s=time.strftime("%H:%M:%S ")+" ".join(str(x) for x in a);print(s,flush=True);open(OUT+"/progress.log","a").write(s+"\n")
d=torch.load(ROOT+"/scratch/cache/cifar10_uint8.pt",weights_only=True);sp=json.load(open(ROOT+"/results/splits.json"))
DATA={s:(d["Xtr"][sp[s]],d["Ytr"][sp[s]]) for s in("train","cal","val")}
def prep(x):
    x=x.to(DEV).float().div_(255.);x=F.interpolate(x,size=(224,224),mode="bilinear",align_corners=False);return (x-.5)/.5
def mp(N,r): return 0 if r<=0 or N<4 else int(min(max(1,int(N*r)),(N+1)//2-1))
def token_merge(tokens,ratio):
    r=mp(tokens.shape[1]-1,ratio);t=tokens;cls,p=t[:,:1],t[:,1:];a,b=p[:,0::2],p[:,1::2]
    S=F.normalize(a,dim=-1)@F.normalize(b,dim=-1).transpose(-1,-2);mx,idx=S.max(-1);o=mx.argsort(-1,descending=True);src,unm=o[:,:r],o[:,r:];dst=idx.gather(-1,src);D=a.shape[-1]
    u=a.gather(1,unm[...,None].expand(-1,-1,D));sv=a.gather(1,src[...,None].expand(-1,-1,D));b=b.scatter_reduce(1,dst[...,None].expand(-1,-1,D),sv,reduce="mean",include_self=True)
    return torch.cat([cls,u,b],1)
MODELS={"b0":{},"merged":{6:.5,8:.25}}; SL=(4,6,8,10)
@torch.no_grad()
def extract(name):
    bb=timm.create_model("vit_tiny_patch16_224",pretrained=False,num_classes=0).to(DEV).eval();bb.load_state_dict(torch.load(CK+f"ft_{name}.pt",weights_only=True,map_location=DEV)["bb"]);M=MODELS[name];out={}
    for s,(X,Y) in DATA.items():
        r={(l,k):[] for l in SL for k in("cls","pool")}
        for i in range(0,len(Y),128):
            t=bb._pos_embed(bb.patch_embed(prep(X[i:i+128])));t=bb.norm_pre(t) if hasattr(bb,"norm_pre") else t
            for l,b in enumerate(bb.blocks,1):
                t=b(t)
                if l in SL:r[l,"cls"].append(t[:,0].cpu());r[l,"pool"].append(t[:,1:].mean(1).cpu())
                if l in M:t=token_merge(t,M[l])
        out[s]={k:torch.cat(v) for k,v in r.items()};log(name,s,"extracted")
    return out
def fit(X,y,dim,seed):
    torch.manual_seed(1000+seed);h=nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,192),nn.GELU(),nn.Linear(192,10)).to(DEV);X=X.to(DEV);y=y.to(DEV);o=torch.optim.AdamW(h.parameters(),lr=2e-3,weight_decay=1e-4);n=len(y);st=10*int(np.ceil(n/128));s=torch.optim.lr_scheduler.OneCycleLR(o,max_lr=2e-3,total_steps=st,pct_start=.1)
    for _ in range(10):
        p=torch.randperm(n,device=DEV)
        for i in range(0,n,128):
            b=p[i:i+128];l=F.cross_entropy(h(X[b]),y[b]);o.zero_grad(set_to_none=True);l.backward();o.step();s.step()
    return h.eval()
GT=torch.logspace(-1.3,1,200)
P={}
yv=DATA["val"][1].numpy();rec=[]
for name in MODELS:
    pp=OUT+f"/probs_{name}.npz"
    if os.path.exists(pp):P[name]=dict(np.load(pp));continue
    F_=extract(name);Pn={}
    for l,kind in itertools.product(SL,("pool","both")):
        get=lambda s:(F_[s][l,"pool"] if kind=="pool" else torch.cat([F_[s][l,"cls"],F_[s][l,"pool"]],1))
        r=[]
        for seed in range(3):
            h=fit(get("train"),DATA["train"][1],192 if kind=="pool" else 384,seed)
            with torch.no_grad():
                lc=h(get("cal").to(DEV)).cpu();lv=h(get("val").to(DEV)).cpu()
            T=float(GT[torch.stack([F.cross_entropy(lc/t,DATA["cal"][1]) for t in GT]).argmin()]);q=F.softmax(lv/T,-1).numpy().astype(np.float32);r.append(q);rec.append(dict(model=name,block=l,head=kind,seed=seed,T=T,val=100*(q.argmax(-1)==yv).mean()))
        Pn[f"{l}|{kind}"]=np.stack(r)
    np.savez(pp,**Pn);P[name]=Pn;log(name,"heads done");del F_;torch.cuda.empty_cache()
if rec:pd.DataFrame(rec).to_csv(OUT+"/heads.csv",index=False)
C={n:dict(np.load(CK+f"probs_{n}.npz")) for n in MODELS}
print("head val % (mean of 3):")
tab=[]
for n in MODELS:
    for l in SL:
        tab.append(dict(model=n,block=l,cls=100*(C[n][f"{l}|mlp192|val"].argmax(-1)==yv).mean(),pool=100*(P[n][f"{l}|pool"].argmax(-1)==yv).mean(),both=100*(P[n][f"{l}|both"].argmax(-1)==yv).mean()))
tab=pd.DataFrame(tab);tab.to_csv(OUT+"/head_acc.csv",index=False);print(tab.round(2).to_string(index=False))
T_=json.load(open(ROOT+"/results/energy_tables.json"))
def icost(tab,bs,b):
    ys=np.array([tab[str(x)]["gross"] for x in bs]);return float(np.interp(np.log(max(b,1.)),np.log(np.array(bs,float)),ys)) if b>=1 else b*ys[0]
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
H=lambda p:-(p*np.log(np.clip(p,1e-12,1))).sum(-1)/np.log(10)
TAU=[.02,.05,.08,.12,.16,.2,.25,.3,.4,.5,.6,.7]
def head(n,l,kind):return C[n][f"{l}|mlp192|val"] if kind=="cls" else P[n][f"{l}|{kind}"]
cfg=[]
perm=np.random.default_rng(0).permutation(4000);folds=[(perm[:2000],perm[2000:]),(perm[2000:],perm[:2000])]
b0pred=C["b0"]["final|lin|val"].argmax(-1)==yv
for n,S in MODELS.items():
    fin=C[n]["final|lin|val"]
    for kind in("cls","pool","both"):
        for g,L in((4,8),(4,10),(6,10),(8,10),(4,None),(6,None),(8,None)):
            A=head(n,g,kind);Bh=head(n,L,kind) if L else None
            for tau in TAU:
                ex1=H(A)<tau
                pr=fin.argmax(-1).copy();e2=np.zeros_like(ex1)
                if L:
                    e2=(~ex1)&(H(Bh)<tau);pr=np.where(e2,Bh.argmax(-1),pr)
                pr=np.where(ex1,A.argmax(-1),pr);cfg.append(dict(model=n,head=kind,g=g,L=L,tau=tau,cor=(pr==yv),ex1=ex1,e2=e2))
rows=[];REG=[("cuda_laptop",32),("cpu_laptop",32),("cpu_laptop",1),("cuda_laptop",1)]
for t,b in REG:
    bc=cost(T_[t],b,None,{},None,0,0)
    for eps in(.0025,.005,.01):
        for n in MODELS:
            for kind in("cls","pool","both"):
                sv=[];dd=[]
                for A,Bf in folds:
                    bestc=None;best=None;ba=b0pred[:,A].mean()
                    for c in cfg:
                        if c["model"]!=n or c["head"]!=kind or c["cor"][:,A].mean()<ba-eps:continue
                        cc=cost(T_[t],b,c["g"],MODELS[n],c["L"],c["ex1"][:,A].mean(),c["e2"][:,A].mean())
                        if bestc is None or cc<bestc:bestc,best=cc,c
                    if best is None:continue
                    sv.append(100*(1-cost(T_[t],b,best["g"],MODELS[n],best["L"],best["ex1"][:,Bf].mean(),best["e2"][:,Bf].mean())/bc));dd.append(100*(best["cor"][:,Bf].mean()-b0pred[:,Bf].mean()))
                rows.append(dict(reg=f"{t}_b{b}",eps=eps*100,model=n,head=kind,saved=np.mean(sv) if sv else np.nan,heldout_diff=np.mean(dd) if dd else np.nan,last=f"g{best['g']} L{best['L']} t{best['tau']}" if best else ""))
o=pd.DataFrame(rows);o.to_csv(OUT+"/selection.csv",index=False);pd.set_option("display.width",200,"display.max_rows",300);print(o.round(2).to_string(index=False));log("DONE")
