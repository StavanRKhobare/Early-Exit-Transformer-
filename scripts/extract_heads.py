# Block 2: forward passes for shared, dedicated merged and dedicated B0, patch-average exit heads, and probabilities on val, test and the 25,000 unused images.
# Features are written per split so a crash loses little, and deleted once probabilities exist. Float32.
import os, json, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, timm, pandas as pd
ROOT=os.path.abspath("outputs");CK=ROOT+"/scratch/ft_pilot/";FE=ROOT+"/scratch/ft_features/";OUT=ROOT+"/results/shared_backbone/";os.makedirs(FE,exist_ok=True);DEV="cuda"
def log(*a):
    s=time.strftime("%H:%M:%S ")+" ".join(str(x) for x in a);print(s,flush=True);open(OUT+"progress.log","a").write(s+"\n")
d=torch.load(ROOT+"/scratch/cache/cifar10_uint8.pt",weights_only=True);sp=json.load(open(ROOT+"/results/splits.json"))
used=set(sp["train"])|set(sp["cal"])|set(sp["val"]);un=sorted(set(range(50000))-used);assert len(un)==25000
DATA={"train":(d["Xtr"][sp["train"]],d["Ytr"][sp["train"]]),"cal":(d["Xtr"][sp["cal"]],d["Ytr"][sp["cal"]]),"val":(d["Xtr"][sp["val"]],d["Ytr"][sp["val"]]),"test":(d["Xte"],d["Yte"]),"unused":(d["Xtr"][un],d["Ytr"][un])}
def prep(x):
    x=x.to(DEV).float().div_(255.);x=F.interpolate(x,size=(224,224),mode="bilinear",align_corners=False);return (x-.5)/.5
def mp(N,r): return 0 if r<=0 or N<4 else int(min(max(1,int(N*r)),(N+1)//2-1))
def token_merge(t,ratio):
    r=mp(t.shape[1]-1,ratio);cls,p=t[:,:1],t[:,1:];a,b=p[:,0::2],p[:,1::2]
    S=F.normalize(a,dim=-1)@F.normalize(b,dim=-1).transpose(-1,-2);mx,idx=S.max(-1);o=mx.argsort(-1,descending=True);src,unm=o[:,:r],o[:,r:];dst=idx.gather(-1,src);D=a.shape[-1]
    u=a.gather(1,unm[...,None].expand(-1,-1,D));sv=a.gather(1,src[...,None].expand(-1,-1,D));b=b.scatter_reduce(1,dst[...,None].expand(-1,-1,D),sv,reduce="mean",include_self=True)
    return torch.cat([cls,u,b],1)
ALL={"none":{}, "8r50":{8:.5}, "6r50":{6:.5}, "6r50_8r25":{6:.5,8:.25}, "4r25_8r25":{4:.25,8:.25}, "4r50":{4:.5}}
MODELS={"shared":list(ALL),"b0":["none"],"merged":["6r50_8r25"]}
EX=(4,6,8,10)
pname=lambda M,l:"_".join(f"{k}r{int(v*100)}" for k,v in sorted(M.items()) if k<l) or "none"
def walk(bb,t,start,names,out):
    for l in range(start+1,13):
        t=bb.blocks[l-1](t)
        if l in EX:out[f"{l}|{pname(ALL[names[0]],l)}"]=t[:,1:].mean(1)
        groups={}
        for n in names:groups.setdefault(ALL[n].get(l,0),[]).append(n)
        if len(groups)>1 or 0 not in groups:
            for r,g in groups.items():walk(bb,token_merge(t,r) if r else t,l,g,out)
            return
    for n in names:out[f"final|{n}"]=bb.norm(t)[:,0]
@torch.no_grad()
def extract(name,split):
    p=FE+f"{name}_{split}.pt"
    if os.path.exists(p):return torch.load(p,weights_only=True)
    s=torch.load(CK+f"ft_{name}.pt",weights_only=True,map_location=DEV);bb=timm.create_model("vit_tiny_patch16_224",pretrained=False,num_classes=0).to(DEV).eval();bb.load_state_dict(s["bb"]);X,_=DATA[split];r={};t0=time.time()
    for i in range(0,len(X),128):
        x=prep(X[i:i+128]);t=bb._pos_embed(bb.patch_embed(x));t=bb.norm_pre(t) if hasattr(bb,"norm_pre") else t;o={};walk(bb,t,0,MODELS[name],o)
        for k,v in o.items():r.setdefault(k,[]).append(v.cpu().half())
    r={k:torch.cat(v) for k,v in r.items()};torch.save(r,p+".tmp");os.replace(p+".tmp",p);log(name,split,f"extracted {len(X)} images in {time.time()-t0:.0f}s, {len(r)} slots");return r
def mk():return nn.Sequential(nn.LayerNorm(192),nn.Linear(192,192),nn.GELU(),nn.Linear(192,10)).to(DEV)
def fit(X,y,seed,final=False):
    torch.manual_seed(1000+seed);h=nn.Linear(192,10).to(DEV) if final else mk();X=X.float().to(DEV);y=y.to(DEV);o=torch.optim.AdamW(h.parameters(),lr=2e-3,weight_decay=1e-4);n=len(y);st=10*int(np.ceil(n/128));s=torch.optim.lr_scheduler.OneCycleLR(o,max_lr=2e-3,total_steps=st,pct_start=.1)
    for _ in range(10):
        p=torch.randperm(n,device=DEV)
        for i in range(0,n,128):
            b=p[i:i+128];l=F.cross_entropy(h(X[b]),y[b]);o.zero_grad(set_to_none=True);l.backward();o.step();s.step()
    return h.eval()
GT=torch.logspace(-1.3,1,200);PS=("val","test","unused");rec=[]
for name in MODELS:
    out=OUT+f"probs_{name}.npz"
    if os.path.exists(out):log(name,"probabilities already exist");continue
    Fx={sp_:extract(name,sp_) for sp_ in DATA};st=torch.load(CK+f"ft_{name}.pt",weights_only=True,map_location="cpu");fh=nn.Linear(192,10);fh.load_state_dict(st["head"]);P={}
    for k in Fx["train"]:
        final=k.startswith("final")
        if final:
            for s in PS:
                with torch.no_grad():P[f"{k}|fthead|{s}"]=F.softmax(fh(Fx[s][k].float()),-1).numpy().astype(np.float16)
        r={s:[] for s in PS}
        for seed in range(3):
            h=fit(Fx["train"][k],DATA["train"][1],seed,final)
            with torch.no_grad():
                lc=h(Fx["cal"][k].float().to(DEV)).cpu();ls={s:h(Fx[s][k].float().to(DEV)).cpu() for s in PS}
            T=float(GT[torch.stack([F.cross_entropy(lc/t,DATA["cal"][1]) for t in GT]).argmin()])
            for s in PS:r[s].append(F.softmax(ls[s]/T,-1).numpy().astype(np.float32))
            rec.append(dict(model=name,slot=k,seed=seed,T=T,val=100*(ls["val"].argmax(-1)==DATA["val"][1]).float().mean().item()))
        for s in PS:P[f"{k}|{s}"]=np.stack(r[s])
    np.savez(out,**P);pd.DataFrame(rec).to_csv(OUT+"heads.csv",index=False);log(name,"heads done, probabilities saved")
    for s in DATA:os.remove(FE+f"{name}_{s}.pt")
np.savez(OUT+"labels.npz",**{s:DATA[s][1].numpy() for s in PS});log("BLOCK2 DONE")
