# Joint training pilot: fine-tune backbone, final head and patch-average exit heads together, merging on (6r50_8r25). Then freeze and retrain heads the same way as the earlier screen.
# Also tests size-weighted pooling on both this and the earlier two-stage model. VALIDATION ONLY, test untouched. Float32. Cost-table predictions. Resumable. FT_SMOKE=1 for a code check.
import os, json, time, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F, timm, itertools
SMOKE=os.environ.get("FT_SMOKE","0")=="1"; AUX=float(os.environ.get("JP_AUX",.3)); KD=float(os.environ.get("JP_KD",0)); TAG=os.environ.get("JP_TAG","joint")
ROOT=os.path.abspath("outputs"); OUT=ROOT+"/results/joint_pilot"+("_smoke" if SMOKE else ""); os.makedirs(OUT,exist_ok=True); CK=ROOT+"/scratch/ft_pilot/"; DEV="cuda"; EPOCHS=1 if SMOKE else 5; BATCH=64
def log(*a):
    s=time.strftime("%H:%M:%S ")+" ".join(str(x) for x in a);print(s,flush=True);open(OUT+"/progress.log","a").write(s+"\n")
d=torch.load(ROOT+"/scratch/cache/cifar10_uint8.pt",weights_only=True);sp=json.load(open(ROOT+"/results/splits.json"))
if SMOKE:sp={k:v[::40] for k,v in sp.items()}
DATA={s:(d["Xtr"][sp[s]],d["Ytr"][sp[s]]) for s in("train","cal","val")}
def prep(x):
    x=x.to(DEV).float().div_(255.);x=F.interpolate(x,size=(224,224),mode="bilinear",align_corners=False);return (x-.5)/.5
def mp(N,r): return 0 if r<=0 or N<4 else int(min(max(1,int(N*r)),(N+1)//2-1))
def token_merge(t,ratio,sz):
    r=mp(t.shape[1]-1,ratio);cls,p=t[:,:1],t[:,1:];a,b=p[:,0::2],p[:,1::2];sa,sb=sz[:,0::2],sz[:,1::2]
    S=F.normalize(a,dim=-1)@F.normalize(b,dim=-1).transpose(-1,-2);mx,idx=S.max(-1);o=mx.argsort(-1,descending=True);src,unm=o[:,:r],o[:,r:];dst=idx.gather(-1,src);D=a.shape[-1]
    u=a.gather(1,unm[...,None].expand(-1,-1,D));sv=a.gather(1,src[...,None].expand(-1,-1,D));b=b.scatter_reduce(1,dst[...,None].expand(-1,-1,D),sv,reduce="mean",include_self=True)
    sb=sb.scatter_add(1,dst,sa.gather(1,src));return torch.cat([cls,u,b],1),torch.cat([sa.gather(1,unm),sb],1)
M={6:.5,8:.25};SL=(4,6,8,10)
def forward(bb,x,sink=None,pools=None):
    t=bb._pos_embed(bb.patch_embed(x));t=bb.norm_pre(t) if hasattr(bb,"norm_pre") else t;sz=torch.ones(t.shape[0],t.shape[1]-1,device=t.device)
    for l,b in enumerate(bb.blocks,1):
        t=b(t)
        if l in SL:
            p=t[:,1:]
            if pools is not None:pools[l]=p.mean(1)
            if sink is not None:sink[l,"pool"]=p.mean(1);sink[l,"wpool"]=(p*sz[...,None]).sum(1)/sz.sum(1,keepdim=True)
        if l in M:t,sz=token_merge(t,M[l],sz)
    return bb.norm(t)[:,0]
def augment(x):
    n=x.shape[0];p=F.pad(x.float(),(4,4,4,4),mode="reflect");ox,oy=torch.randint(0,9,(n,),device=x.device),torch.randint(0,9,(n,),device=x.device);a=torch.arange(32,device=x.device)
    ix,iy=ox[:,None]+a[None],oy[:,None]+a[None];b=torch.arange(n,device=x.device)[:,None,None,None];c=torch.arange(3,device=x.device)[None,:,None,None]
    out=p[b,c,iy[:,None,:,None],ix[:,None,None,:]];f=torch.rand(n,device=x.device)<.5;out[f]=out[f].flip(-1);return out
def mk(dim=192):return nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,192),nn.GELU(),nn.Linear(192,10)).to(DEV)
def train_joint():
    path=CK+f"ft_{TAG}.pt";torch.manual_seed(0);bb=timm.create_model("vit_tiny_patch16_224",pretrained=True,num_classes=0).to(DEV);head=nn.Linear(192,10).to(DEV);aux=nn.ModuleDict({str(l):mk() for l in SL})
    if os.path.exists(path) and not SMOKE:
        bb.load_state_dict(torch.load(path,weights_only=True,map_location=DEV)["bb"]);log("loaded",TAG);return bb.eval()
    G=[{"params":list(head.parameters())+list(aux.parameters()),"lr":1e-3},{"params":list(bb.norm.parameters()),"lr":1e-4}]+[{"params":list(b.parameters()),"lr":1e-4*.8**(12-l)} for l,b in enumerate(bb.blocks,1)]+[{"params":list(bb.patch_embed.parameters())+[bb.cls_token,bb.pos_embed],"lr":1e-4*.8**12}]
    opt=torch.optim.AdamW(G,weight_decay=.01);X,Y=DATA["train"];steps=max(1,EPOCHS*(len(Y)//BATCH));sch=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=[g["lr"] for g in G],total_steps=steps,pct_start=.3 if SMOKE else .1)
    g=torch.Generator().manual_seed(0);log(f"{TAG}: start, aux {AUX}, kd {KD}, {steps} steps")
    Xv,Yv=DATA["val"]
    for ep in range(EPOCHS):
        bb.train();perm=torch.randperm(len(Y),generator=g);t0=time.time();tot=0;n=0
        for i in range(0,len(Y)-BATCH+1,BATCH):
            idx=perm[i:i+BATCH];x=prep(augment(X[idx].to(DEV)));y=Y[idx].to(DEV);pools={};f=forward(bb,x,pools=pools);lf=head(f);loss=F.cross_entropy(lf,y)
            for l in SL:
                lg=aux[str(l)](pools[l]);a=F.cross_entropy(lg,y)
                if KD>0:a=(1-KD)*a+KD*F.kl_div(F.log_softmax(lg,-1),F.softmax(lf.detach(),-1),reduction="batchmean")
                loss=loss+AUX*a
            opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(bb.parameters(),1.);opt.step();sch.step();tot+=loss.item();n+=1
        bb.eval();acc={"final":0};[acc.setdefault(l,0) for l in SL]
        with torch.no_grad():
            for i in range(0,len(Yv),128):
                p={};f=forward(bb,prep(Xv[i:i+128]),pools=p);y=Yv[i:i+128].to(DEV);acc["final"]+=(head(f).argmax(-1)==y).sum().item()
                for l in SL:acc[l]+=(aux[str(l)](p[l]).argmax(-1)==y).sum().item()
        log(f"{TAG}: epoch {ep+1}/{EPOCHS} loss {tot/max(n,1):.3f} {time.time()-t0:.0f}s | val "+" ".join(f"{k}:{100*v/len(Yv):.1f}" for k,v in acc.items()))
    if not SMOKE:torch.save({"bb":bb.state_dict(),"head":head.state_dict(),"aux":aux.state_dict()},path)
    return bb.eval()
@torch.no_grad()
def extract(bb):
    out={}
    for s,(X,Y) in DATA.items():
        r={}
        for i in range(0,len(Y),128):
            k={};k["final"]=forward(bb,prep(X[i:i+128]),sink=k)
            for a,v in k.items():r.setdefault(a,[]).append(v.cpu())
        out[s]={a:torch.cat(v) for a,v in r.items()}
    return out
def fit(X,y,seed,final=False):
    torch.manual_seed(1000+seed);h=(nn.Linear(192,10).to(DEV) if final else mk());X=X.to(DEV);y=y.to(DEV);o=torch.optim.AdamW(h.parameters(),lr=2e-3,weight_decay=1e-4);n=len(y);st=10*int(np.ceil(n/128));s=torch.optim.lr_scheduler.OneCycleLR(o,max_lr=2e-3,total_steps=st,pct_start=.3 if SMOKE else .1)
    for _ in range(10):
        p=torch.randperm(n,device=DEV)
        for i in range(0,n,128):
            b=p[i:i+128];l=F.cross_entropy(h(X[b]),y[b]);o.zero_grad(set_to_none=True);l.backward();o.step();s.step()
    return h.eval()
GT=torch.logspace(-1.3,1,200);yv=DATA["val"][1].numpy()
def heads(F_):
    P={}
    for key in [(l,k) for l in SL for k in("pool","wpool")]+["final"]:
        r=[]
        for seed in range(3):
            h=fit(F_["train"][key],DATA["train"][1],seed,final=key=="final")
            with torch.no_grad():lc=h(F_["cal"][key].to(DEV)).cpu();lv=h(F_["val"][key].to(DEV)).cpu()
            T=float(GT[torch.stack([F.cross_entropy(lc/t,DATA["cal"][1]) for t in GT]).argmin()]);r.append(F.softmax(lv/T,-1).numpy().astype(np.float32))
        P[f"{key[0]}|{key[1]}" if key!="final" else "final"]=np.stack(r)
    return P
P={}
for name in ("merged",TAG):
    pp=OUT+f"/probs_{name}.npz"
    if os.path.exists(pp) and not SMOKE:P[name]=dict(np.load(pp));continue
    if name=="merged":
        bb=timm.create_model("vit_tiny_patch16_224",pretrained=False,num_classes=0).to(DEV).eval();bb.load_state_dict(torch.load(CK+"ft_merged.pt",weights_only=True,map_location=DEV)["bb"])
    else:bb=train_joint()
    P[name]=heads(extract(bb));log(name,"heads done")
    if not SMOKE:np.savez(pp,**P[name])
    del bb;torch.cuda.empty_cache()
b0=dict(np.load(CK+"probs_b0.npz"))["final|lin|val"]
if SMOKE:b0=b0[:,::40]
b0c=b0.argmax(-1)==yv;log("B0 val",round(100*b0c.mean(),2))
acc=[dict(model=n,**{k:100*(v.argmax(-1)==yv).mean() for k,v in P[n].items()}) for n in P];a=pd.DataFrame(acc);a.to_csv(OUT+f"/head_acc_{TAG}.csv",index=False);pd.set_option("display.width",220,"display.max_rows",300);print(a.round(2).T.to_string())
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
n=len(yv);perm=np.random.default_rng(0).permutation(n);folds=[(perm[:n//2],perm[n//2:]),(perm[n//2:],perm[:n//2])]
cfg=[]
for name in P:
    for kind in("pool","wpool"):
        for g,L in((4,8),(4,10),(6,10),(8,10),(4,None),(6,None),(8,None)):
            A=P[name][f"{g}|{kind}"];Bh=P[name][f"{L}|{kind}"] if L else None;fin=P[name]["final"]
            for tau in TAU:
                ex1=H(A)<tau;pr=fin.argmax(-1);e2=np.zeros_like(ex1)
                if L:e2=(~ex1)&(H(Bh)<tau);pr=np.where(e2,Bh.argmax(-1),pr)
                pr=np.where(ex1,A.argmax(-1),pr);cfg.append(dict(model=name,head=kind,g=g,L=L,tau=tau,cor=pr==yv,ex1=ex1,e2=e2))
rows=[]
for t,b in [("cuda_laptop",32),("cpu_laptop",32),("cpu_laptop",1),("cuda_laptop",1)]:
    bc=cost(T_[t],b,None,{},None,0,0)
    for eps in(.0025,.005,.01):
        for name in P:
            for kind in("pool","wpool"):
                sv=[];dd=[];pick=None
                for Af,Bf in folds:
                    best=None;bestc=None;ba=b0c[:,Af].mean()
                    for c in cfg:
                        if c["model"]!=name or c["head"]!=kind or c["cor"][:,Af].mean()<ba-eps:continue
                        cc=cost(T_[t],b,c["g"],M,c["L"],c["ex1"][:,Af].mean(),c["e2"][:,Af].mean())
                        if bestc is None or cc<bestc:bestc,best=cc,c
                    if best is None:continue
                    sv.append(100*(1-cost(T_[t],b,best["g"],M,best["L"],best["ex1"][:,Bf].mean(),best["e2"][:,Bf].mean())/bc));dd.append(100*(best["cor"][:,Bf].mean()-b0c[:,Bf].mean()));pick=f"g{best['g']} L{best['L']} t{best['tau']}"
                rows.append(dict(reg=f"{t}_b{b}",eps=eps*100,model=name,head=kind,saved=np.mean(sv) if sv else np.nan,heldout_diff=np.mean(dd) if dd else np.nan,n_folds=len(sv),last=pick))
o=pd.DataFrame(rows);o.to_csv(OUT+f"/selection_{TAG}.csv",index=False);print(o.round(2).to_string(index=False));log("DONE")
