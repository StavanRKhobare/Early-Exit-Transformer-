# Screen token-merge policies on the frozen backbone: can exit-head evidence let us merge earlier and harder? VALIDATION ONLY, test set untouched. Float32.
# Policies change WHICH tokens merge, never how many, so block costs are the same as plain ToMe at the same schedule.
import os, json, time, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F, timm
SMOKE=os.environ.get("FT_SMOKE","0")=="1"; ROOT=os.path.abspath("outputs"); OUT=ROOT+"/results/merge_screen"+("_smoke" if SMOKE else ""); os.makedirs(OUT,exist_ok=True); DEV="cuda"
def log(*a):
    s=time.strftime("%H:%M:%S ")+" ".join(str(x) for x in a);print(s,flush=True);open(OUT+"/progress.log","a").write(s+"\n")
d=torch.load(ROOT+"/scratch/cache/cifar10_uint8.pt",weights_only=True);sp=json.load(open(ROOT+"/results/splits.json"))
step=40 if SMOKE else 2
tr=sp["train"][::step];va=sp["val"][::(10 if SMOKE else 1)]
Xt,Yt=d["Xtr"][tr],d["Ytr"][tr];Xv,Yv=d["Xtr"][va],d["Ytr"][va]
def prep(x):
    x=x.to(DEV).float().div_(255.);x=F.interpolate(x,size=(224,224),mode="bilinear",align_corners=False);return (x-.5)/.5
bb=timm.create_model("vit_tiny_patch16_224",pretrained=True,num_classes=0).to(DEV).eval()
for p in bb.parameters():p.requires_grad=False
LAYERS=(2,4,6)
ATT={}
for l in LAYERS:
    bb.blocks[l-1].attn.fused_attn=False
    bb.blocks[l-1].attn.attn_drop.register_forward_hook(lambda m,i,o,l=l:ATT.__setitem__(l,i[0].mean(1)[:,0,1:].detach()))
def emb(x):
    t=bb._pos_embed(bb.patch_embed(x));return bb.norm_pre(t) if hasattr(bb,"norm_pre") else t
@torch.no_grad()
def pass1(X):
    out={l:[] for l in LAYERS}
    for i in range(0,len(X),128):
        t=emb(prep(X[i:i+128]))
        for l,b in enumerate(bb.blocks[:6],1):
            t=b(t)
            if l in LAYERS:out[l].append(t[:,1:].mean(1).cpu())
    return {l:torch.cat(v) for l,v in out.items()}
def fit(X,y,arch=None,final=False,seed=0,ep=10):
    torch.manual_seed(1000+seed);h=nn.Sequential(nn.Identity() if final else nn.LayerNorm(192),nn.Linear(192,10)).to(DEV);X=X.to(DEV);y=y.to(DEV)
    o=torch.optim.AdamW(h.parameters(),lr=2e-3,weight_decay=1e-4);n=len(y);st=ep*int(np.ceil(n/128));s=torch.optim.lr_scheduler.OneCycleLR(o,max_lr=2e-3,total_steps=st,pct_start=.3 if SMOKE else .1)
    for _ in range(ep):
        p=torch.randperm(n,device=DEV)
        for i in range(0,n,128):
            b=p[i:i+128];l=F.cross_entropy(h(X[b]),y[b]);o.zero_grad(set_to_none=True);l.backward();o.step();s.step()
    return h.eval()
log("pass 1: patch-evidence heads");F1=pass1(Xt);HEAD={l:fit(F1[l],Yt) for l in LAYERS};F1v=pass1(Xv)
for l in LAYERS:
    log(f"  patch head block {l}: val {100*(HEAD[l](F1v[l].to(DEV)).argmax(-1).cpu()==Yv).float().mean().item():.1f}%")
def z(x):return (x-x.mean(1,keepdim=True))/(x.std(1,keepdim=True)+1e-6)
def pairs(Np,ratio):return int(min(max(1,int(Np*ratio)),(Np+1)//2-1))
def merge(t,ratio,policy,lam,l):
    r=pairs(t.shape[1]-1,ratio);cls,p=t[:,:1],t[:,1:];a,b=p[:,0::2],p[:,1::2];B=t.shape[0]
    S=F.normalize(a,dim=-1)@F.normalize(b,dim=-1).transpose(-1,-2)
    if policy=="random":S=torch.rand_like(S)
    elif policy=="attn":
        s=z(ATT[l]);S=S-lam*(s[:,0::2,None]+s[:,1::2][:,None,:])/2
    elif policy in("ev_protect","ev_diff","logit_sim"):
        lg=HEAD[l](p);m=lg.mean(1);top=m.topk(2,-1).indices;e=lg.gather(2,top[:,:1,None].expand(-1,p.shape[1],-1)-0)[...,0]-lg.gather(2,top[:,1:2,None].expand(-1,p.shape[1],-1))[...,0]
        if policy=="ev_protect":
            s=z(e.abs());S=S-lam*(s[:,0::2,None]+s[:,1::2][:,None,:])/2
        elif policy=="ev_diff":
            s=e/(e.std(1,keepdim=True)+1e-6);S=S-lam*(s[:,0::2,None]-s[:,1::2][:,None,:]).abs()
        else:
            q=F.normalize(lg-m[:,None],dim=-1);S=q[:,0::2]@q[:,1::2].transpose(-1,-2)
    mx,idx=S.max(-1);o=mx.argsort(-1,descending=True);src,unm=o[:,:r],o[:,r:];dst=idx.gather(-1,src);D=a.shape[-1]
    u=a.gather(1,unm[...,None].expand(-1,-1,D));sv=a.gather(1,src[...,None].expand(-1,-1,D))
    b2=b.scatter_reduce(1,dst[...,None].expand(-1,-1,D),sv,reduce="mean",include_self=True)
    return torch.cat([cls,u,b2],1)
POL=[("tome",0),("random",0),("attn",.1),("attn",.3),("ev_protect",.1),("ev_protect",.3),("ev_diff",.1),("ev_diff",.3),("logit_sim",0)]
SCHED=[(2,.5),(4,.5),(6,.5)]
@torch.no_grad()
def pass2(X):
    out={"none":[]};[out.setdefault((l,r,p,lam),[]) for l,r in SCHED for p,lam in POL]
    for i in range(0,len(X),128):
        t=emb(prep(X[i:i+128]))
        for l,blk in enumerate(bb.blocks,1):
            t=blk(t)
            for L,r in SCHED:
                if L==l:
                    for p,lam in POL:
                        c=merge(t,r,p,lam,l)
                        for bk in bb.blocks[l:]:c=bk(c)
                        out[(L,r,p,lam)].append(bb.norm(c)[:,0].cpu())
        out["none"].append(bb.norm(t)[:,0].cpu())
    return {k:torch.cat(v) for k,v in out.items()}
torch.manual_seed(0);log("pass 2: train images");Ft=pass2(Xt);log("pass 2: validation images");Fv=pass2(Xv)
pred={}
for k in Ft:
    pred[k]=torch.stack([fit(Ft[k],Yt,final=True,seed=s)(Fv[k].to(DEV)).argmax(-1).cpu() for s in range(3)])==Yv
base=pred["none"];rows=[]
for k in pred:
    if k=="none":continue
    L,r,p,lam=k;t=pred[(L,r,"tome",0)];rng=np.random.default_rng(0);di=(pred[k].float()-t.float()).numpy();m=di.mean(0)
    bo=[m[rng.integers(0,len(m),len(m))].mean()*100 for _ in range(1000)]
    rows.append(dict(block=L,ratio=r,policy=p,lam=lam,val_acc=100*pred[k].float().mean().item(),vs_b0=100*(pred[k].float().mean()-base.float().mean()).item(),vs_tome=100*m.mean(),ci_low=np.percentile(bo,2.5),ci_high=np.percentile(bo,97.5)))
o=pd.DataFrame(rows);o.to_csv(OUT+"/screen.csv",index=False)
log("B0 val",round(100*base.float().mean().item(),2));print(o.round(2).to_string(index=False));log("DONE")
