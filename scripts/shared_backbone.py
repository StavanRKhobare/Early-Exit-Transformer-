# One backbone trained across merge schedules: each step samples one schedule uniformly. Float32, seed 0, same data and optimiser as finetune_pilot.py. Validation only.
# Saves a checkpoint every epoch and resumes from it. FT_SMOKE=1 for a code check.
import os, json, time, subprocess, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F, timm
SMOKE=os.environ.get("FT_SMOKE","0")=="1"; ROOT=os.path.abspath("outputs"); OUT=ROOT+"/results/shared_backbone"+("_smoke" if SMOKE else ""); CK=ROOT+"/scratch/ft_pilot/"
os.makedirs(OUT,exist_ok=True); DEV="cuda"; EPOCHS=1 if SMOKE else 5; BATCH=64
def log(*a):
    s=time.strftime("%H:%M:%S ")+" ".join(str(x) for x in a);print(s,flush=True);open(OUT+"/progress.log","a").write(s+"\n")
def gpu(): return subprocess.run(["nvidia-smi","--query-gpu=temperature.gpu,clocks.gr,power.draw","--format=csv,noheader,nounits"],capture_output=True,text=True).stdout.strip()
d=torch.load(ROOT+"/scratch/cache/cifar10_uint8.pt",weights_only=True);sp=json.load(open(ROOT+"/results/splits.json"))
if SMOKE:sp={k:v[::40] for k,v in sp.items()}
X,Y=d["Xtr"][sp["train"]],d["Ytr"][sp["train"]];Xv,Yv=d["Xtr"][sp["val"]],d["Ytr"][sp["val"]]
def prep(x):
    x=x.to(DEV).float().div_(255.);x=F.interpolate(x,size=(224,224),mode="bilinear",align_corners=False);return (x-.5)/.5
def mp(N,r): return 0 if r<=0 or N<4 else int(min(max(1,int(N*r)),(N+1)//2-1))
def token_merge(t,ratio):
    r=mp(t.shape[1]-1,ratio);cls,p=t[:,:1],t[:,1:];a,b=p[:,0::2],p[:,1::2]
    S=F.normalize(a,dim=-1)@F.normalize(b,dim=-1).transpose(-1,-2);mx,idx=S.max(-1);o=mx.argsort(-1,descending=True);src,unm=o[:,:r],o[:,r:];dst=idx.gather(-1,src);D=a.shape[-1]
    u=a.gather(1,unm[...,None].expand(-1,-1,D));sv=a.gather(1,src[...,None].expand(-1,-1,D));b=b.scatter_reduce(1,dst[...,None].expand(-1,-1,D),sv,reduce="mean",include_self=True)
    return torch.cat([cls,u,b],1)
SCHED={"none":{}, "8r50":{8:.5}, "6r50":{6:.5}, "6r50_8r25":{6:.5,8:.25}, "4r25_8r25":{4:.25,8:.25}, "4r50":{4:.5}}
NAMES=list(SCHED)
def forward(bb,x,M):
    t=bb._pos_embed(bb.patch_embed(x));t=bb.norm_pre(t) if hasattr(bb,"norm_pre") else t
    for l,b in enumerate(bb.blocks,1):
        t=b(t)
        if l in M:t=token_merge(t,M[l])
    return bb.norm(t)[:,0]
def augment(x):
    n=x.shape[0];p=F.pad(x.float(),(4,4,4,4),mode="reflect");ox,oy=torch.randint(0,9,(n,),device=x.device),torch.randint(0,9,(n,),device=x.device);a=torch.arange(32,device=x.device)
    ix,iy=ox[:,None]+a[None],oy[:,None]+a[None];b=torch.arange(n,device=x.device)[:,None,None,None];c=torch.arange(3,device=x.device)[None,:,None,None]
    out=p[b,c,iy[:,None,:,None],ix[:,None,None,:]];f=torch.rand(n,device=x.device)<.5;out[f]=out[f].flip(-1);return out
@torch.no_grad()
def acc(bb,head,M,XX,YY):
    c=0
    for i in range(0,len(YY),128):c+=(head(forward(bb,prep(XX[i:i+128]),M)).argmax(-1).cpu()==YY[i:i+128]).sum().item()
    return 100*c/len(YY)
torch.manual_seed(0);np.random.seed(0)
bb=timm.create_model("vit_tiny_patch16_224",pretrained=True,num_classes=0).to(DEV);head=nn.Linear(192,10).to(DEV)
G=[{"params":list(head.parameters()),"lr":1e-3},{"params":list(bb.norm.parameters()),"lr":1e-4}]+[{"params":list(b.parameters()),"lr":1e-4*.8**(12-l)} for l,b in enumerate(bb.blocks,1)]+[{"params":list(bb.patch_embed.parameters())+[bb.cls_token,bb.pos_embed],"lr":1e-4*.8**12}]
opt=torch.optim.AdamW(G,weight_decay=.01);steps=max(2,EPOCHS*(len(Y)//BATCH));sch=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=[g["lr"] for g in G],total_steps=steps,pct_start=.3 if SMOKE else .1)
path=CK+("smoke_shared.pt" if SMOKE else "ft_shared_ckpt.pt");start=0
if os.path.exists(path) and not SMOKE:
    s=torch.load(path,map_location=DEV,weights_only=False);bb.load_state_dict(s["bb"]);head.load_state_dict(s["head"]);opt.load_state_dict(s["opt"]);sch.load_state_dict(s["sch"]);start=s["epoch"];torch.set_rng_state(s["rng"].cpu());log("resumed at epoch",start)
rs=np.random.RandomState(0);rs.randint(0,len(NAMES),size=start*(len(Y)//BATCH))
log(f"shared: start at epoch {start}, schedules {NAMES}, {steps} steps, gpu {gpu()}")
for ep in range(start,EPOCHS):
    bb.train();g=torch.Generator().manual_seed(1000+ep);perm=torch.randperm(len(Y),generator=g);t0=time.time();tot=0;n=0
    for i in range(0,len(Y)-BATCH+1,BATCH):
        idx=perm[i:i+BATCH];x=prep(augment(X[idx].to(DEV)));y=Y[idx].to(DEV);M=SCHED[NAMES[rs.randint(0,len(NAMES))]]
        loss=F.cross_entropy(head(forward(bb,x,M)),y);opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(bb.parameters(),1.);opt.step();sch.step();tot+=loss.item();n+=1
    bb.eval();head.eval();res={k:acc(bb,head,SCHED[k],Xv,Yv) for k in ("none","6r50_8r25","4r50")}
    log(f"shared: epoch {ep+1}/{EPOCHS} loss {tot/max(n,1):.3f} {time.time()-t0:.0f}s | val "+" ".join(f"{k}:{v:.2f}" for k,v in res.items())+f" | temp,clock,W {gpu()}")
    if not SMOKE:
        torch.save({"bb":bb.state_dict(),"head":head.state_dict(),"opt":opt.state_dict(),"sch":sch.state_dict(),"epoch":ep+1,"rng":torch.get_rng_state()},path+".tmp");os.replace(path+".tmp",path)
if not SMOKE:torch.save({"bb":bb.state_dict(),"head":head.state_dict()},CK+"ft_shared.pt")
rows=[]
def load(n):
    m=timm.create_model("vit_tiny_patch16_224",pretrained=False,num_classes=0).to(DEV).eval();h=nn.Linear(192,10).to(DEV);s=torch.load(CK+f"ft_{n}.pt",weights_only=True,map_location=DEV);m.load_state_dict(s["bb"]);h.load_state_dict(s["head"]);return m,h.eval()
for k,M in SCHED.items():rows.append(dict(model="shared",schedule=k,val=acc(bb,head,M,Xv,Yv)))
if not SMOKE:
    for n,k in(("b0","none"),("merged","6r50_8r25")):
        m,h=load(n);rows.append(dict(model=f"dedicated {n}",schedule=k,val=acc(m,h,SCHED[k],Xv,Yv)));del m,h
o=pd.DataFrame(rows);o.to_csv(OUT+"/val_by_schedule.csv",index=False);print(o.round(2).to_string(index=False));log("DONE")
