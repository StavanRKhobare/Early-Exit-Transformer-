# Block 4: real energy of the chosen designs against dedicated B0. Measured, not predicted. Float32 on GPU, RAPL package on CPU.
# Designs come from Block 3 by a rule fixed before any measurement (see pick_designs). Images are 1,000 unused CIFAR-10 train images, so the test set stays out.
# Round-robin over regimes, so stopping at the deadline still leaves balanced repeats. Rows are saved after every measurement.
import os, sys, re, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _load_notebook import notebook_source
SMOKE_RUN="--smoke" in sys.argv
os.environ.update(GV_AMP="0", GV_HW="laptopfp32", GV_CPU_REGIME="0", GV_STAGES="", MPLBACKEND="Agg")
exec(compile(notebook_source(), "experiment.ipynb", "exec"))
ROOT=os.path.abspath("outputs");CK=ROOT+"/scratch/ft_pilot/";KEEP=ROOT+"/scratch/ft_keep/";OUT=ROOT+"/results/real_energy/"+("smoke/" if SMOKE_RUN else "");os.makedirs(OUT,exist_ok=True)
DEADLINE=time.mktime(time.strptime(time.strftime("%Y-%m-%d ")+"04:40","%Y-%m-%d %H:%M"))
if SMOKE_RUN:DEADLINE=time.time()+600
WINDOW,N_IMG,REPS,COOL_C,COOL_MAX=(1.,40,1,200,1) if SMOKE_RUN else (15.,1000,6,70,240)
REGIMES=[("cuda",32),("cpu",32),("cpu",1),("cuda",1)]
def log(*a):
    s=time.strftime("%H:%M:%S ")+" ".join(str(x) for x in a);print(s,flush=True);open(OUT+"progress.log","a").write(s+"\n")
ALLS={"none":{}, "8r50":{8:.5}, "6r50":{6:.5}, "6r50_8r25":{6:.5,8:.25}, "4r25_8r25":{4:.25,8:.25}, "4r50":{4:.5}}
pn=lambda M,l:"_".join(f"{k}r{int(v*100)}" for k,v in sorted(M.items()) if k<l) or "none"
def mp(N,r):return 0 if r<=0 or N<4 else int(min(max(1,int(N*r)),(N+1)//2-1))
def merge_same(t,ratio):
    # The same merge used in training and extract_heads.py.
    r=mp(t.shape[1]-1,ratio);cls,p=t[:,:1],t[:,1:];a,b=p[:,0::2],p[:,1::2]
    S=F.normalize(a,dim=-1)@F.normalize(b,dim=-1).transpose(-1,-2);mx,idx=S.max(-1);o=mx.argsort(-1,descending=True);src,unm=o[:,:r],o[:,r:];dst=idx.gather(-1,src);D=a.shape[-1]
    u=a.gather(1,unm[...,None].expand(-1,-1,D));sv=a.gather(1,src[...,None].expand(-1,-1,D));b=b.scatter_reduce(1,dst[...,None].expand(-1,-1,D),sv,reduce="mean",include_self=True)
    return torch.cat([cls,u,b],1)
def parse(d):
    m=re.match(r"(\S+)(?: g(\d+)@([\d.]+))?(?: L(\d+)@([\d.]+))?",d);return dict(sched=m[1],ex=[(int(m[k]),float(m[k+1])) for k in(2,4) if m[k]])
def pick_designs():
    # Fixed rule. Exact-guarantee design at the 0.5-pt limit for each device regime; the z rule if exact finds none.
    # Then the exit-only design from the same shared model on CPU batch 32, to show what merging adds. Up to 5 designs plus B0.
    C=json.load(open(ROOT+"/results/certified/chosen_seed0.json"));out=[]
    def get(fam,reg,eps):
        for rule in("Pareto FST exact (CP)","Pareto FST z"):
            d=C.get(f"{fam}|{reg}|{eps}|{rule}")
            if d and "no exit" not in d:return d,rule
        return None,None
    for reg in("cuda_laptop_b32","cpu_laptop_b32","cpu_laptop_b1"):
        for eps in(.005,.01):
            d,r=get("shared",reg,eps)
            if d and d not in [x[0] for x in out]:out.append((d,f"shared {reg} limit {eps*100:g} {r}"))
    d,r=get("shared_nomerge","cpu_laptop_b32",.005)
    if d and d not in [x[0] for x in out]:out.append((d,f"exit only cpu_laptop_b32 limit 0.5 {r}"))
    return out[:5]
class M(nn.Module):
    def __init__(s,bb,fh,sched,ex,hd):
        super().__init__();s.bb,s.fh,s.merges,s.ex=bb,fh,ALLS[sched],{l:(hd[(sched,l)][0],hd[(sched,l)][1],t) for l,t in ex}
    @torch.no_grad()
    def forward(s,x,tau):
        n=x.shape[0];pred=torch.zeros(n,dtype=torch.long,device=x.device);alive=torch.arange(n,device=x.device);t=embed(s.bb,x);counts={}
        for l,block in enumerate(s.bb.blocks,1):
            t=block(t)
            if l in s.ex:
                h,T,tt=s.ex[l];lg=h(t[:,1:].mean(1));p=F.softmax(lg/T,-1);m=-(p*p.clamp_min(1e-12).log()).sum(-1)/math.log(10)<tt;k=int(m.sum().item());counts[l]=k
                if k:
                    pred[alive[m]]=lg[m].argmax(-1)
                    if k==alive.shape[0]:return pred,counts
                    alive,t=alive[~m],t[~m]
            if l in s.merges:t=merge_same(t,s.merges[l])
        pred[alive]=s.fh(s.bb.norm(t)[:,0]).argmax(-1);return pred,counts
d_=torch.load(ROOT+"/scratch/cache/cifar10_uint8.pt",weights_only=True);sp=json.load(open(ROOT+"/results/splits.json"))
used=set(sp["train"])|set(sp["cal"])|set(sp["val"]);un=sorted(set(range(50000))-used);Ytr=d_["Ytr"];Xtr=d_["Xtr"]
idx=np.random.RandomState(2026).permutation(25000)[10000:10000+N_IMG];Xu=Xtr[[un[i] for i in idx]];Yu=Ytr[[un[i] for i in idx]]
# Exit heads: refit from saved features with the seed-0 recipe of extract_heads.py, then checked against its saved predictions.
os.makedirs(KEEP,exist_ok=True);Yt={s:Ytr[sp[s]] for s in("train","cal","val")};EX=(4,6,8,10)
if SMOKE_RUN:Yt={s:v[::25] for s,v in Yt.items()}
def walk(bb,t,start,names,o):
    for l in range(start+1,13):
        t=bb.blocks[l-1](t)
        if l in EX:o[f"{l}|{pn(ALLS[names[0]],l)}"]=t[:,1:].mean(1)
        gr={}
        for n in names:gr.setdefault(ALLS[n].get(l,0),[]).append(n)
        if len(gr)>1 or 0 not in gr:
            for r,g in gr.items():walk(bb,merge_same(t,r) if r else t,l,g,o)
            return
    for n in names:o[f"final|{n}"]=bb.norm(t)[:,0]
@torch.no_grad()
def features(names):
    # Rebuilds the features the exit heads train on, for the chosen schedules only. Saved so a crash loses little.
    tag="smoke_" if SMOKE_RUN else "";r={};bb=timm.create_model("vit_tiny_patch16_224",pretrained=False,num_classes=0);bb.load_state_dict(torch.load(CK+"ft_shared.pt",weights_only=True,map_location="cpu")["bb"]);bb=bb.to("cuda").eval()
    for s in Yt:
        p=KEEP+f"{tag}{'+'.join(sorted(names))}_{s}.pt"
        if os.path.exists(p):r[s]=torch.load(p,weights_only=True);continue
        ids=sp[s][::25] if SMOKE_RUN else sp[s];X=Xtr[ids];o={};t0=time.time()
        for i in range(0,len(ids),128):
            x=prepare(X[i:i+128],"cuda");e={};walk(bb,embed(bb,x),0,names,e)
            for k,v in e.items():o.setdefault(k,[]).append(v.cpu().half())
        r[s]={k:torch.cat(v) for k,v in o.items()};torch.save(r[s],p+".tmp");os.replace(p+".tmp",p);log("features",s,len(ids),f"{time.time()-t0:.0f}s")
    del bb;torch.cuda.empty_cache();return r
GT=torch.logspace(-1.3,1,200)
def fit(X,y):
    torch.manual_seed(1000);h=nn.Sequential(nn.LayerNorm(192),nn.Linear(192,192),nn.GELU(),nn.Linear(192,10)).to("cuda");X=X.float().to("cuda");y=y.to("cuda");o=torch.optim.AdamW(h.parameters(),lr=2e-3,weight_decay=1e-4);n=len(y);s_=torch.optim.lr_scheduler.OneCycleLR(o,max_lr=2e-3,total_steps=10*int(np.ceil(n/128)),pct_start=.1)
    for _ in range(10):
        p=torch.randperm(n,device="cuda")
        for i in range(0,n,128):
            b=p[i:i+128];l=F.cross_entropy(h(X[b]),y[b]);o.zero_grad(set_to_none=True);l.backward();o.step();s_.step()
    return h.eval()
designs=pick_designs();log("designs",designs)
assert designs,"Block 3 chose nothing to measure"
Ft=features(sorted({parse(d)["sched"] for d,_ in designs}))
PS=dict(np.load(ROOT+"/results/shared_backbone/probs_shared.npz"));hd={};variant=json.load(open(ROOT+"/results/certified/final_variant.json")).get("shared","fthead")
for d,_ in designs:
    q=parse(d)
    for l,_t in q["ex"]:
        if (q["sched"],l) in hd:continue
        key=f"{l}|{pn(ALLS[q['sched']],l)}";h=fit(Ft["train"][key],Yt["train"])
        with torch.no_grad():
            lc=h(Ft["cal"][key].float().to("cuda")).cpu();pv=h(Ft["val"][key].float().to("cuda")).argmax(-1).cpu().numpy()
        T=float(GT[torch.stack([F.cross_entropy(lc/t,Yt["cal"]) for t in GT]).argmin()]);ag=(pv==PS[f"{key}|val"][0][::25 if SMOKE_RUN else 1].argmax(-1)).mean();hd[(q["sched"],l)]=(h,T);log("head",key,"T",round(T,3),"agreement with saved",round(float(ag),4))
fheads={}
for n in {parse(d)["sched"] for d,_ in designs}:
    fh=nn.Linear(192,10)
    if variant=="fthead":fh.load_state_dict(torch.load(CK+"ft_shared.pt",weights_only=True,map_location="cpu")["head"])
    else:
        k=f"final|{n}";torch.manual_seed(1000);X=Ft["train"][k].float().to("cuda");y=Yt["train"].to("cuda");fh=fh.to("cuda");o=torch.optim.AdamW(fh.parameters(),lr=2e-3,weight_decay=1e-4);N=len(y);s_=torch.optim.lr_scheduler.OneCycleLR(o,max_lr=2e-3,total_steps=10*int(np.ceil(N/128)),pct_start=.1)
        for _ in range(10):
            p=torch.randperm(N,device="cuda")
            for i in range(0,N,128):
                b=p[i:i+128];l=F.cross_entropy(fh(X[b]),y[b]);o.zero_grad(set_to_none=True);l.backward();o.step();s_.step()
    fheads[n]=fh.cpu().eval()
del Ft
def backbone(name,dev):
    b=timm.create_model("vit_tiny_patch16_224",pretrained=False,num_classes=0);b.load_state_dict(torch.load(CK+f"ft_{name}.pt",weights_only=True,map_location="cpu")["bb"]);return b.to(dev).eval()
b0_fh=nn.Linear(192,10);b0_fh.load_state_dict(torch.load(CK+"ft_b0.pt",weights_only=True,map_location="cpu")["head"]);b0_fh.eval()
import copy
ctx={}
rapl_ok=make_rapl() is not None or SMOKE_RUN
if not rapl_ok:log("WARNING: CPU energy is not readable, so CPU regimes are skipped")
for dev in("cuda","cpu"):
    if dev=="cpu" and not rapl_ok:continue
    bs=backbone("shared",dev);bz=backbone("b0",dev);hh={k:(copy.deepcopy(v[0]).to(dev).eval(),v[1]) for k,v in hd.items()}
    ms={"B0":M(bz,copy.deepcopy(b0_fh).to(dev),"none",[],hh)}
    for d,_ in designs:
        q=parse(d);ms[d]=M(bs,copy.deepcopy(fheads[q["sched"]]).to(dev),q["sched"],q["ex"],hh)
    x=torch.cat([prepare(Xu[i:i+64],dev) for i in range(0,N_IMG,64)]);ctx[dev]=dict(models=ms,x=x,meter=make_meter(dev),aux=make_rapl() if dev=="cuda" else None,bb=bs)
log("ready: GPU",gpu_state(),"RAPL",rapl_ok)
out=OUT+"runs.csv";rows=[];rng=np.random.RandomState(7)
def summarise():
    R=pd.DataFrame(rows);R.to_csv(out,index=False);res=[]
    for (dv,b),g in R.groupby(["dev","batch"]):
        pv=g.pivot_table(index="rep",columns="design",values="gross_per_sample")
        if "B0" not in pv:continue
        for d in pv.columns:
            if d=="B0":continue
            r=(1-pv[d]/pv["B0"]).dropna()*100;gd=g[g.design==d];res.append(dict(dev=dv,batch=b,design=d,reps=len(r),saved_pct=r.mean(),sd=r.std(),ci95={2:12.71,3:4.303,4:3.182,5:2.776,6:2.571}.get(len(r),np.nan)*r.std()/np.sqrt(len(r)) if len(r)>1 else np.nan,acc=gd.acc.mean(),b0_acc=g[g.design=="B0"].acc.mean(),j_per_image=gd.gross_per_sample.mean(),b0_j=g[g.design=="B0"].gross_per_sample.mean(),temp_max=gd.temp_end.max() if "temp_end" in gd else np.nan))
    S=pd.DataFrame(res);S.to_csv(OUT+"summary.csv",index=False);return S
stop=False
for rep in range(REPS):
    for dev,batch in REGIMES:
        if dev not in ctx:continue
        c=ctx[dev];names=list(c["models"])
        for i in rng.permutation(len(names)):
            if time.time()>DEADLINE:stop=True;break
            nm=names[i];t0=time.time()
            if dev=="cuda":
                while gpu_state().get("temp_c",0)>COOL_C and time.time()-t0<COOL_MAX:time.sleep(3)
            else:time.sleep(3)
            start=gpu_state();time.sleep(1)
            r=real_once(c["models"][nm],0.,c["x"],Yu,batch,dev,c["meter"],c["aux"],WINDOW);end=gpu_state()
            rows.append(dict(rep=rep,dev=dev,batch=batch,design=nm,temp_start=start.get("temp_c"),temp_end=end.get("temp_c"),clock_end=end.get("clock_mhz"),slowdown=end.get("sw_thermal_slowdown"),waited_s=time.time()-t0,**r));summarise()
            log(rep,dev,batch,nm,"J/img",round(r["gross_per_sample"],4),"acc",r["acc"],"temp",start.get("temp_c"),end.get("temp_c"))
        if stop:break
    if stop:break
S=summarise();print(S.round(3).to_string(index=False));log("BLOCK4 DONE",len(rows),"measurements","stopped at deadline" if stop else "all repeats finished")
