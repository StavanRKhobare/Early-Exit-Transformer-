# Does a larger selection split remove the overshoot? Static exit + merge, linear head, seed 0, GPU float16 batch 32 cost ordering.
# Selection sizes drawn from 10,000 pooled images; a separate random 4,000 is the report set in each of 300 repeats. Cost-table predictions only.
import os, numpy as np, pandas as pd
_src=open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"ens.py")).read().split("scs=[S for S")[0]
_file=__file__
exec(_src)
df=pd.read_pickle("ens_cfg.pkl")
y=Ys; b0v=(sig("none__final__lin")["val_pred"]==y["val"]).mean()
yall=np.r_[y["val"],y["test"]]
base=np.r_[sig("none__final__lin")["val_pred"][0],sig("none__final__lin")["test_pred"][0]]==yall
d=df[df.mg&(df.fin=="lin")&(df.w==0)&(df.val>=b0v-.03)].sort_values("c_cuda_laptop_32").reset_index(drop=True)
C=np.stack([np.r_[run(r.g,r.tau,r.S,r.L,r.fin,r.w,"val")[0][0],run(r.g,r.tau,r.S,r.L,r.fin,r.w,"test")[0][0]] for r in d.itertuples()])
Di=C.astype(np.int8)-base.astype(np.int8)
cost=d["c_cuda_laptop_32"].values;b0=b0c("cuda_laptop",32,"test")
print("candidates",len(C),"| B0 accuracy noise pts at n=1000,4000,10000:",[round(100*np.sqrt(.92*.08/n),2) for n in(1000,4000,10000)])
# paired noise of a typical design against B0
i=np.argmin(np.abs(Di.mean(1)+.009));print("paired noise, design near -0.9 pts, n=4000:",round(100*Di[i].std()/np.sqrt(4000),2),"pts")
true=Di.mean(1)  # best available estimate of each design's real gap, from all 14,000
rng=np.random.default_rng(1);rows=[]
for eps in(.0025,.005,.01):
  for n in(1000,2000,4000,7000,10000):
    for rule,z in(("naive",0),("margin z=1.64",1.645)):
      v=[];sv=[];rep=[];tr=[]
      for _ in range(300):
        p=rng.permutation(14000);r,pool=p[:4000],p[4000:];sel=pool[:n]
        ds=Di[:,sel];ok=ds.mean(1)-z*ds.std(1)/np.sqrt(n)>=-eps
        if not ok.any():v.append(0);sv.append(0);rep.append(0);tr.append(0);continue
        k=np.where(ok)[0][0];g=Di[k,r].mean();v.append(g<-eps);sv.append(100*(1-cost[k]/b0));rep.append(100*g);tr.append(100*true[k])
      rows.append(dict(eps_pts=eps*100,n_select=n,rule=rule,overshoot_pct=100*np.mean(v),saved=np.mean(sv),report_diff=np.mean(rep),pool14k_diff=np.mean(tr)))
o=pd.DataFrame(rows);o.to_csv("split_size_test.csv",index=False);print(o.round(2).to_string(index=False))
