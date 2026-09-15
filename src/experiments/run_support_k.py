"""理论的可证伪预言: 例外覆盖率应随 k (该用户在该属性上的例外支持数) 单调下降."""
import numpy as np, sys, gc
from collections import defaultdict
sys.path.insert(0,'.')
import metrics as M, baselines_ext as B
from data import build_records, per_user_folds
def sp(a,b):
    ra=np.argsort(np.argsort(a)).astype(float); rb=np.argsort(np.argsort(b)).astype(float)
    ra=(ra-ra.mean())/(ra.std()+1e-9); rb=(rb-rb.mean())/(rb.std()+1e-9); return float((ra*rb).mean())
records,_=build_records(); folds=per_user_folds(records,5,42)
K,OV=[],[]
for f in range(5):
    tr=[records[i] for i in range(len(records)) if folds[i]!=f]
    te=[records[i] for i in range(len(records)) if folds[i]==f]
    pr=B.fm_predict(tr,te,objective="bce"); prtr=B.fm_predict(tr,tr,objective="bce")
    y=np.asarray(pr["y"]); p=np.asarray(pr["p_allow"])
    thr=float(M.equal_error_threshold(np.asarray(prtr["y"]),np.asarray(prtr["p_allow"])))
    yh=(p>=thr).astype(int)
    ur=defaultdict(lambda:[0,0])
    for r in tr:
        a=ur[r["user_id"]]; a[0]+=r["label"]; a[1]+=1
    mu={u:(1 if a[0]/max(a[1],1)>=0.5 else 0) for u,a in ur.items()}
    # k = 该用户在训练集里、该请求的 domain 上、同方向的例外数
    cnt=defaultdict(int)
    for r in tr:
        u=r["user_id"]
        if u in mu and r["label"]!=mu[u]:
            cnt[(u,r["domain"])]+=1
    u_=np.array([r["user_id"] for r in te])
    urate={uu: float(y[u_==uu].mean()) for uu in np.unique(u_)}
    m_=np.array([1 if urate[uu]>=0.5 else 0 for uu in u_])
    for i,r in enumerate(te):
        if y[i]==m_[i]: continue          # 只看例外
        K.append(cnt[(r["user_id"],r["domain"])])
        OV.append(float(yh[i]!=y[i]))
    del pr,prtr; gc.collect()
K,OV=np.array(K,float),np.array(OV)
_OUT={'n_exceptions':int(len(K)),'bins':{}}
print(f"测试集例外 {len(K)} 个\n")
print(f"{'k (该用户在该domain上的例外支持数)':>36}{'n':>7}{'例外被覆盖':>12}")
for lo,hi,lab in ((0,0,'k=0 (无支持)'),(1,1,'k=1'),(2,3,'k=2-3'),(4,6,'k=4-6'),(7,99,'k>=7')):
    m=(K>=lo)&(K<=hi)
    if m.sum()<15: continue
    print(f"{lab:>36}{m.sum():7d}{100*OV[m].mean():11.1f}%")
    _OUT['bins'][lab]=dict(n=int(m.sum()),override=float(100*OV[m].mean()))
r=sp(-K,OV)
rng=np.random.default_rng(0)
null=np.array([sp(-K,rng.permutation(OV)) for _ in range(5000)])
p_=float((null>=r).mean())
print(f"\nrho(支持数 k 越小, 例外越易被覆盖) = {r:+.3f}   p = {p_:.4f}")
_OUT['rho']=float(sp(K,OV)); _OUT['p']=p_
import json as _json, os as _os
_json.dump(_OUT, open(_os.path.join(_os.environ.get("PERM_RESULTS_DIR","../results"),"table50_support_k.json"),"w"), indent=1)
print("saved table50_support_k.json")
print("=> " + ("✓ 理论预言成立 —— 支持越少, 例外越易被抹掉" if p_<0.01
               else "✗ 理论预言不成立"))
