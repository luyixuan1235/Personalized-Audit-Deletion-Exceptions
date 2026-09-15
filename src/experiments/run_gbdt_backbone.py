"""Cross-backbone check (reviewer request): a non-neural family (boosted trees).
Output: results/table40_gbdt.json"""
import numpy as np, sys, os, json, gc
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import baselines_ext as B, metrics as M
from data import build_records, per_user_folds
records,_=build_records(); folds=per_user_folds(records,5,42)
Y,P,MU,THR=[],[],[],[]
for f in range(5):
    tr=[records[i] for i in range(len(records)) if folds[i]!=f]
    te=[records[i] for i in range(len(records)) if folds[i]==f]
    pr=B.gbdt_predict(tr,te); prtr=B.gbdt_predict(tr,tr)
    THR.append(float(M.equal_error_threshold(np.asarray(prtr["y"]),np.asarray(prtr["p_allow"]))))
    ur={}
    for r in tr: ur.setdefault(r["user_id"],[]).append(r["label"])
    mu={u:(1 if np.mean(v)>=0.5 else 0) for u,v in ur.items()}
    Y+=list(pr["y"]); P+=list(pr["p_allow"]); MU+=[mu.get(r["user_id"],1) for r in te]
    gc.collect()
y=np.array(Y); p=np.array(P); mu=np.array(MU); thr=float(np.mean(THR))
yh=(p>=thr).astype(int); ov=(yh!=y); exc=(y!=mu); pr_=(mu==1)&(y==0)
res=dict(acc=round(100*float((yh==y).mean()),1), exc=round(100*float(ov[exc].mean()),1),
         routine=round(100*float(ov[~exc].mean()),1),
         ratio=round(float(ov[exc].mean()/ov[~exc].mean()),2),
         perm_refusals=round(100*float(ov[pr_].mean()),1))
json.dump(res, open('../results/table40_gbdt.json','w'), indent=1)
print(res)
