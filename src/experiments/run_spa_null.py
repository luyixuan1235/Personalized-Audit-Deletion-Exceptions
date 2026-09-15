"""SPA counterpart of the within-user permutation null: does the FM's 2.4x exceed
what label imbalance alone would produce on SPA? Same construction as Wu
(run_review_stats E3): permute each user's error indicators within-user, holding
error counts fixed. Output: results_spa/table72_spa_null.json"""
import json, os, sys
import numpy as np
sys.path.insert(0,'.')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import metrics as M
import baselines_ext as B
from data import build_records, per_user_folds
from collections import defaultdict

records,_=build_records(); folds=per_user_folds(records,5,42)
U,Y,OV,MU=[],[],[],[]
for f in range(5):
    tr=[records[i] for i in range(len(records)) if folds[i]!=f]
    te=[records[i] for i in range(len(records)) if folds[i]==f]
    pr=B.fm_predict(tr,te,objective="bce"); ptr=B.fm_predict(tr,tr,objective="bce")
    thr=float(M.equal_error_threshold(np.asarray(ptr["y"]),np.asarray(ptr["p_allow"])))
    ur=defaultdict(list)
    for r in tr: ur[r["user_id"]].append(r["label"])
    mu={u:(1 if np.mean(v)>=0.5 else 0) for u,v in ur.items()}
    for r,p in zip(te,pr["p_allow"]):
        if r["user_id"] not in mu: continue
        U.append(r["user_id"]); Y.append(r["label"])
        OV.append(float((p>=thr)!=r["label"])); MU.append(mu[r["user_id"]])
    print(f"fold {f} done", flush=True)
U=np.array(U); Y=np.array(Y); OV=np.array(OV); MU=np.array(MU)
exc=(Y!=MU)
obs=OV[exc].mean()/OV[~exc].mean()
rng=np.random.default_rng(0)
uu=np.unique(U); idx_by_u={x:np.where(U==x)[0] for x in uu}
nulls=[]
for t in range(500):
    ovp=OV.copy()
    for x in uu:
        ix=idx_by_u[x]; ovp[ix]=rng.permutation(ovp[ix])
    e=ovp[exc].mean(); r=ovp[~exc].mean()
    nulls.append(e/r)
    if (t+1)%100==0: print(f"perm {t+1}/500", flush=True)
N=np.array(nulls)
out=dict(observed=round(float(obs),2), null_mean=round(float(N.mean()),2),
         null_95=[round(float(q),2) for q in np.percentile(N,[2.5,97.5])],
         p=round(float((N>=obs).mean()),4), n_exc=int(exc.sum()))
print(out)
json.dump(out,open(os.environ.get("PERM_RESULTS_DIR","../results_spa")+"/table72_spa_null.json","w"),indent=1)
print("saved table72_spa_null.json")
