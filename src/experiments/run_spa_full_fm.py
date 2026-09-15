"""Reference: full-SPA FM/bce EOR ratio, same protocol as the scale gradient.
Output: results_spa/table71_spa_full_fm.json"""
import json, os, sys
import numpy as np
sys.path.insert(0,'.')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import metrics as M
import baselines_ext as B
from data import build_records, per_user_folds
from collections import defaultdict
records,_=build_records(); folds=per_user_folds(records,5,42)
EXC,ROU=[],[]
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
        (EXC if r["label"]!=mu[r["user_id"]] else ROU).append(float((p>=thr)!=r["label"]))
    print(f"fold {f} done", flush=True)
out=dict(eor=round(100*float(np.mean(EXC)),1),routine=round(100*float(np.mean(ROU)),1),
         ratio=round(float(np.mean(EXC)/max(np.mean(ROU),1e-9)),2),n_exc=len(EXC),n=len(records))
print(out)
json.dump(out,open(os.environ.get("PERM_RESULTS_DIR","../results_spa")+"/table71_spa_full_fm.json","w"),indent=1)
print("saved table71_spa_full_fm.json")
