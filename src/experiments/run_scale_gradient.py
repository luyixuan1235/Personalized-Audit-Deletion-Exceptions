"""Scale gradient on SPA: does the bilinear habit force track total corpus size?
Random user subsets (composition preserved) at 458/700/1000/1400, plus the full
corpus reference. Output: results_spa/table70_scale_gradient.json"""
import json, os, sys
import numpy as np
sys.path.insert(0,'.')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import metrics as M
import baselines_ext as B
from data import build_records, per_user_folds
from collections import defaultdict

records,_=build_records()
users=sorted({r["user_id"] for r in records})
rng=np.random.default_rng(42)
out={}
for n in (458,700,1000,1400):
    sub=set(rng.choice(users,size=n,replace=False))
    recs=[r for r in records if r["user_id"] in sub]
    folds=per_user_folds(recs,5,42)
    EXC,ROU=[],[]
    for f in range(5):
        tr=[recs[i] for i in range(len(recs)) if folds[i]!=f]
        te=[recs[i] for i in range(len(recs)) if folds[i]==f]
        pr=B.fm_predict(tr,te,objective="bce"); ptr=B.fm_predict(tr,tr,objective="bce")
        thr=float(M.equal_error_threshold(np.asarray(ptr["y"]),np.asarray(ptr["p_allow"])))
        ur=defaultdict(list)
        for r in tr: ur[r["user_id"]].append(r["label"])
        mu={u:(1 if np.mean(v)>=0.5 else 0) for u,v in ur.items()}
        for r,p in zip(te,pr["p_allow"]):
            if r["user_id"] not in mu: continue
            ov=float((p>=thr)!=r["label"])
            (EXC if r["label"]!=mu[r["user_id"]] else ROU).append(ov)
    out[f"n={n}"]=dict(eor=round(100*float(np.mean(EXC)),1),
                       routine=round(100*float(np.mean(ROU)),1),
                       ratio=round(float(np.mean(EXC)/max(np.mean(ROU),1e-9)),2),
                       n_dec=len(recs))
    print(f"n={n}:", out[f"n={n}"], flush=True)
rd=os.environ.get("PERM_RESULTS_DIR","../results_spa")
json.dump(out, open(os.path.join(rd,"table70_scale_gradient.json"),"w"), indent=1)
print("saved table70_scale_gradient.json")
