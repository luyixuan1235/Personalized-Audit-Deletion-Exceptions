"""Exception stability (reviewer request): does an exception recur?
P(held-out decision is an exception | user has a training-fold exception on the same cell)
vs the base rate. Cells: (domain), (tool, data-type). -> results/table38_exception_stability.json"""
import numpy as np, sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import build_records, per_user_folds
records,_=build_records(); folds=per_user_folds(records,5,42)
res={}
for cellname,key in [("domain",lambda r:(r["user_id"],r["domain"])),
                     ("tool_dtype",lambda r:(r["user_id"],r["tool"],r["data_type"]))]:
    with_p, wo_p, base = [],[],[]
    for f in range(5):
        tr=[records[i] for i in range(len(records)) if folds[i]!=f]
        te=[records[i] for i in range(len(records)) if folds[i]==f]
        ur={}
        for r in tr: ur.setdefault(r["user_id"],[]).append(r["label"])
        mu={u:(1 if np.mean(v)>=0.5 else 0) for u,v in ur.items()}
        prec={key(r) for r in tr if r["user_id"] in mu and r["label"]!=mu[r["user_id"]]}
        for r in te:
            u=r["user_id"]
            if u not in mu: continue
            is_exc=(r["label"]!=mu[u]); base.append(is_exc)
            (with_p if key(r) in prec else wo_p).append(is_exc)
    res[cellname]=dict(p_exc_given_precedent=round(100*float(np.mean(with_p)),1), n_with=len(with_p),
                       p_exc_no_precedent=round(100*float(np.mean(wo_p)),1), n_without=len(wo_p),
                       base_rate=round(100*float(np.mean(base)),1),
                       lift=round(float(np.mean(with_p)/np.mean(base)),2))
if __name__=='__main__':
    json.dump(res, open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'..','results','table38_exception_stability.json') if False else '../results/table38_exception_stability.json','w'), indent=1)
    print(json.dumps(res,indent=1))
