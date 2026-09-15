"""Does the 'unexplained residual' decompose into the novel-exception floor?
Deregularized, converged FM (wd=1e-6, 2000 epochs). Stratify exception override by the
same-user precedent count k at two granularities. Prediction: the residual concentrates on
k=0 (novel) exceptions; precedented exceptions approach the regularized model's k>=7 floor.
Output: results/table45_residual_decomp.json"""
import numpy as np, sys, os, json, gc
sys.path.insert(0,'.'); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import baselines_ext as B, metrics as M
from data import build_records, per_user_folds
from collections import defaultdict
records,_=build_records(); folds=per_user_folds(records,5,42)
buckets=defaultdict(list); agg=[]
for f in range(5):
    tr=[records[i] for i in range(len(records)) if folds[i]!=f]
    te=[records[i] for i in range(len(records)) if folds[i]==f]
    pr=B.fm_predict(tr,te,objective="bce",wd=1e-6,epochs=2000)
    prtr=B.fm_predict(tr,tr,objective="bce",wd=1e-6,epochs=2000)
    y=np.asarray(pr["y"]); p=np.asarray(pr["p_allow"])
    ytr=np.asarray(prtr["y"]); ptr=np.asarray(prtr["p_allow"])
    thr=float(M.equal_error_threshold(ytr,ptr)); yh=(p>=thr).astype(int)
    ur={}
    for r in tr: ur.setdefault(r["user_id"],[]).append(r["label"])
    mu={u:(1 if np.mean(v)>=0.5 else 0) for u,v in ur.items()}
    # same-user, same-direction exception precedents at two granularities
    cnt_dom=defaultdict(int); cnt_cell=defaultdict(int)
    for r in tr:
        u=r["user_id"]
        if u in mu and r["label"]!=mu[u]:
            cnt_dom[(u,r["domain"],r["label"])]+=1
            cnt_cell[(u,r["tool"],r["data_type"],r["label"])]+=1
    for i,r in enumerate(te):
        u=r["user_id"]
        if u not in mu or y[i]==mu[u]: continue
        ov=float(yh[i]!=y[i]); agg.append(ov)
        kd=cnt_dom.get((u,r["domain"],y[i]),0)
        kc=cnt_cell.get((u,r["tool"],r["data_type"],y[i]),0)
        buckets[("domain","k=0" if kd==0 else ("k=1-2" if kd<=2 else "k>=3"))].append(ov)
        buckets[("cell","k=0" if kc==0 else "k>=1")].append(ov)
    print(f"fold {f+1}/5 done", flush=True)
    del pr,prtr; gc.collect()
out={"aggregate_exception_override":round(100*float(np.mean(agg)),1), "n_exceptions":len(agg)}
for (g,k),v in sorted(buckets.items()):
    out[f"{g}/{k}"]=dict(override=round(100*float(np.mean(v)),1), n=len(v))
json.dump(out, open('../results/table45_residual_decomp.json','w'), indent=1)
print(json.dumps(out,indent=1))
