"""Completes the decision layer against the novelty floor: defer FIRST-OF-KIND requests.
Triggers compared at matched coverage: margin-only | novelty-only | margin+novelty hybrid.
Novelty = user has NO training record on this (tool,dtype) [any label].
Output: results/table46_novelty_defer.json"""
import numpy as np, sys, os, json, gc
sys.path.insert(0,'.'); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import baselines_ext as B, metrics as M
from data import build_records, per_user_folds
records,_=build_records(); folds=per_user_folds(records,5,42)
pool={"y":[],"p":[],"mu":[],"novel":[],"dom_novel":[],"thr":[]}
for f in range(5):
    tr=[records[i] for i in range(len(records)) if folds[i]!=f]
    te=[records[i] for i in range(len(records)) if folds[i]==f]
    pr=B.fm_predict(tr,te,objective="bce"); prtr=B.fm_predict(tr,tr,objective="bce")
    y=np.asarray(pr["y"]); p=np.asarray(pr["p_allow"])
    pool["thr"].append(float(M.equal_error_threshold(np.asarray(prtr["y"]),np.asarray(prtr["p_allow"]))))
    ur={}
    for r in tr: ur.setdefault(r["user_id"],[]).append(r["label"])
    mu={u:(1 if np.mean(v)>=0.5 else 0) for u,v in ur.items()}
    seen=set((r["user_id"],r["tool"],r["data_type"]) for r in tr)
    seend=set((r["user_id"],r["domain"]) for r in tr)
    pool["y"]+=list(y); pool["p"]+=list(p)
    pool["mu"]+=[mu.get(r["user_id"],1) for r in te]
    pool["novel"]+=[(r["user_id"],r["tool"],r["data_type"]) not in seen for r in te]
    pool["dom_novel"]+=[(r["user_id"],r["domain"]) not in seend for r in te]
    del pr,prtr; gc.collect()
    print(f"fold {f+1}/5", flush=True)
y=np.array(pool["y"]); p=np.array(pool["p"]); mu=np.array(pool["mu"])
novel=np.array(pool["novel"]); thr=float(np.mean(pool["thr"]))
yh=(p>=thr).astype(int); ov=(yh!=y); exc=(y!=mu); margin=np.abs(p-thr)
print(f"novel share of test decisions: {100*novel.mean():.1f}%  | novel share of exceptions: {100*novel[exc].mean():.1f}%")

def report(defer, label):
    auto=~defer
    dn=auto&(y==0); pr_=exc&(y==0)
    r=dict(coverage=round(100*float(auto.mean()),1),
           exception=round(100*float(ov[auto&exc].mean()),1),
           routine=round(100*float(ov[auto&~exc].mean()),1),
           perm_refusals=round(100*float(ov[auto&pr_].mean()),1) if (auto&pr_).any() else None,
           fgr=round(100*float(yh[dn].mean()),1) if dn.any() else None)
    print(f"{label:34s} {r}", flush=True)
    return r

out={}
out["none"]=report(np.zeros(len(y),bool),"no deferral")
out["novelty_cell"]=report(novel,"novelty only (tool,dtype)")
# domain-level novelty: user has no record in this domain
dom_novel=np.array(pool["dom_novel"])
out["novelty_domain"]=report(dom_novel,"novelty only (domain)")
# hybrid @40%: defer the lowest-margin decisions AMONG the novel ones, 40% total
q=np.quantile(margin[novel], min(1.0, 0.40/max(1e-9,novel.mean())))
out["novel_and_margin_40"]=report(novel&(margin<=q),"novel AND low-margin @40%")
out["margin_40"]=report(margin<=np.quantile(margin,0.40),"margin only @40%")
out["_novel_share"]={"decisions":round(100*float(novel.mean()),1),"exceptions":round(100*float(novel[exc].mean()),1),
                     "domain_novel_decisions":round(100*float(dom_novel.mean()),1)}
json.dump(out, open('../results/table46_novelty_defer.json','w'), indent=1)
print("saved table46_novelty_defer.json")
