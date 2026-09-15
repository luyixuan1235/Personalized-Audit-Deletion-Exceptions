"""tab:defer signed row: calibrated signed model, margin deferral at 40% budget.
Run: cd src && python3 experiments/run_signed_defer.py
Output: results/table55_signed_defer.json"""
import json, os, sys
import numpy as np
sys.path.insert(0,'.')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import metrics as M
import baselines_ext as B
from data import build_records, per_user_folds, Graph
from train import predict
records,_=build_records(); folds=per_user_folds(records,5,42)
gr={}
for r in records: gr.setdefault(r["user_id"],[]).append(r["label"])
gr={u:float(np.mean(v)) for u,v in gr.items()}
mu={u:(1 if g>=0.5 else 0) for u,g in gr.items()}
out={}
FULL,EXC40,FG40,EXC=[],[],[],[]
for f in range(5):
    tr=[records[i] for i in range(len(records)) if folds[i]!=f]
    te=[records[i] for i in range(len(records)) if folds[i]==f]
    g=Graph(tr,records)
    m=B.train_signed(g,tr,use_sideinfo=True,objective="bce")
    pr=predict(m,g,te); ptr=predict(m,g,tr)
    y=np.asarray(pr["y"]); p=np.asarray(pr["p_allow"])
    thr=float(M.equal_error_threshold(np.asarray(ptr["y"]),np.asarray(ptr["p_allow"])))
    yh=(p>=thr).astype(int)
    exc=np.array([r["label"]!=mu[r["user_id"]] for r in te])
    ov=(yh!=y)
    EXC+=list(ov[exc].astype(float))
    margin=np.abs(p-thr); cut=np.quantile(margin,0.40); auto=margin>cut
    if (auto&exc).sum(): EXC40+=list(ov[auto&exc].astype(float))
    dn=auto&(y==0)
    if dn.sum(): FG40+=list(yh[dn].astype(float))
    print(f"fold {f} done", flush=True)
out=dict(full_coverage_eor=float(100*np.mean(EXC)),
         defer40_exception=float(100*np.mean(EXC40)),
         defer40_fgr=float(100*np.mean(FG40)))
print(out)
rd=os.environ.get("PERM_RESULTS_DIR","../results")
json.dump(out, open(os.path.join(rd,"table55_signed_defer.json"),"w"), indent=1)
print("saved table55_signed_defer.json")
