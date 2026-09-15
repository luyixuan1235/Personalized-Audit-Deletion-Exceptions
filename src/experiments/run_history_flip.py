"""#2b: paired counterfactual histories. Flip 20 random users' TRAINING labels
(grant<->deny), retrain, compare their held-out predictions with the baseline
model: do predictions track the inverted majority? Controls: untouched users.
Output: results/table58_history_flip.json"""
import json, os, sys
import numpy as np
sys.path.insert(0,'.')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import metrics as M
import baselines_ext as B
from data import build_records, per_user_folds

records,_=build_records(); folds=per_user_folds(records,5,42)
rng=np.random.default_rng(42)
users=sorted({r["user_id"] for r in records})
flip=set(rng.choice(users,size=20,replace=False))
f=0
tr=[records[i] for i in range(len(records)) if folds[i]!=f]
te=[records[i] for i in range(len(records)) if folds[i]==f]
tr_flip=[dict(r,label=1-r["label"]) if r["user_id"] in flip else r for r in tr]
out={}
pb=B.fm_predict(tr,te,objective="bce"); pf=B.fm_predict(tr_flip,te,objective="bce")
p0=np.asarray(pb["p_allow"]); p1=np.asarray(pf["p_allow"])
u=np.array([r["user_id"] for r in te])
mf=np.isin(u,list(flip))
out["flipped_users"]=dict(
    n_users=20, n_test=int(mf.sum()),
    mean_p_before=float(p0[mf].mean()), mean_p_after=float(p1[mf].mean()),
    mean_shift=float((p1[mf]-p0[mf]).mean()))
out["control_users"]=dict(
    n_test=int((~mf).sum()),
    mean_p_before=float(p0[~mf].mean()), mean_p_after=float(p1[~mf].mean()),
    mean_abs_shift=float(np.abs(p1[~mf]-p0[~mf]).mean()))
# per flipped user: does prediction majority follow the inverted history majority?
gr={}
for r in tr: gr.setdefault(r["user_id"],[]).append(r["label"])
followed=0
for uu in flip:
    m0=1 if np.mean(gr[uu])>=0.5 else 0
    mm=(u==uu)
    if mm.sum()==0: continue
    pred_maj_after=1 if (p1[mm]>=0.5).mean()>=0.5 else 0
    if pred_maj_after==1-m0: followed+=1
out["flipped_pred_majority_follows_inverted_history"]=f"{followed}/20"
print(json.dumps(out,indent=1))
json.dump(out, open(os.path.join(os.environ.get("PERM_RESULTS_DIR","../results"),"table58_history_flip.json"),"w"), indent=1)
print("saved table58_history_flip.json")
