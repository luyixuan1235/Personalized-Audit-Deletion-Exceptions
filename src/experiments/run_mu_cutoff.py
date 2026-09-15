"""Reviewer: does the gap depend on the 0.5 majority cutoff? Redefine
m_u = 1[grant_rate_u >= c] for c in {0.3,...,0.7} and recompute EOR/routine on
the FM (bce) predictions. Output: results/table65_mu_cutoff.json"""
import json, os, sys
import numpy as np
sys.path.insert(0, '.')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import metrics as M
import baselines_ext as B
from data import build_records, per_user_folds
from collections import defaultdict

records, _ = build_records(); folds = per_user_folds(records, 5, 42)
P, Y, GR = [], [], []
for f in range(5):
    tr = [records[i] for i in range(len(records)) if folds[i] != f]
    te = [records[i] for i in range(len(records)) if folds[i] == f]
    pr = B.fm_predict(tr, te, objective="bce"); ptr = B.fm_predict(tr, tr, objective="bce")
    thr = float(M.equal_error_threshold(np.asarray(ptr["y"]), np.asarray(ptr["p_allow"])))
    ur = defaultdict(list)
    for r in tr: ur[r["user_id"]].append(r["label"])
    g = {u: float(np.mean(v)) for u, v in ur.items()}
    for r, p in zip(te, pr["p_allow"]):
        P.append(float(p >= thr)); Y.append(r["label"]); GR.append(g.get(r["user_id"], 0.5))
P, Y, GR = map(np.asarray, (P, Y, GR))
ov = (P != Y).astype(float)
out = {}
for c in (0.3, 0.4, 0.5, 0.6, 0.7):
    mu = (GR >= c).astype(int)
    exc = Y != mu
    out[f"cutoff={c}"] = dict(eor=round(float(100*ov[exc].mean()),1),
                              routine=round(float(100*ov[~exc].mean()),1),
                              ratio=round(float(ov[exc].mean()/max(ov[~exc].mean(),1e-9)),2),
                              n_exc=int(exc.sum()))
    print(f"c={c}: EOR={out[f'cutoff={c}']['eor']} routine={out[f'cutoff={c}']['routine']} "
          f"ratio={out[f'cutoff={c}']['ratio']} n={out[f'cutoff={c}']['n_exc']}", flush=True)
rd = os.environ.get("PERM_RESULTS_DIR", "../results")
json.dump(out, open(os.path.join(rd, "table65_mu_cutoff.json"), "w"), indent=1)
print("saved table65_mu_cutoff.json")
