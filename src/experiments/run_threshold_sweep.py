"""Reviewer check: is the 2-3x EOR/routine ratio an artifact of the equal-error
operating point? Sweep the decision threshold over the deployable range on the
FM (bce) predictions and report EOR, routine, ratio, FGR, accuracy at each.
Output: results/table64_threshold_sweep.json"""
import json, os, sys
import numpy as np
sys.path.insert(0, '.')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import metrics as M
import baselines_ext as B
from data import build_records, per_user_folds
from collections import defaultdict

records, _ = build_records(); folds = per_user_folds(records, 5, 42)
P, Y, U, MU = [], [], [], {}
for f in range(5):
    tr = [records[i] for i in range(len(records)) if folds[i] != f]
    te = [records[i] for i in range(len(records)) if folds[i] == f]
    pr = B.fm_predict(tr, te, objective="bce")
    ur = defaultdict(list)
    for r in tr: ur[r["user_id"]].append(r["label"])
    mu = {u: (1 if np.mean(v) >= 0.5 else 0) for u, v in ur.items()}
    for r, p in zip(te, pr["p_allow"]):
        P.append(p); Y.append(r["label"]); U.append(r["user_id"]); MU[(f, r["user_id"])] = mu.get(r["user_id"])
    for i, r in enumerate(te): pass
    # store per-decision mu
    if f == 0: MUlist = []
for f in range(5): pass
# recompute mu per decision aligned
P = np.asarray(P); Y = np.asarray(Y)
# rebuild mu per decision (train-fold mu stored above per fold)
mu_dec = []
idx = 0
for f in range(5):
    te = [records[i] for i in range(len(records)) if folds[i] == f]
    tr = [records[i] for i in range(len(records)) if folds[i] != f]
    ur = defaultdict(list)
    for r in tr: ur[r["user_id"]].append(r["label"])
    mu = {u: (1 if np.mean(v) >= 0.5 else 0) for u, v in ur.items()}
    for r in te: mu_dec.append(mu[r["user_id"]])
mu_dec = np.asarray(mu_dec)
is_exc = (Y != mu_dec)
out = {}
for thr in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    yh = (P >= thr).astype(int)
    ov = (yh != Y).astype(float)
    eor = float(100 * ov[is_exc].mean()); rou = float(100 * ov[~is_exc].mean())
    fgr = float(100 * yh[Y == 0].mean()); acc = float(100 * (yh == Y).mean())
    out[f"thr={thr}"] = dict(eor=round(eor,1), routine=round(rou,1),
                             ratio=round(eor/max(rou,1e-9),2), fgr=round(fgr,1), acc=round(acc,1))
    print(f"thr={thr}: EOR={eor:.1f} routine={rou:.1f} ratio={eor/rou:.2f} FGR={fgr:.1f} acc={acc:.1f}")
rd = os.environ.get("PERM_RESULTS_DIR", "../results")
json.dump(out, open(os.path.join(rd, "table64_threshold_sweep.json"), "w"), indent=1)
print("saved table64_threshold_sweep.json")
