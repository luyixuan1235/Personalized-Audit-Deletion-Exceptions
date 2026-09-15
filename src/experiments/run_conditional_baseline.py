"""Reviewer check: is high EOR just a missing context-conditioned rule?
Redefine the exception set against per-user CONDITIONAL majorities
(m_{u,data_type}, training-fold, fallback m_u where the cell is unseen) and
recompute the override rates on the same FM (bce) predictions. Also report the
conditional-majority baseline's own accuracy vs the flat habit baseline.
Output: results/table62_conditional_baseline.json"""
import json, os, sys
import numpy as np
sys.path.insert(0, '.')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import metrics as M
import baselines_ext as B
from data import build_records, per_user_folds
from collections import defaultdict

records, _ = build_records(); folds = per_user_folds(records, 5, 42)
EXC_f, ROU_f, EXC_c, ROU_c = [], [], [], []
acc_flat, acc_cond = [], []
n_cell = 0; n_tot = 0
for f in range(5):
    tr = [records[i] for i in range(len(records)) if folds[i] != f]
    te = [records[i] for i in range(len(records)) if folds[i] == f]
    pr = B.fm_predict(tr, te, objective="bce"); ptr = B.fm_predict(tr, tr, objective="bce")
    thr = float(M.equal_error_threshold(np.asarray(ptr["y"]), np.asarray(ptr["p_allow"])))
    y = np.asarray(pr["y"]); yh = (np.asarray(pr["p_allow"]) >= thr).astype(int)
    ur = defaultdict(list); uc = defaultdict(list)
    for r in tr:
        ur[r["user_id"]].append(r["label"])
        uc[(r["user_id"], r["data_type"])].append(r["label"])
    mu = {u: (1 if np.mean(v) >= 0.5 else 0) for u, v in ur.items()}
    mc = {k: (1 if np.mean(v) >= 0.5 else 0) for k, v in uc.items()}
    for i, r in enumerate(te):
        u = r["user_id"]
        if u not in mu: continue
        key = (u, r["data_type"]); n_tot += 1
        m_cond = mc.get(key, mu[u])
        if key in mc: n_cell += 1
        ov = float(yh[i] != y[i])
        (EXC_f if y[i] != mu[u] else ROU_f).append(ov)
        (EXC_c if y[i] != m_cond else ROU_c).append(ov)
        acc_flat.append(float(mu[u] == y[i])); acc_cond.append(float(m_cond == y[i]))
out = dict(
    flat=dict(eor=float(100*np.mean(EXC_f)), routine=float(100*np.mean(ROU_f)),
              ratio=float(np.mean(EXC_f)/np.mean(ROU_f)), n_exc=len(EXC_f),
              baseline_acc=float(100*np.mean(acc_flat))),
    conditional=dict(eor=float(100*np.mean(EXC_c)), routine=float(100*np.mean(ROU_c)),
                     ratio=float(np.mean(EXC_c)/np.mean(ROU_c)), n_exc=len(EXC_c),
                     baseline_acc=float(100*np.mean(acc_cond)),
                     cell_coverage=float(n_cell/n_tot)))
print(json.dumps(out, indent=1))
rd = os.environ.get("PERM_RESULTS_DIR", "../results")
json.dump(out, open(os.path.join(rd, "table62_conditional_baseline.json"), "w"), indent=1)
print("saved table62_conditional_baseline.json")
