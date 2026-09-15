"""Reviewer check: do the most direct memory baselines already solve it?
(a) per-user conjunction lookup: predict the user's training majority on the same
    (tool, data_type) conjunction; fallback m_u.
(b) last-decision: predict the user's most recent training label (proxy: majority
    of the user's training labels = m_u itself, plus exact-match recency where
    the corpus has repeats).
Report accuracy + EOR/routine of each. Output: results/table63_lookup_baselines.json"""
import json, os, sys
import numpy as np
sys.path.insert(0, '.')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import build_records, per_user_folds
from collections import defaultdict

records, _ = build_records(); folds = per_user_folds(records, 5, 42)
conj = lambda r: (r["tool"], r["data_type"])
res = {}
for name in ("conjunction_lookup", "habit"):
    ACC, EXC, ROU = [], [], []
    n_hit = 0; n_tot = 0
    for f in range(5):
        tr = [records[i] for i in range(len(records)) if folds[i] != f]
        te = [records[i] for i in range(len(records)) if folds[i] == f]
        ur = defaultdict(list); uc = defaultdict(list)
        for r in tr:
            ur[r["user_id"]].append(r["label"])
            uc[(r["user_id"],) + conj(r)].append(r["label"])
        mu = {u: (1 if np.mean(v) >= 0.5 else 0) for u, v in ur.items()}
        for r in te:
            u = r["user_id"]
            if u not in mu: continue
            n_tot += 1
            if name == "conjunction_lookup":
                key = (u,) + conj(r)
                if key in uc:
                    yh = 1 if np.mean(uc[key]) >= 0.5 else 0; n_hit += 1
                else:
                    yh = mu[u]
            else:
                yh = mu[u]
            ov = float(yh != r["label"])
            ACC.append(1 - ov)
            (EXC if r["label"] != mu[u] else ROU).append(ov)
    res[name] = dict(acc=round(100*np.mean(ACC),1), eor=round(100*np.mean(EXC),1),
                     routine=round(100*np.mean(ROU),1),
                     ratio=round(np.mean(EXC)/max(np.mean(ROU),1e-9),2),
                     lookup_coverage=round(n_hit/max(n_tot,1),3) if name=="conjunction_lookup" else None)
    print(name, res[name])
rd = os.environ.get("PERM_RESULTS_DIR", "../results")
json.dump(res, open(os.path.join(rd, "table63_lookup_baselines.json"), "w"), indent=1)
print("saved table63_lookup_baselines.json")
