"""P1-4: is SPA's missing habit force (bilinear models) carried by the corpus's
deny-majority USER COMPOSITION? Train the FM on (A) grant-majority users only
(composition inverted, n=458) vs (B) a size-matched random user subset
(composition preserved), and compare exception/routine ratios.
Mirrors the Wu-side composition-inversion intervention (Table A16, bottom).
Run: cd src && PERM_DATA_DIR=../data_spa PERM_RESULTS_DIR=../results_spa \
     python3 experiments/run_spa_composition.py
Output: results_spa/table68_spa_composition.json"""
import json, os, sys
import numpy as np
sys.path.insert(0, '.')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import metrics as M
import baselines_ext as B
from data import build_records, per_user_folds
from collections import defaultdict

records, _ = build_records()
gr_all = defaultdict(list)
for r in records: gr_all[r["user_id"]].append(r["label"])
gr_all = {u: float(np.mean(v)) for u, v in gr_all.items()}
gm_users = sorted(u for u, g in gr_all.items() if g >= 0.5)
rng = np.random.default_rng(42)
rand_users = set(rng.choice(sorted(gr_all), size=len(gm_users), replace=False))

def lens(subset, tag):
    recs = [r for r in records if r["user_id"] in subset]
    folds = per_user_folds(recs, 5, 42)
    EXC, ROU = [], []
    for f in range(5):
        tr = [recs[i] for i in range(len(recs)) if folds[i] != f]
        te = [recs[i] for i in range(len(recs)) if folds[i] == f]
        pr = B.fm_predict(tr, te, objective="bce"); ptr = B.fm_predict(tr, tr, objective="bce")
        thr = float(M.equal_error_threshold(np.asarray(ptr["y"]), np.asarray(ptr["p_allow"])))
        ur = defaultdict(list)
        for r in tr: ur[r["user_id"]].append(r["label"])
        mu = {u: (1 if np.mean(v) >= 0.5 else 0) for u, v in ur.items()}
        for r, p in zip(te, pr["p_allow"]):
            u = r["user_id"]
            if u not in mu: continue
            ov = float((p >= thr) != r["label"])
            (EXC if r["label"] != mu[u] else ROU).append(ov)
    out = dict(n_users=len(subset), n_dec=len(recs),
               grant_majority_share=round(float(np.mean([gr_all[u] >= 0.5 for u in subset])), 3),
               eor=round(100*float(np.mean(EXC)), 1), routine=round(100*float(np.mean(ROU)), 1),
               ratio=round(float(np.mean(EXC)/max(np.mean(ROU), 1e-9)), 2), n_exc=len(EXC))
    print(tag, out, flush=True)
    return out

res = dict(inverted_grant_majority=lens(set(gm_users), "A(grant-maj):"),
           matched_random=lens(rand_users, "B(random):"))
rd = os.environ.get("PERM_RESULTS_DIR", "../results_spa")
json.dump(res, open(os.path.join(rd, "table68_spa_composition.json"), "w"), indent=1)
print("saved table68_spa_composition.json")
