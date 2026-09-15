"""Continuous lambda -> EOR curve on the synthetic control (reviewer: show the
shrinkage is continuous, not three regimes). Same generative model and estimator
as run_synth_regimes, wd swept on a grid, 150 epochs, 3 generation seeds.
Output: results/table66_lambda_curve.json"""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, '.')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import metrics as M
from data import per_user_folds
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_synth_regimes import make_synth, regime

GRID = [0.0, 1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
out = {}
for wd in GRID:
    vals = []
    for sd in (0, 1, 2):
        recs = make_synth(seed=sd)
        folds = per_user_folds(recs, 5, 42)
        users = sorted({r["user_id"] for r in recs}); items = sorted({r["request_key"] for r in recs})
        ui = {u: i for i, u in enumerate(users)}; qi = {q: i+len(users) for i, q in enumerate(items)}
        gr = {}
        for r in recs: gr.setdefault(r["user_id"], []).append(r["label"])
        gr = {u: float(np.mean(v)) for u, v in gr.items()}
        mu = {u: (1 if g >= 0.5 else 0) for u, g in gr.items()}
        res = regime(recs, folds, ui, qi, len(users)+len(items), gr, mu, epochs=150, wd=wd)
        vals.append(res["eor_exc"])
    out[f"wd={wd}"] = dict(eor_mean=round(float(np.mean(vals)),1), eor_sd=round(float(np.std(vals)),1))
    print(f"wd={wd}: EOR={np.mean(vals):.1f} ± {np.std(vals):.1f}", flush=True)
rd = os.environ.get("PERM_RESULTS_DIR", "../results")
json.dump(out, open(os.path.join(rd, "table66_lambda_curve.json"), "w"), indent=1)
print("saved table66_lambda_curve.json")
