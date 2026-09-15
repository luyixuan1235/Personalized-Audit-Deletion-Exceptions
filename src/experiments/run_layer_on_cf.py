"""P0: does the decision layer transfer to the released CF itself?
Uses the released CF's own test predictions (continuous scores, cached).
Protocol: per user, alternate decisions into halves A/B; fit global EER threshold
and per-user thresholds (>=4 decisions, both classes) on A, margin-defer on B at a
40% budget; swap and average. Evaluation half never touches fitting.
Output: results/table67_layer_on_cf.json"""
import json, os, sys
import numpy as np
from collections import defaultdict
sys.path.insert(0, '.')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import metrics as M

d = json.load(open('../results/cf_only_predictions.json'))['cf_only']['test_predictions']
by_u = defaultdict(list)
for r in d: by_u[r['userID']].append((float(r['prediction']), int(r['rating'])))
rows = []   # (u, s, y, half)
for u, lst in by_u.items():
    for i, (s, y) in enumerate(lst): rows.append((u, s, y, i % 2))
S = np.array([r[1] for r in rows]); Y = np.array([r[2] for r in rows])
U = np.array([r[0] for r in rows]); H = np.array([r[3] for r in rows])
gr = {u: np.mean([y for (_, y) in lst]) for u, lst in by_u.items()}
mu = {u: (1 if g >= 0.5 else 0) for u, g in gr.items()}
BUDGET = 0.4
def run(fit_h):
    ev = H != fit_h; ft = H == fit_h
    thr_g = float(M.equal_error_threshold(Y[ft], S[ft]))
    tau = {}
    for u in by_u:
        m = ft & (U == u)
        if m.sum() >= 4 and len(set(Y[m])) == 2:
            # accuracy-preserving, FGR-minimizing threshold on fit half
            cand = np.unique(S[m]); best = (1e9, thr_g)
            for t in cand:
                yh = (S[m] >= t).astype(int)
                acc = (yh == Y[m]).mean()
                fgr = yh[Y[m] == 0].mean() if (Y[m] == 0).any() else 0
                if (1 - acc, fgr) < best[0:1] + (best[1],):
                    pass
            # simpler: pick t minimizing FGR subject to accuracy >= global-thr accuracy
            yg = (S[m] >= thr_g).astype(int); acc0 = (yg == Y[m]).mean()
            ok = []
            for t in cand:
                yh = (S[m] >= t).astype(int)
                if (yh == Y[m]).mean() >= acc0:
                    ok.append((yh[Y[m] == 0].mean() if (Y[m] == 0).any() else 0, t))
            if ok: tau[u] = float(sorted(ok)[0][1])
    t_u = np.array([tau.get(u, thr_g) for u in U])
    margin = np.abs(S - t_u)
    q = np.quantile(margin[ev], BUDGET)
    auto = ev & (margin > q)
    yh = (S >= t_u).astype(int)
    ov = (yh != Y).astype(float)
    exc = np.array([Y[i] != mu[U[i]] for i in range(len(Y))])
    def stats(mask):
        harmed = mask & (Y == 0) & np.array([gr[u] >= 0.6 for u in U])
        return dict(exc=round(float(100*ov[mask & exc].mean()),1),
                    routine=round(float(100*ov[mask & ~exc].mean()),1),
                    fgr=round(float(100*yh[mask & (Y==0)].mean()),1),
                    harmed=round(float(100*ov[harmed].mean()),1) if harmed.sum() else None,
                    n=int(mask.sum()), n_harmed=int(harmed.sum()))
    return dict(baseline=stats(ev), layer=stats(auto),
                coverage=round(float(auto.sum()/ev.sum()),2), n_tau=len(tau))
r0, r1 = run(0), run(1)
def avg(k1, k2):
    return {kk: (round((r0[k1][kk]+r1[k1][kk])/2,1) if isinstance(r0[k1][kk],(int,float)) and r0[k1][kk] is not None and r1[k1][kk] is not None else r0[k1][kk])
            for kk in r0[k1]}
out = dict(baseline=avg('baseline','baseline'), layer=avg('layer','layer'),
           coverage=(r0['coverage']+r1['coverage'])/2, n_tau=[r0['n_tau'], r1['n_tau']],
           note="released CF's own test scores; alternate-half calibration, swap-averaged; 40% budget")
print(json.dumps(out, indent=1))
json.dump(out, open('../results/table67_layer_on_cf.json','w'), indent=1)
print("saved table67_layer_on_cf.json")
