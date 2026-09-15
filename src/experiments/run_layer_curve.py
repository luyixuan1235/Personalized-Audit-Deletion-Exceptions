"""The exception-aware decision layer: do the paper's working parts COMPOSE?

Stack, every stage fitted on held-out data (protocol of run_peruser_honest):
  backbone : signed/calibrated model (the best substrate, tab:defer)
  rule     : per-user threshold where estimable (>=4 cal records, both classes),
             global EER otherwise
  deferral : defer the smallest |p - thr_u(x)| at a fixed budget (25% / 40%)

Conditions: global | global+defer | per-user | per-user+defer (the layer).
Outputs: table33 + table56 (curve/CIs/sensitivity)
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C, baselines_ext as B, metrics as M
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import Graph, build_records, per_user_folds
from train import predict

KFOLD, CAL_FRAC, SEED = 5, 0.25, 42
MIN_CAL = int(__import__('os').environ.get('MIN_CAL', '4'))  # MIN_CAL_ENV
SUFFIX = '' if MIN_CAL == 4 else f'_mincal{MIN_CAL}'
EP = 15 if C.QUICK else int(os.environ.get("PERM_EPOCHS", 200))

def _fgr(s, y, thr):
    dn = (y == 0)
    return float((s[dn] >= thr).mean()) if dn.any() else float("nan")

def _best_thr(s, y, thr0):
    acc0 = float(((s >= thr0).astype(int) == y).mean())
    best, bf = thr0, _fgr(s, y, thr0)
    if np.isnan(bf): return thr0
    for t in np.linspace(s.min(), s.max(), 200):
        if float(((s >= t).astype(int) == y).mean()) >= acc0:
            f = _fgr(s, y, t)
            if not np.isnan(f) and f < bf: bf, best = f, t
    return best

records, _ = build_records()
folds = per_user_folds(records, KFOLD, SEED)
rng = np.random.default_rng(SEED)

pool = dict(y=[], p=[], u=[], thr_u=[], thr_g=[], mu=[])
cov_users = []
for f in range(KFOLD):
    tr_all = [records[i] for i in range(len(records)) if folds[i] != f]
    te = [records[i] for i in range(len(records)) if folds[i] == f]
    perm = rng.permutation(len(tr_all))
    ncal = int(round(CAL_FRAC * len(tr_all)))
    cal = [tr_all[i] for i in perm[:ncal]]; fit = [tr_all[i] for i in perm[ncal:]]
    g = Graph(fit, records)
    print(f"[fold {f+1}/{KFOLD}] fit={len(fit)} cal={len(cal)} test={len(te)}", flush=True)
    m = B.train_signed(g, fit, epochs=EP, use_sideinfo=True, objective="bce")
    pr = predict(m, g, cal + te); prtr = predict(m, g, fit)
    p_all, y_all = np.asarray(pr["p_allow"]), np.asarray(pr["y"])
    nc = len(cal)
    p_cal, y_cal, p_te, y_te = p_all[:nc], y_all[:nc], p_all[nc:], y_all[nc:]
    u_cal = np.array([r["user_id"] for r in cal])
    u_te  = np.array([r["user_id"] for r in te])
    thr_g = float(M.equal_error_threshold(np.asarray(prtr["y"]), np.asarray(prtr["p_allow"])))
    # habit prior m_u from the TRAINING fold (fit+cal)
    ytr_all = {}
    for r in tr_all: ytr_all.setdefault(r["user_id"], []).append(r["label"])
    mu_map = {u: (1 if np.mean(v) >= 0.5 else 0) for u, v in ytr_all.items()}
    ok = 0
    thr_u = np.full(len(te), thr_g)
    for uu in np.unique(u_te):
        sc, yc = p_cal[u_cal == uu], y_cal[u_cal == uu]
        if len(sc) >= MIN_CAL and len(set(yc)) == 2:
            t = _best_thr(sc, yc, thr_g); ok += 1
        else:
            t = thr_g
        thr_u[u_te == uu] = t
    cov_users.append(ok / len(np.unique(u_te)))
    pool["y"] += list(y_te); pool["p"] += list(p_te); pool["u"] += list(u_te)
    pool["thr_u"] += list(thr_u); pool["thr_g"] += [thr_g]*len(te)
    pool["mu"] += [mu_map.get(uu, 1) for uu in u_te]

y  = np.array(pool["y"]); p = np.array(pool["p"]); u = np.array(pool["u"])
tu = np.array(pool["thr_u"]); tg = np.array(pool["thr_g"]); mu = np.array(pool["mu"])
exc = (y != mu)

def cell(thr, budget, label):
    yh = (p >= thr).astype(int); ov = (yh != y)
    margin = np.abs(p - thr)
    if budget > 0:
        cut = np.quantile(margin, budget); auto = margin > cut
    else:
        auto = np.ones(len(y), bool)
    dn = auto & (y == 0); pr_ = exc & (y == 0); rc_ = exc & (y == 1)
    r = dict(coverage=float(auto.mean()),
             routine=float(ov[auto & ~exc].mean()),
             exception=float(ov[auto & exc].mean()),
             fgr=float(yh[dn].mean()) if dn.any() else None,
             perm_refusals=float(ov[auto & pr_].mean()) if (auto & pr_).any() else None,
             restr_consents=float(ov[auto & rc_].mean()) if (auto & rc_).any() else None,
             exc_deferred=float(1 - auto[exc].mean()))
    print(f"{label:34s} cov {100*r['coverage']:5.1f}%  routine {100*r['routine']:5.1f}%  "
          f"exc {100*r['exception']:5.1f}%  FGR {100*r['fgr']:5.1f}%  perm.ref {100*r['perm_refusals']:5.1f}%", flush=True)
    return r

out = {"user_threshold_coverage": float(np.mean(cov_users))}
out["global"]            = cell(tg, 0.0,  "global thr (baseline)")
out["global_defer40"]    = cell(tg, 0.40, "global thr + defer 40%")
out["peruser"]           = cell(tu, 0.0,  "per-user thr")
out["peruser_defer25"]   = cell(tu, 0.25, "LAYER: per-user + defer 25%")
out["peruser_defer40"]   = cell(tu, 0.40, "LAYER: per-user + defer 40%")
json.dump(out, open(os.environ.get("PERM_RESULTS_DIR","../results")+f"/table33_decision_layer{SUFFIX}.json","w"), indent=1)
print("saved table33_decision_layer.json")

# ---- appended: full risk-coverage curve, cluster CIs, deferred-error sensitivity (table56)
out56={"curve":{}}
for b in (0.0,0.1,0.2,0.25,0.3,0.4,0.5,0.6):
    c=cell(tu,b,f"curve budget={b}")
    margin=np.abs(p-tu)
    if b>0:
        cut=np.quantile(margin,b); dfr=margin<=cut
    else:
        dfr=np.zeros(len(y),bool)
    c["deferred_true_denials_share_of_all"]=float(((y==0)&dfr).mean())
    exc_dn=exc&(y==0)
    c["n_exc"]=int((exc&(margin>(np.quantile(margin,b) if b>0 else -1))).sum())
    c["n_perm_ref"]=int((exc_dn&(margin>(np.quantile(margin,b) if b>0 else -1))).sum())
    out56["curve"][f"budget={b}"]=c
# system-level harmful grants under deferred-prompt error e (40% budget)
margin=np.abs(p-tu); cut=np.quantile(margin,0.4); auto=margin>cut
yh=(p>=tu).astype(int)
auto_fg_all=float(((yh==1)&(y==0)&auto).mean())
defer_dn=float(((y==0)&~auto).mean())
base_fg_all=float(((yh==1)&(y==0)).mean())
out56["system_sensitivity_40"]={f"e={e}":dict(harmful_grant_rate=float(auto_fg_all+e*defer_dn))
                                for e in (0.0,0.05,0.10,0.20)}
out56["system_sensitivity_40"]["no_deferral_baseline"]=base_fg_all
# user-cluster bootstrap CIs at 40% budget
users=np.unique(u); rng2=np.random.default_rng(1)
def stat(picked):
    cnt={}
    for uu in picked: cnt[uu]=cnt.get(uu,0)+1
    w=np.array([cnt.get(uu,0) for uu in u],float)
    m=auto&(w>0)
    ov=(yh!=y)
    exc_m=exc&m; dn_m=m&(y==0); prm=exc&(y==0)&m; rcm=exc&(y==1)&m
    def wm(mask,val):
        ww=w[mask]
        return float((val[mask]*ww).sum()/ww.sum()) if ww.sum() else np.nan
    return (wm(exc_m,ov.astype(float)), wm(dn_m,yh.astype(float)),
            wm(prm,ov.astype(float)), wm(rcm,ov.astype(float)))
boots=np.array([stat(rng2.choice(users,len(users),replace=True)) for _ in range(1000)])
out56["cluster_ci_40"]={k:[float(np.nanpercentile(boots[:,i],2.5)),
                           float(np.nanpercentile(boots[:,i],97.5))]
                        for i,k in enumerate(("exception","fgr","perm_refusals","restr_consents"))}
out56["n_at_40"]=dict(n_exc=out56["curve"]["budget=0.4"]["n_exc"],
                      n_perm_ref=out56["curve"]["budget=0.4"]["n_perm_ref"])
json.dump(out56, open(os.environ.get("PERM_RESULTS_DIR","../results")+f"/table56_layer_curve{SUFFIX}.json","w"), indent=1)
print("saved table56_layer_curve.json")
