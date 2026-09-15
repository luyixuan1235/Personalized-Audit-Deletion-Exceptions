"""Is per-user thresholding a REAL remedy, or an oracle artifact?

A previous version of this analysis fitted each user's threshold on the very records it
then scored, which is not a method but an upper bound -- and an optimistically biased one,
since most users contribute only a handful of denials. This script replaces that oracle
with an honest, deployable protocol and reports both, so the gap between them is visible.

Protocol (per fold):
  * train the model on the training fold
  * carve a per-user CALIBRATION split out of the training fold
  * fit each user's threshold on THAT user's calibration records only (falling back to
    the global threshold when the user has too few, or lacks both classes)
  * evaluate on the held-out test fold

Reported per model:
  global      -- one equal-error threshold for everyone (what the field does)
  per-user    -- honest per-user thresholds, fitted on held-out per-user data
  oracle      -- per-user thresholds fitted on the test records themselves (UPPER BOUND,
                 not attainable; shown only to bound what per-user thresholding could
                 ever buy)
  coverage    -- fraction of users with enough calibration data to get their own threshold

If `per-user` lands near `global`, per-user thresholding is not a usable remedy at this
data scale, and the paper must say so rather than quote the oracle.
"""
import os
import sys

import numpy as np

import _common as C
import baselines_ext as B
import metrics as M

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import Graph, build_records, per_user_folds  # noqa: E402
from train import predict, train  # noqa: E402

KFOLD = 5
CAL_FRAC = 0.25       # share of the TRAINING fold held out to fit per-user thresholds
MIN_CAL = 4           # a user needs this many calibration records (with both classes)
MIN_DENIES = 3        # ... and this many test denials to get an FGR estimate
SEED = 42

CELLS = [
    ("atomic / ranking",    ("atomic", "bpr")),
    ("FM     / ranking",    ("fm", "bpr")),
    ("FM     / calibrated", ("fm", "bce")),
    ("signed / calibrated", ("signed", "bce")),
]


def _epochs():
    return 15 if C.QUICK else int(os.environ.get("PERM_EPOCHS", 200))


def _fit(arch, obj, g, tr, ev, ep):
    if arch == "fm":
        return B.fm_predict(tr, ev, objective=obj)
    if arch == "atomic":
        m = train(g, tr, mode="allow", loss=obj, use_sideinfo=False, epochs=ep)
        return predict(m, g, ev)
    m = B.train_signed(g, tr, epochs=ep, use_sideinfo=True, objective=obj)
    return predict(m, g, ev)


def _fgr(s, y, thr):
    dn = (y == 0)
    return float((s[dn] >= thr).mean()) if dn.any() else float("nan")


def _best_thr(s, y, thr0):
    """Threshold minimising FGR without costing accuracy, measured on (s, y)."""
    acc0 = float(((s >= thr0).astype(int) == y).mean())
    best, bf = thr0, _fgr(s, y, thr0)
    if np.isnan(bf):
        return thr0
    for t in np.linspace(s.min(), s.max(), 200):
        if float(((s >= t).astype(int) == y).mean()) >= acc0:
            f = _fgr(s, y, t)
            if not np.isnan(f) and f < bf:
                bf, best = f, t
    return best


def main():
    records, _ = build_records()
    ep = _epochs()
    folds = per_user_folds(records, KFOLD, SEED)
    rng = np.random.default_rng(SEED)
    res = {n: dict(g=[], h=[], o=[], cov=[]) for n, _ in CELLS}

    for f in range(KFOLD):
        tr_all = [records[i] for i in range(len(records)) if folds[i] != f]
        te = [records[i] for i in range(len(records)) if folds[i] == f]
        # per-user calibration split carved out of the TRAINING fold
        perm = rng.permutation(len(tr_all))
        ncal = int(round(CAL_FRAC * len(tr_all)))
        cal = [tr_all[i] for i in perm[:ncal]]
        fit = [tr_all[i] for i in perm[ncal:]]
        g = Graph(fit, records)
        print(f"[fold {f+1}/{KFOLD}] fit={len(fit)} cal={len(cal)} test={len(te)}")

        for name, (arch, obj) in CELLS:
            both = cal + te
            pr = _fit(arch, obj, g, fit, both, ep)
            prtr = _fit(arch, obj, g, fit, fit, ep)
            p_all = np.asarray(pr["p_allow"]); y_all = np.asarray(pr["y"])
            nc = len(cal)
            p_cal, y_cal = p_all[:nc], y_all[:nc]
            p_te, y_te = p_all[nc:], y_all[nc:]
            u_cal = np.array([r["user_id"] for r in cal])
            u_te = np.array([r["user_id"] for r in te])

            thr_g = float(M.equal_error_threshold(np.asarray(prtr["y"]),
                                                  np.asarray(prtr["p_allow"])))

            n_ok, n_tot = 0, 0
            gg, hh, oo = [], [], []
            for uu in np.unique(u_te):
                mt = (u_te == uu)
                dn = mt & (y_te == 0)
                if dn.sum() < MIN_DENIES:
                    continue
                n_tot += 1
                st, yt = p_te[mt], y_te[mt]

                # honest: fit this user's threshold on THEIR calibration records
                mc = (u_cal == uu)
                thr_u = thr_g
                if mc.sum() >= MIN_CAL and len(np.unique(y_cal[mc])) == 2:
                    thr_u = _best_thr(p_cal[mc], y_cal[mc], thr_g)
                    n_ok += 1

                gg.append(100 * _fgr(st, yt, thr_g))
                hh.append(100 * _fgr(st, yt, thr_u))
                oo.append(100 * _fgr(st, yt, _best_thr(st, yt, thr_g)))  # ORACLE

            res[name]["g"].append(np.mean(gg)); res[name]["h"].append(np.mean(hh))
            res[name]["o"].append(np.mean(oo))
            res[name]["cov"].append(100 * n_ok / max(n_tot, 1))

    out = {}
    print("\n" + "=" * 78)
    print("Per-user thresholds: ORACLE (test-fitted) vs HONEST (held-out-fitted)")
    print("=" * 78)
    print(f"{'model':22s}{'global':>9}{'per-user':>10}{'oracle':>9}"
          f"{'honest gain':>13}{'users w/ own thr':>18}")
    for name, _ in CELLS:
        r = res[name]
        g, h, o, cv = (float(np.mean(r[k])) for k in ("g", "h", "o", "cov"))
        out[name] = dict(global_fgr=g, peruser_fgr=h, oracle_fgr=o,
                         honest_gain=g - h, oracle_gain=g - o, coverage=cv)
        print(f"{name:22s}{g:9.1f}{h:10.1f}{o:9.1f}{g - h:12.1f}p{cv:17.1f}%")

    print("\nIf 'per-user' ~ 'global', per-user thresholding does not work at this data")
    print("scale, and the oracle column must NOT be quoted as an achievable remedy.")
    C.save(out, "table23_peruser_honest.json")
    print("Saved -> table23_peruser_honest.json")


if __name__ == "__main__":
    main()
