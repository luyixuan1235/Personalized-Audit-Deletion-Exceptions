"""Who does the remedy actually help?

The controlled ablation (run_denial_ablation.py) showed something uncomfortable: every
improvement the field has made -- calibrated objectives, denial modelling, graph
structure -- raises accuracy, lowers the AGGREGATE false-grant rate, and protects the
users who were already safe, while leaving the harmed users almost exactly where they
were (permissive users' refusals: 23.6% overridden -> 24.0%).

One intervention did move the aggregate a lot: fitting the decision threshold PER USER
instead of globally. But the ablation is a warning -- an intervention can improve every
aggregate number and still not touch the people who bear the harm. So we ask the only
question that matters:

    does per-user thresholding help the PERMISSIVE users whose rare refusals are
    being overridden, or does it too just help the users who were already fine?

Protocol: per fold, carve a per-user calibration split out of the training fold, fit each
user's threshold on THAT user's held-out records (falling back to the global threshold if
they have too few), and evaluate on the test fold. Break the result out by the user's
grant rate. Unit of analysis is the DENIAL, so no user is excluded.

If the permissive column does not move, the paper has a diagnosis and no remedy, and must
say so.
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
CAL_FRAC = 0.25
MIN_CAL = 4
SEED = 42

CELLS = [("atomic / ranking", ("atomic", "bpr")),
         ("FM     / calibrated", ("fm", "bce")),
         ("signed / calibrated", ("signed", "bce"))]

BINS = [(0.0, 0.2, "restrictive"), (0.2, 0.4, "mid-low"), (0.4, 0.6, "mid"),
        (0.6, 0.8, "permissive"), (0.8, 1.01, "very permissive")]


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
    """Threshold that minimises this user's false grants without costing them accuracy."""
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
    acc = {n: dict(g=[], glob=[], peru=[], fitted=[]) for n, _ in CELLS}

    for f in range(KFOLD):
        tr_all = [records[i] for i in range(len(records)) if folds[i] != f]
        te = [records[i] for i in range(len(records)) if folds[i] == f]
        perm = rng.permutation(len(tr_all))
        ncal = int(round(CAL_FRAC * len(tr_all)))
        cal = [tr_all[i] for i in perm[:ncal]]
        fit = [tr_all[i] for i in perm[ncal:]]
        g = Graph(fit, records)
        print(f"[fold {f+1}/{KFOLD}]")

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

            for uu in np.unique(u_te):
                mt = (u_te == uu)
                dn = mt & (y_te == 0)
                if not dn.any():
                    continue
                grate = float(y_te[mt].mean())
                mc = (u_cal == uu)
                thr_u, fitted = thr_g, 0
                if mc.sum() >= MIN_CAL and len(np.unique(y_cal[mc])) == 2:
                    thr_u = _best_thr(p_cal[mc], y_cal[mc], thr_g)
                    fitted = 1
                k = int(dn.sum())
                acc[name]["g"] += [grate] * k
                acc[name]["glob"] += list((p_te[dn] >= thr_g).astype(float))
                acc[name]["peru"] += list((p_te[dn] >= thr_u).astype(float))
                acc[name]["fitted"] += [fitted] * k

    out = {}
    print("\n" + "=" * 78)
    print("Who does per-user thresholding actually help?  (unit = one denial)")
    print("=" * 78)
    for name, _ in CELLS:
        a = acc[name]
        G = np.array(a["g"]); GL = np.array(a["glob"]); PU = np.array(a["peru"])
        FT = np.array(a["fitted"])
        print(f"\n### {name}   ({len(G)} denials, "
              f"{100 * FT.mean():.0f}% of them from users with a fitted threshold)")
        print(f"    {'user grant rate':>18}{'denials':>9}{'global':>9}"
              f"{'per-user':>10}{'change':>9}")
        row = {}
        for lo, hi, lab in BINS:
            m = (G >= lo) & (G < hi)
            if m.sum() == 0:
                continue
            gl, pu = 100 * GL[m].mean(), 100 * PU[m].mean()
            row[lab] = dict(n=int(m.sum()), glob=gl, peru=pu, delta=pu - gl)
            flag = "  <-- the harmed" if lab.endswith("permissive") else ""
            print(f"    {lab:>18}{m.sum():9d}{gl:8.1f}%{pu:9.1f}%"
                  f"{pu - gl:+8.1f}p{flag}")
        overall_g, overall_p = 100 * GL.mean(), 100 * PU.mean()
        print(f"    {'ALL (aggregate)':>18}{len(G):9d}{overall_g:8.1f}%"
              f"{overall_p:9.1f}%{overall_p - overall_g:+8.1f}p")
        out[name] = dict(bins=row, aggregate_global=overall_g,
                         aggregate_peruser=overall_p, n_denials=int(len(G)))

    print("\nThe question: does the 'permissive' row move as much as the aggregate?")
    print("If it does not, per-user thresholding is another improvement that helps")
    print("the users who were already safe -- and the paper has no remedy to offer.")
    C.save(out, "table25_remedy_who.json")
    print("\nSaved -> table25_remedy_who.json")


if __name__ == "__main__":
    main()
