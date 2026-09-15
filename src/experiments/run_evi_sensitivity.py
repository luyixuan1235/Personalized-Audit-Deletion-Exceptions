"""Hyperparameter sensitivity of the evidential model (reviewer: is it under-tuned?).

The paper reports that evidential learning "adds none" and, on SPA, costs 5.2 points
of accuracy (83.3 -> 78.1) relative to the calibrated signed model. But the evidential
KL/refusal regularizer weight `lam_kl_max` is a SINGLE global default (0.1 in
train.DEFAULTS), used unchanged on both corpora -- and SPA is ~33x larger than Wu with
the opposite class balance. Evidential deep learning is notoriously sensitive to this
coefficient, so the accuracy drop may be under-tuning rather than a property of the
method. That would weaken the paper's negative result.

This sweeps lam_kl_max on a grid and reports accuracy and HC-FPR (5-fold), so the
negative result can be stated as "no favourable accuracy/safety trade-off ANYWHERE on
the grid" rather than "at one arbitrary coefficient".
"""
import os
import sys

import numpy as np

import _common as C
import metrics as M

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import Graph, build_records, per_user_folds  # noqa: E402
from train import predict, train  # noqa: E402

KFOLD = 5
# 0.1 = the paper's untuned default. The grid is deliberately extended well past the
# point where accuracy is still rising (SPA accuracy climbs monotonically up to 1.0),
# so the negative result cannot be dismissed as "you stopped searching too early".
GRID = [float(x) for x in os.environ.get(
    "PERM_LAMKL_GRID", "0.0,0.01,0.03,0.1,0.3,1.0,3.0,10.0").split(",")]


def _epochs():
    return 15 if C.QUICK else int(os.environ.get("PERM_EPOCHS", 200))


def main():
    records, _ = build_records()
    ep = _epochs()
    folds = per_user_folds(records, KFOLD, 42)
    per = {lk: {"acc": [], "hc": []} for lk in GRID}

    for f in range(KFOLD):
        tr = [records[i] for i in range(len(records)) if folds[i] != f]
        te = [records[i] for i in range(len(records)) if folds[i] == f]
        g = Graph(tr, records)
        print(f"[fold {f+1}/{KFOLD}] train={len(tr)} test={len(te)}")
        for lk in GRID:
            m = train(g, tr, mode="dual", loss="evi_refuse", use_sideinfo=True,
                      epochs=ep, lam_kl_max=lk)
            pr = predict(m, g, te)
            y = np.asarray(pr["y"]); p = np.asarray(pr["p_allow"])
            hc, _ = M.hc_fpr(y, p, 0.50)
            per[lk]["acc"].append(100 * M.basic(y, p, thr=0.5)["acc"])
            per[lk]["hc"].append(100 * hc)

    out = {}
    print("\n=== evidential KL weight sensitivity (5-fold mean +/- std) ===")
    print(f"{'lam_kl_max':>12}{'Acc':>18}{'HC-FPR':>18}")
    for lk in GRID:
        a = np.array(per[lk]["acc"]); h = np.array(per[lk]["hc"])
        out[str(lk)] = dict(acc_mean=float(a.mean()), acc_std=float(a.std(ddof=1)),
                            hc_mean=float(h.mean()), hc_std=float(h.std(ddof=1)),
                            acc_folds=a.tolist(), hc_folds=h.tolist())
        tag = "  <- paper's default" if lk == 0.1 else ""
        print(f"{lk:12.2f}{a.mean():11.1f} +/-{a.std(ddof=1):4.1f}"
              f"{h.mean():11.1f} +/-{h.std(ddof=1):4.1f}{tag}")

    best = max(GRID, key=lambda k: out[str(k)]["acc_mean"])
    print(f"\nBest accuracy at lam_kl_max={best} "
          f"({out[str(best)]['acc_mean']:.1f}%, HC-FPR {out[str(best)]['hc_mean']:.1f})")
    C.save(out, "table18_evi_sensitivity.json")
    print("Saved -> table18_evi_sensitivity.json")


if __name__ == "__main__":
    main()
