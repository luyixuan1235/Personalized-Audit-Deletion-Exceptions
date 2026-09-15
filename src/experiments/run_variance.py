"""Per-fold variance for the headline HC-FPR comparisons (reviewer M1).

Unlike the other scripts, which POOL predictions across folds and compute one number,
this computes Accuracy and HC-FPR SEPARATELY on each held-out fold and reports
mean +/- std across folds. This directly supports the "overlapping levers" discussion
(the 2x2 cells 10.3 vs 10.9 on Wu and 2.2 vs 4.7 on SPA): with per-fold spread one can
judge which differences are within noise and which are not.

Cells = the objective x architecture 2x2 (Table 4) plus the calibrated inductive-bias
ladder point (evidential/dual-graph), so the numbers line up with the main tables.
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
# label -> (arch, objective) ; arch in {fm, signed, dual}
CELLS = [
    ("FM-bpr",        ("fm", "bpr")),
    ("FM-bce",        ("fm", "bce")),
    ("Signed-bpr",    ("signed", "bpr")),
    ("Signed-bce",    ("signed", "bce")),
    ("dual-evidential", ("dual", "evi")),
]


def _epochs():
    return 15 if C.QUICK else int(os.environ.get("PERM_EPOCHS", 200))


def _fit(arch, obj, g, tr, te, ep):
    if arch == "fm":
        return B.fm_predict(tr, te, objective=obj)
    if arch == "dual":
        m = train(g, tr, mode="dual", loss="evi_refuse", use_sideinfo=True, epochs=ep)
        return predict(m, g, te)
    m = B.train_signed(g, tr, epochs=ep, use_sideinfo=True, objective=obj)
    return predict(m, g, te)


def main():
    records, _ = build_records()
    ep = _epochs()
    folds = per_user_folds(records, KFOLD, 42)
    per = {name: {"acc": [], "hc": []} for name, _ in CELLS}

    for f in range(KFOLD):
        tr = [records[i] for i in range(len(records)) if folds[i] != f]
        te = [records[i] for i in range(len(records)) if folds[i] == f]
        g = Graph(tr, records)
        print(f"[fold {f+1}/{KFOLD}] train={len(tr)} test={len(te)}")
        for name, (arch, obj) in CELLS:
            pr = _fit(arch, obj, g, tr, te, ep)
            y = np.asarray(pr["y"]); p = np.asarray(pr["p_allow"])
            hc, _ = M.hc_fpr(y, p, 0.50)
            per[name]["acc"].append(100 * M.basic(y, p, thr=0.5)["acc"])
            per[name]["hc"].append(100 * hc)

    out = {}
    print("\n=== Per-fold mean +/- std (k=%d) ===" % KFOLD)
    print(f"{'cell':16s}{'Acc (mean+/-sd)':>20}{'HC-FPR (mean+/-sd)':>22}")
    for name, _ in CELLS:
        a = np.array(per[name]["acc"]); h = np.array(per[name]["hc"])
        out[name] = dict(acc_mean=float(a.mean()), acc_std=float(a.std(ddof=1)),
                         hc_mean=float(h.mean()), hc_std=float(h.std(ddof=1)),
                         hc_folds=h.tolist(), acc_folds=a.tolist())
        print(f"{name:16s}{a.mean():9.1f} +/-{a.std(ddof=1):5.1f}"
              f"{h.mean():13.1f} +/-{h.std(ddof=1):5.1f}")
    C.save(out, "table12_variance.json")
    print("\nUse the HC-FPR mean+/-std to judge which 2x2 gaps exceed cross-fold noise.")


if __name__ == "__main__":
    main()
