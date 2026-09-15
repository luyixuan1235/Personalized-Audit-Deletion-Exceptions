"""Cost-sensitive learning vs. plain calibration (reviewer: "why not cost-sensitive?").

The obvious remedy for imbalance is to reweight the loss. We compare, on the SAME
factorization-machine architecture, four objectives:
  bpr    -- ranking (the failure baseline)
  bce    -- plain calibrated classification (our recommended fix)
  wbce   -- class-weighted BCE (cost-sensitive: minority up-weighted)
  focal  -- focal loss (down-weights easy examples)
Reporting Acc / HC-FPR / deny-hiconf-recall lets us see whether reweighting adds
anything over a plain calibrated objective, or whether calibration alone suffices.
"""
import os
import sys

import numpy as np

import _common as C
import baselines_ext as B
import metrics as M

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import build_records, per_user_folds  # noqa: E402

OBJS = ["bpr", "bce", "wbce", "focal"]
KFOLD = 5
LO = 0.2


def main():
    records, _ = build_records()
    folds = per_user_folds(records, KFOLD, 42)
    pool = {o: {"y": [], "p": []} for o in OBJS}
    for f in range(KFOLD):
        tr = [records[i] for i in range(len(records)) if folds[i] != f]
        te = [records[i] for i in range(len(records)) if folds[i] == f]
        print(f"[fold {f+1}/{KFOLD}] train={len(tr)} test={len(te)}")
        for o in OBJS:
            pr = B.fm_predict(tr, te, objective=o)
            pool[o]["y"].extend(pr["y"]); pool[o]["p"].extend(pr["p_allow"])

    out = {}
    print("\n=== FM: cost-sensitive objectives vs. plain calibration ===")
    print(f"{'objective':10s}{'Acc':>7}{'HC-FPR':>8}{'deny_hiconf':>13}{'ECE':>8}")
    for o in OBJS:
        y = np.array(pool[o]["y"]); p = np.array(pool[o]["p"])
        b = M.basic(y, p, thr=0.5); hc, _ = M.hc_fpr(y, p, 0.50)
        deny = (y == 0)
        rec = 100 * float((p[deny] <= LO).mean()) if deny.any() else float("nan")
        out[o] = dict(acc=100*b["acc"], hc_fpr=100*hc, deny_hiconf_rec=rec,
                      ece=M.ece(y, p))
        print(f"{o:10s}{100*b['acc']:7.1f}{100*hc:8.1f}{rec:12.1f}%{M.ece(y,p):8.3f}")
    C.save(out, "table13_costsensitive.json")
    print("\nIf wbce/focal do not beat plain bce on HC-FPR, calibration -- not "
          "reweighting -- is the effective fix.")


if __name__ == "__main__":
    main()
