"""3x2 objective x architecture grid -- extends the 2x2 with the ATOMIC-ID rung.

Why: the paper's Table 1 ("accuracy is deceptive", HC-FPR = 100%) uses a one-sided
ATOMIC-ID collaborative predictor, while the controlled 2x2 (Table 4) starts from a
FACTORIZATION MACHINE, whose worst corner is only ~46% HC-FPR. The two architectures
were never crossed, so the effect attributed to "architecture" in the paper covers
only the FM -> signed-graph step and silently omits the (much larger) atomic -> FM
step. This script closes the gap by running the full grid

    architecture in {atomic-ID CF, FM (features), signed graph}
       x objective in {ranking (BPR), calibrated (BCE)}

on the same 5 per-user folds (seed 42), so every cell is comparable and the total
architecture effect can be accounted for honestly.

Cells:
  atomic-bpr = exactly the Table 1 / Section 3 model (mode=allow, no side info, BPR)
  atomic-bce = the missing cell: same architecture, calibrated objective
  FM-*, Signed-* = the existing 2x2 (reproduced here on the same folds)
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

# label -> (architecture, objective)
CELLS = [
    ("atomic-bpr", ("atomic", "bpr")),   # == Table 1 model (the 100% HC-FPR one)
    ("atomic-bce", ("atomic", "bce")),   # the cell the paper never ran
    ("FM-bpr",     ("fm", "bpr")),
    ("FM-bce",     ("fm", "bce")),
    ("Signed-bpr", ("signed", "bpr")),
    ("Signed-bce", ("signed", "bce")),
]


def _epochs():
    return 15 if C.QUICK else int(os.environ.get("PERM_EPOCHS", 200))


def _fit(arch, obj, g, tr, te, ep):
    """Returns (test preds, train preds) -- train preds are needed to fit the EER
    threshold on the training fold, matching the Table 1 / Table 8 protocol."""
    if arch == "fm":
        return B.fm_predict(tr, te, objective=obj), B.fm_predict(tr, tr, objective=obj)
    if arch == "atomic":
        # one-sided, atomic request IDs (no structured features) == the Table 1 recipe
        m = train(g, tr, mode="allow", loss=obj, use_sideinfo=False, epochs=ep)
        return predict(m, g, te), predict(m, g, tr)
    m = B.train_signed(g, tr, epochs=ep, use_sideinfo=True, objective=obj)
    return predict(m, g, te), predict(m, g, tr)


def main():
    records, _ = build_records()
    ep = _epochs()
    folds = per_user_folds(records, KFOLD, 42)
    per = {name: {"acc": [], "acc_eer": [], "hc": [], "lo": [], "hi": []}
           for name, _ in CELLS}

    for f in range(KFOLD):
        tr = [records[i] for i in range(len(records)) if folds[i] != f]
        te = [records[i] for i in range(len(records)) if folds[i] == f]
        g = Graph(tr, records)
        print(f"[fold {f+1}/{KFOLD}] train={len(tr)} test={len(te)}")
        for name, (arch, obj) in CELLS:
            pr, prtr = _fit(arch, obj, g, tr, te, ep)
            y = np.asarray(pr["y"]); p = np.asarray(pr["p_allow"])
            ytr = np.asarray(prtr["y"]); ptr = np.asarray(prtr["p_allow"])
            hc, _ = M.hc_fpr(y, p, 0.50)          # HC-FPR always decides at p >= 0.5
            thr_eer = float(M.equal_error_threshold(ytr, ptr))
            per[name]["acc"].append(100 * M.basic(y, p, thr=0.5)["acc"])
            per[name]["acc_eer"].append(100 * M.basic(y, p, thr=thr_eer)["acc"])
            per[name]["hc"].append(100 * hc)
            per[name]["lo"].append(float(p.min()))
            per[name]["hi"].append(float(p.max()))

    out = {}
    print("\n=== 3x2 grid: HC-FPR (5-fold mean +/- std) ===")
    print("Acc@.5  = accuracy at the fixed 0.5 decision point (Table 4 protocol)")
    print("Acc@EER = accuracy at the train-fold equal-error threshold (Table 1 protocol)")
    print(f"\n{'cell':14s}{'Acc@.5':>9}{'Acc@EER':>10}{'HC-FPR':>18}{'score range':>20}")
    for name, _ in CELLS:
        a = np.array(per[name]["acc"]); ae = np.array(per[name]["acc_eer"])
        h = np.array(per[name]["hc"])
        lo = float(np.mean(per[name]["lo"])); hi = float(np.mean(per[name]["hi"]))
        out[name] = dict(acc_mean=float(a.mean()), acc_std=float(a.std(ddof=1)),
                         acc_eer_mean=float(ae.mean()), acc_eer_std=float(ae.std(ddof=1)),
                         hc_mean=float(h.mean()), hc_std=float(h.std(ddof=1)),
                         hc_folds=h.tolist(), acc_folds=a.tolist(),
                         acc_eer_folds=ae.tolist(), p_min=lo, p_max=hi)
        print(f"{name:14s}{a.mean():9.1f}{ae.mean():10.1f}"
              f"{h.mean():11.1f} +/-{h.std(ddof=1):4.1f}"
              f"      [{lo:.2f}, {hi:.2f}]")

    print("\n--- architecture effect, objective held at RANKING (the omitted step) ---")
    for a, b in (("atomic-bpr", "FM-bpr"), ("FM-bpr", "Signed-bpr")):
        print(f"  {a:12s} -> {b:12s}  HC-FPR {out[a]['hc_mean']:6.1f} -> "
              f"{out[b]['hc_mean']:6.1f}   (delta {out[b]['hc_mean']-out[a]['hc_mean']:+.1f})")
    print("--- objective effect, architecture held ---")
    for a, b in (("atomic-bpr", "atomic-bce"), ("FM-bpr", "FM-bce"),
                 ("Signed-bpr", "Signed-bce")):
        print(f"  {a:12s} -> {b:12s}  HC-FPR {out[a]['hc_mean']:6.1f} -> "
              f"{out[b]['hc_mean']:6.1f}   (delta {out[b]['hc_mean']-out[a]['hc_mean']:+.1f})")

    C.save(out, "table16_grid3.json")
    print("\nSaved -> table16_grid3.json")


if __name__ == "__main__":
    main()
