"""Does evidential learning help on the COVERAGE-AWARE metric, at ANY lam_kl?

Closes the last honest gap in the negative result. The paper rejects evidential
learning partly because it costs accuracy -- but the paper's own thesis is that
accuracy is the wrong yardstick, so that argument is self-undermining. The coherent
test is to judge it in the currency the paper actually advocates: risk-coverage.

It also cannot be judged at the single untuned lam_kl=0.1 default, since tuning lam_kl
DOES buy HC-FPR (Wu 5.1 -> 3.4, SPA 1.2 -> 0.6). So we sweep lam_kl and, at each point,
report AURC (area under the risk-coverage curve, lower = better) and AUAC (selective
accuracy, higher = better) alongside HC-FPR -- against the calibrated signed-graph
model trained on the identical folds.

Claim to be tested: at NO point on the lam_kl grid does evidential learning improve on
the calibrated signed model's risk-coverage curve.

Protocol matches run_evi_sensitivity.py (per-user 5-fold, seed 42). AURC/AUAC use the
train-fold EER threshold, as in the appendix ladder table.
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
COV_GRID = np.linspace(0.05, 1.0, 20)
GRID = [float(x) for x in os.environ.get(
    "PERM_LAMKL_GRID", "0.0,0.01,0.03,0.1,0.3,1.0,3.0,10.0").split(",")]


def _epochs():
    return 15 if C.QUICK else int(os.environ.get("PERM_EPOCHS", 200))


def _curve_areas(y, p, thr):
    """AUAC (selective accuracy) and AURC (false-grant risk) over the coverage grid."""
    conf = np.abs(2 * p - 1)
    order = np.argsort(-conf, kind="stable")
    y, p = np.asarray(y)[order], np.asarray(p)[order]
    pred = (p >= thr).astype(float)
    n = len(y)
    accs, risks = [], []
    for c in COV_GRID:
        k = max(1, int(round(c * n)))
        yy, pp = y[:k], pred[:k]
        deny = (yy == 0)
        accs.append(float((pp == yy).mean()))
        risks.append(float(pp[deny].mean()) if deny.any() else np.nan)
    return 100 * float(np.nanmean(accs)), 100 * float(np.nanmean(risks))


def _eval(pr, prtr):
    y = np.asarray(pr["y"]); p = np.asarray(pr["p_allow"])
    thr = float(M.equal_error_threshold(np.asarray(prtr["y"]),
                                        np.asarray(prtr["p_allow"])))
    auac, aurc = _curve_areas(y, p, thr)
    hc, _ = M.hc_fpr(y, p, 0.50)
    return dict(acc=100 * M.basic(y, p, thr=thr)["acc"], hc=100 * hc,
                aurc=aurc, auac=auac)


def main():
    records, _ = build_records()
    ep = _epochs()
    folds = per_user_folds(records, KFOLD, 42)

    cells = ["signed-bce"] + [f"evi:lam={lk}" for lk in GRID]
    per = {c: {k: [] for k in ("acc", "hc", "aurc", "auac")} for c in cells}

    for f in range(KFOLD):
        tr = [records[i] for i in range(len(records)) if folds[i] != f]
        te = [records[i] for i in range(len(records)) if folds[i] == f]
        g = Graph(tr, records)
        print(f"[fold {f+1}/{KFOLD}] train={len(tr)} test={len(te)}")

        m = B.train_signed(g, tr, epochs=ep, use_sideinfo=True, objective="bce")
        r = _eval(predict(m, g, te), predict(m, g, tr))
        for k in r:
            per["signed-bce"][k].append(r[k])

        for lk in GRID:
            m = train(g, tr, mode="dual", loss="evi_refuse", use_sideinfo=True,
                      epochs=ep, lam_kl_max=lk)
            r = _eval(predict(m, g, te), predict(m, g, tr))
            for k in r:
                per[f"evi:lam={lk}"][k].append(r[k])

    out = {}
    print("\n=== evidential vs calibrated signed, on the COVERAGE-AWARE metrics ===")
    print("(AURC lower is better; AUAC higher is better; 5-fold means)")
    print(f"\n{'cell':16s}{'Acc':>8}{'HC-FPR':>9}{'AURC':>9}{'AUAC':>9}")
    for c in cells:
        d = {k: np.array(v) for k, v in per[c].items()}
        out[c] = {k: [float(v.mean()), float(v.std(ddof=1))] for k, v in d.items()}
        out[c]["aurc_folds"] = d["aurc"].tolist()
        print(f"{c:16s}{d['acc'].mean():8.1f}{d['hc'].mean():9.1f}"
              f"{d['aurc'].mean():9.1f}{d['auac'].mean():9.1f}")

    base = out["signed-bce"]["aurc"][0]
    better = [c for c in cells[1:] if out[c]["aurc"][0] < base]
    print(f"\nsigned-bce AURC = {base:.1f}")
    if better:
        print("!! evidential IMPROVES AURC at: " + ", ".join(
            f"{c} ({out[c]['aurc'][0]:.1f})" for c in better))
        print("   -> the negative result does NOT hold on the coverage-aware metric.")
    else:
        print("No lam_kl on the grid improves AURC over the calibrated signed model")
        print("   -> the negative result HOLDS, stated in the paper's own currency.")
    C.save(out, "table20_evi_aurc.json")
    print("Saved -> table20_evi_aurc.json")


if __name__ == "__main__":
    main()
