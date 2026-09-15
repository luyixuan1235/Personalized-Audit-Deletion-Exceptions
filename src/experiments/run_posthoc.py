"""Post-hoc calibration baselines on the COLLAPSED ranking model (missing control).

The paper claims a calibrated TRAINING objective is the fix, but never tests the
cheapest alternative a reviewer will ask for: just Platt/isotonic-scale the ranking
model's scores after the fact. This script runs that control -- and turns it into a
direct test of Proposition 1.

Proposition 1 says the ranking loss is invariant to a PER-USER additive shift c_u:
each user's scores can float by a DIFFERENT amount. A global Platt/isotonic map is a
SINGLE monotone transform applied to every sample, so it can undo a shared offset but
provably cannot undo per-user offsets that differ across users. The sharp prediction:

  * global post-hoc calibration  -> helps, but does NOT restore safety
  * per-user post-hoc calibration -> restores safety, but needs enough per-user
    labelled calibration data (the practical catch: sparse users fall back to global)

Protocol (per fold): carve a 20% CALIBRATION split out of the training fold, fit the
model on the remaining 80%, fit the calibrator on the calibration split, evaluate on
the held-out test fold. Calibrators map the model's logit z = logit(p):
  none        : raw p
  platt       : sigmoid(A z + B), A,B fit globally by MLE
  isotonic    : global monotone step function (PAVA)
  platt-user  : per-user (A_u, B_u); users lacking both classes fall back to global

No sklearn dependency: Platt is 2-parameter logistic regression (Newton/GD) and
isotonic is the pool-adjacent-violators algorithm, both implemented here.
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
CAL_FRAC = 0.20
SEED = 42
MIN_USER_CAL = 4      # a user needs >= this many calibration points ...
                      # ... AND both classes present, else fall back to global


def _epochs():
    return 15 if C.QUICK else int(os.environ.get("PERM_EPOCHS", 200))


def _logit(p, eps=1e-6):
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    return np.log(p / (1 - p))


# ---------------- calibrators ----------------
def fit_platt(z, y, iters=200, lr=0.5):
    """2-param logistic regression sigmoid(A z + B) by gradient descent on NLL."""
    z = np.asarray(z, float); y = np.asarray(y, float)
    if len(z) == 0 or len(np.unique(y)) < 2:
        return None
    s = z.std() or 1.0
    zc = (z - z.mean()) / s
    A, Bb = 1.0, 0.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(A * zc + Bb)))
        gA = np.mean((p - y) * zc); gB = np.mean(p - y)
        A -= lr * gA; Bb -= lr * gB
    # fold the standardization back in so the map applies to raw z
    return (A / s, Bb - A * z.mean() / s)


def apply_platt(par, z):
    A, Bb = par
    return 1.0 / (1.0 + np.exp(-(A * np.asarray(z, float) + Bb)))


def fit_isotonic(z, y):
    """Pool-adjacent-violators: returns (sorted_z, fitted_p) defining a step function."""
    z = np.asarray(z, float); y = np.asarray(y, float)
    if len(z) == 0:
        return None
    o = np.argsort(z, kind="stable")
    zs, ys = z[o], y[o]
    # PAVA on ys
    vals = list(ys); wts = [1.0] * len(ys)
    i = 0
    while i < len(vals) - 1:
        if vals[i] <= vals[i + 1] + 1e-12:
            i += 1
            continue
        # pool i and i+1, then backtrack
        tot_w = wts[i] + wts[i + 1]
        pooled = (vals[i] * wts[i] + vals[i + 1] * wts[i + 1]) / tot_w
        vals[i:i + 2] = [pooled]; wts[i:i + 2] = [tot_w]
        i = max(i - 1, 0)
    # expand back to per-sample fitted values
    fitted = np.concatenate([np.full(int(w), v) for v, w in zip(vals, wts)])
    return (zs, fitted)


def apply_isotonic(iso, z):
    zs, fitted = iso
    return np.interp(np.asarray(z, float), zs, fitted, left=fitted[0], right=fitted[-1])


# ---------------- evaluation ----------------
def _row(y, p):
    hc, _ = M.hc_fpr(y, p, 0.50)
    deny = (np.asarray(y) == 0)
    # deny high-confidence recall: true denials confidently refused (p <= 0.5)
    dr = float((np.asarray(p)[deny] <= 0.5).mean()) if deny.any() else float("nan")
    return dict(acc=100 * M.basic(y, p, thr=0.5)["acc"], hc=100 * hc,
                deny_rec=100 * dr, pmin=float(np.min(p)), pmax=float(np.max(p)),
                ece=float(M.ece(y, p)))


def _fit_model(arch, g, fit_recs, eval_recs, ep):
    """Train on fit_recs, predict on eval_recs (a single concatenated batch)."""
    if arch == "fm":
        pr = B.fm_predict(fit_recs, eval_recs, objective="bpr")
        return np.asarray(pr["p_allow"]), np.asarray(pr["y"])
    m = train(g, fit_recs, mode="allow", loss="bpr", use_sideinfo=False, epochs=ep)
    pr = predict(m, g, eval_recs)
    return np.asarray(pr["p_allow"]), np.asarray(pr["y"])


def main():
    records, _ = build_records()
    ep = _epochs()
    folds = per_user_folds(records, KFOLD, SEED)
    rng = np.random.default_rng(SEED)

    ARCHS = [("atomic-CF (Table 1 model)", "atomic"), ("FM", "fm")]
    METHODS = ["none", "platt", "isotonic", "platt-user"]
    acc = {a: {m: {"acc": [], "hc": [], "deny_rec": [], "pmin": [], "pmax": [], "ece": []}
               for m in METHODS} for a, _ in ARCHS}
    user_cov = []

    for f in range(KFOLD):
        tr_all = [records[i] for i in range(len(records)) if folds[i] != f]
        te = [records[i] for i in range(len(records)) if folds[i] == f]
        perm = rng.permutation(len(tr_all))
        ncal = int(round(CAL_FRAC * len(tr_all)))
        cal = [tr_all[i] for i in perm[:ncal]]
        fit = [tr_all[i] for i in perm[ncal:]]
        g = Graph(fit, records)
        print(f"[fold {f+1}/{KFOLD}] fit={len(fit)} cal={len(cal)} test={len(te)}")

        for label, arch in ARCHS:
            both = cal + te
            p_all, y_all = _fit_model(arch, g, fit, both, ep)
            nc = len(cal)
            z_cal, y_cal = _logit(p_all[:nc]), y_all[:nc]
            z_te, y_te = _logit(p_all[nc:]), y_all[nc:]
            u_cal = np.array([r["user_id"] for r in cal])
            u_te = np.array([r["user_id"] for r in te])

            # --- global calibrators
            gp = fit_platt(z_cal, y_cal)
            iso = fit_isotonic(z_cal, y_cal)

            preds = {"none": p_all[nc:]}
            preds["platt"] = apply_platt(gp, z_te) if gp else p_all[nc:]
            preds["isotonic"] = apply_isotonic(iso, z_te) if iso else p_all[nc:]

            # --- per-user Platt, falling back to the global map
            pu = np.array(preds["platt"], float, copy=True)
            n_ok = 0; users = np.unique(u_te)
            for uu in users:
                mcal = (u_cal == uu)
                if mcal.sum() >= MIN_USER_CAL and len(np.unique(y_cal[mcal])) == 2:
                    par = fit_platt(z_cal[mcal], y_cal[mcal])
                    if par:
                        mte = (u_te == uu)
                        pu[mte] = apply_platt(par, z_te[mte])
                        n_ok += 1
            preds["platt-user"] = pu
            if arch == "atomic":
                user_cov.append(100.0 * n_ok / max(len(users), 1))

            for m in METHODS:
                r = _row(y_te, preds[m])
                for k in r:
                    acc[label][m][k].append(r[k])

    out = {}
    for label, _ in ARCHS:
        print(f"\n=== post-hoc calibration of the RANKING model: {label} ===")
        print(f"{'calibrator':14s}{'Acc':>8}{'HC-FPR':>16}{'deny hi-conf':>14}"
              f"{'score range':>18}{'ECE':>8}")
        out[label] = {}
        for m in METHODS:
            d = {k: np.array(v) for k, v in acc[label][m].items()}
            out[label][m] = {k: [float(v.mean()), float(v.std(ddof=1))] for k, v in d.items()}
            out[label][m]["hc_folds"] = d["hc"].tolist()
            print(f"{m:14s}{d['acc'].mean():8.1f}"
                  f"{d['hc'].mean():10.1f} +/-{d['hc'].std(ddof=1):4.1f}"
                  f"{d['deny_rec'].mean():13.1f}%"
                  f"     [{d['pmin'].mean():.2f}, {d['pmax'].mean():.2f}]"
                  f"{d['ece'].mean():8.3f}")
    out["_pct_users_individually_calibratable"] = [float(np.mean(user_cov)),
                                                   float(np.std(user_cov, ddof=1))]
    print(f"\nUsers with enough calibration data to fit their own Platt map: "
          f"{np.mean(user_cov):.1f}% (rest fall back to the global map)")
    print("Proposition 1 prediction: a GLOBAL monotone map cannot undo PER-USER "
          "offsets -> global calibration should leave HC-FPR high; per-user should not.")
    C.save(out, "table17_posthoc.json")
    print("Saved -> table17_posthoc.json")


if __name__ == "__main__":
    main()
