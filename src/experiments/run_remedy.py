"""Does the remedy actually work ON THE NEW METRIC?

The paper's diagnosis is: a ranking objective leaves a per-user offset free, so the one
global threshold that must then be fitted is systematically wrong per user, and the harm
concentrates on a minority. The proposed remedy is to train with a calibrated pointwise
objective, which anchors scores to labels and removes the free offset at its source.

That remedy has never been tested against the diagnosis. This script does it. For every
cell of the objective x architecture grid, evaluated AT THE THRESHOLD IT WOULD ACTUALLY
BE DEPLOYED WITH (train-fold equal-error, exactly as the released system does -- never
at a fixed 0.5, which is meaningless for a ranking score), we report:

  * eta                -- offset heterogeneity of the trained model (no labels needed)
  * aggregate FGR      -- the number a paper would report
  * per-user FGR       -- median vs mean, and the burden borne by the worst quartile
  * avoidable FGR      -- what a per-user threshold would recover at no accuracy cost

The prediction, if the mechanism is right:
    calibrated objective  ->  lower eta  ->  less concentration  ->  less avoidable loss
If a calibrated objective leaves eta and the concentration untouched, the mechanism is
wrong and we must say so.
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
MIN_DENIES = 3

CELLS = [
    ("atomic  / ranking",   ("atomic", "bpr")),
    ("atomic  / calibrated", ("atomic", "bce")),
    ("FM      / ranking",   ("fm", "bpr")),
    ("FM      / calibrated", ("fm", "bce")),
    ("signed  / ranking",   ("signed", "bpr")),
    ("signed  / calibrated", ("signed", "bce")),
]


def _epochs():
    return 15 if C.QUICK else int(os.environ.get("PERM_EPOCHS", 200))


def _fit(arch, obj, g, tr, te, ep):
    if arch == "fm":
        return B.fm_predict(tr, te, objective=obj), B.fm_predict(tr, tr, objective=obj)
    if arch == "atomic":
        m = train(g, tr, mode="allow", loss=obj, use_sideinfo=False, epochs=ep)
        return predict(m, g, te), predict(m, g, tr)
    m = B.train_signed(g, tr, epochs=ep, use_sideinfo=True, objective=obj)
    return predict(m, g, te), predict(m, g, tr)


def _gini(x):
    x = np.sort(np.asarray(x, float)); n = len(x); c = np.cumsum(x)
    return float((n + 1 - 2 * np.sum(c) / c[-1]) / n) if c[-1] > 0 else 0.0


def _top25(x):
    x = np.sort(np.asarray(x, float))[::-1]
    k = max(1, len(x) // 4)
    return float(100 * x[:k].sum() / x.sum()) if x.sum() > 0 else float("nan")


def _fgr(s, y, thr):
    dn = (y == 0)
    return float((s[dn] >= thr).mean()) if dn.any() else float("nan")


def _oracle(s, y, thr):
    """Best per-user threshold that costs this user no accuracy."""
    acc0 = float(((s >= thr).astype(int) == y).mean())
    best = _fgr(s, y, thr)
    for t in np.linspace(s.min(), s.max(), 200):
        if float(((s >= t).astype(int) == y).mean()) >= acc0:
            best = min(best, _fgr(s, y, t))
    return best


def main():
    records, _ = build_records()
    ep = _epochs()
    folds = per_user_folds(records, KFOLD, 42)
    acc = {n: dict(eta_b=[], eta_w=[], agg=[], per=[], glob=[], orac=[], accy=[])
           for n, _ in CELLS}

    for f in range(KFOLD):
        tr = [records[i] for i in range(len(records)) if folds[i] != f]
        te = [records[i] for i in range(len(records)) if folds[i] == f]
        g = Graph(tr, records)
        u = np.array([r["user_id"] for r in te])
        print(f"[fold {f+1}/{KFOLD}]")
        for name, (arch, obj) in CELLS:
            pr, prtr = _fit(arch, obj, g, tr, te, ep)
            y = np.asarray(pr["y"]); p = np.asarray(pr["p_allow"])
            # the threshold a deployed system would actually use
            thr = float(M.equal_error_threshold(np.asarray(prtr["y"]),
                                                np.asarray(prtr["p_allow"])))
            a = acc[name]
            a["agg"].append(100 * _fgr(p, y, thr))
            a["accy"].append(100 * M.basic(y, p, thr=thr)["acc"])

            z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
            mus, sds = [], []
            for uu in np.unique(u):
                mu = (u == uu)
                if mu.sum() >= 3:
                    mus.append(z[mu].mean()); sds.append(z[mu].std())
                dn = mu & (y == 0)
                if dn.sum() >= MIN_DENIES:
                    a["per"].append(100 * _fgr(p[mu], y[mu], thr))
                    a["glob"].append(100 * _fgr(p[mu], y[mu], thr))
                    a["orac"].append(100 * _oracle(p[mu], y[mu], thr))
            a["eta_b"].append(np.std(mus)); a["eta_w"].append(np.mean(sds))

    out = {}
    print("\n" + "=" * 84)
    print("Evaluated at the DEPLOYED (train-fold equal-error) threshold, never at 0.5")
    print("=" * 84)
    print(f"{'cell':22s}{'Acc':>6}{'eta':>7}{'aggFGR':>8}{'median':>8}{'mean':>7}"
          f"{'worst25%':>10}{'avoidable':>11}")
    for name, _ in CELLS:
        a = acc[name]
        eta = float(np.mean(a["eta_b"]) / np.mean(a["eta_w"]))
        per = np.array(a["per"])
        gl, orc = float(np.mean(a["glob"])), float(np.mean(a["orac"]))
        out[name] = dict(acc=float(np.mean(a["accy"])), eta=eta,
                         agg_fgr=float(np.mean(a["agg"])),
                         median=float(np.median(per)), mean=float(per.mean()),
                         gini=_gini(per), top25=_top25(per),
                         global_fgr=gl, oracle_fgr=orc, avoidable=gl - orc,
                         n_users=len(per))
        o = out[name]
        print(f"{name:22s}{o['acc']:6.1f}{eta:7.2f}{o['agg_fgr']:8.1f}"
              f"{o['median']:8.1f}{o['mean']:7.1f}{o['top25']:9.1f}%"
              f"{o['avoidable']:10.1f}p")

    print("\nPrediction: calibrated objective -> lower eta -> less concentration"
          " -> less avoidable loss.")
    for a_r, a_c in (("atomic  / ranking", "atomic  / calibrated"),
                     ("FM      / ranking", "FM      / calibrated"),
                     ("signed  / ranking", "signed  / calibrated")):
        r, c = out[a_r], out[a_c]
        print(f"  {a_r.split('/')[0].strip():8s}: eta {r['eta']:.2f} -> {c['eta']:.2f}"
              f"   worst25% {r['top25']:.0f}% -> {c['top25']:.0f}%"
              f"   avoidable {r['avoidable']:.1f}p -> {c['avoidable']:.1f}p")

    C.save(out, "table22_remedy.json")
    print("\nSaved -> table22_remedy.json")


if __name__ == "__main__":
    main()
