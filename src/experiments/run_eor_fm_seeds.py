"""P0-10: 5 split-seed robustness of train-majority EOR on the FM backbone.

Wu's released hybrid cannot be re-split without retraining the LLM. This script
re-splits OUR factorization-machine estimator (the mechanism substrate) with
five within-user fold seeds and reports mean ± std of EOR under train-only
majority (cross-fitted: majority from the training fold only).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import _p0_metrics as M  # noqa: E402
import baselines_ext as B  # noqa: E402
from data import build_records, per_user_folds  # noqa: E402

SEEDS = [42, 123, 456, 789, 1024]
KFOLD = 5
EPOCHS = 15 if os.environ.get("PERM_QUICK", "0") == "1" else int(os.environ.get("PERM_EPOCHS", 150))


def fold_majority(tr):
    by = {}
    for r in tr:
        by.setdefault(r["user_id"], []).append(r["label"])
    return {u: (1 if float(np.mean(v)) >= 0.5 else 0) for u, v in by.items()}


def eor_from_fold(tr, te, y, yh):
    mu_map = fold_majority(tr)
    users = np.array([r["user_id"] for r in te])
    keep = np.array([u in mu_map for u in users])
    users = users[keep]; y = y[keep]; yh = yh[keep]
    mu, n_fb = M.mu_vector(users, mu_map, {})
    stats = M.rates(y, yh, mu)
    stats["n_test_original"] = int(len(te))
    stats["n_evaluable_train_majority"] = int(len(y))
    stats["n_excluded_cold_start"] = int(len(te) - len(y))
    stats["n_fallback"] = int(n_fb)
    return stats


def run_seed(records, seed):
    folds = per_user_folds(records, KFOLD, seed)
    ys, yhs, us, mus = [], [], [], []
    per_fold = []
    for f in range(KFOLD):
        tr = [records[i] for i in range(len(records)) if folds[i] != f]
        te = [records[i] for i in range(len(records)) if folds[i] == f]
        print(f"  [seed {seed} fold {f+1}/{KFOLD}] train={len(tr)} test={len(te)}",
              flush=True)
        pr = B.fm_predict(tr, te, objective="bce", epochs=EPOCHS, seed=seed + f)
        y = np.asarray(pr["y"])
        p = np.asarray(pr["p_allow"])
        yh = (p >= 0.5).astype(int)
        st = eor_from_fold(tr, te, y, yh)
        per_fold.append({k: st[k] for k in ("acc", "exception", "routine", "ratio", "n_exc", "n_test_original", "n_evaluable_train_majority", "n_excluded_cold_start")})
        mu_map = fold_majority(tr)
        users = np.array([r["user_id"] for r in te])
        keep = np.array([u in mu_map for u in users])
        users = users[keep]; y = y[keep]; yh = yh[keep]
        mu, _ = M.mu_vector(users, mu_map, {})
        ys.append(y); yhs.append(yh); us.append(users); mus.append(mu)
    y = np.concatenate(ys); yh = np.concatenate(yhs)
    users = np.concatenate(us); mu = np.concatenate(mus)
    pooled = M.rates(y, yh, mu)
    return dict(pooled=pooled, per_fold=per_fold)


def summarize(vals):
    a = np.array(vals, float)
    return dict(mean=round(float(a.mean()), 2), std=round(float(a.std(ddof=1)), 2),
                values=[round(float(v), 2) for v in a])


def main():
    records, _ = build_records()
    per_seed = {}
    for seed in SEEDS:
        print(f"==== seed {seed} ====", flush=True)
        per_seed[str(seed)] = run_seed(records, seed)
        p = per_seed[str(seed)]["pooled"]
        print(f"  pooled  acc={p['acc']:.1f}  EOR={p['exception']:.1f}  "
              f"routine={p['routine']:.1f}  ratio={p['ratio']:.2f}", flush=True)

    keys = ("acc", "exception", "routine", "ratio")
    agg = {k: summarize([per_seed[str(s)]["pooled"][k] for s in SEEDS]) for k in keys}
    out = dict(
        seeds=SEEDS, epochs=EPOCHS, backbone="FM-bce", majority="train-fold (cross-fitted)",
        aggregate=agg, per_seed={s: dict(pooled=per_seed[s]["pooled"],
                                         per_fold=per_seed[s]["per_fold"])
                                 for s in per_seed},
        note=("Wu hybrid predictions are a fixed released split and are not re-seeded here. "
              "User-cluster CIs for that artifact are in user_cluster_uncertainty.json."),
    )
    M.save(out, "robustness_5seeds.json")
    print("aggregate:", json_dumps := __import__("json").dumps(agg, indent=2))
    print(json_dumps)


if __name__ == "__main__":
    main()
