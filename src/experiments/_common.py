"""Shared harness for the experiments."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import Graph, build_records, leave_user_out_folds, per_user_folds  # noqa: E402
from train import predict, train  # noqa: E402

QUICK = os.environ.get("PERM_QUICK", "0") == "1"
RESULTS = os.environ.get("PERM_RESULTS_DIR",
                         os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                      "..", "results"))


def _epochs(cfg):
    if QUICK:
        return {**cfg, "epochs": 15}
    e = os.environ.get("PERM_EPOCHS")
    if e:
        return {**cfg, "epochs": int(e)}
    return cfg


def kfold(records, cfg, fold_type="within", k=5, seed=42, want_train=False):
    """Train per fold and pool test predictions. fold_type: 'within' | 'luo'."""
    cfg = _epochs(cfg)
    folds = (per_user_folds(records, k, seed) if fold_type == "within"
             else leave_user_out_folds(records, k, seed))
    keys = ["p_allow", "u", "y", "e_allow", "e_deny"]
    pool = {kk: [] for kk in keys + ["domain", "data_type"]}
    trpool = {kk: [] for kk in keys + ["domain", "data_type"]} if want_train else None
    for f in range(k):
        tr = [records[i] for i in range(len(records)) if folds[i] != f]
        te = [records[i] for i in range(len(records)) if folds[i] == f]
        if not tr or not te:
            continue
        g = Graph(tr, records)
        print(f"[fold {f+1}/{k}] train={len(tr)} test={len(te)} "
              f"users={g.n_users} reqs={g.n_reqs}")
        m = train(g, tr, **cfg)
        pr = predict(m, g, te)
        for kk in pool:
            pool[kk].extend(pr[kk])
        if want_train:
            prtr = predict(m, g, tr)
            for kk in trpool:
                trpool[kk].extend(prtr[kk])
    out = _arrayify(pool)
    return (out, _arrayify(trpool)) if want_train else out


def _arrayify(pool):
    num = ("p_allow", "u", "y", "e_allow", "e_deny")
    return {kk: (np.array(v) if kk in num else v) for kk, v in pool.items()}


def save(obj, name):
    import json
    os.makedirs(RESULTS, exist_ok=True)
    p = os.path.join(RESULTS, name)
    json.dump(obj, open(p, "w"), indent=2, default=float)
    print(f"saved -> {p}")
