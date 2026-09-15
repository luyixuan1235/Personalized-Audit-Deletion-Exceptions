"""Compositional generalization to UNSEEN request types (the method paper's novelty).

A permission request is a structured tuple (domain, tool, data_type). We hold out
ENTIRE request combinations (request_key) from training and test on them -- but only
keep held-out combos that are RECOMBINATIONS of seen parts (each of their domain,
tool, data_type appears in some training combo). This isolates *compositional*
generalization (a novel triple of familiar factors) from pure out-of-vocabulary.

Atomic-ID CF (the prior, e.g. Wu-style) has no embedding for an unseen request node
-> it collapses. A structured (CI-preserving) request embedding composes the unseen
node from its seen domain/tool/data_type factors -> it should still generalize.

Configs (struct vs atomic x one-sided vs signed):
  CF-atomic     : one-sided preference CF, atomic request id   (prior baseline)
  CF-struct     : one-sided preference CF, structured embedding
  Signed-atomic : signed two-sided evidence, atomic request id
  Signed-struct : signed two-sided evidence, structured embedding   (ours)
"""
import os
import sys

import numpy as np

import _common as C
import baselines_ext as B
import metrics as M

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import Graph, build_records  # noqa: E402
from train import predict, train  # noqa: E402

CONFIGS = [
    ("CF-atomic",           dict(kind="cf", use_sideinfo=False)),   # prior atomic-ID CF
    ("Signed-atomic",       dict(kind="signed", use_sideinfo=False)),
    ("GBDT-feature",        dict(kind="gbdt")),                     # strong non-graph, feature-based
    ("FM-feature",          dict(kind="fm")),                      # canonical feature-based CF
    ("Signed-struct (ours)", dict(kind="signed", use_sideinfo=True)),
]


def _epochs():
    if C.QUICK:
        return 15
    return int(os.environ.get("PERM_EPOCHS", 200))


def combo_folds(records, k=5, seed=42):
    """Assign each record a fold by its request_key (whole combos move together)."""
    keys = sorted({r["request_key"] for r in records})
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(keys))
    fold_of = {keys[perm[i]]: i % k for i in range(len(keys))}
    return np.array([fold_of[r["request_key"]] for r in records])


def _recombinable(test, tr):
    """Keep held-out records whose domain, tool AND data_type each appear in train."""
    sd = {r["domain"] for r in tr}; st = {r["tool"] for r in tr}
    sdt = {r["data_type"] for r in tr}
    return [r for r in test if r["domain"] in sd and r["tool"] in st
            and r["data_type"] in sdt]


def _fit_predict(cfg, g, tr, te, ep):
    if cfg["kind"] == "gbdt":
        return B.gbdt_predict(tr, te), B.gbdt_predict(tr, tr)
    if cfg["kind"] == "fm":
        return B.fm_predict(tr, te), B.fm_predict(tr, tr)
    if cfg["kind"] == "cf":
        m = train(g, tr, mode="allow", loss="bpr",
                  use_sideinfo=cfg["use_sideinfo"], epochs=ep)
    else:
        m = B.train_signed(g, tr, epochs=ep, use_sideinfo=cfg["use_sideinfo"])
    return predict(m, g, te), predict(m, g, tr)


def main():
    records, _ = build_records()
    ep = _epochs()
    folds = combo_folds(records, 5, 42)
    pool = {n: {"y": [], "p": [], "ytr": [], "ptr": []} for n, _ in CONFIGS}
    n_test_total = 0

    for f in range(5):
        tr = [records[i] for i in range(len(records)) if folds[i] != f]
        te_all = [records[i] for i in range(len(records)) if folds[i] == f]
        te = _recombinable(te_all, tr)
        if not tr or not te:
            continue
        n_test_total += len(te)
        g = Graph(tr, records)
        print(f"[combo fold {f+1}/5] train={len(tr)} held={len(te_all)} "
              f"recombinable-test={len(te)}")
        for name, cfg in CONFIGS:
            pte, ptr = _fit_predict(cfg, g, tr, te, ep)
            pool[name]["y"].extend(pte["y"]); pool[name]["p"].extend(pte["p_allow"])
            pool[name]["ytr"].extend(ptr["y"]); pool[name]["ptr"].extend(ptr["p_allow"])

    rows = {}
    print(f"\n=== Compositional generalization to UNSEEN combos "
          f"(n_test={n_test_total}) ===")
    print("config                  Acc   FPR  HC-FPR")
    for name, _ in CONFIGS:
        d = pool[name]
        if not d["y"]:
            continue
        y = np.array(d["y"]); p = np.array(d["p"])
        thr = M.equal_error_threshold(np.array(d["ytr"]), np.array(d["ptr"]))
        b = M.basic(y, p, thr=thr)
        hc, _ = M.hc_fpr(y, p, 0.50)
        rows[name] = dict(acc=100*b["acc"], fpr=100*b["fpr"], hc_fpr=100*hc,
                          n_test=n_test_total)
        print(f"{name:22s}{rows[name]['acc']:6.1f}{rows[name]['fpr']:6.1f}"
              f"{rows[name]['hc_fpr']:8.1f}")
    C.save(rows, "table7_compogen.json")
    print("\nIf structured >> atomic here, compositional request modeling is the "
          "method's real, winnable contribution.")


if __name__ == "__main__":
    main()
