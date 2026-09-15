"""Table A: the dual-graph model vs. EXTERNAL baselines (not self-ablations).
Methods: global/per-user majority (floors), GBDT (non-graph), CF/LightGCN (the prior
graph-CF approach), Signed-GCN (single-space signed propagation, the closest prior to
our dual-graph idea), and the dual-graph model (ours). Reports Acc / FPR / high-confidence FPR, with the
decision threshold calibrated on the training fold."""
import os
import sys

import numpy as np

import _common as C
import baselines_ext as B
import metrics as M

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import Graph, build_records, per_user_folds  # noqa: E402
from train import predict, train  # noqa: E402

CF  = dict(mode="allow", use_sideinfo=False, loss="bpr")
OURS = dict(mode="dual",  use_sideinfo=True,  loss="evi_refuse")


def _ep(cfg):
    if C.QUICK:
        return {**cfg, "epochs": 15}
    e = os.environ.get("PERM_EPOCHS")
    return {**cfg, "epochs": int(e)} if e else cfg


def _signed_epochs():
    if C.QUICK:
        return 15
    return int(os.environ.get("PERM_EPOCHS", 200))


def main():
    records, _ = build_records()
    folds = per_user_folds(records, 5, 42)
    names = ["Global-majority", "Per-user-majority", "GBDT", "CF/LightGCN",
             "Signed-GCN", "dual-graph (ours)"]
    pool = {n: {"y": [], "p_allow": [], "domain": [], "data_type": []} for n in names}
    trpool = {n: {"y": [], "p_allow": []} for n in names}

    for f in range(5):
        tr = [records[i] for i in range(len(records)) if folds[i] != f]
        te = [records[i] for i in range(len(records)) if folds[i] == f]
        g = Graph(tr, records)
        print(f"\n[fold {f+1}/5] train={len(tr)} test={len(te)}")

        cf = train(g, tr, **_ep(CF)); ours = train(g, tr, **_ep(OURS))
        sg = B.train_signed(g, tr, epochs=_signed_epochs())
        preds = {
            "Global-majority":   (B.maj_predict(tr, te, "global"), B.maj_predict(tr, tr, "global")),
            "Per-user-majority": (B.maj_predict(tr, te, "user"),   B.maj_predict(tr, tr, "user")),
            "GBDT":              (B.gbdt_predict(tr, te),          B.gbdt_predict(tr, tr)),
            "CF/LightGCN":       (predict(cf, g, te),              predict(cf, g, tr)),
            "Signed-GCN":        (predict(sg, g, te),              predict(sg, g, tr)),
            "dual-graph (ours)": (predict(ours, g, te),            predict(ours, g, tr)),
        }
        for n, (pte, ptr) in preds.items():
            for k in pool[n]:
                pool[n][k].extend(pte[k])
            trpool[n]["y"].extend(ptr["y"]); trpool[n]["p_allow"].extend(ptr["p_allow"])

    rows = {}
    for n in names:
        y = np.array(pool[n]["y"]); p = np.array(pool[n]["p_allow"])
        thr = M.equal_error_threshold(np.array(trpool[n]["y"]), np.array(trpool[n]["p_allow"]))
        b = M.basic(y, p, pool[n]["data_type"], thr=thr)
        hc, _ = M.hc_fpr(y, p, 0.50)
        rows[n] = dict(acc=100*b["acc"], fpr=100*b["fpr"], fnr=100*b["fnr"],
                       hc_fpr=100*hc, sens_fpr=100*b.get("sens_fpr", float("nan")),
                       ece=M.ece(y, p), brier=M.brier(y, p))
    C.save(rows, "tableA_baselines.json")
    print("\n=== Table A: dual-graph vs. external baselines (%) ===")
    print("method              Acc   FPR  FNR  HC-FPR  ECE   Brier")
    for n in names:
        r = rows[n]
        print(f"{n:19s}{r['acc']:5.1f}{r['fpr']:5.1f}{r['fnr']:5.1f}{r['hc_fpr']:7.1f}"
              f"{r['ece']:7.3f}{r['brier']:7.3f}")


if __name__ == "__main__":
    main()
