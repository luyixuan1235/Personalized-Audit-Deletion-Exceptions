"""SPA subsampled toward Wu's per-user sparsity (paper: coverage-matrix note b,
"SPA, in full", Table app:corpora).

Full SPA (~146 decisions/user) shows the habit force absent through the
refusal-side lens (restrictive vs permissive users' refusals, ratio ~1.1x).
Subsampling each user to <=42 decisions should restore it (~1.4x): with sparse
histories the model falls back on a scalar of how permissive the user is.

Model: ranking FM (BPR objective), matching Table app:corpora's "ranking CF" lens.
Run: cd src && PERM_DATA_DIR=../data_spa PERM_RESULTS_DIR=../results_spa \
       python3 experiments/run_spa_subsample.py
Output: results_spa/table48_spa_subsample.json
"""
import json, os, sys
import numpy as np

sys.path.insert(0, '.')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import metrics as M
from data import build_records, per_user_folds, Graph
from train import predict as g_predict, train as g_train
import baselines_ext as B
MODEL = os.environ.get("PERM_MODEL", "fm")

KFOLD, SEED, PER_USER = 5, 42, 42
EP = int(os.environ.get("PERM_EPOCHS", 100))
# CAPS: cap each user's TRAINING records (test fold stays full) -- the original
# density intervention. 0 = uncapped. Set CAPS="" to skip and use whole-corpus mode.
CAPS = [int(x) for x in os.environ.get("CAPS", "").split(",") if x != ""]


def subsample(records, per_user, seed):
    rng = np.random.default_rng(seed)
    by_u = {}
    for i, r in enumerate(records):
        by_u.setdefault(r["user_id"], []).append(i)
    keep = []
    for u, ix in by_u.items():
        ix = np.asarray(ix)
        keep.extend(ix if len(ix) <= per_user
                    else rng.choice(ix, size=per_user, replace=False))
    return [records[i] for i in sorted(keep)]


def lens(records, cap=0):
    folds = per_user_folds(records, KFOLD, SEED)
    rng = np.random.default_rng(SEED)
    gr = {}
    for r in records:
        gr.setdefault(r["user_id"], []).append(r["label"])
    gr = {u: float(np.mean(v)) for u, v in gr.items()}
    G, V, EXC, ROU = [], [], [], []
    for f in range(KFOLD):
        tr = [records[i] for i in range(len(records)) if folds[i] != f]
        te = [records[i] for i in range(len(records)) if folds[i] == f]
        if cap > 0:
            per = {}
            for r in tr:
                per.setdefault(r["user_id"], []).append(r)
            tr = [r for v in per.values()
                  for r in (v if len(v) <= cap else
                            [v[i] for i in rng.permutation(len(v))[:cap]])]
        if MODEL == "atomic":
            g = Graph(tr, records)
            m = g_train(g, tr, mode="allow", loss="bpr", use_sideinfo=False, epochs=EP)
            pr, ptr = g_predict(m, g, te), g_predict(m, g, tr)
        else:
            pr = B.fm_predict(tr, te, epochs=EP, objective="bpr")
            ptr = B.fm_predict(tr, tr, epochs=EP, objective="bpr")
        thr = float(M.equal_error_threshold(np.asarray(ptr["y"]),
                                            np.asarray(ptr["p_allow"])))
        y = np.asarray(pr["y"]); p = np.asarray(pr["p_allow"])
        yh = (p >= thr).astype(int)
        mu = {u: (1 if g >= 0.5 else 0) for u, g in gr.items()}
        for r, yy, hh in zip(te, y, yh):
            ov = float(hh != yy)
            (EXC if yy != mu[r["user_id"]] else ROU).append(ov)
            if yy == 0:
                G.append(gr[r["user_id"]]); V.append(ov)
        print(f"fold {f} done", flush=True)
    G, V = np.asarray(G), np.asarray(V)
    lo = 100 * V[G < 0.2].mean(); hi = 100 * V[G >= 0.6].mean()
    return dict(restrictive=float(lo), permissive=float(hi),
                refusal_ratio=float(hi / max(lo, 0.1)),
                eor_exc=float(100 * np.mean(EXC)), eor_rout=float(100 * np.mean(ROU)),
                eor_ratio=float(np.mean(EXC) / max(np.mean(ROU), 1e-9)),
                n=int(len(records)))


def main():
    records, _ = build_records()
    out = {}
    if CAPS:
        for cap in CAPS:
            out[f"train_cap={cap or 'none'}"] = lens(records, cap=cap)
            print(f"cap={cap}:", out[f"train_cap={cap or 'none'}"], flush=True)
    else:
        sub = subsample(records, PER_USER, SEED)
        print(f"SPA {len(records)} -> subsampled {len(sub)} "
              f"({len(sub)/len({r['user_id'] for r in sub}):.1f}/user)", flush=True)
        out["whole_corpus_subsample"] = lens(sub)
        print(out["whole_corpus_subsample"], flush=True)
    rd = os.environ.get("PERM_RESULTS_DIR", "../results_spa")
    with open(os.path.join(rd, f"table48_spa_subsample_{MODEL}.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print("saved table48_spa_subsample.json")


if __name__ == "__main__":
    main()
