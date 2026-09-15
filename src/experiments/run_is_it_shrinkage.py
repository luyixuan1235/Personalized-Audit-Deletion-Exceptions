"""Is the effect a defect, or just shrinkage?

We observe that permission models override the rare refusals of PERMISSIVE users far more
often than those of restrictive ones. Before calling this a defect of the field, we must
rule out the most deflationary explanation available:

    Any estimator fitted on few observations per user shrinks that user's predictions
    toward their own base rate. A user who grants 90% of requests is therefore predicted
    to grant almost everything -- and their few refusals are overridden. This is not a
    bug; it is the bias-variance tradeoff working exactly as designed.

If that is the whole story, the paper has rediscovered shrinkage, and must say so.

Three tests, each answering a different question:

  WITHIN   -- Inside a single corpus, does the effect weaken for users who happen to have
              MORE training data? (Tests the sparsity account without crossing datasets,
              so corpus differences cannot confound it.)

  SYNTH    -- On synthetic data where each user's TRUE per-request preferences are known
              and are NOT a function of their permissiveness, does a standard model
              reproduce the effect anyway? If it does, the effect is a generic property of
              shrinkage under sparsity, not a defect of these systems. If it does NOT, the
              effect requires something the real data has and the synthetic data lacks.

  COST     -- Cost-sensitive learning (class-weighted BCE, focal) is the textbook answer
              to asymmetric error costs. We previously found it does not improve the
              AGGREGATE. But we never asked whether it helps the users who bear the harm.
              If shrinkage toward grants is the mechanism, up-weighting denials should
              counteract it exactly where it hurts.

Unit of analysis is the DENIAL throughout, so no user is excluded.
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
SEED = 42


def _epochs():
    return 15 if C.QUICK else int(os.environ.get("PERM_EPOCHS", 200))


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra = (ra - ra.mean()) / (ra.std() + 1e-9)
    rb = (rb - rb.mean()) / (rb.std() + 1e-9)
    return float((ra * rb).mean())


def summarise(G, V, label, indent=4):
    """G = user's grant rate for each denial; V = was that denial overridden."""
    G, V = np.asarray(G), np.asarray(V)
    lo = 100 * V[G < 0.2].mean() if (G < 0.2).any() else float("nan")
    hi = 100 * V[G >= 0.6].mean() if (G >= 0.6).any() else float("nan")
    r = spearman(G, V)
    pad = " " * indent
    print(f"{pad}{label:26s} n={len(G):6d}  restrictive={lo:5.1f}%  "
          f"permissive={hi:5.1f}%  ratio={hi / max(lo, 0.1):5.1f}x  rho={r:+.3f}")
    return dict(n=int(len(G)), restrictive=lo, permissive=hi,
                ratio=hi / max(lo, 0.1), rho=r)


def collect(records, folds, fit_fn, ep, extra=None):
    """Returns (grant_rate_per_denial, overridden_per_denial, n_train_records_per_denial)."""
    G, V, N = [], [], []
    for f in range(KFOLD):
        tr = [records[i] for i in range(len(records)) if folds[i] != f]
        te = [records[i] for i in range(len(records)) if folds[i] == f]
        g = Graph(tr, records)
        ntr = {}
        for r in tr:
            ntr[r["user_id"]] = ntr.get(r["user_id"], 0) + 1
        pr, prtr = fit_fn(g, tr, te, ep)
        y = np.asarray(pr["y"]); p = np.asarray(pr["p_allow"])
        u = np.array([r["user_id"] for r in te])
        thr = float(M.equal_error_threshold(np.asarray(prtr["y"]),
                                            np.asarray(prtr["p_allow"])))
        yh = (p >= thr).astype(int)
        gr = {uu: float(y[u == uu].mean()) for uu in np.unique(u)}
        dn = (y == 0)
        G += [gr[uu] for uu in u[dn]]
        V += list(yh[dn].astype(float))
        N += [ntr.get(uu, 0) for uu in u[dn]]
    return np.array(G), np.array(V), np.array(N)


def fm_bce(g, tr, te, ep):
    return B.fm_predict(tr, te, objective="bce"), B.fm_predict(tr, tr, objective="bce")


def fm_obj(obj):
    def f(g, tr, te, ep):
        return (B.fm_predict(tr, te, objective=obj),
                B.fm_predict(tr, tr, objective=obj))
    return f


def atomic_bpr(g, tr, te, ep):
    m = train(g, tr, mode="allow", loss="bpr", use_sideinfo=False, epochs=ep)
    return predict(m, g, te), predict(m, g, tr)


# --------------------------------------------------------------- synthetic corpus
def make_synth(n_users=181, n_items=180, per_user=42, seed=0):
    """Each user has a permissiveness p_u, and each REQUEST has an intrinsic
    sensitivity s_q. The user's true label is drawn from a preference that combines the
    two -- crucially, a user's *exceptions* are driven by request sensitivity, NOT by how
    permissive they are. So there is nothing in the ground truth that says 'permissive
    users' refusals are special'. If a model trained on this still overrides permissive
    users' refusals more, the effect is pure shrinkage."""
    rng = np.random.default_rng(seed)
    p_u = rng.uniform(0.05, 0.95, n_users)          # permissiveness
    s_q = rng.normal(0, 1.0, n_items)               # request sensitivity
    recs = []
    for u in range(n_users):
        qs = rng.choice(n_items, size=per_user, replace=False)
        # logit of granting: user's own level + how benign the request is
        base = np.log(p_u[u] / (1 - p_u[u]))
        pr = 1 / (1 + np.exp(-(base - s_q[qs])))
        ys = (rng.random(per_user) < pr).astype(int)
        for q, y in zip(qs, ys):
            recs.append(dict(user_id=f"U{u}", domain=f"d{q % 8}", tool=f"t{q % 20}",
                             data_type=f"c{q % 12}", request_key=f"q{q}", label=int(y)))
    return recs


def main():
    ep = _epochs()
    out = {}

    # ---------------------------------------------------------------- REAL corpus
    records, _ = build_records()
    folds = per_user_folds(records, KFOLD, SEED)
    print("=" * 84)
    print("TEST 1 (WITHIN): does the effect weaken for users with MORE training data?")
    print("=" * 84)
    G, V, N = collect(records, folds, atomic_bpr, ep)
    out["within"] = {}
    qs = np.percentile(N, [33, 67])
    for lab, m in (("fewest records", N <= qs[0]),
                   ("middle", (N > qs[0]) & (N <= qs[1])),
                   ("most records", N > qs[1])):
        if m.sum() > 20:
            med = int(np.median(N[m]))
            out["within"][lab] = summarise(G[m], V[m], f"{lab} (~{med} recs/user)")
    print("\n    If sparsity is the cause, the effect should SHRINK from top row to bottom.")

    # ---------------------------------------------------------------- SYNTHETIC
    print("\n" + "=" * 84)
    print("TEST 2 (SYNTH): does a model reproduce the effect on data where, by")
    print("construction, permissiveness does NOT make a user's refusals special?")
    print("=" * 84)
    srecs = make_synth(per_user=42)
    sfolds = per_user_folds(srecs, KFOLD, SEED)
    Gs, Vs, _ = collect(srecs, sfolds, atomic_bpr, ep)
    out["synth_sparse"] = summarise(Gs, Vs, "synthetic, 42 recs/user")
    srecs2 = make_synth(per_user=150, seed=1)
    sfolds2 = per_user_folds(srecs2, KFOLD, SEED)
    Gs2, Vs2, _ = collect(srecs2, sfolds2, atomic_bpr, ep)
    out["synth_dense"] = summarise(Gs2, Vs2, "synthetic, 150 recs/user")
    print("\n    If the SPARSE synthetic run shows the effect and the DENSE one does not,")
    print("    the phenomenon is shrinkage under sparsity -- a statistical inevitability,")
    print("    not a defect peculiar to these systems.")

    # ---------------------------------------------------------------- COST-SENSITIVE
    print("\n" + "=" * 84)
    print("TEST 3 (COST): does up-weighting denials protect the users who are harmed?")
    print("=" * 84)
    out["cost"] = {}
    for obj, lab in (("bce", "plain calibrated BCE"),
                     ("wbce", "class-weighted BCE"),
                     ("focal", "focal loss")):
        Gc, Vc, _ = collect(records, folds, fm_obj(obj), ep)
        out["cost"][obj] = summarise(Gc, Vc, lab)
    print("\n    We already know reweighting does not move the AGGREGATE. The question is")
    print("    whether it moves the PERMISSIVE column -- the only column that matters.")

    C.save(out, "table26_shrinkage.json")
    print("\nSaved -> table26_shrinkage.json")


if __name__ == "__main__":
    main()
