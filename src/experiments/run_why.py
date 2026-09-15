"""Why are permissive users' refusals overridden? Three decisive tests.

Established so far: a permission model overrides a permissive user's rare refusals far
more often than a restrictive user's (3% vs 30% in the released system), and the refusals
it overrides are on requests the crowd ALSO tends to deny -- so the signal is present in
the data and the model is failing to use it. Six explanations have been tested and
rejected (ranking objective, denial representation, sparsity, cost-sensitive loss,
idiosyncrasy, and a synthetic replication).

The synthetic test, however, had a flaw: its labels were drawn STOCHASTICALLY from the
preference probability, so a restrictive user's "denials" contained many samples whose
true probability was moderate -- pure label noise, which inflates their false-grant rate
and destroys the contrast. Real stated preferences are DETERMINISTIC. Test 1 redoes it.

TEST 1 (SYNTH-DET) -- deterministic labels: y = 1[base_u > s_q]. If the effect now appears,
    the earlier "synthetic does not reproduce it" conclusion was an artifact of our own
    design, and the phenomenon IS generic to an additive score with a global threshold.

TEST 2 (VARIANCE) -- decompose the model's score into a user component and a request
    component. If the user component dominates, then a permissive user's high baseline
    swamps the request signal, and no amount of request-level evidence can pull their
    score below a GLOBAL threshold. This is the quantitative form of the hypothesis.

TEST 3 (INTERACTION) -- an additive/bilinear score f(user)+g(request) structurally cannot
    say "even though this user is permissive, THIS request is one they refuse." A model
    that represents interactions (gradient-boosted trees over the same features) can. If
    the harmed users improve under GBDT, the additive structure is the culprit and the
    remedy is architectural, not a threshold patch.
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


def summarise(G, V, label):
    G, V = np.asarray(G), np.asarray(V)
    lo = 100 * V[G < 0.2].mean() if (G < 0.2).any() else float("nan")
    hi = 100 * V[G >= 0.6].mean() if (G >= 0.6).any() else float("nan")
    r = spearman(G, V)
    print(f"    {label:30s} n={len(G):6d}  restrictive={lo:5.1f}%  "
          f"permissive={hi:5.1f}%  ratio={hi / max(lo, 0.1):5.1f}x  rho={r:+.3f}")
    return dict(n=int(len(G)), restrictive=float(lo), permissive=float(hi),
                ratio=float(hi / max(lo, 0.1)), rho=r)


def evaluate(records, folds, fit, ep):
    G, V = [], []
    for f in range(KFOLD):
        tr = [records[i] for i in range(len(records)) if folds[i] != f]
        te = [records[i] for i in range(len(records)) if folds[i] == f]
        g = Graph(tr, records)
        pr, prtr = fit(g, tr, te, ep)
        y = np.asarray(pr["y"]); p = np.asarray(pr["p_allow"])
        u = np.array([r["user_id"] for r in te])
        thr = float(M.equal_error_threshold(np.asarray(prtr["y"]),
                                            np.asarray(prtr["p_allow"])))
        yh = (p >= thr).astype(int)
        gr = {uu: float(y[u == uu].mean()) for uu in np.unique(u)}
        dn = (y == 0)
        G += [gr[uu] for uu in u[dn]]
        V += list(yh[dn].astype(float))
    return np.array(G), np.array(V)


def fm_bce(g, tr, te, ep):
    return B.fm_predict(tr, te, objective="bce"), B.fm_predict(tr, tr, objective="bce")


def atomic_bpr(g, tr, te, ep):
    m = train(g, tr, mode="allow", loss="bpr", use_sideinfo=False, epochs=ep)
    return predict(m, g, te), predict(m, g, tr)


_MLP_FIELDS = ("user_id", "domain", "tool", "data_type")


def mlp_predict(tr, te, dim=16, hidden=64, epochs=150, lr=0.05, wd=1e-6, seed=42):
    """Same features, same optimizer, same budget as the factorization machine -- the ONLY
    difference is that an MLP over the concatenated embeddings can represent an
    INTERACTION between who the user is and what the request is, whereas the FM's score is
    a sum of a linear term and pairwise dot products. If the harmed users improve here and
    nowhere else, the additive/bilinear form is the culprit."""
    import torch
    torch.manual_seed(seed); np.random.seed(seed)
    idx = {f: {} for f in _MLP_FIELDS}
    for r in tr:
        for f in _MLP_FIELDS:
            idx[f].setdefault(r[f], len(idx[f]) + 1)
    sizes = [len(idx[f]) + 1 for f in _MLP_FIELDS]

    def enc(recs):
        return torch.tensor([[idx[f].get(r[f], 0) for f in _MLP_FIELDS] for r in recs],
                            dtype=torch.long)
    offs = torch.tensor(np.cumsum([0] + sizes[:-1]), dtype=torch.long)
    emb = torch.nn.Embedding(int(sum(sizes)), dim)
    torch.nn.init.normal_(emb.weight, std=0.01)
    net = torch.nn.Sequential(torch.nn.Linear(dim * len(_MLP_FIELDS), hidden),
                              torch.nn.ReLU(), torch.nn.Linear(hidden, 1))
    opt = torch.optim.Adam(list(emb.parameters()) + list(net.parameters()),
                           lr=lr, weight_decay=wd)
    Xtr, ytr = enc(tr) + offs, torch.tensor([r["label"] for r in tr], dtype=torch.float32)
    bce = torch.nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        opt.zero_grad()
        h = emb(Xtr).reshape(len(tr), -1)
        loss = bce(net(h).squeeze(-1), ytr)
        loss.backward(); opt.step()
    with torch.no_grad():
        Xte = enc(te) + offs
        p = torch.sigmoid(net(emb(Xte).reshape(len(te), -1)).squeeze(-1)).numpy()
    return {"y": np.array([r["label"] for r in te]), "p_allow": p}


def mlp(g, tr, te, ep):
    return mlp_predict(tr, te), mlp_predict(tr, tr)


# ------------------------------------------------------------------ synthetic
def make_synth(n_users=181, n_items=180, per_user=42, seed=0, deterministic=True):
    rng = np.random.default_rng(seed)
    p_u = rng.uniform(0.05, 0.95, n_users)
    s_q = rng.normal(0, 1.0, n_items)
    recs = []
    for u in range(n_users):
        qs = rng.choice(n_items, size=per_user, replace=False)
        base = np.log(p_u[u] / (1 - p_u[u]))
        z = base - s_q[qs]
        if deterministic:
            ys = (z > 0).astype(int)          # <-- no label noise: a stated preference
        else:
            ys = (rng.random(per_user) < 1 / (1 + np.exp(-z))).astype(int)
        for q, y in zip(qs, ys):
            recs.append(dict(user_id=f"U{u}", domain=f"d{q % 8}", tool=f"t{q % 20}",
                             data_type=f"c{q % 12}", request_key=f"q{q}", label=int(y)))
    return recs


def main():
    ep = _epochs()
    out = {}
    records, _ = build_records()
    folds = per_user_folds(records, KFOLD, SEED)

    # ---------------------------------------------------- TEST 1
    print("=" * 88)
    print("TEST 1 (SYNTH-DET): our earlier synthetic used STOCHASTIC labels, which inject")
    print("label noise into restrictive users' denials. Real preferences are deterministic.")
    print("=" * 88)
    out["synth"] = {}
    for det, lab in ((False, "stochastic labels (our old, flawed design)"),
                     (True, "deterministic labels (as in real data)")):
        srecs = make_synth(per_user=42, deterministic=det)
        sfolds = per_user_folds(srecs, KFOLD, SEED)
        G, V = evaluate(srecs, sfolds, atomic_bpr, ep)
        out["synth"]["det" if det else "stoch"] = summarise(G, V, lab)
    print("\n    If the deterministic row shows the effect, the phenomenon is generic to an")
    print("    additive score + a global threshold -- and our earlier refutation was our bug.")

    # ---------------------------------------------------- TEST 2
    print("\n" + "=" * 88)
    print("TEST 2 (VARIANCE): how much of the model's score is the USER, and how much is")
    print("the REQUEST? If the user term dominates, a permissive user's baseline swamps")
    print("any request-level evidence, and no global threshold can catch their refusals.")
    print("=" * 88)
    out["variance"] = {}
    for corpus_lab, recs, flds in (("real (Wu)", records, folds),):
        for mlab, fit in (("ranking CF", atomic_bpr), ("calibrated FM", fm_bce)):
            ub, qb = [], []
            for f in range(KFOLD):
                tr = [recs[i] for i in range(len(recs)) if flds[i] != f]
                te = [recs[i] for i in range(len(recs)) if flds[i] == f]
                g = Graph(tr, recs)
                pr, _ = fit(g, tr, te, ep)
                p = np.asarray(pr["p_allow"])
                z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
                u = np.array([r["user_id"] for r in te])
                q = np.array([f"{r['domain']}|{r['tool']}|{r['data_type']}" for r in te])
                # variance of the group means = variance explained by that factor
                um = np.array([z[u == uu].mean() for uu in np.unique(u)])
                qm = np.array([z[q == qq].mean() for qq in np.unique(q)])
                ub.append(um.var()); qb.append(qm.var())
            uv, qv = float(np.mean(ub)), float(np.mean(qb))
            share = 100 * uv / (uv + qv)
            out["variance"][mlab] = dict(user_var=uv, request_var=qv, user_share=share)
            print(f"    {mlab:20s} user-term var={uv:6.3f}  request-term var={qv:6.3f}"
                  f"   -> USER explains {share:.0f}% of the between-group score spread")

    # ---------------------------------------------------- TEST 3
    print("\n" + "=" * 88)
    print("TEST 3 (INTERACTION): an additive/bilinear score cannot say 'permissive user,")
    print("but THIS request is one they refuse'. Gradient-boosted trees can. Do the harmed")
    print("users improve?")
    print("=" * 88)
    out["interaction"] = {}
    for mlab, fit in (("FM (bilinear)", fm_bce), ("MLP (interactions)", mlp)):
        try:
            G, V = evaluate(records, folds, fit, ep)
            out["interaction"][mlab] = summarise(G, V, mlab)
        except Exception as e:  # noqa: BLE001
            print(f"    {mlab}: unavailable ({str(e)[:60]})")
    print("\n    If the MLP's 'permissive' column falls well below the FM's, the additive structure")
    print("    is the culprit and the remedy is architectural, not a threshold patch.")

    C.save(out, "table27_why.json")
    print("\nSaved -> table27_why.json")


if __name__ == "__main__":
    main()
