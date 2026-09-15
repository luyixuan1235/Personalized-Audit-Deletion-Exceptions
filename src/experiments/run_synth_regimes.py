"""Synthetic control, three estimator regimes (paper Table tab:synth).

Generative model: each user a latent threshold b_u, each request a latent
sensitivity s_q, label y = 1[b_u > s_q] -- deterministic, no exception structure.
Estimator: pooled logistic regression over user + request indicator features
(the exact design of Proposition 1), trained with the repo's standard recipe
(full-batch Adam, lr 0.05). What varies is the regime:
  1. unregularized, trained to convergence   (wd=0,    epochs=5000)
  2. unregularized, limited budget           (wd=0,    epochs=150)
  3. L2-regularized, limited budget          (wd=1e-2, epochs=150)
Metrics, at the train-fold equal-error threshold: refusals overridden by the
refusing user's grant rate (restrictive < 0.2 / permissive >= 0.6), and the
exception/routine override rates (y != m_u vs y == m_u).

Run: cd src && python3 experiments/run_synth_regimes.py
Output: results/table47_synth_regimes.json
"""
import json, os, sys
import numpy as np
import torch

sys.path.insert(0, '.')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import metrics as M
from data import per_user_folds

KFOLD, SEED = 5, 42
torch.set_num_threads(4)


def make_synth(n_users=181, n_items=180, per_user=42, seed=0):
    rng = np.random.default_rng(seed)
    p_u = rng.uniform(0.05, 0.95, n_users)
    s_q = rng.normal(0, 1.0, n_items)
    recs = []
    for u in range(n_users):
        qs = rng.choice(n_items, size=per_user, replace=False)
        base = np.log(p_u[u] / (1 - p_u[u]))
        ys = ((base - s_q[qs]) > 0).astype(int)      # deterministic label
        for q, y in zip(qs, ys):
            recs.append(dict(user_id=f"U{u}", domain=f"d{q % 8}", tool=f"t{q % 20}",
                             data_type=f"c{q % 12}", request_key=f"q{q}", label=int(y)))
    return recs


def regime(recs, folds, ui, qi, D, gr, mu, epochs, wd, lr=0.05):
    G, V, EXC, ROU, INS = [], [], [], [], []
    for f in range(KFOLD):
        tr = [recs[i] for i in range(len(recs)) if folds[i] != f]
        te = [recs[i] for i in range(len(recs)) if folds[i] == f]
        torch.manual_seed(SEED)
        w = torch.zeros(D, requires_grad=True); b = torch.zeros(1, requires_grad=True)
        opt = torch.optim.Adam([w, b], lr=lr, weight_decay=wd)
        Xtr = torch.tensor([[ui[r["user_id"]], qi[r["request_key"]]] for r in tr])
        ytr = torch.tensor([float(r["label"]) for r in tr])
        for _ in range(epochs):
            opt.zero_grad()
            torch.nn.functional.binary_cross_entropy_with_logits(
                w[Xtr].sum(1) + b, ytr).backward()
            opt.step()
        with torch.no_grad():
            ptr = torch.sigmoid(w[Xtr].sum(1) + b).numpy()
            Xte = torch.tensor([[ui[r["user_id"]], qi[r["request_key"]]] for r in te])
            p = torch.sigmoid(w[Xte].sum(1) + b).numpy()
        thr = float(M.equal_error_threshold(
            np.array([r["label"] for r in tr]), ptr))
        for r, pp, yy in zip(tr, ptr, [x["label"] for x in tr]):
            if yy != mu[r["user_id"]]:
                INS.append(float((pp >= thr) != yy))
        for r, pp in zip(te, p):
            ov = float((pp >= thr) != r["label"])
            (EXC if r["label"] != mu[r["user_id"]] else ROU).append(ov)
            if r["label"] == 0:
                G.append(gr[r["user_id"]]); V.append(float(pp >= thr))
    G, V = np.asarray(G), np.asarray(V)
    return dict(restrictive=float(100 * V[G < 0.2].mean()),
                permissive=float(100 * V[G >= 0.6].mean()),
                eor_exc=float(100 * np.mean(EXC)), eor_rout=float(100 * np.mean(ROU)),
                insample_exc=float(100 * np.mean(INS)), n_refusals=int(len(V)))


def main():
    seeds = [int(x) for x in os.environ.get("SYNTH_SEEDS", "0,1,2,3,4").split(",")]
    per = {}
    for sd in seeds:
        print(f"--- generation seed {sd}")
        for key, d in _one(sd, save=(len(seeds) == 1)).items():
            per.setdefault(key, []).append(dict(d, gen_seed=sd))
    if len(seeds) > 1:
        summ = {k: {m: [round(float(np.mean([r[m] for r in rows])), 1),
                        round(float(np.std([r[m] for r in rows])), 1)]
                    for m in ("restrictive", "permissive", "eor_exc",
                              "eor_rout", "insample_exc")}
                for k, rows in per.items()}
        for k, v in summ.items():
            print(k, v)
        rd = os.environ.get("PERM_RESULTS_DIR", "../results")
        with open(os.path.join(rd, "table47_synth_regimes.json"), "w") as fh:
            json.dump({"per_seed": per, "mean_sd": summ}, fh, indent=1)
        print("saved table47_synth_regimes.json (multi-seed)")


def _one(gen_seed=0, save=True):
    recs = make_synth(seed=gen_seed)
    folds = per_user_folds(recs, KFOLD, SEED)
    users = sorted({r["user_id"] for r in recs})
    items = sorted({r["request_key"] for r in recs})
    ui = {u: i for i, u in enumerate(users)}
    qi = {q: i + len(users) for i, q in enumerate(items)}
    D = len(users) + len(items)
    gr = {}
    for r in recs:
        gr.setdefault(r["user_id"], []).append(r["label"])
    gr = {u: float(np.mean(v)) for u, v in gr.items()}
    mu = {u: (1 if g >= 0.5 else 0) for u, g in gr.items()}
    out = {}
    for key, ep, wd in (("unregularized_converged", 5000, 0.0),
                        ("unregularized_limited",   150,  0.0),
                        ("l2_limited",              150,  1e-2)):
        out[key] = regime(recs, folds, ui, qi, D, gr, mu, ep, wd)
        print(key, {k: round(v, 1) if isinstance(v, float) else v
                    for k, v in out[key].items()}, flush=True)
    if save:
        rd = os.environ.get("PERM_RESULTS_DIR", "../results")
        with open(os.path.join(rd, "table47_synth_regimes.json"), "w") as fh:
            json.dump(out, fh, indent=1)
        print("saved table47_synth_regimes.json")
    return out


if __name__ == "__main__":
    main()
