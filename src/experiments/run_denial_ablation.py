"""Controlled ablation: does REPRESENTING DENIAL protect a user's rare refusals?

The observation (Section: Observation) is that a permission model overrides the refusals
of PERMISSIVE users far more often than those of restrictive ones -- 3% vs 30% of denials
in the released system. The proposed cause is that a model which never learns what a user
REFUSES can only learn what they WANT, so a permissive user's appetite drowns out their
few, specific boundaries.

Across configurations this appears as a gradient. This script reports a configuration ladder,
not a single-factor ablation: architecture, objective, and negative sampling differ across
steps while each step changes how denial enters training:

  discard  -- denials thrown away entirely; negatives sampled from UNOBSERVED requests.
              This is the released system's recipe (their code: train[train.rating > 0]),
              i.e. implicit feedback / ImplicitCF.
  bpr      -- the user's actual denials used as the negatives of the pairwise loss.
  bce      -- denials used as pointwise labels (calibrated objective).
  signed   -- denials given their own evidence channel (signed graph).

Measured on DENIALS, not users (no user is excluded, and no per-user rate is formed, so
users with a single rare refusal -- exactly the ones the hypothesis is about -- are in the
sample):

    rho( user's grant rate , this denial was overridden )
    override rate for restrictive users (grant rate < 0.2) vs permissive (>= 0.6)

Descriptive comparison: report the measured sequence across the four configurations.
Because multiple training choices differ, the sequence is not interpreted as a causal
effect of denial representation alone.
"""
import os
import sys
from collections import defaultdict

import numpy as np
import torch

import _common as C
import baselines_ext as B
import metrics as M

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import Graph, build_records, per_user_folds  # noqa: E402
from losses import bpr_loss  # noqa: E402
from model import DualGraph  # noqa: E402
from train import predict, train  # noqa: E402

KFOLD = 5
SEED = 42


def _epochs():
    return 15 if C.QUICK else int(os.environ.get("PERM_EPOCHS", 200))


def train_discard(graph, tr, epochs, seed=SEED):
    """The released system's recipe: DISCARD the denials. Train BPR on grants only, with
    negatives drawn from requests the user has NOT interacted with (implicit feedback).
    The model therefore never sees a single refusal."""
    torch.manual_seed(seed); np.random.seed(seed)
    u, q, y = graph.encode(tr)
    model = DualGraph(graph, dim=32, layers=3, mode="allow", comp=(False,) * 3, device="cpu")
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)

    pos = defaultdict(list)
    seen = defaultdict(set)
    for i in range(len(u)):
        seen[int(u[i])].add(int(q[i]))
        if y[i] == 1:                       # <-- grants only; denials discarded
            pos[int(u[i])].append(int(q[i]))
    n_items = int(max(q)) + 1
    users = [uu for uu in pos if len(pos[uu]) > 0]
    rng = np.random.default_rng(seed)

    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        bu, bp, bn = [], [], []
        for uu in users:
            bu.append(uu)
            bp.append(rng.choice(pos[uu]))
            for _try in range(20):          # negative = an UNOBSERVED request
                cand = int(rng.integers(0, n_items))
                if cand not in seen[uu]:
                    break
            bn.append(cand)
        bu = torch.tensor(bu)
        sp_a, sp_d = model.scores(bu, torch.tensor(bp))
        sn_a, sn_d = model.scores(bu, torch.tensor(bn))
        loss = bpr_loss(sp_a - sp_d, sn_a - sn_d)
        loss.backward(); opt.step()
    return model


CELLS = ["discard", "bpr", "bce", "signed"]


def _fit(cell, g, tr, te, ep):
    if cell == "discard":
        m = train_discard(g, tr, ep)
        return predict(m, g, te), predict(m, g, tr)
    if cell == "signed":
        m = B.train_signed(g, tr, epochs=ep, use_sideinfo=False, objective="bce")
        return predict(m, g, te), predict(m, g, tr)
    m = train(g, tr, mode="allow", loss=cell, use_sideinfo=False, epochs=ep)
    return predict(m, g, te), predict(m, g, tr)


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra = (ra - ra.mean()) / (ra.std() + 1e-9)
    rb = (rb - rb.mean()) / (rb.std() + 1e-9)
    return float((ra * rb).mean())


def main():
    records, _ = build_records()
    ep = _epochs()
    folds = per_user_folds(records, KFOLD, SEED)
    acc = {c: dict(g=[], v=[], accy=[], fgr=[]) for c in CELLS}

    for f in range(KFOLD):
        tr = [records[i] for i in range(len(records)) if folds[i] != f]
        te = [records[i] for i in range(len(records)) if folds[i] == f]
        g = Graph(tr, records)
        u = np.array([r["user_id"] for r in te])
        print(f"[fold {f+1}/{KFOLD}]")
        for cell in CELLS:
            pr, prtr = _fit(cell, g, tr, te, ep)
            y = np.asarray(pr["y"]); p = np.asarray(pr["p_allow"])
            thr = float(M.equal_error_threshold(np.asarray(prtr["y"]),
                                                np.asarray(prtr["p_allow"])))
            yh = (p >= thr).astype(int)
            gr = {uu: float(y[u == uu].mean()) for uu in np.unique(u)}
            dn = (y == 0)
            acc[cell]["g"] += [gr[uu] for uu in u[dn]]      # the user's grant rate
            acc[cell]["v"] += list(yh[dn].astype(float))    # was this denial overridden?
            acc[cell]["accy"].append(100 * M.basic(y, p, thr=thr)["acc"])
            acc[cell]["fgr"].append(100 * float(yh[dn].mean()))

    out = {}
    print("\n" + "=" * 82)
    print("Unit of analysis = ONE DENIAL. No user excluded, no per-user rate formed.")
    print("=" * 82)
    print(f"{'denial representation':24s}{'Acc':>6}{'FGR':>7}{'restrict.':>11}"
          f"{'permiss.':>10}{'ratio':>8}{'rho':>8}{'p':>8}")
    for cell in CELLS:
        a = acc[cell]
        G = np.array(a["g"]); V = np.array(a["v"])
        r = spearman(G, V)
        rng = np.random.default_rng(1)
        null = np.array([spearman(G, rng.permutation(V)) for _ in range(2000)])
        pv = float((null >= r).mean())
        lo = 100 * V[G < 0.2].mean() if (G < 0.2).any() else float("nan")
        hi = 100 * V[G >= 0.6].mean() if (G >= 0.6).any() else float("nan")
        out[cell] = dict(acc=float(np.mean(a["accy"])), fgr=float(np.mean(a["fgr"])),
                         restrictive=lo, permissive=hi, ratio=hi / max(lo, 0.1),
                         rho=r, p=pv, n_denials=int(len(G)))
        print(f"{cell:24s}{out[cell]['acc']:6.1f}{out[cell]['fgr']:7.1f}"
              f"{lo:10.1f}%{hi:9.1f}%{hi / max(lo, 0.1):8.1f}x{r:8.3f}{pv:8.4f}")

    print("\nDescriptive sequence across denial-training configurations:")
    print("   discard > bpr > bce > signed")
    seq = [out[c]["ratio"] for c in CELLS]
    mono = all(seq[i] >= seq[i + 1] for i in range(len(seq) - 1))
    print(f"   observed ratios: " + " > ".join(f"{s:.1f}x" for s in seq))
    print(f"   -> monotone sequence: {'YES' if mono else 'NO'} (descriptive; not single-factor)")
    if not mono:
        print("   The displayed configuration sequence is not monotone.")
    C.save(out, "table24_denial_ablation.json")
    print("\nSaved -> table24_denial_ablation.json")


if __name__ == "__main__":
    main()
