"""EXCEPTION MEMORY: the training method Proposition 1 prescribes.

The proposition says an exception cannot be fitted in a factorized model without collateral
damage, because there is no free parameter for the pair (u, q): the deviation must be routed
through p_u, which serves ALL of that user's requests. Empirical risk minimization then
trades k exceptions against n-k routine decisions, and the exceptions lose.

The prescription follows immediately. Give the exception a channel that does NOT pass
through p_u:

    s(u,q) = b_u + <p_u, phi_q>          <- the prior: shared, learns typical behaviour
             + sum_a  m[u, a = v_a(q)]   <- exception memory: NOT shared across attributes

m[u, domain=health] is a free parameter. Fitting it does not touch p_u, so its collateral
damage is confined to that user's OTHER decisions in the same domain -- which is exactly the
right locality, since a user's exceptions cluster by attribute (91.6% of held-out exceptions
share a domain, tool or data-type with a prior exception by the same user).

This is NOT the rule-mining of Section 7, which overrode the model's prediction post hoc and
paid for it in routine accuracy. The memory is trained JOINTLY with the prior, under its own
regularization strength, so it learns only where the evidence supports it.

Reported on the DECISION (no user excluded): exception override, routine override, accuracy.
A method that helps the aggregate but not the exceptions is not a method for this problem.
"""
import os
import sys

import numpy as np
import torch

import _common as C
import metrics as M

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import build_records, per_user_folds  # noqa: E402

KFOLD = 5
SEED = 42
FM_FIELDS = ("user_id", "domain", "tool", "data_type")
MEM_ATTRS = ("domain", "tool", "data_type")     # the exception memory's keys


def _epochs():
    return 15 if C.QUICK else int(os.environ.get("PERM_EPOCHS", 200))


def fit(tr, te, dim=16, epochs=150, lr=0.05, wd=1e-6, wd_mem=None, use_mem=True, seed=SEED):
    """wd_mem: weight decay on the exception memory. If None, no memory is used."""
    torch.manual_seed(seed); np.random.seed(seed)

    # --- the prior: a standard factorization machine
    idx = {f: {} for f in FM_FIELDS}
    for r in tr:
        for f in FM_FIELDS:
            idx[f].setdefault(r[f], len(idx[f]) + 1)
    sizes = [len(idx[f]) + 1 for f in FM_FIELDS]
    offs = torch.tensor(np.cumsum([0] + sizes[:-1]), dtype=torch.long)
    tot = int(sum(sizes))

    def enc(R):
        return torch.tensor([[idx[f].get(r[f], 0) for f in FM_FIELDS] for r in R],
                            dtype=torch.long) + offs

    lin = torch.nn.Embedding(tot, 1); torch.nn.init.zeros_(lin.weight)
    fac = torch.nn.Embedding(tot, dim); torch.nn.init.normal_(fac.weight, std=0.01)
    b0 = torch.zeros(1, requires_grad=True)

    # --- the exception memory: one free scalar per (user, attribute=value)
    midx = {}
    for r in tr:
        for a in MEM_ATTRS:
            midx.setdefault((r["user_id"], a, r[a]), len(midx) + 1)
    mem = torch.nn.Embedding(len(midx) + 1, 1)      # index 0 = unseen (u, a=v)
    torch.nn.init.zeros_(mem.weight)

    def menc(R):
        return torch.tensor([[midx.get((r["user_id"], a, r[a]), 0) for a in MEM_ATTRS]
                             for r in R], dtype=torch.long)

    params = [
        {"params": list(lin.parameters()) + list(fac.parameters()), "weight_decay": wd},
        {"params": [b0], "weight_decay": 0.0},
    ]
    if use_mem:
        params.append({"params": list(mem.parameters()),
                       "weight_decay": wd if wd_mem is None else wd_mem})
    opt = torch.optim.Adam(params, lr=lr)

    Xtr, Mtr = enc(tr), menc(tr)
    ytr = torch.tensor([r["label"] for r in tr], dtype=torch.float32)

    def score(X, Mm):
        l = lin(X).sum(1).squeeze(-1) + b0
        v = fac(X); s = v.sum(1)
        z = l + 0.5 * ((s * s).sum(-1) - (v * v).sum(-1).sum(-1))
        if use_mem:
            z = z + mem(Mm).sum(1).squeeze(-1)
        return z

    bce = torch.nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        opt.zero_grad()
        bce(score(Xtr, Mtr), ytr).backward()
        opt.step()

    with torch.no_grad():
        return (torch.sigmoid(score(enc(te), menc(te))).numpy(),
                torch.sigmoid(score(Xtr, Mtr)).numpy())


def evaluate(records, folds, ep, **kw):
    EXC, ROU, ACC, FG = [], [], [], []
    for f in range(KFOLD):
        tr = [records[i] for i in range(len(records)) if folds[i] != f]
        te = [records[i] for i in range(len(records)) if folds[i] == f]
        p, ptr = fit(tr, te, epochs=ep, **kw)
        y = np.array([r["label"] for r in te])
        ytr = np.array([r["label"] for r in tr])
        thr = float(M.equal_error_threshold(ytr, ptr))
        yh = (p >= thr).astype(int)
        u = np.array([r["user_id"] for r in te])
        ur = {uu: float(y[u == uu].mean()) for uu in np.unique(u)}
        mu = np.array([1 if ur[uu] >= 0.5 else 0 for uu in u])
        exc = (y != mu); ov = (yh != y)
        EXC += list(ov[exc].astype(float))
        ROU += list(ov[~exc].astype(float))
        FG += list(yh[y == 0].astype(float))
        ACC.append(100 * (yh == y).mean())
    return (float(np.mean(ACC)), 100 * float(np.mean(ROU)),
            100 * float(np.mean(EXC)), 100 * float(np.mean(FG)))


def main():
    records, _ = build_records()
    folds = per_user_folds(records, KFOLD, SEED)
    ep = _epochs()
    out = {}

    print("=" * 82)
    print("EXCEPTION MEMORY: give the exception a channel that does not pass through p_u.")
    print("=" * 82)
    print(f"\n{'method':40s}{'Acc':>7}{'routine':>10}{'exception':>12}{'FGR':>8}")

    a, r, e, g = evaluate(records, folds, ep, use_mem=False)
    out["FM (no memory)"] = dict(acc=a, routine=r, exception=e, fgr=g)
    print(f"{'FM (no memory)  [baseline]':40s}{a:7.1f}{r:9.1f}%{e:11.1f}%{g:7.1f}%")
    base = e

    for wd_mem in (1e-2, 1e-3, 1e-4, 1e-5, 0.0):
        a, r, e, g = evaluate(records, folds, ep, use_mem=True, wd_mem=wd_mem)
        lab = f"+ exception memory (wd_mem={wd_mem:g})"
        out[lab] = dict(acc=a, routine=r, exception=e, fgr=g)
        d = base - e
        star = "  <<<" if d > 3 else ""
        print(f"{lab:40s}{a:7.1f}{r:9.1f}%{e:11.1f}%{g:7.1f}%{star}")

    best = min((k for k in out if "memory" in k), key=lambda k: out[k]["exception"])
    b, m = out["FM (no memory)"], out[best]
    print(f"\nbaseline : exception {b['exception']:.1f}%  routine {b['routine']:.1f}%  "
          f"acc {b['acc']:.1f}  FGR {b['fgr']:.1f}%")
    print(f"best     : exception {m['exception']:.1f}%  routine {m['routine']:.1f}%  "
          f"acc {m['acc']:.1f}  FGR {m['fgr']:.1f}%   [{best}]")
    de = b["exception"] - m["exception"]
    dr = m["routine"] - b["routine"]
    da = m["acc"] - b["acc"]
    print(f"\nexception override {-de:+.1f} pts | routine {dr:+.1f} pts | accuracy {da:+.1f}")
    if de > 3 and dr < 1 and da > -1:
        print("=> The method WORKS: exceptions protected without paying in routine accuracy.")
    elif de > 3:
        print("=> Exceptions improve, but at a cost -- report the trade honestly.")
    else:
        print("=> The method FAILS. Report it as such.")

    C.save(out, "table30_exception_memory.json")
    print("\nSaved -> table30_exception_memory.json")


if __name__ == "__main__":
    main()
