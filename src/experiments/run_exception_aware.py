"""Exception-aware learning: a method derived from the diagnosis, not bolted onto it.

The mechanism (Sections 4-5) says a personalized model erases exceptions because it is,
by construction, an estimator of the user's typical behaviour. Two facts from the
falsification section tell us where an intervention must act:

  * Reweighting FAILS. Up-weighting each user's minority class changes the SIZE of the
    gradient on exceptions but not the fact that the model's capacity is being spent on
    the prior. (Real result: permissive users' refusals 30.5% -> 31.3%.)
  * Regularization AMPLIFIES. Relaxing weight decay from 1e-2 to 1e-4 halves the exception
    override rate. Under L2, the *prior* term is well-supported and survives shrinkage;
    the *deviation* term is thinly supported and is shrunk to death.

That points at the fix. Do not ask one term to carry both the prior and the deviation from
it, and do not regularize them alike. Decompose the score explicitly:

    s(u,q) = PRIOR(u,q)  +  DEV(u,q)

where PRIOR is the thing the model would learn anyway -- the user's own base rate and the
crowd's base rate for this kind of request -- and DEV is what is left: precisely the
exception signal. Fit PRIOR in closed form (no capacity spent, nothing to shrink), and
give DEV the model's capacity with its OWN regularization strength.

Three variants, so the credit is attributable:

  base        -- plain calibrated FM (the baseline)
  residual    -- fit the closed-form prior, then train the FM on the RESIDUAL y - prior
  residual+   -- as above, and do not regularize the deviation term (wd_dev = 0)

Reported on the DECISION (no user excluded): the exception override rate, the routine
override rate, and accuracy. A method that helps only the aggregate is not a method for
this problem.
"""
import os
import sys
from collections import defaultdict

import numpy as np
import torch

import _common as C
import metrics as M

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import build_records, per_user_folds  # noqa: E402

KFOLD = 5
SEED = 42
FIELDS = ("user_id", "domain", "tool", "data_type")


def _epochs():
    return 15 if C.QUICK else int(os.environ.get("PERM_EPOCHS", 200))


def _logit(p, eps=1e-3):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def closed_form_prior(tr, ev, alpha=5.0):
    """The two priors, in closed form. No parameters, nothing for L2 to shrink.
    user prior : the user's own grant rate (smoothed toward the global rate)
    crowd prior: the grant rate of this (tool, data-type) conjunction among OTHER users
    Returns the prior as a LOGIT, which is what the model's score is measured in."""
    glob = float(np.mean([r["label"] for r in tr]))
    ur = defaultdict(lambda: [0.0, 0.0])
    cr = defaultdict(lambda: [0.0, 0.0])
    for r in tr:
        a = ur[r["user_id"]]; a[0] += r["label"]; a[1] += 1
        b = cr[(r["tool"], r["data_type"])]; b[0] += r["label"]; b[1] += 1

    def p_of(rec):
        a = ur.get(rec["user_id"], [0.0, 0.0])
        b = cr.get((rec["tool"], rec["data_type"]), [0.0, 0.0])
        pu = (a[0] + alpha * glob) / (a[1] + alpha)          # user prior
        pc = (b[0] + alpha * glob) / (b[1] + alpha)          # crowd prior
        return 0.5 * (_logit(pu) + _logit(pc))               # both, in logit space
    return np.array([p_of(r) for r in ev], dtype=np.float32)


def fm(tr, te, mode, dim=16, epochs=150, lr=0.05, wd=1e-6, wd_dev=None, seed=SEED):
    """mode: 'base'    -- predict y directly (the standard model)
             'residual'-- predict y with the closed-form prior as a fixed OFFSET, so the
                          model's parameters are free to carry only the DEVIATION.
       wd_dev: weight decay on the deviation term (defaults to wd)."""
    torch.manual_seed(seed); np.random.seed(seed)
    idx = {f: {} for f in FIELDS}
    for r in tr:
        for f in FIELDS:
            idx[f].setdefault(r[f], len(idx[f]) + 1)
    sizes = [len(idx[f]) + 1 for f in FIELDS]

    def enc(R):
        return torch.tensor([[idx[f].get(r[f], 0) for f in FIELDS] for r in R],
                            dtype=torch.long)
    offs = torch.tensor(np.cumsum([0] + sizes[:-1]), dtype=torch.long)
    tot = int(sum(sizes))
    lin = torch.nn.Embedding(tot, 1); torch.nn.init.zeros_(lin.weight)
    fac = torch.nn.Embedding(tot, dim); torch.nn.init.normal_(fac.weight, std=0.01)
    b0 = torch.zeros(1, requires_grad=True)

    wd_dev = wd if wd_dev is None else wd_dev
    opt = torch.optim.Adam([
        {"params": list(lin.parameters()) + list(fac.parameters()), "weight_decay": wd_dev},
        {"params": [b0], "weight_decay": 0.0},
    ], lr=lr)

    Xtr = enc(tr) + offs
    ytr = torch.tensor([r["label"] for r in tr], dtype=torch.float32)
    off_tr = torch.tensor(closed_form_prior(tr, tr)) if mode != "base" else torch.zeros(len(tr))

    def dev(X):
        l = lin(X).sum(1).squeeze(-1) + b0
        v = fac(X); s = v.sum(1)
        return l + 0.5 * ((s * s).sum(-1) - (v * v).sum(-1).sum(-1))

    bce = torch.nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        opt.zero_grad()
        bce(dev(Xtr) + off_tr, ytr).backward()
        opt.step()

    with torch.no_grad():
        def score(R):
            o = torch.tensor(closed_form_prior(tr, R)) if mode != "base" else torch.zeros(len(R))
            return torch.sigmoid(dev(enc(R) + offs) + o).numpy()
        return score(te), score(tr)


CELLS = [
    ("base FM (calibrated)",              dict(mode="base")),
    ("+ prior as offset (residual)",      dict(mode="residual")),
    ("+ prior offset, deviation unregularized", dict(mode="residual", wd_dev=0.0)),
]


def main():
    records, _ = build_records()
    folds = per_user_folds(records, KFOLD, SEED)
    ep = _epochs()
    out = {}

    print("=" * 84)
    print("Exception-aware learning: fit the PRIOR in closed form, give the model's")
    print("capacity (and its regularization budget) to the DEVIATION.")
    print("=" * 84)
    print(f"\n{'method':44s}{'Acc':>7}{'routine':>10}{'exception':>12}{'ratio':>8}")
    for name, kw in CELLS:
        EXC, ROU, ACC = [], [], []
        for f in range(KFOLD):
            tr = [records[i] for i in range(len(records)) if folds[i] != f]
            te = [records[i] for i in range(len(records)) if folds[i] == f]
            p, ptr = fm(tr, te, epochs=ep, **kw)
            y = np.array([r["label"] for r in te])
            ytr = np.array([r["label"] for r in tr])
            thr = float(M.equal_error_threshold(ytr, ptr))
            yh = (p >= thr).astype(int)
            u = np.array([r["user_id"] for r in te])
            ur = {uu: float(y[u == uu].mean()) for uu in np.unique(u)}
            mu = np.array([1 if ur[uu] >= 0.5 else 0 for uu in u])
            exc = (y != mu)
            ov = (yh != y)
            EXC += list(ov[exc].astype(float))
            ROU += list(ov[~exc].astype(float))
            ACC.append(100 * (yh == y).mean())
        e, r_ = 100 * np.mean(EXC), 100 * np.mean(ROU)
        out[name] = dict(acc=float(np.mean(ACC)), exception=float(e), routine=float(r_),
                         ratio=float(e / max(r_, 0.1)), n_exc=len(EXC))
        print(f"{name:44s}{np.mean(ACC):7.1f}{r_:9.1f}%{e:11.1f}%{e / max(r_, .1):7.1f}x")

    b = out["base FM (calibrated)"]
    best = min(out, key=lambda k: out[k]["exception"])
    print(f"\nbaseline exception override = {b['exception']:.1f}%")
    print(f"best method                 = {out[best]['exception']:.1f}%  ({best})")
    d = b["exception"] - out[best]["exception"]
    print(f"-> exception override {'REDUCED by %.1f points' % d if d > 1 else 'NOT reduced'}"
          f"; accuracy {b['acc']:.1f} -> {out[best]['acc']:.1f}")
    if d <= 1:
        print("   The method fails. Report it as such.")
    C.save(out, "table29_exception_aware.json")
    print("\nSaved -> table29_exception_aware.json")


if __name__ == "__main__":
    main()
