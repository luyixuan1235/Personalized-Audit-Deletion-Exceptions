"""Shared metrics for P0/P1 hardening: train-only majority and user-cluster CIs.

Wu released predictions are a fixed train/test split. Train-only majority is read
from processed_dataset.json's training fold (not from the evaluation labels).
Our own k-fold models should compute majority from the training fold of each split.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
RESULTS = Path(__import__("os").environ.get("PERM_RESULTS_DIR", str(ROOT / "results"))).resolve()
WU_RESULTS = ROOT / "wu-repo" / "results"
DATA = Path(__import__("os").environ.get("PERM_DATA_DIR", ROOT / "data")).resolve()

ALLOWED = {"Yes, always share": 1, "No, never share": 0}
N_BOOT = 2000
N_PERM = 5000
MIN_DENIES = 3


def pred_path(name: str) -> Path:
    fn = {
        "CF": "cf_only_predictions.json",
        "IC": "ic_only_predictions.json",
        "IC+CF": "ic_cf_predictions.json",
    }[name]
    for base in (RESULTS, WU_RESULTS):
        p = base / fn
        if p.exists():
            return p
    raise FileNotFoundError(fn)


def load_cf(path):
    from analyze_wu_peruser import load_cf as _load
    return _load(str(path))[:3]


def load_llm(path):
    from analyze_wu_peruser import load_llm as _load
    return _load(str(path))[:3]


def load_model(name: str):
    path = pred_path(name)
    u, y, yh = (load_cf if name == "CF" else load_llm)(path)
    return u, y, yh, path


def corpus_majorities():
    """Per-user majority from the Wu corpus splits (not from model predictions)."""
    data = json.loads((DATA / "processed_dataset.json").read_text(encoding="utf-8"))
    train, full, n_train, n_full = {}, {}, {}, {}
    for uid, ud in data.items():
        tr, te = [], []
        for ex in ud.get("training", []):
            for lab in ex.get("answer", {}).values():
                if lab in ALLOWED:
                    tr.append(ALLOWED[lab])
        for ex in ud.get("testing", []):
            for lab in ex.get("answer", {}).values():
                if lab in ALLOWED:
                    te.append(ALLOWED[lab])
        both = tr + te
        if tr:
            train[uid] = 1 if float(np.mean(tr)) >= 0.5 else 0
            n_train[uid] = len(tr)
        if both:
            full[uid] = 1 if float(np.mean(both)) >= 0.5 else 0
            n_full[uid] = len(both)
    return dict(train=train, full=full, n_train=n_train, n_full=n_full)


def test_majority(users, y):
    um = {uu: float(y[users == uu].mean()) for uu in np.unique(users)}
    return {uu: (1 if r >= 0.5 else 0) for uu, r in um.items()}


def mu_vector(users, mapping, fallback):
    """Map each decision to a majority label. fallback is used only if user missing."""
    out = np.empty(len(users), dtype=int)
    n_fb = 0
    for i, uu in enumerate(users):
        if uu in mapping:
            out[i] = mapping[uu]
        else:
            out[i] = fallback[uu]
            n_fb += 1
    return out, n_fb


def rates(y, yh, mu):
    exc = y != mu
    ov = yh != y
    n_exc = int(exc.sum())
    n_rout = int((~exc).sum())
    k_exc = int(ov[exc].sum()) if n_exc else 0
    k_rout = int(ov[~exc].sum()) if n_rout else 0
    e = (k_exc / n_exc) if n_exc else float("nan")
    r = (k_rout / n_rout) if n_rout else float("nan")
    return dict(
        exception=100 * e, routine=100 * r,
        ratio=(e / r if r and r > 0 else float("nan")),
        n_exc=n_exc, n_rout=n_rout, k_exc=k_exc, k_rout=k_rout,
        acc=100 * float((yh == y).mean()),
        fgr=100 * float(yh[y == 0].mean()) if (y == 0).any() else float("nan"),
    )


def gini(x):
    x = np.sort(np.asarray(x, float))
    n = len(x)
    c = np.cumsum(x)
    return float((n + 1 - 2 * np.sum(c) / c[-1]) / n) if c[-1] > 0 else 0.0


def top_share(x, frac=0.25):
    x = np.sort(np.asarray(x, float))[::-1]
    k = max(1, int(len(x) * frac))
    return float(100 * x[:k].sum() / x.sum()) if x.sum() > 0 else float("nan")


def top_share_counts(false_grants, denial_counts, frac=0.25):
    """Share of false-grant counts borne by highest-FPR users."""
    false_grants = np.asarray(false_grants, float)
    denial_counts = np.asarray(denial_counts, float)
    order = np.argsort(-(false_grants / denial_counts))
    k = max(1, int(len(order) * frac))
    total = false_grants.sum()
    return float(100 * false_grants[order[:k]].sum() / total) if total > 0 else float("nan")

def per_user_fpr(users, y, yh, min_denies=MIN_DENIES):
    per, ns, uids = [], [], []
    for u in np.unique(users):
        dn = (users == u) & (y == 0)
        if dn.sum() >= min_denies:
            per.append(100 * float(yh[dn].mean()))
            ns.append(int(dn.sum()))
            uids.append(u)
    return np.array(per), np.array(ns), np.array(uids)


def cluster_bootstrap_eor(users, y, yh, mu, rng, n=N_BOOT):
    uu = np.unique(users)
    buckets = {x: np.where(users == x)[0] for x in uu}
    boots = []
    for _ in range(n):
        pick = rng.choice(uu, size=len(uu), replace=True)
        idx = np.concatenate([buckets[x] for x in pick])
        boots.append(rates(y[idx], yh[idx], mu[idx]))
    exc = np.array([b["exception"] for b in boots])
    rout = np.array([b["routine"] for b in boots])
    ratio = np.array([b["ratio"] for b in boots])
    return dict(
        exception_ci=[round(float(q), 1) for q in np.nanpercentile(exc, [2.5, 97.5])],
        routine_ci=[round(float(q), 1) for q in np.nanpercentile(rout, [2.5, 97.5])],
        ratio_ci=[round(float(q), 2) for q in np.nanpercentile(ratio, [2.5, 97.5])],
        exception_mean=round(float(np.nanmean(exc)), 1),
        ratio_mean=round(float(np.nanmean(ratio)), 2),
        n_boot=n,
    )


def cluster_bootstrap_concentration(users, y, yh, rng, n=N_BOOT):
    """Resample users, recompute worst-quartile false-grant share."""
    uu = np.unique(users)
    boots = []
    for _ in range(n):
        pick = rng.choice(uu, size=len(uu), replace=True)
        # rebuild per-user FPR on the resampled multiset of users
        per, counts, denials = [], [], []
        for x in pick:
            dn = (users == x) & (y == 0)
            if dn.sum() >= MIN_DENIES:
                per.append(100 * float(yh[dn].mean()))
                counts.append(int(yh[dn].sum()))
                denials.append(int(dn.sum()))
        if len(per) < 8:
            continue
        boots.append(top_share_counts(np.array(counts), np.array(denials), 0.25))
    arr = np.array(boots)
    return dict(
        top25_mean=round(float(np.nanmean(arr)), 1),
        top25_ci=[round(float(q), 1) for q in np.nanpercentile(arr, [2.5, 97.5])],
        n_boot=int(len(arr)),
    )


def permutation_top25(per, ns, agg_fpr, rng, n=N_PERM):
    """Permutation null using false-grant counts, ranked by user FPR."""
    ns = np.asarray(ns, int)
    observed_counts = np.rint(np.asarray(per, float) * ns / 100.0).astype(int)
    obs = top_share_counts(observed_counts, ns, 0.25)
    p = agg_fpr / 100.0
    null = np.array([
        top_share_counts(rng.binomial(ns, p), ns, 0.25)
        for _ in range(n)
    ])
    return dict(
        observed=round(float(obs), 1),
        null_mean=round(float(null.mean()), 1),
        null_p95=round(float(np.percentile(null, 95)), 1),
        p=round(float((null >= obs).mean()), 4),
    )


def save(obj, name):
    RESULTS.mkdir(parents=True, exist_ok=True)
    p = RESULTS / name
    p.write_text(json.dumps(obj, indent=2, default=float), encoding="utf-8")
    print(f"saved -> {p}")
    return p
