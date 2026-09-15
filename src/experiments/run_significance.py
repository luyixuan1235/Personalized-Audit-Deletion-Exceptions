"""Paired significance tests on the 2x2 HC-FPR cells (reviewer: overlapping levers).

The 2x2 discussion in the paper turns on whether the two single-fix cells
(calibrated objective on FM vs. ranking objective on the signed graph) really are
indistinguishable. Eyeballing whether mean+/-std intervals overlap is not a test:
the folds are PAIRED (run_variance.py uses the same per_user_folds(seed=42) split
for every cell), and a paired test is far more powerful than an overlap check.

This script reads the per-fold HC-FPR values already stored by run_variance.py
(table12_variance.json -> hc_folds) and reports, for each headline comparison:
  * paired t-test (t, two-sided p)
  * paired bootstrap 95% CI of the mean difference
  * exact sign-flip permutation p

Note on n=5: an exact two-sided sign-flip permutation test has 2^5 = 32 sign
assignments, so its smallest attainable p-value is 2/32 = 0.0625 -- it can never
reject at 0.05 regardless of effect size. We report it for completeness but read
significance off the paired t-test and the bootstrap CI.

No scipy dependency: the Student-t tail is evaluated via the regularized
incomplete beta function (continued fraction, Numerical Recipes betacf).
"""
import itertools
import json
import math
import os
import sys

import numpy as np

import _common as C

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

B_ITERS = 20000
SEED = 0

# (cell_a, cell_b, what the comparison decides)
COMPARISONS = [
    ("FM-bce", "Signed-bpr",
     "objective-fix vs architecture-fix  [the 'overlapping levers' claim]"),
    ("FM-bpr", "FM-bce",
     "ranking -> calibrated, architecture held (objective lever)"),
    ("FM-bpr", "Signed-bpr",
     "FM -> signed graph, objective held (architecture lever)"),
    ("FM-bce", "Signed-bce",
     "calibrated: + signed graph (claimed second-order gain)"),
    ("Signed-bce", "dual-evidential",
     "+ evidential on top of calibrated signed (claimed no gain)"),
]


# ---- Student-t two-sided p, no scipy ---------------------------------------
def _betacf(a, b, x, itmax=200, eps=3e-16, fpmin=1e-300):
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _betainc(a, b, x):
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
             + a * math.log(x) + b * math.log1p(-x))
    front = math.exp(lbeta)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_sf_two_sided(t, df):
    """Two-sided p-value for Student-t statistic t with df degrees of freedom."""
    if df <= 0:
        return float("nan")
    return _betainc(0.5 * df, 0.5, df / (df + t * t))


def paired_t(x, y):
    d = np.asarray(x, float) - np.asarray(y, float)
    n = len(d)
    sd = d.std(ddof=1)
    if sd == 0:
        return float("inf") * np.sign(d.mean()), 0.0
    t = d.mean() / (sd / math.sqrt(n))
    return t, t_sf_two_sided(t, n - 1)


def boot_ci(d, iters=B_ITERS, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(d)
    means = np.array([rng.choice(d, n, replace=True).mean() for _ in range(iters)])
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def perm_p(d):
    """Exact two-sided sign-flip permutation p (all 2^n sign assignments)."""
    n = len(d)
    obs = abs(d.mean())
    hits = sum(1 for s in itertools.product([1, -1], repeat=n)
               if abs((np.asarray(s) * d).mean()) >= obs - 1e-12)
    return hits / (2 ** n)


def main():
    res_dir = os.environ.get("PERM_RESULTS_DIR", "../results")
    path = os.path.join(res_dir, "table12_variance.json")
    with open(path) as fh:
        var = json.load(fh)

    folds = {k: np.asarray(v["hc_folds"], float) for k, v in var.items()}
    out = {}
    print(f"\n=== Paired significance on per-fold HC-FPR  ({path}) ===")
    print("(paired across the same 5 per-user folds, seed 42)\n")
    for a, b, desc in COMPARISONS:
        if a not in folds or b not in folds:
            print(f"  SKIP {a} vs {b} (missing)")
            continue
        x, y = folds[a], folds[b]
        d = x - y
        t, p = paired_t(x, y)
        lo, hi = boot_ci(d)
        pp = perm_p(d)
        verdict = "significant" if p < 0.05 else "n.s."
        out[f"{a}_vs_{b}"] = dict(
            mean_a=float(x.mean()), mean_b=float(y.mean()), mean_diff=float(d.mean()),
            t=float(t), p_paired_t=float(p), p_signflip_exact=float(pp),
            ci95_lo=lo, ci95_hi=hi, significant=bool(p < 0.05), folds_diff=d.tolist())
        print(f"  {a:14s} vs {b:14s}  {desc}")
        print(f"    mean {x.mean():6.2f} vs {y.mean():6.2f}   diff {d.mean():+6.2f}"
              f"   95% CI [{lo:+.2f}, {hi:+.2f}]")
        print(f"    paired t({len(d)-1}) = {t:+.2f}   p = {p:.4f}  -> {verdict}"
              f"    (exact sign-flip p = {pp:.4f}; floor 0.0625 at n=5)\n")

    C.save(out, "table15_significance.json")
    print("Saved -> table15_significance.json")


if __name__ == "__main__":
    main()
