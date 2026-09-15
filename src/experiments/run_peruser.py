"""Per-user false-grant dispersion under a single GLOBAL decision threshold.

This is the paper's central empirical claim, restated correctly.

A pairwise ranking objective leaves a PER-USER score offset unidentified,
so the deployed decision boundary must be fitted externally: the practitioner must
fit one externally, and in practice fits a SINGLE GLOBAL threshold (Wu et al. scan
the raw dot-product scale for the point where FPR = FNR). But if the offsets the
objective left free are heterogeneous across users, one global constant cannot serve
them all -- it will be systematically wrong for the users whose offsets sit far from
the mean, and their false grants are then averaged away by every aggregate metric.

The theory makes a falsifiable cross-dataset prediction. Define the offset
heterogeneity of the trained model (computable without labels):

    eta = sd_u( mean_{x in u} s(x) ) / mean_u( sd_{x in u} s(x) )

We measure eta = 0.99 on Wu and 0.60 on SPA. The prediction is therefore that the
per-user false-grant rate should be markedly MORE dispersed on Wu than on SPA. This
script tests exactly that, on both corpora, with the same model and protocol.

Reported per dataset:
  * global EER threshold and the aggregate FPR it yields (the number a paper reports)
  * the DISTRIBUTION of per-user FPR under that one threshold
  * a permutation test against the null "all users share one false-grant probability
    and differ only by sampling noise" -- so the dispersion cannot be dismissed as a
    small-denominator artifact
  * the false grants recoverable by a per-user threshold that costs that user no
    accuracy (the price of insisting on a single global constant)
"""
import os
import sys

import numpy as np

import _common as C
import metrics as M

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import Graph, build_records, per_user_folds  # noqa: E402
from train import predict, train  # noqa: E402

KFOLD = 5
MIN_DENIES = 3          # a user needs this many true denials to get an FPR estimate
N_PERM = 5000
SEED = 42
SEEDS = [42, 123, 456, 789, 1024]


def _epochs():
    return 15 if C.QUICK else int(os.environ.get("PERM_EPOCHS", 200))


def _fpr(s, y, thr):
    dn = (y == 0)
    return float((s[dn] >= thr).mean()) if dn.any() else float("nan")


def _oracle_fpr(s, y, thr_global):
    """Lowest FPR reachable for THIS user by moving the threshold, without costing
    the user any accuracy relative to the global threshold."""
    acc0 = float(((s >= thr_global).astype(int) == y).mean())
    best = _fpr(s, y, thr_global)
    for t in np.linspace(s.min(), s.max(), 200):
        if float(((s >= t).astype(int) == y).mean()) >= acc0:
            best = min(best, _fpr(s, y, t))
    return best


def run_one(seed, save_table21=True):
    records, _ = build_records()
    ep = _epochs()
    folds = per_user_folds(records, KFOLD, seed)
    rng = np.random.default_rng(seed)

    all_fpr, all_n, all_false, all_glob, all_orac, thrs, agg = [], [], [], [], [], [], []
    fold_thresholds, audit_predictions = [], []
    off_between, off_within = [], []

    for f in range(KFOLD):
        tr = [records[i] for i in range(len(records)) if folds[i] != f]
        te = [records[i] for i in range(len(records)) if folds[i] == f]
        g = Graph(tr, records)
        # the ranking recipe the field uses: pairwise BPR, one-sided, atomic IDs
        m = train(g, tr, mode="allow", loss="bpr", use_sideinfo=False, epochs=ep,
                  seed=seed)
        pr, prtr = predict(m, g, te), predict(m, g, tr)
        y = np.asarray(pr["y"]); p = np.asarray(pr["p_allow"])
        u = np.array([r["user_id"] for r in te])

        # ONE global threshold, fitted on the training fold (equal error rate) --
        # exactly the protocol the published system uses.
        thr = float(M.equal_error_threshold(np.asarray(prtr["y"]),
                                            np.asarray(prtr["p_allow"])))
        thrs.append(thr)
        fold_thresholds.append(dict(fold_id=int(f), threshold=thr, n_train=len(tr), n_test=len(te)))
        yh = (p >= thr).astype(int)
        audit_predictions.extend(
            dict(fold_id=int(f), user_id=str(r["user_id"]), true_label=int(yy),
                 p_allow=float(pp), threshold=thr, prediction=int(pred))
            for r, yy, pp, pred in zip(te, y, p, yh)
        )
        agg.append(100 * _fpr(p, y, thr))

        z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
        mus, sds = [], []
        for uu in np.unique(u):
            mu = (u == uu)
            zz = z[mu]
            if len(zz) >= 3:
                mus.append(zz.mean()); sds.append(zz.std())
            dn = mu & (y == 0)
            if dn.sum() >= MIN_DENIES:
                all_fpr.append(100 * _fpr(p[mu], y[mu], thr))
                all_n.append(int(dn.sum()))
                all_false.append(int((p[dn] >= thr).sum()))
                all_glob.append(100 * _fpr(p[mu], y[mu], thr))
                all_orac.append(100 * _oracle_fpr(p[mu], y[mu], thr))
        off_between.append(np.std(mus)); off_within.append(np.mean(sds))

    fpr = np.array(all_fpr); ns = np.array(all_n); false_counts = np.array(all_false)
    eta = float(np.mean(off_between) / np.mean(off_within))
    aggregate = float(np.mean(agg))

    print("\n" + "=" * 66)
    print(f"global EER threshold  = {np.mean(thrs):.3f}")
    print(f"AGGREGATE false-grant rate (what a paper reports) = {aggregate:.1f}%")
    print(f"offset heterogeneity  eta = {eta:.2f}")
    print("=" * 66)
    print(f"\nPer-user FPR under that ONE global threshold "
          f"({len(fpr)} users with >={MIN_DENIES} denials):")
    print(f"   median {np.median(fpr):5.1f}%    mean {fpr.mean():5.1f}%    "
          f"sd {fpr.std():5.1f}%")
    print("   deciles " + " ".join(f"{np.percentile(fpr, q):.0f}"
                                   for q in (10, 25, 50, 75, 90, 100)))
    for t in (30, 50, 75):
        print(f"   users with FPR >= {t:3d}% : {(fpr >= t).sum():4d} / {len(fpr)}"
              f"  ({100 * (fpr >= t).mean():.1f}%)")

    # ---- Is the harm CONCENTRATED, or merely noisy?
    # The concentration statistics (Gini, worst-decile burden share) must be compared
    # against the null, not read raw: a LOWER base rate mechanically permits more
    # zeros and hence a higher Gini, so a raw Gini gap across datasets with different
    # base rates proves nothing. We therefore simulate the null "all users share one
    # false-grant probability p, and differ only by sampling noise" using each user's
    # OWN denial count, and report observed-vs-null for every statistic.
    def _gini(x):
        x = np.sort(np.asarray(x, float))
        n = len(x)
        c = np.cumsum(x)
        return float((n + 1 - 2 * np.sum(c) / c[-1]) / n) if c[-1] > 0 else 0.0

    def _topshare_counts(counts, denoms, frac):
        """Share of false-grant counts borne by users ranked by per-user FPR."""
        counts = np.asarray(counts, int)
        denoms = np.asarray(denoms, int)
        order = np.argsort(-(counts / denoms))
        k = max(1, int(len(order) * frac))
        total = counts.sum()
        return float(100 * counts[order[:k]].sum() / total) if total > 0 else float("nan")

    p_glob = aggregate / 100.0
    obs = dict(sd=float(fpr.std()), gini=_gini(fpr),
               top10=_topshare_counts(false_counts, ns, 0.10),
               top25=_topshare_counts(false_counts, ns, 0.25),
               median=float(np.median(fpr)))
    sims = {k: [] for k in obs}
    for _ in range(N_PERM):
        sim_counts = np.array([rng.binomial(n, p_glob) for n in ns])
        sim = sim_counts / ns * 100
        sims["sd"].append(sim.std()); sims["gini"].append(_gini(sim))
        sims["top10"].append(_topshare_counts(sim_counts, ns, 0.10))
        sims["top25"].append(_topshare_counts(sim_counts, ns, 0.25))
        sims["median"].append(np.median(sim))
    print(f"\nNull: every user shares one false-grant probability p={p_glob:.3f},")
    print(f"      differing only by sampling noise (each user's own denial count).")
    print(f"\n{'statistic':>10}{'observed':>10}{'null mean':>11}{'null 95th':>11}{'p':>8}")
    pvals = {}
    for k in ("sd", "gini", "top10", "top25"):
        nl = np.array(sims[k])
        pvals[k] = float((nl >= obs[k]).mean())
        print(f"{k:>10}{obs[k]:>10.1f}{nl.mean():>11.1f}"
              f"{np.percentile(nl, 95):>11.1f}{pvals[k]:>8.4f}")
    nl = np.array(sims["median"])
    print(f"{'median':>10}{obs['median']:>10.1f}{nl.mean():>11.1f}"
          f"{'':>11}{'':>8}  (a median far BELOW the mean = concealed tail)")
    pval = pvals["sd"]

    gl, orc = float(np.mean(all_glob)), float(np.mean(all_orac))
    print(f"\nCost of insisting on ONE global threshold:")
    print(f"   per-user FPR, global threshold      = {gl:.1f}%")
    print(f"   per-user FPR, per-user threshold    = {orc:.1f}%  "
          f"(no accuracy cost to that user)")
    print(f"   -> avoidable false grants           = {gl - orc:.1f} points")

    payload = dict(seed=seed, eta=eta, aggregate_fpr=aggregate, thr=float(np.mean(thrs)),
                   n_users=len(fpr), fpr_median=float(np.median(fpr)),
                   fpr_mean=float(fpr.mean()),
                   observed=obs, null_mean={k: float(np.mean(v)) for k, v in sims.items()},
                   null_p95={k: float(np.percentile(v, 95)) for k, v in sims.items()},
                   perm_p=pvals,
                   frac_ge30=float((fpr >= 30).mean()), frac_ge50=float((fpr >= 50).mean()),
                   global_fpr=gl, oracle_fpr=orc,
                   per_user_fpr=fpr.tolist(), per_user_ndeny=ns.tolist(),
                   per_user_false_grants=false_counts.tolist(),
                   concentration_definition="count-weighted false-grant share, users ranked by per-user FPR",
                   fold_thresholds=fold_thresholds, predictions=audit_predictions)
    if save_table21:
        C.save(payload, "table21_peruser.json")
        print("\nSaved -> table21_peruser.json")
    return payload


def _summary(payload):
    return dict(seed=payload["seed"], eta=payload["eta"],
                aggregate_fpr=payload["aggregate_fpr"],
                fpr_median=payload["fpr_median"], fpr_mean=payload["fpr_mean"],
                top25=payload["observed"]["top25"],
                perm_p_top25=payload["perm_p"]["top25"],
                n_users=payload["n_users"])


def main():
    want_multi = os.environ.get("PERM_MULTI_SEED", "1") == "1"
    seeds = SEEDS if want_multi else [SEED]
    per_seed = {}
    for seed in seeds:
        print(f"\n######## per-user FPR  seed={seed}  ########", flush=True)
        payload = run_one(seed, save_table21=(seed == SEED))
        per_seed[str(seed)] = _summary(payload)
    if len(seeds) > 1:
        import json
        top25 = np.array([per_seed[str(s)]["top25"] for s in seeds])
        med = np.array([per_seed[str(s)]["fpr_median"] for s in seeds])
        agg = np.array([per_seed[str(s)]["aggregate_fpr"] for s in seeds])
        out = dict(
            seeds=seeds,
            per_seed=per_seed,
            aggregate=dict(
                top25_mean=float(top25.mean()), top25_std=float(top25.std(ddof=1)),
                median_mean=float(med.mean()), median_std=float(med.std(ddof=1)),
                aggregate_fpr_mean=float(agg.mean()),
                aggregate_fpr_std=float(agg.std(ddof=1)),
            ),
            note="seed=42 is the primary table21_peruser.json used by fig-peruser-fpr.",
        )
        C.save(out, "table_wu_peruser_concentration_5seeds.json")
        print("\n5-seed summary:", json.dumps(out["aggregate"], indent=2))


if __name__ == "__main__":
    main()
