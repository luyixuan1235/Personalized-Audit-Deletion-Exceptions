"""Measure the PER-USER OFFSET DISPERSION of the collapsed ranking model.

Why: post-hoc calibration turned out to behave very differently on the two corpora
(global Platt fixes SPA: HC-FPR 100 -> 3.3; but not Wu: 100 -> 60.7, unstable).
Proposition 1 explains why this MUST be dataset-dependent rather than guaranteed:

  the ranking loss leaves a PER-USER offset c_u free. A global monotone calibrator
  can only undo the COMMON part of those offsets. So it restores safety exactly to
  the extent the fitted offsets are HOMOGENEOUS across users -- a quantity the
  objective does not control and the practitioner cannot observe in advance.

This script measures that quantity directly on the trained ranking model: for each
user, the mean logit of their requests (their empirical offset c_u), then the spread
of c_u across users, normalised by the within-user spread of logits. If the story is
right, the offsets should be much more heterogeneous on Wu than on SPA.

  het = std_across_users(mean_u logit) / mean_across_users(std_within_u logit)

het >> 1  : offsets dominate -> a single global map cannot align users -> post-hoc fails
het << 1  : offsets are near-common -> one global map suffices -> post-hoc works
"""
import os
import sys

import numpy as np

import _common as C

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import Graph, build_records, per_user_folds  # noqa: E402
from train import predict, train  # noqa: E402

KFOLD = 5
MIN_PER_USER = 3


def _epochs():
    return 15 if C.QUICK else int(os.environ.get("PERM_EPOCHS", 200))


def _logit(p, eps=1e-6):
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    return np.log(p / (1 - p))


def main():
    records, _ = build_records()
    ep = _epochs()
    folds = per_user_folds(records, KFOLD, 42)
    hets, betw, wthn = [], [], []

    for f in range(KFOLD):
        tr = [records[i] for i in range(len(records)) if folds[i] != f]
        te = [records[i] for i in range(len(records)) if folds[i] == f]
        g = Graph(tr, records)
        print(f"[fold {f+1}/{KFOLD}]")
        # the collapsed Table-1 model: one-sided, atomic IDs, ranking objective
        m = train(g, tr, mode="allow", loss="bpr", use_sideinfo=False, epochs=ep)
        pr = predict(m, g, te)
        z = _logit(pr["p_allow"])
        users = np.array([r["user_id"] for r in te])

        mu, sd = [], []
        for uu in np.unique(users):
            zz = z[users == uu]
            if len(zz) >= MIN_PER_USER:
                mu.append(zz.mean()); sd.append(zz.std())
        mu = np.array(mu); sd = np.array(sd)
        between = float(mu.std())          # spread of the per-user offsets c_u
        within = float(sd.mean())          # typical spread of scores WITHIN a user
        het = between / max(within, 1e-9)
        betw.append(between); wthn.append(within); hets.append(het)
        print(f"   between-user offset sd = {between:.3f}   "
              f"within-user sd = {within:.3f}   heterogeneity = {het:.2f}")

    out = dict(between_user_sd=[float(np.mean(betw)), float(np.std(betw, ddof=1))],
               within_user_sd=[float(np.mean(wthn)), float(np.std(wthn, ddof=1))],
               heterogeneity=[float(np.mean(hets)), float(np.std(hets, ddof=1))],
               het_folds=hets)
    print(f"\n=== per-user offset heterogeneity (ranking model, 5-fold) ===")
    print(f"  between-user offset sd : {np.mean(betw):.3f} +/- {np.std(betw, ddof=1):.3f}")
    print(f"  within-user score sd   : {np.mean(wthn):.3f} +/- {np.std(wthn, ddof=1):.3f}")
    print(f"  heterogeneity ratio    : {np.mean(hets):.2f} +/- {np.std(hets, ddof=1):.2f}")
    print("\n  >>1 : per-user offsets dominate -> NO single global calibrator can align")
    print("        users -> post-hoc calibration must fail (Proposition 1).")
    print("  <<1 : offsets nearly common -> one global map suffices -> post-hoc works.")
    C.save(out, "table19_offset_spread.json")
    print("Saved -> table19_offset_spread.json")


if __name__ == "__main__":
    main()
