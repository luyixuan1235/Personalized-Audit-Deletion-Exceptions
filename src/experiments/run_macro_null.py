"""Macro-user EOR ratio against its own macro-level permutation null.
Addresses the reviewer point that the macro-averaged exception/routine ratios
(2.4 / 1.8 / 1.6) sit near the decision-level mechanical floor (1.5-1.6): the
decision-level null does not directly bound the macro statistic, so we build the
null at the macro level itself. Within each user the error COUNT is fixed and the
errors are scattered uniformly over that user's own decisions; the macro ratio is
recomputed per replicate (2,000 reps). Also reports a user-cluster bootstrap 95% CI
of the observed macro ratio.
Writes results/table74_macro_null.json
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

def load(path):
    d = json.load(open(path)); users, y, yh = [], [], []
    if "cf_only" in d:
        blob = d["cf_only"]; thr = blob["metrics"]["threshold"]
        for p in blob["test_predictions"]:
            users.append(p["userID"]); y.append(int(p["rating"]))
            yh.append(1 if p["prediction"] >= thr else 0)
    else:
        for pid, blob in d.items():
            for pred, gt in zip(blob["predictions"], blob["ground_truth"]):
                pp = pred.get("permission", {})
                for dtype, gt_label in gt.get("answer", {}).items():
                    if dtype not in pp: continue
                    users.append(pid); y.append(1 if "Yes" in gt_label else 0)
                    yh.append(1 if "Yes" in pp[dtype].get("label", "") else 0)
    return np.array(users), np.array(y), np.array(yh)

FILES = {"CF":    os.path.join(ROOT, "wu-repo/results/cf_only_predictions.json"),
         "IC":    os.path.join(ROOT, "wu-repo/results/ic_only_predictions.json"),
         "IC+CF": os.path.join(ROOT, "wu-repo/results/ic_cf_predictions.json")}

def macro_ratio(u, exc, ov, uu):
    per_e, per_r = [], []
    for x in uu:
        m = (u == x)
        if (m & exc).any(): per_e.append(ov[m & exc].mean())
        if (m & ~exc).any(): per_r.append(ov[m & ~exc].mean())
    me, mr = float(np.mean(per_e)), float(np.mean(per_r))
    return me, mr, (me / mr if mr > 0 else np.nan)

rng = np.random.default_rng(42)
REPS_NULL, REPS_BOOT = 2000, 2000
out = {}
for name, path in FILES.items():
    u, y, yh = load(path)
    uu = np.unique(u)
    gr = {x: y[u == x].mean() for x in uu}
    mu = np.array([1 if gr[x] >= 0.5 else 0 for x in u])
    exc = (y != mu); ov = (yh != y)
    me, mr, obs = macro_ratio(u, exc, ov, uu)

    # macro-level permutation null: per user, fix error count, scatter errors
    idx = {x: np.where(u == x)[0] for x in uu}
    errs = {x: int(ov[idx[x]].sum()) for x in uu}
    null = []
    for _ in range(REPS_NULL):
        ov_p = np.zeros(len(u), bool)
        for x in uu:
            k = errs[x]
            if k: ov_p[rng.choice(idx[x], size=k, replace=False)] = True
        null.append(macro_ratio(u, exc, ov_p, uu)[2])
    null = np.array([v for v in null if np.isfinite(v)])
    p = float((null >= obs).mean())

    # user-cluster bootstrap CI of the observed macro ratio
    boot = []
    for _ in range(REPS_BOOT):
        pick = rng.choice(uu, size=len(uu), replace=True)
        per_e, per_r = [], []
        for x in pick:
            m = (u == x)
            if (m & exc).any(): per_e.append(ov[m & exc].mean())
            if (m & ~exc).any(): per_r.append(ov[m & ~exc].mean())
        if per_e and per_r and np.mean(per_r) > 0:
            boot.append(float(np.mean(per_e)) / float(np.mean(per_r)))
    lo, hi = np.percentile(boot, [2.5, 97.5])

    out[name] = dict(
        macro_exc=round(100 * me, 1), macro_routine=round(100 * mr, 1),
        macro_ratio=round(obs, 2),
        null_mean=round(float(null.mean()), 2),
        null_95=[round(float(np.percentile(null, 2.5)), 2),
                 round(float(np.percentile(null, 97.5)), 2)],
        p_perm=round(p, 4),
        boot_ci95=[round(float(lo), 2), round(float(hi), 2)],
        n_users=len(uu), reps_null=REPS_NULL, reps_boot=REPS_BOOT, seed=42)
    print(name, out[name])

dst = os.path.join(ROOT, "results/table74_macro_null.json")
json.dump(out, open(dst, "w"), indent=1)
print("wrote", dst)
