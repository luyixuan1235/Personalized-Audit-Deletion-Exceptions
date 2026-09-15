"""Per-user harm concentration in Wu et al.'s ACTUAL released system.

Runs on the outputs of the authors' own code (llm-platform-security/ai-agent-permissions),
unmodified, on their own released data. Analyses all three of their models:

  cf_only  - LightGCN + BPR. Decides by thresholding a raw dot-product score at ONE
             GLOBAL threshold (fitted by scanning for FPR=FNR). This is the model our
             mechanism predicts should concentrate harm on a minority of users.
  ic_only  - LLM in-context learning. Decides per item from the user's own history;
             there is NO global threshold, so the mechanism should NOT apply.
  ic_cf    - The hybrid, and the paper's headline system (~85% accuracy). The LLM
             decides, but is fed the CF model's top-5 "requests the user may ALLOW".
             Note the CF channel can only ever transmit grants: it is structurally
             incapable of recommending a denial.

The question this script exists to answer, honestly, in either direction:

   Does the deployed hybrid inherit the CF component's per-user concentration,
   or does the LLM half repair it?

If the hybrid is flat, our claim must shrink to the CF component in isolation, and we
say so. We compute, for each model:

  * aggregate false-grant rate (the number a paper reports)
  * the DISTRIBUTION of per-user false-grant rate: median vs mean, Gini, and the share
    of all false grants borne by the worst quartile of users
  * a permutation test against the null "all users share one false-grant probability
    and differ only by sampling noise", so a low base rate cannot masquerade as
    concentration (a lower rate mechanically permits more zeros, hence a higher Gini --
    the raw statistic alone proves nothing)

Usage (from src/, after running the authors' scripts):
    python analyze_wu_peruser.py --results ../results
"""
import argparse
import json
import os
import sys

import numpy as np

MIN_DENIES = 3
N_PERM = 5000
SEED = 0


# ---------- loaders: one per output format the authors' scripts emit ----------
def load_cf(path):
    """cf_only_predictions.json: flat records + one global threshold."""
    with open(path) as fh:
        d = json.load(fh)
    blk = d["cf_only"]
    thr = float(blk["metrics"]["threshold"])
    users, y, yhat = [], [], []
    for r in blk["test_predictions"]:
        users.append(r["userID"])
        y.append(1 if r["rating"] > 0 else 0)
        yhat.append(1 if float(r["prediction"]) >= thr else 0)
    return np.array(users), np.array(y), np.array(yhat), f"global thr={thr:.3f}"


def load_llm(path):
    """ic_only / ic_cf predictions.json: per-participant LLM labels (no threshold)."""
    with open(path) as fh:
        d = json.load(fh)
    users, y, yhat = [], [], []
    for pid, data in d.items():
        if not isinstance(data, dict) or data.get("skipped") or "error" in data:
            continue
        preds = {p.get("id"): p.get("permission", {})
                 for p in data.get("predictions", [])}
        for gt in data.get("ground_truth", []):
            pp = preds.get(gt["id"])
            if not pp:
                continue
            for dtype, gt_label in gt.get("answer", {}).items():
                if dtype not in pp:
                    continue
                users.append(pid)
                y.append(1 if "Yes" in gt_label else 0)
                yhat.append(1 if "Yes" in pp[dtype].get("label", "") else 0)
    return np.array(users), np.array(y), np.array(yhat), "LLM label (no threshold)"


# ---------- concentration statistics ----------
def gini(x):
    x = np.sort(np.asarray(x, float))
    n = len(x)
    c = np.cumsum(x)
    return float((n + 1 - 2 * np.sum(c) / c[-1]) / n) if c[-1] > 0 else 0.0


def top_share(x, frac):
    x = np.sort(np.asarray(x, float))[::-1]
    k = max(1, int(len(x) * frac))
    return float(100 * x[:k].sum() / x.sum()) if x.sum() > 0 else float("nan")


def top_share_counts(false_grants, denial_counts, frac):
    """Share of false-grant counts, ranking users by per-user FPR."""
    false_grants = np.asarray(false_grants, float)
    denial_counts = np.asarray(denial_counts, float)
    order = np.argsort(-(false_grants / denial_counts))
    k = max(1, int(len(order) * frac))
    total = false_grants.sum()
    return float(100 * false_grants[order[:k]].sum() / total) if total > 0 else float("nan")

def analyse(name, users, y, yhat, note):
    ok = len(y) > 0
    if not ok:
        print(f"\n### {name}: no usable predictions found -- skipping")
        return None
    agg = 100 * float(yhat[y == 0].mean()) if (y == 0).any() else float("nan")
    acc = 100 * float((yhat == y).mean())

    per, ns = [], []
    for u in np.unique(users):
        dn = (users == u) & (y == 0)
        if dn.sum() >= MIN_DENIES:
            per.append(100 * float(yhat[dn].mean()))
            ns.append(int(dn.sum()))
    per, ns = np.array(per), np.array(ns)
    if len(per) == 0:
        print(f"\n### {name}: no user has >={MIN_DENIES} denials -- skipping")
        return None

    obs = dict(median=float(np.median(per)), mean=float(per.mean()),
               sd=float(per.std()), gini=gini(per), top25=top_share(per, 0.25))

    rng = np.random.default_rng(SEED)
    p = agg / 100.0
    null = {k: [] for k in obs}
    for _ in range(N_PERM):
        counts = np.array([rng.binomial(n, p) for n in ns])
        sim = counts / ns * 100
        null["median"].append(np.median(sim)); null["mean"].append(sim.mean())
        null["sd"].append(sim.std()); null["gini"].append(gini(sim))
        null["top25"].append(top_share_counts(counts, ns, 0.25))

    print(f"\n### {name}   [{note}]")
    print(f"    accuracy {acc:.1f}%   AGGREGATE false-grant rate {agg:.1f}%  "
          f"({len(per)} users with >={MIN_DENIES} denials)")
    print(f"    {'statistic':>8}{'observed':>10}{'null mean':>11}{'null p95':>10}{'p':>8}")
    res = {}
    for k in ("median", "sd", "gini", "top25"):
        nl = np.array(null[k])
        # median: concealment shows as observed BELOW null; others: observed ABOVE
        pv = float((nl <= obs[k]).mean()) if k == "median" else float((nl >= obs[k]).mean())
        res[k] = dict(obs=obs[k], null=float(nl.mean()),
                      p95=float(np.percentile(nl, 95)), p=pv)
        print(f"    {k:>8}{obs[k]:>10.1f}{nl.mean():>11.1f}"
              f"{np.percentile(nl, 95):>10.1f}{pv:>8.4f}")

    concealed = obs["median"] < 0.5 * obs["mean"] and res["median"]["p"] < 0.05
    conc = res["top25"]["p"] < 0.05 and obs["top25"] > res["top25"]["p95"]
    print(f"    -> harm concentrated beyond noise : {'YES' if conc else 'no'}"
          f"   (worst 25% of users bear {obs['top25']:.1f}% of all false grants)")
    print(f"    -> aggregate metric CONCEALS it   : {'YES' if concealed else 'no'}"
          f"   (median user {obs['median']:.1f}% vs mean {obs['mean']:.1f}%)")
    return dict(name=name, accuracy=acc, aggregate_fpr=agg, n_users=len(per),
                stats=res, concentrated=bool(conc), concealed=bool(concealed),
                per_user_fpr=per.tolist(), per_user_ndeny=ns.tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="../results",
                    help="Wu et al. results dir containing *_predictions.json")
    ap.add_argument("--out", default="wu_peruser_analysis.json")
    a = ap.parse_args()

    jobs = [
        ("CF only (LightGCN+BPR, global threshold)", "cf_only_predictions.json", load_cf),
        ("IC only (LLM in-context)", "ic_only_predictions.json", load_llm),
        ("IC+CF hybrid (the deployed system)", "ic_cf_predictions.json", load_llm),
    ]
    out = []
    for name, fn, loader in jobs:
        path = os.path.join(a.results, fn)
        if not os.path.exists(path):
            print(f"\n### {name}: {fn} not found -- run the authors' script first")
            continue
        try:
            r = analyse(name, *loader(path))
            if r:
                out.append(r)
        except Exception as e:  # noqa: BLE001
            print(f"\n### {name}: could not parse {fn} ({e})")

    if not out:
        sys.exit("No results parsed.")

    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)
    for r in out:
        tag = ("concentrated AND concealed" if r["concealed"] and r["concentrated"]
               else "concentrated, not concealed" if r["concentrated"]
               else "no concentration beyond noise")
        print(f"  {r['name'][:44]:46s} {tag}")
    hyb = next((r for r in out if r["name"].startswith("IC+CF")), None)
    cf = next((r for r in out if r["name"].startswith("CF only")), None)
    if hyb and cf:
        print()
        if hyb["concentrated"]:
            print("  The DEPLOYED hybrid inherits the concentration. The claim about the")
            print("  published system holds, and is not confined to an ablated component.")
        else:
            print("  The deployed hybrid does NOT concentrate harm: the LLM half repairs")
            print("  what the CF half breaks. Our claim must be scoped to the CF component,")
            print("  and the paper must say so plainly.")

    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nSaved -> {a.out}")


if __name__ == "__main__":
    main()
