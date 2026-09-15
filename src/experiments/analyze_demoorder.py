"""Did re-presenting the SAME demonstrations change what the model erases?

Four conditions, identical decisions available to the model in three of them (the fourth,
`balanced`, removes surplus grant-only queries and therefore removes information -- if it
still helps, the imbalance was hurting).

If the exception erasure is driven by the in-context majority-label bias of Zhao et al.
(ICML 2021), then presenting the same history with its refusals made salient -- moved to the
end (recency) or balanced against the grants -- should reduce the exception override rate
without any new information. If it does not, that mechanism is not what is doing the damage
here, and we say so.
"""
import json
import sys

import numpy as np


def load(path):
    """Returns per-decision arrays: user, truth, prediction, LLM confidence."""
    with open(path) as fh:
        d = json.load(fh)
    U, Y, YH, CF = [], [], [], []
    for pid, dat in d.items():
        if not isinstance(dat, dict) or dat.get("skipped") or "error" in dat:
            continue
        preds = {p.get("id"): p.get("permission", {}) for p in dat.get("predictions", [])}
        for gt in dat.get("ground_truth", []):
            pp = preds.get(gt["id"])
            if not pp:
                continue
            for dt, gl in gt.get("answer", {}).items():
                if dt not in pp:
                    continue
                U.append(pid)
                Y.append(1 if "Yes" in gl else 0)
                YH.append(1 if "Yes" in pp[dt].get("label", "") else 0)
                CF.append(float(pp[dt].get("score", 0.5)))
    return map(np.array, (U, Y, YH, CF))


def report(name, path):
    try:
        U, Y, YH, CF = load(path)
    except FileNotFoundError:
        print(f"{name:26s}  (not run)")
        return None
    if len(Y) == 0:
        print(f"{name:26s}  (no parsable predictions)")
        return None
    um = {u: float(Y[U == u].mean()) for u in np.unique(U)}
    m = np.array([1 if um[u] >= 0.5 else 0 for u in U])
    exc = (Y != m)
    ov = (YH != Y)
    acc = 100 * (YH == Y).mean()
    fgr = 100 * YH[Y == 0].mean() if (Y == 0).any() else float("nan")
    r = 100 * ov[~exc].mean()
    e = 100 * ov[exc].mean()
    conf = CF[exc & ov].mean() if (exc & ov).any() else float("nan")
    print(f"{name:26s}{acc:7.1f}{fgr:8.1f}%{r:10.1f}%{e:12.1f}%{e / max(r, .1):7.1f}x"
          f"{conf:9.3f}{len(Y):8d}")
    return dict(acc=acc, fgr=fgr, routine=r, exception=e, conf=float(conf), n=int(len(Y)))


if __name__ == "__main__":
    print("Same demonstrations, different presentation. Unit = one decision.\n")
    print(f"{'condition':26s}{'Acc':>7}{'FGR':>9}{'routine':>10}{'exception':>12}"
          f"{'ratio':>7}{'conf':>9}{'n':>8}")
    out = {}
    for name, path in (
        ("natural (as released)", "../results/ic_only_predictions.json"),
        ("deny_last (recency)", "../results/ic_deny_last_predictions.json"),
        ("deny_first", "../results/ic_deny_first_predictions.json"),
        ("balanced (majority-label)", "../results/ic_balanced_predictions.json"),
    ):
        r = report(name, path)
        if r:
            out[name] = r

    base = out.get("natural (as released)")
    if base:
        print()
        for k, v in out.items():
            if k.startswith("natural"):
                continue
            d = base["exception"] - v["exception"]
            verdict = ("REDUCES exception erasure" if d > 3 else
                       "no material change" if abs(d) <= 3 else
                       "WORSENS it")
            print(f"  {k:26s} exception {base['exception']:.1f} -> {v['exception']:.1f} "
                  f"({-d:+.1f})  {verdict}")
        print("\nIf none of these moves the exception column, the in-context majority-label")
        print("bias is not what is erasing exceptions here, and we report that.")
    with open("demoorder_analysis.json", "w") as fh:
        json.dump(out, fh, indent=1)
