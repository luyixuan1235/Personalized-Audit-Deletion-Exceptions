"""Analyze the Qwen IC runs (Wu anchor + full SPA): EOR, two-sided rates, the
two-prior 2x2, elicited-confidence pathology, and the aggregate/exception ECE split.
Run: cd src && python3 experiments/analyze_qwen_ic.py
Output: results/table59_qwen_wu.json, results_spa/table60_qwen_spa.json
"""
import json, os, sys
import numpy as np
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
POS = "Yes, always share"

def load(corpus):
    rdir = "results" if corpus == "wu" else "results_spa"
    path = os.path.join(ROOT, rdir, f"qwen_ic_predictions_{corpus}.json")
    blob = json.load(open(path))
    rows = []           # (user, y, yhat, conf)
    for rec in blob["results"]:
        uid = rec["participant_id"]
        truth = {}
        for ex in rec.get("ground_truth", []):
            for k, v in ex.get("answer", {}).items():
                truth[(ex["id"], k)] = 1 if v == POS else 0
        for pr in rec.get("predictions", []):
            if not isinstance(pr, dict):
                continue
            for k, v in (pr.get("permission") or {}).items():
                key = (pr.get("id"), k)
                if key not in truth or not isinstance(v, dict):
                    continue
                lab = v.get("label"); sc = v.get("score")
                if lab is None:
                    continue
                yh = 1 if lab == POS else 0
                try: c = float(sc)
                except Exception: c = np.nan
                rows.append((uid, truth[key], yh, c, k))
    return rows, blob.get("model")

def ece(conf, correct, bins=10):
    conf, correct = np.asarray(conf), np.asarray(correct)
    m = ~np.isnan(conf)
    conf, correct = conf[m], correct[m]
    if not len(conf): return None
    edges = np.linspace(0.5, 1.0, bins + 1); e = 0.0
    for i in range(bins):
        sel = (conf >= edges[i]) & (conf < edges[i+1] + (1e-9 if i == bins-1 else 0))
        if sel.sum():
            e += sel.mean() * abs(correct[sel].mean() - conf[sel].mean())
    return float(e)

def analyze(rows, model, corpus):
    users = defaultdict(list)
    for u, y, yh, c, k in rows: users[u].append(y)
    mu = {u: (1 if np.mean(v) >= 0.5 else 0) for u, v in users.items()}
    gr = {u: float(np.mean(v)) for u, v in users.items()}
    # crowd label: leave-one-out majority on the request key
    ck = defaultdict(lambda: [0, 0])
    for u, y, yh, c, k in rows:
        ck[k][0] += y; ck[k][1] += 1
    exc, rout, conf_e, corr_e, conf_a, corr_a = [], [], [], [], [], []
    pr_ref, rc_con, cells = [], [], defaultdict(list)
    for u, y, yh, c, k in rows:
        ov = float(yh != y)
        is_exc = y != mu[u]
        (exc if is_exc else rout).append(ov)
        conf_a.append(c); corr_a.append(1 - ov)
        if is_exc: conf_e.append(c); corr_e.append(1 - ov)
        if y == 0 and gr[u] >= 0.6: pr_ref.append(ov)
        if y == 1 and gr[u] <= 0.4: rc_con.append(ov)
        s, n = ck[k]
        if n > 1:
            cm = 1 if (s - y) / (n - 1) >= 0.5 else 0
            cells[("contra_habit" if is_exc else "agree_habit",
                   "contra_crowd" if y != cm else "agree_crowd")].append(ov)
    erasing = [c for (u, y, yh, c, k) in rows if yh != y and y != mu[u] and not np.isnan(c)]
    out = dict(model=model, corpus=corpus, n=len(rows), n_users=len(users),
               acc=float(1 - np.mean([float(yh != y) for u, y, yh, c, k in rows])),
               eor_exc=float(100*np.mean(exc)), eor_rout=float(100*np.mean(rout)),
               ratio=float(np.mean(exc)/max(np.mean(rout), 1e-9)),
               n_exc=len(exc),
               perm_refusals=float(100*np.mean(pr_ref)) if pr_ref else None,
               restr_consents=float(100*np.mean(rc_con)) if rc_con else None,
               conf_while_erasing=float(np.mean(erasing)) if erasing else None,
               conf_overall=float(np.nanmean(conf_a)),
               ECE_all=ece(conf_a, corr_a), ECE_exceptions=ece(conf_e, corr_e),
               two_by_two={f"{a}/{b}": dict(override=float(100*np.mean(v)), n=len(v))
                           for (a, b), v in sorted(cells.items())})
    return out

def main():
    res = {}
    for corpus, rdir, tab in (("wu", "results", "table59_qwen_wu.json"),
                              ("spa", "results_spa", "table60_qwen_spa.json")):
        path = os.path.join(ROOT, rdir, f"qwen_ic_predictions_{corpus}.json")
        if not os.path.exists(path):
            print(f"skip {corpus}: no predictions yet"); continue
        rows, model = load(corpus)
        if not rows:
            print(f"skip {corpus}: no parsed rows"); continue
        out = analyze(rows, model, corpus)
        json.dump(out, open(os.path.join(ROOT, rdir, tab), "w"), indent=1)
        res[corpus] = out
        print(f"=== {corpus} ({model}) n={out['n']} users={out['n_users']}")
        print(f"    acc={out['acc']*100:.1f}  EOR={out['eor_exc']:.1f} routine={out['eor_rout']:.1f} "
              f"ratio={out['ratio']:.2f}x  n_exc={out['n_exc']}")
        print(f"    perm_refusals={out['perm_refusals']}  restr_consents={out['restr_consents']}")
        print(f"    conf_while_erasing={out['conf_while_erasing']}  conf_overall={out['conf_overall']:.3f}")
        print(f"    ECE_all={out['ECE_all']}  ECE_exceptions={out['ECE_exceptions']}")
        for k, v in out["two_by_two"].items():
            print(f"      {k}: {v['override']:.1f}% (n={v['n']})")
        print(f"    saved {tab}")
if __name__ == "__main__":
    main()
