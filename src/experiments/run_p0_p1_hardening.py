"""P0-2 / P0-10 (Wu released artifact): train-only majority as PRIMARY EOR,
plus user-cluster bootstrap CIs for EOR and worst-quartile concentration.

Does not retrain models. 5 split-seeds of the Wu hybrid require the LLM API
and are out of scope here; split-seed robustness is run on our FM / atomic
models (run_eor_fm_seeds.py, run_peruser.py).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _p0_metrics as M  # noqa: E402

MODELS = ("CF", "IC", "IC+CF")
RNG = np.random.default_rng(42)


def main():
    maps = M.corpus_majorities()
    out_eor = {"primary": "m_u=train", "note": (
        "Train-only majority is computed from processed_dataset.json training "
        "answers. Evaluation-split majority is retained only as sensitivity."
    )}
    sensitivity = {}
    cluster = {}
    concentration = []

    print(f"{'model':8s}{'m_u':8s}{'Acc':>7}{'FGR':>8}{'routine':>9}{'exc':>8}{'ratio':>8}{'n_exc':>7}")
    for name in MODELS:
        users, y, yh, path = M.load_model(name)
        test_map = M.test_majority(users, y)
        row = {"source": str(path)}
        for kind in ("train", "test", "full"):
            mapping = maps[kind] if kind != "test" else test_map
            mu, n_fb = M.mu_vector(users, mapping, test_map)
            stats = M.rates(y, yh, mu)
            stats["n_fallback_to_test"] = int(n_fb)
            row[f"m_u={kind}"] = {k: (round(v, 4) if isinstance(v, float) else v)
                                  for k, v in stats.items()}
            s = stats
            print(f"{name:8s}{kind:8s}{s['acc']:7.1f}{s['fgr']:8.1f}"
                  f"{s['routine']:9.1f}{s['exception']:8.1f}{s['ratio']:8.2f}{s['n_exc']:7d}")
        mu_train, _ = M.mu_vector(users, maps["train"], test_map)
        ci = M.cluster_bootstrap_eor(users, y, yh, mu_train, RNG)
        conc_ci = M.cluster_bootstrap_concentration(users, y, yh, RNG)
        per, ns, _ = M.per_user_fpr(users, y, yh)
        perm = M.permutation_top25(per, ns, row["m_u=train"]["fgr"], RNG)
        row["cluster_CI_train"] = ci
        row["cluster_CI_top25"] = conc_ci
        row["permutation_top25"] = perm
        out_eor[name] = row["m_u=train"]
        out_eor[name]["cluster_CI"] = ci
        sensitivity[name] = {k: row[k] for k in ("m_u=test", "m_u=full")}
        cluster[name] = dict(eor=ci, top25=conc_ci, permutation_top25=perm,
                             n_users=int(len(per)),
                             fpr_median=round(float(np.median(per)), 1),
                             fpr_mean=round(float(per.mean()), 1),
                             top25_obs=round(float(M.top_share_counts(np.rint(per * ns / 100.0), ns, 0.25)), 1))
        concentration.append(dict(
            name=name,
            accuracy=row["m_u=train"]["acc"],
            aggregate_fgr=row["m_u=train"]["fgr"],
            n_users=int(len(per)),
            median=float(np.median(per)),
            top25=float(M.top_share_counts(np.rint(per * ns / 100.0), ns, 0.25)),
            permutation=perm,
            cluster_CI_top25=conc_ci,
            per_user_fpr=per.tolist(),
            per_user_ndeny=ns.tolist(),
        ))
        print(f"         cluster EOR ratio CI {ci['ratio_ci']}  "
              f"top25 {perm['observed']}% (null {perm['null_mean']}%, p={perm['p']})  "
              f"cluster top25 CI {conc_ci['top25_ci']}")

    out_eor["sensitivity"] = sensitivity
    M.save(out_eor, "table_eor_train_majority.json")
    M.save(cluster, "user_cluster_uncertainty.json")
    M.save(concentration, "table_wu_peruser_concentration.json")

    # Primary deployed-EOR table used by figures: TRAIN majority.
    table53 = {}
    for name in MODELS:
        s = out_eor[name]
        table53[name] = dict(
            acc=s["acc"], fgr=s["fgr"], routine=s["routine"],
            exception=s["exception"], n_exc=s["n_exc"], ratio=s["ratio"],
            majority="train",
        )
    table53["_sensitivity_test"] = {
        name: dict(
            acc=sensitivity[name]["m_u=test"]["acc"],
            fgr=sensitivity[name]["m_u=test"]["fgr"],
            routine=sensitivity[name]["m_u=test"]["routine"],
            exception=sensitivity[name]["m_u=test"]["exception"],
            n_exc=sensitivity[name]["m_u=test"]["n_exc"],
            ratio=sensitivity[name]["m_u=test"]["ratio"],
            majority="test",
        ) for name in MODELS
    }
    table53["_primary"] = "m_u=train"
    M.save(table53, "table53_eor_deployed.json")
    print("primary EOR is now m_u=train (table53_eor_deployed.json)")


if __name__ == "__main__":
    main()
