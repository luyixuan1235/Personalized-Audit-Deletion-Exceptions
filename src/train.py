"""Training and prediction for the dual-graph model.

train(graph, train_records, cfg) -> model, fitting either:
  * the evidential refusal objective ('evi' / 'evi_refuse'), pointwise; or
  * BPR on the net signed score ('bpr'), pairwise (the un-identified ablation).
predict(model, graph, records) -> arrays (p_allow, u, y, domain, data_type).
"""
from collections import defaultdict

import numpy as np
import torch

from losses import bpr_loss, evidential_loss
from model import DualGraph
from thresholds import risk_target

DEFAULTS = dict(dim=32, layers=3, mode="dual", use_sideinfo=True, loss="evi_refuse",
                epochs=200, lr=0.01, wd=1e-5, lam_kl_max=0.1, risk_loss=True, seed=42)


def _omega(graph, q, risk_loss):
    if not risk_loss:
        return np.ones(len(q), np.float32)
    return np.array([min(3.0, 0.10 / risk_target(graph.req_dom_name[qi])) for qi in q],
                    np.float32)


def train(graph, train_records, **kw):
    cfg = {**DEFAULTS, **kw}
    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    dev = cfg.get("device", "cpu")
    u, q, y = graph.encode(train_records)
    comp = cfg.get("comp") or (cfg["use_sideinfo"],) * 3
    model = DualGraph(graph, dim=cfg["dim"], layers=cfg["layers"], mode=cfg["mode"],
                comp=comp, combine=cfg.get("combine", "sum"), device=dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    ut = torch.tensor(u, device=dev); qt = torch.tensor(q, device=dev)
    yt = torch.tensor(y, device=dev)
    om = torch.tensor(_omega(graph, q, cfg["risk_loss"]), device=dev)

    if cfg["loss"] == "bpr":
        pos, neg = defaultdict(list), defaultdict(list)
        for i in range(len(u)):
            (pos if y[i] == 1 else neg)[int(u[i])].append(int(q[i]))
        users = [uu for uu in pos if neg.get(uu)]
        rng = np.random.default_rng(cfg["seed"])

    model.train()
    for ep in range(1, cfg["epochs"] + 1):
        opt.zero_grad()
        if cfg["loss"] == "bpr":
            bu, bp, bn = [], [], []
            for uu in users:
                bu.append(uu); bp.append(rng.choice(pos[uu])); bn.append(rng.choice(neg[uu]))
            bu = torch.tensor(bu, device=dev)
            sp_a, sp_d = model.scores(bu, torch.tensor(bp, device=dev))
            sn_a, sn_d = model.scores(bu, torch.tensor(bn, device=dev))
            loss = bpr_loss(sp_a - sp_d, sn_a - sn_d)
        elif cfg["loss"] == "bce":
            # Pointwise calibrated objective on the SAME architecture as the 'bpr'
            # path -- lets us cross any architecture (incl. the one-sided atomic-ID CF
            # of Table 1) with either objective, so the grid isolates the objective.
            p, _, _, _ = model.opinion(ut, qt)
            loss = torch.nn.functional.binary_cross_entropy(
                p.clamp(1e-6, 1 - 1e-6), yt.float())
        else:
            lam = 0.0 if cfg["loss"] == "evi" else \
                cfg["lam_kl_max"] * min(1.0, ep / (0.5 * cfg["epochs"]))  # anneal
            _, _, e_a, e_d = model.opinion(ut, qt)
            loss = evidential_loss(e_a, e_d, yt, omega=om, lam_kl=lam)
        loss.backward(); opt.step()
        if ep == 1 or ep % 50 == 0 or ep == cfg["epochs"]:
            print(f"  epoch {ep:4d}/{cfg['epochs']}  loss={float(loss.detach()):.4f}")
    return model


@torch.no_grad()
def predict(model, graph, records):
    model.eval()
    u, q, y = graph.encode(records)
    dev = model.device
    p, uu, e_a, e_d = model.opinion(torch.tensor(u, device=dev), torch.tensor(q, device=dev))
    return {
        "p_allow": p.cpu().numpy(), "u": uu.cpu().numpy(),
        "e_allow": e_a.cpu().numpy(), "e_deny": e_d.cpu().numpy(),
        "y": y, "user": u, "req": q,
        "domain": [graph.req_dom_name[qi] for qi in q],
        "data_type": [graph.req_dt_name[qi] for qi in q],
    }
