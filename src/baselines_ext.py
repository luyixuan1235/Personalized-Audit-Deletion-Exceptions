"""External baselines for the comparison table:
  * global / per-user majority (trivial floors),
  * GBDT over structured features (a strong non-graph classifier),
  * Signed-GCN (Derr et al. 2018 style single-space signed propagation).
Each predictor returns the same dict shape as train.predict."""
from collections import defaultdict

import numpy as np


def _pack(te, p):
    return {"y": np.array([r["label"] for r in te]), "p_allow": np.asarray(p, float),
            "domain": [r["domain"] for r in te], "data_type": [r["data_type"] for r in te]}


def maj_predict(tr, te, mode="global"):
    glob = float(np.mean([r["label"] for r in tr]))
    if mode == "global":
        return _pack(te, [glob] * len(te))
    by = defaultdict(list)
    for r in tr:
        by[r["user_id"]].append(r["label"])
    rate = {u: float(np.mean(v)) for u, v in by.items()}
    return _pack(te, [rate.get(r["user_id"], glob) for r in te])


def gbdt_predict(tr, te, seed=42):
    """Gradient-boosted trees over (domain, tool, data_type, user-allow-rate)."""
    from sklearn.ensemble import GradientBoostingClassifier

    def enc(field):
        m = {}
        for r in tr:
            m.setdefault(r[field], len(m))
        return m
    dm, tm, dtm = enc("domain"), enc("tool"), enc("data_type")
    by = defaultdict(list)
    for r in tr:
        by[r["user_id"]].append(r["label"])
    rate = {u: float(np.mean(v)) for u, v in by.items()}
    glob = float(np.mean([r["label"] for r in tr]))

    def feats(recs):
        return np.array([[dm.get(r["domain"], -1), tm.get(r["tool"], -1),
                          dtm.get(r["data_type"], -1), rate.get(r["user_id"], glob)]
                         for r in recs], float)
    ytr = np.array([r["label"] for r in tr])
    clf = GradientBoostingClassifier(random_state=seed)
    clf.fit(feats(tr), ytr)
    return _pack(te, clf.predict_proba(feats(te))[:, 1])


_FM_FIELDS = ("user_id", "domain", "tool", "data_type")


def fm_predict(tr, te, dim=16, epochs=150, lr=0.05, wd=1e-6, seed=42, device="cpu",
               objective="bce"):
    """Factorization Machine over (user, domain, tool, data_type) -- the canonical
    FEATURE-BASED CF that generalizes to unseen request COMBINATIONS by construction.
    objective='bce' (calibrated pointwise) or 'bpr' (pairwise ranking). The FM
    ARCHITECTURE is identical across objectives, so any HC-FPR gap is due to the
    OBJECTIVE, not the architecture. Index 0 per field = OOV."""
    import torch
    from collections import defaultdict
    torch.manual_seed(seed); np.random.seed(seed)
    idx = {f: {} for f in _FM_FIELDS}
    for r in tr:
        for f in _FM_FIELDS:
            idx[f].setdefault(r[f], len(idx[f]) + 1)   # 1..n ; 0 reserved for OOV
    sizes = [len(idx[f]) + 1 for f in _FM_FIELDS]

    def enc(recs):
        return torch.tensor([[idx[f].get(r[f], 0) for f in _FM_FIELDS] for r in recs],
                            dtype=torch.long, device=device)
    offs = torch.tensor(np.cumsum([0] + sizes[:-1]), dtype=torch.long, device=device)
    tot = int(sum(sizes))
    lin = torch.nn.Embedding(tot, 1).to(device); torch.nn.init.zeros_(lin.weight)
    fac = torch.nn.Embedding(tot, dim).to(device)
    torch.nn.init.normal_(fac.weight, std=0.01)
    b0 = torch.zeros(1, requires_grad=True, device=device)
    opt = torch.optim.Adam(list(lin.parameters()) + list(fac.parameters()) + [b0],
                           lr=lr, weight_decay=wd)
    ytr = torch.tensor([r["label"] for r in tr], dtype=torch.float32, device=device)
    bce = torch.nn.BCEWithLogitsLoss()
    if objective == "wbce":   # class-weighted BCE (cost-sensitive): up-weight the minority
        npos = float(ytr.sum()); nneg = float(len(ytr) - ytr.sum())
        pw = torch.tensor([nneg / max(npos, 1.0)], device=device)
        wbce = torch.nn.BCEWithLogitsLoss(pos_weight=pw)

    def logits(X):
        l = b0 + lin(X).sum(1).squeeze(-1)
        v = fac(X)                                   # (N, F, dim)
        inter = 0.5 * ((v.sum(1) ** 2) - (v ** 2).sum(1)).sum(-1)
        return l + inter

    def focal(z, y, gamma=2.0):                       # focal loss (Lin et al. 2017)
        p = torch.sigmoid(z)
        pt = torch.where(y == 1, p, 1 - p).clamp(1e-6, 1.0)
        return (-((1 - pt) ** gamma) * torch.log(pt)).mean()

    if objective == "bpr":
        from losses import bpr_loss
        pos, neg = defaultdict(list), defaultdict(list)
        for i, r in enumerate(tr):
            (pos if r["label"] == 1 else neg)[r["user_id"]].append(i)
        users = [u for u in pos if neg.get(u)]
        rng = np.random.default_rng(seed)
        Xall = enc(tr) + offs
    else:
        Xtr = enc(tr) + offs

    for ep in range(1, epochs + 1):
        opt.zero_grad()
        if objective == "bpr":
            pi = [rng.choice(pos[u]) for u in users]
            ni = [rng.choice(neg[u]) for u in users]
            loss = bpr_loss(logits(Xall[pi]), logits(Xall[ni]))
        elif objective == "wbce":
            loss = wbce(logits(Xtr), ytr)
        elif objective == "focal":
            loss = focal(logits(Xtr), ytr)
        else:  # bce
            loss = bce(logits(Xtr), ytr)
        loss.backward(); opt.step()
        if ep == 1 or ep % 50 == 0 or ep == epochs:
            print(f"  [fm/{objective}] epoch {ep:4d}/{epochs}  loss={float(loss.detach()):.4f}")
    with torch.no_grad():
        p = torch.sigmoid(logits(enc(te) + offs)).cpu().numpy()
    return _pack(te, p)


def train_signed(graph, tr, epochs=200, lr=0.01, wd=1e-5, seed=42, device="cpu",
                 use_sideinfo=True, objective="bce"):
    """SignedGCN trained with objective='bce' (calibrated) or 'bpr' (ranking). The
    ARCHITECTURE (signed graph + single score) is identical across objectives, so any
    HC-FPR gap isolates the effect of the OBJECTIVE."""
    import torch
    from collections import defaultdict
    from model import SignedGCN
    torch.manual_seed(seed); np.random.seed(seed)
    u, q, y = graph.encode(tr)
    m = SignedGCN(graph, use_sideinfo=use_sideinfo, device=device)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
    ut = torch.tensor(u, device=device); qt = torch.tensor(q, device=device)
    yt = torch.tensor(y, dtype=torch.float32, device=device)
    bce = torch.nn.BCELoss()
    if objective == "bpr":
        from losses import bpr_loss
        pos, neg = defaultdict(list), defaultdict(list)
        for i in range(len(u)):
            (pos if y[i] == 1 else neg)[int(u[i])].append(int(q[i]))
        users = [uu for uu in pos if neg.get(uu)]
        rng = np.random.default_rng(seed)
    m.train()
    for ep in range(1, epochs + 1):
        opt.zero_grad()
        if objective == "bpr":
            bu = torch.tensor(users, device=device)
            bp = torch.tensor([rng.choice(pos[uu]) for uu in users], device=device)
            bn = torch.tensor([rng.choice(neg[uu]) for uu in users], device=device)
            loss = bpr_loss(m.scores(bu, bp), m.scores(bu, bn))
        else:
            p, _, _, _ = m.opinion(ut, qt)
            loss = bce(p.clamp(1e-6, 1 - 1e-6), yt)
        loss.backward(); opt.step()
        if ep == 1 or ep % 50 == 0 or ep == epochs:
            print(f"  [signed/{objective}] epoch {ep:4d}/{epochs}  loss={float(loss.detach()):.4f}")
    return m
