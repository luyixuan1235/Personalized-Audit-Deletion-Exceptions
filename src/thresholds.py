"""Risk-calibrated decision thresholds (paper Sec. 4.4 / T0-T5).

A decision band (t_allow, t_deny) maps p_allow -> allow / deny / defer. Bands are
calibrated on a held-out split to a per-domain target false-share rate tau_d, so
high-stakes domains get a stricter budget (risk-aware = T2r).
"""
import numpy as np

# Per-domain target false-share rate (deployment policy). Lower = stricter.
RISK_KW = [("financ", 0.02), ("bank", 0.02), ("payment", 0.02), ("tax", 0.02),
           ("health", 0.03), ("medical", 0.03), ("identity", 0.03),
           ("social", 0.07), ("smart", 0.07),
           ("entertain", 0.10), ("travel", 0.10)]
DEFAULT_TAU = 0.10


def risk_target(domain):
    d = str(domain).lower()
    for kw, tau in RISK_KW:
        if kw in d:
            return tau
    return DEFAULT_TAU


def calibrate_band(p, y, tau):
    """Smallest t_allow with auto-allow FPR<=tau; largest t_deny with auto-deny
    FNR<=tau. Returns (t_allow, t_deny)."""
    p = np.asarray(p, float); y = np.asarray(y)
    if len(p) == 0:
        return 2.0, -1.0
    idx = np.argsort(-p); ps, ys = p[idx], y[idx]
    n = np.arange(1, len(ps) + 1); fpr = np.cumsum(ys == 0) / n
    ok = np.where(fpr <= tau)[0]
    t_allow = float(ps[ok.max()]) if len(ok) else 2.0
    idx2 = np.argsort(p); ps2, ys2 = p[idx2], y[idx2]
    n2 = np.arange(1, len(ps2) + 1); fnr = np.cumsum(ys2 == 1) / n2
    ok2 = np.where(fnr <= tau)[0]
    t_deny = float(ps2[ok2.max()]) if len(ok2) else -1.0
    return t_allow, t_deny


def calibrate(p, dom, y, mode="global", min_n=30, uniform_tau=0.05):
    """Return a calibration object: per-domain bands + a global fallback.
    mode: 'global' (T1), 'uniform' (T2, same tau per domain),
          'risk' (T2r, risk_target per domain)."""
    p = np.asarray(p, float); y = np.asarray(y); dom = np.asarray(dom, dtype=object)
    gtau = uniform_tau if mode != "risk" else DEFAULT_TAU
    gband = calibrate_band(p, y, gtau)
    bands = {}
    if mode in ("uniform", "risk"):
        for d in np.unique(dom):
            m = dom == d
            if m.sum() >= min_n:
                tau = uniform_tau if mode == "uniform" else risk_target(d)
                bands[d] = calibrate_band(p[m], y[m], tau)
    return {"mode": mode, "global": gband, "bands": bands}


def calibrate_user(p, user, y, tau=0.05, k=20):
    """Per-user band (T4) with shrinkage toward the global band:
    t_u = w*t_raw + (1-w)*t_global, w = n/(n+k)."""
    p = np.asarray(p, float); y = np.asarray(y); user = np.asarray(user)
    gband = calibrate_band(p, y, tau)
    bands = {}
    for uu in np.unique(user):
        m = user == uu; n = int(m.sum())
        raw = calibrate_band(p[m], y[m], tau)
        w = n / (n + k)
        bands[uu] = (w * raw[0] + (1 - w) * gband[0],
                     w * raw[1] + (1 - w) * gband[1])
    return {"mode": "user", "global": gband, "bands": bands, "key": "user"}


def decide(p, key_val, cal):
    """Return decisions in {1 allow, 0 deny, -1 defer}. key_val is the per-item
    domain (T1/T2/T2r) or user id (T4), matching the calibration's keys."""
    p = np.asarray(p, float); key_val = np.asarray(key_val, dtype=object)
    out = np.full(len(p), -1, int)
    for i in range(len(p)):
        ta, td = cal["bands"].get(key_val[i], cal["global"])
        if p[i] >= ta:
            out[i] = 1
        elif p[i] <= td:
            out[i] = 0
    return out


def evaluate(p, dom, y, cal, data_types=None):
    """Coverage, overall FPR on auto-decided, and per-domain FPR."""
    d = decide(p, dom, cal)
    auto = d >= 0
    y = np.asarray(y)
    cov = float(auto.mean())
    res = {"coverage": cov, "query_rate": 1 - cov}
    if auto.any():
        ya, da = y[auto], d[auto]
        fp = int(((da == 1) & (ya == 0)).sum()); tn = int(((da == 0) & (ya == 0)).sum())
        res["fpr"] = fp / (fp + tn) if (fp + tn) else float("nan")
        res["acc"] = float((da == ya).mean())
        perdom = {}
        dom = np.asarray(dom, dtype=object)
        for dd in np.unique(dom[auto]):
            mm = auto & (dom == dd)
            yy, ddc = y[mm], d[mm]
            f = int(((ddc == 1) & (yy == 0)).sum()); t = int(((ddc == 0) & (yy == 0)).sum())
            perdom[str(dd)] = f / (f + t) if (f + t) else float("nan")
        res["per_domain_fpr"] = perdom
    return res
