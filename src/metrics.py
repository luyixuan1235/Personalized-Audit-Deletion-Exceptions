"""Evaluation metrics: accuracy, false-share rate (FPR), high-confidence
FPR, calibration (ECE, Brier), and Sens-FPR on high-sensitivity data types."""
import numpy as np

from sensitivity import tier_of


def _conf(y, yhat):
    y, yhat = np.asarray(y), np.asarray(yhat)
    tp = int(((yhat == 1) & (y == 1)).sum()); fp = int(((yhat == 1) & (y == 0)).sum())
    tn = int(((yhat == 0) & (y == 0)).sum()); fn = int(((yhat == 0) & (y == 1)).sum())
    return tp, fp, tn, fn


def fpr_of(y, yhat):
    _, fp, tn, _ = _conf(y, yhat)
    return fp / (fp + tn) if (fp + tn) else float("nan")


def basic(y, p, data_types=None, thr=0.5):
    y = np.asarray(y); p = np.asarray(p); yhat = (p >= thr).astype(int)
    tp, fp, tn, fn = _conf(y, yhat)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    out = {
        "acc": float((yhat == y).mean()),
        "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "fpr": fp / (fp + tn) if fp + tn else float("nan"),
        "fnr": fn / (fn + tp) if fn + tp else float("nan"),
    }
    if data_types is not None:
        hi = np.array([tier_of(d) == "high" for d in data_types])
        out["sens_fpr"] = fpr_of(y[hi], yhat[hi]) if hi.any() else float("nan")
    return out


def equal_error_threshold(y, p):
    """Decision threshold (calibrated on TRAIN) where FPR == FNR -- the operating
    point used for full-coverage metrics. BPR/ranking scores are not centered at
    0.5, so a fixed 0.5 would collapse to all-allow / all-deny."""
    y = np.asarray(y); p = np.asarray(p)
    ts = np.unique(p)
    if len(ts) > 256:
        ts = np.quantile(p, np.linspace(0.02, 0.98, 256))
    best_t, best_d = 0.5, 2.0
    for t in ts:
        yhat = (p >= t).astype(int)
        tp, fp, tn, fn = _conf(y, yhat)
        fpr = fp / (fp + tn + 1e-9); fnr = fn / (fn + tp + 1e-9)
        d = abs(fpr - fnr)
        if d < best_d:
            best_d, best_t = d, float(t)
    return best_t


def hc_fpr(y, p, coverage=0.5):
    """False-share rate within the top-`coverage` most confident predictions."""
    y = np.asarray(y); p = np.asarray(p); conf = np.abs(2 * p - 1)
    if len(y) == 0:
        return float("nan"), float("nan")
    thr = np.quantile(conf, 1 - coverage)
    m = conf >= thr
    yhat = (p[m] >= 0.5).astype(int)
    return fpr_of(y[m], yhat), float((yhat == y[m]).mean())


def ece(y, p, bins=10):
    y = np.asarray(y); p = np.asarray(p)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= edges[i + 1])
        if m.any():
            conf = p[m].mean(); acc = (y[m] == 1).mean()
            e += m.mean() * abs(conf - acc)
    return float(e)


def brier(y, p):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))
