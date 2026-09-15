"""The Rare Exception / Conditional Policy hypothesis.

Proposed by an outside reviewer after nine of our own hypotheses failed. The claim:

    A user's permission policy is not a threshold on a scalar preference. It is
    "grant by default, EXCEPT for a small number of conditional rules" --
    e.g. IF data_type = Medical THEN deny; IF tool = third-party AND data = Finance
    THEN deny. Those rules have tiny support in training, so the gradient is dominated
    by the user's grants and the rules are never learned. The MORE permissive a user is,
    the fewer such rules they have, and the more completely those rules are drowned out.

This is the first hypothesis that would also explain our own failures: it explains why our
synthetic data (a one-dimensional threshold model with NO rare rules) did not reproduce
the effect, why global class-reweighting did not help (a permissive user's denials are
rare *within their own history*, not globally), and why an MLP did not help (no inductive
bias for a rare conjunction).

But it is still a story until it is tested. Two tests, neither of which requires us to
believe anything:

TEST A (SUPPORT) -- pure data + one model. If denials are overridden because they sit on
    rarely-seen feature conjunctions, then (i) override rate should rise as the support of
    a denial's conjunction falls, and (ii) permissive users' denials should sit on
    lower-support conjunctions. If overrides are spread evenly across support levels, the
    hypothesis is dead.

TEST B (RULE-SYNTH) -- the synthetic our earlier one should have been. Each user grants by
    default and holds k deny-rules over feature conjunctions; permissiveness is set by k.
    There is no shared "sensitivity axis" and no per-user threshold. If the effect appears
    here and did not appear in the threshold-based synthetic, the difference between the
    two generative models IS the mechanism.
"""
import os
import sys
from collections import defaultdict

import numpy as np

import _common as C
import baselines_ext as B
import metrics as M

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import Graph, build_records, per_user_folds  # noqa: E402
from train import predict, train  # noqa: E402

KFOLD = 5
SEED = 42


def _epochs():
    return 15 if C.QUICK else int(os.environ.get("PERM_EPOCHS", 200))


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra = (ra - ra.mean()) / (ra.std() + 1e-9)
    rb = (rb - rb.mean()) / (rb.std() + 1e-9)
    return float((ra * rb).mean())


def perm_p(x, y, r, iters=3000, seed=0):
    rng = np.random.default_rng(seed)
    null = np.array([spearman(x, rng.permutation(y)) for _ in range(iters)])
    return float((null >= r).mean())


def fm_bce(g, tr, te, ep):
    return B.fm_predict(tr, te, objective="bce"), B.fm_predict(tr, tr, objective="bce")


def atomic_bpr(g, tr, te, ep):
    m = train(g, tr, mode="allow", loss="bpr", use_sideinfo=False, epochs=ep)
    return predict(m, g, te), predict(m, g, tr)


def collect(records, folds, fit, ep, conj):
    """Per DENIAL: user's grant rate, whether it was overridden, and the SUPPORT of the
    conjunction it sits on (how many training records share that conjunction, and how many
    of THIS USER's training records do)."""
    G, V, S, SU = [], [], [], []
    for f in range(KFOLD):
        tr = [records[i] for i in range(len(records)) if folds[i] != f]
        te = [records[i] for i in range(len(records)) if folds[i] == f]
        g = Graph(tr, records)
        sup = defaultdict(int)
        sup_u = defaultdict(int)
        for r in tr:
            sup[conj(r)] += 1
            sup_u[(r["user_id"], conj(r))] += 1
        pr, prtr = fit(g, tr, te, ep)
        y = np.asarray(pr["y"]); p = np.asarray(pr["p_allow"])
        thr = float(M.equal_error_threshold(np.asarray(prtr["y"]),
                                            np.asarray(prtr["p_allow"])))
        yh = (p >= thr).astype(int)
        u = np.array([r["user_id"] for r in te])
        gr = {uu: float(y[u == uu].mean()) for uu in np.unique(u)}
        for i, r in enumerate(te):
            if y[i] != 0:
                continue
            G.append(gr[r["user_id"]]); V.append(float(yh[i]))
            S.append(sup[conj(r)]); SU.append(sup_u[(r["user_id"], conj(r))])
    return (np.array(G), np.array(V), np.array(S, float), np.array(SU, float))


def summarise(G, V, label):
    lo = 100 * V[G < 0.2].mean() if (G < 0.2).any() else float("nan")
    hi = 100 * V[G >= 0.6].mean() if (G >= 0.6).any() else float("nan")
    r = spearman(G, V)
    print(f"    {label:34s} n={len(G):6d}  restrictive={lo:5.1f}%  "
          f"permissive={hi:5.1f}%  ratio={hi / max(lo, 0.1):5.1f}x  rho={r:+.3f}")
    return dict(n=int(len(G)), restrictive=float(lo), permissive=float(hi),
                ratio=float(hi / max(lo, 0.1)), rho=r)


# --------------------------------------------------------- rule-based synthetic
def make_rule_synth(n_users=181, per_user=42, seed=0):
    """A user grants BY DEFAULT, and holds k deny-rules over feature conjunctions.
    Permissiveness is set by k -- no shared sensitivity axis, no per-user threshold.
    A permissive user (small k) has very few, very specific refusals."""
    rng = np.random.default_rng(seed)
    DOM = [f"d{i}" for i in range(8)]
    TOOL = [f"t{i}" for i in range(20)]
    DT = [f"c{i}" for i in range(12)]
    recs = []
    for uidx in range(n_users):
        k = int(rng.integers(1, 9))                    # 1..8 deny-rules
        rules = set()
        for _ in range(k):
            rules.add((TOOL[rng.integers(len(TOOL))], DT[rng.integers(len(DT))]))
        for _ in range(per_user):
            d = DOM[rng.integers(len(DOM))]
            t = TOOL[rng.integers(len(TOOL))]
            c = DT[rng.integers(len(DT))]
            y = 0 if (t, c) in rules else 1            # grant unless a rule fires
            recs.append(dict(user_id=f"U{uidx}", domain=d, tool=t, data_type=c,
                             request_key=f"{d}|{t}|{c}", label=int(y)))
    return recs


def main():
    ep = _epochs()
    out = {}
    conj = lambda r: (r["tool"], r["data_type"])  # noqa: E731

    # ------------------------------------------------------------------- TEST A
    print("=" * 90)
    print("TEST A (SUPPORT): are the overridden denials the ones on RARELY-SEEN")
    print("feature conjunctions (tool x data_type)?  [pure data + one model]")
    print("=" * 90)
    records, _ = build_records()
    folds = per_user_folds(records, KFOLD, SEED)
    out["support"] = {}
    for mlab, fit in (("ranking CF", atomic_bpr), ("calibrated FM", fm_bce)):
        G, V, S, SU = collect(records, folds, fit, ep, conj)
        print(f"\n  --- {mlab}  ({len(G)} denials)")
        print(f"    {'conjunction support':>22}{'denials':>9}{'overridden':>12}")
        qs = np.percentile(S, [25, 50, 75])
        bins = [(-1, qs[0], "lowest quartile"), (qs[0], qs[1], "2nd"),
                (qs[1], qs[2], "3rd"), (qs[2], 1e9, "highest quartile")]
        rows = {}
        for lo, hi, lab in bins:
            m = (S > lo) & (S <= hi)
            if m.sum() == 0:
                continue
            rows[lab] = dict(n=int(m.sum()), rate=100 * float(V[m].mean()),
                             median_support=float(np.median(S[m])))
            bar = "█" * int(100 * V[m].mean() / 3)
            print(f"    {lab:>22}{m.sum():9d}{100 * V[m].mean():11.1f}%  {bar}")
        r_sv = spearman(-S, V)          # low support -> overridden?
        p_sv = perm_p(-S, V, r_sv)
        r_gs = spearman(G, -S)          # permissive users on low-support conjunctions?
        p_gs = perm_p(G, -S, r_gs)
        print(f"\n    rho(LOW support, overridden)          = {r_sv:+.3f}  p={p_sv:.4f}"
              f"   {'<-- core prediction' if p_sv < 0.01 else '<-- FAILS'}")
        print(f"    rho(permissive user, LOW support)     = {r_gs:+.3f}  p={p_gs:.4f}")
        out["support"][mlab] = dict(bins=rows, rho_support_override=r_sv, p_support=p_sv,
                                    rho_permissive_lowsupport=r_gs, p_perm_lowsupport=p_gs)

    # ------------------------------------------------------------------- TEST B
    print("\n" + "=" * 90)
    print("TEST B (RULE-SYNTH): a user grants by default and holds k deny-RULES over")
    print("conjunctions. No sensitivity axis, no per-user threshold. Does the effect")
    print("appear here, where the threshold-based synthetic gave the OPPOSITE?")
    print("=" * 90)
    out["rule_synth"] = {}
    srecs = make_rule_synth()
    sfolds = per_user_folds(srecs, KFOLD, SEED)
    gr = defaultdict(lambda: [0, 0])
    for r in srecs:
        a = gr[r["user_id"]]; a[0] += r["label"]; a[1] += 1
    rates = np.array([a[0] / a[1] for a in gr.values()])
    print(f"    synthetic users: grant rate 10/50/90 pct = "
          f"{np.percentile(rates, 10):.2f}/{np.percentile(rates, 50):.2f}/"
          f"{np.percentile(rates, 90):.2f}\n")
    for mlab, fit in (("ranking CF", atomic_bpr), ("calibrated FM", fm_bce)):
        G, V, _, _ = collect(srecs, sfolds, fit, ep, conj)
        out["rule_synth"][mlab] = summarise(G, V, f"rule-based synthetic, {mlab}")
    print("\n    Recall the THRESHOLD-based synthetic gave ratio 0.2x (reversed).")
    print("    If this one gives a ratio well above 1, the generative structure --")
    print("    rare conditional rules -- is the mechanism.")

    C.save(out, "table28_rare_rule.json")
    print("\nSaved -> table28_rare_rule.json")


if __name__ == "__main__":
    main()
