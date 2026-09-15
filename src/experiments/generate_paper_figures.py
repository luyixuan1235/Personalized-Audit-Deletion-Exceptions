"""Regenerate the five manuscript figures from reproduced result files.

Run from src/experiments or src:
    python experiments/generate_paper_figures.py

Writes PDF (vector) + PNG preview into journal-manuscript/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.colors import LinearSegmentedColormap

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = ROOT / "results"
MS = ROOT.parents[1] / "journal-manuscript"
sys.path.insert(0, str(HERE))
from analyze_wu_peruser import load_cf, load_llm  # noqa: E402

# Okabe–Ito (colorblind-safe)
BLUE = "#0077BB"
VERM = "#CC3311"
ORANGE = "#EE7733"
TEAL = "#009988"
NAVY = "#334155"
ROUTINE = "#7FBFA5"   # soft green (original figure)
EXCEPTION = "#C25E5E" # soft red   (original figure)
INK = "#1F2937"
MUTED = "#6B7280"
FILL = "#F8FAFC"
HARM = "#9F1239"
SAFE = "#0F766E"


def setup():
    # Springer submission requires all fonts embedded in figure PDFs.
    # Use DejaVu Serif (bundled with matplotlib, embeds as TrueType) rather
    # than Times New Roman, which resolves as a non-embedded Type-3/CID font.
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
    })


def wilson(k, n, z=1.96):
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return 100 * (c - h) / d, 100 * (c + h) / d


def save(fig, name, pdf=True, png=True, dpi=300):
    MS.mkdir(parents=True, exist_ok=True)
    pdf_path = MS / f"{name}.pdf"
    png_path = MS / f"{name}.png"
    written = []
    if pdf:
        fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.03)
        written.append(pdf_path.name)
    if png:
        fig.savefig(png_path, dpi=dpi, bbox_inches="tight", pad_inches=0.03)
        written.append(png_path.name)
    plt.close(fig)
    print("wrote " + "  ".join(written))
    return png_path if png else pdf_path


def eor_from_preds(majority="train"):
    """Headline EOR bars. Default majority is train-only (P0-2)."""
    import _p0_metrics as P0
    maps = P0.corpus_majorities()
    rows = []
    for name in ("CF", "IC", "IC+CF"):
        users, y, yh, _ = P0.load_model(name)
        test_map = P0.test_majority(users, y)
        mapping = maps["train"] if majority == "train" else test_map
        mu, _ = P0.mu_vector(users, mapping, test_map)
        st = P0.rates(y, yh, mu)
        n_exc, n_rout = st["n_exc"], st["n_rout"]
        k_exc, k_rout = st["k_exc"], st["k_rout"]
        rows.append(dict(
            name=name, exc=st["exception"], rout=st["routine"],
            n_exc=n_exc, n_rout=n_rout,
            el=wilson(k_exc, n_exc), rl=wilson(k_rout, n_rout),
            ratio=st["ratio"], majority=majority,
        ))
    return rows


def fig_eor_main():
    rows = eor_from_preds()
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(6.1, 2.95))
    h = 0.34
    for i, d in enumerate(rows):
        ax.barh(y[i] + h, d["rout"], h, color=ROUTINE, edgecolor=INK,
                linewidth=0.6, zorder=3, label="routine" if i == 0 else None)
        ax.barh(y[i], d["exc"], h, color=EXCEPTION, edgecolor=INK,
                linewidth=0.6, zorder=3, label="exceptions" if i == 0 else None)
        ax.text(d["rout"] + 0.6, y[i] + h, f'{d["rout"]:.1f}', va="center",
                ha="left", fontsize=9, color=INK)
        ax.text(d["exc"] + 0.6, y[i], f'{d["exc"]:.1f}', va="center",
                ha="left", fontsize=9, color=INK)
    ax.set_yticks(y + h / 2)
    ax.set_yticklabels([d["name"] for d in rows], fontsize=10)
    ax.set_xlim(0, 47)
    ax.set_xlabel("overridden (%), train-only majority", fontsize=10)
    ax.xaxis.grid(True, linestyle="-", linewidth=0.4, color="#D9D9D9", zorder=0)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.7)
        sp.set_color(INK)
    ax.legend(frameon=True, facecolor="white", edgecolor="#D9D9D9", loc="upper right", fontsize=9)
    return save(fig, "fig-eor-main")


def fig_two_priors():
    """Grouped horizontal bars: 3 models × 4 habit×crowd combos (original style)."""
    data = json.loads((RESULTS / "table52_two_priors.json").read_text(encoding="utf-8"))
    combos = [
        ("agree_habit/agree_crowd", "agree habit\n· agree crowd"),
        ("agree_habit/contra_crowd", "agree habit\n· contra crowd"),
        ("contra_habit/agree_crowd", "contra habit\n· agree crowd"),
        ("contra_habit/contra_crowd", "contra habit\n· contra crowd"),
    ]
    models = [
        ("CF only (LightGCN+BPR)", "CF", "#5B8DB8"),
        ("IC only (LLM)", "IC", "#7FBFA5"),
        ("IC+CF (deployed, 85.1%)", "IC+CF", "#C98BAE"),
    ]
    fig, ax = plt.subplots(figsize=(6.4, 3.55))
    n_c, n_m = len(combos), len(models)
    group_h, bh = 1.15, 0.26
    ypos = np.arange(n_c)[::-1] * group_h
    for mi, (src, lab, col) in enumerate(models):
        off = (mi - 1) * bh
        for ci, (key, _) in enumerate(combos):
            v = data[src][key]["override"]
            ax.barh(ypos[ci] + off, v, bh * 0.92, color=col, edgecolor="white",
                    linewidth=0.4, zorder=3, label=lab if ci == 0 else None)
            ax.text(v + 0.7, ypos[ci] + off, f"{v:.1f}", va="center",
                    ha="left", fontsize=8, color=INK)
    ax.set_yticks(ypos)
    ax.set_yticklabels([c[1] for c in combos], fontsize=9.5, color=INK, linespacing=1.05)
    ax.tick_params(axis="y", length=3.5, pad=7, labelsize=9.5, colors=INK)
    ax.tick_params(axis="x", labelsize=9, colors=INK)
    ax.set_xlim(0, 74)
    ax.set_xlabel("overridden (%)", fontsize=10)
    ax.xaxis.grid(True, linestyle="-", linewidth=0.4, color="#D9D9D9", zorder=0)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_linewidth(0.7); sp.set_color(INK)
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5),
              fontsize=8.5, title="model", title_fontsize=8.5)
    fig.subplots_adjust(left=0.26, right=0.80, top=0.96, bottom=0.16)
    return save(fig, "fig-two-priors")


def fig_peruser_dist():
    """Figure 1: per-user false-grant-rate distribution (boxplot + hybrid ref line)."""
    d = json.loads((RESULTS / "table21_peruser.json").read_text(encoding="utf-8"))
    fpr = np.array(d["per_user_fpr"], dtype=float)
    hybrid = json.loads((RESULTS / "table53_eor_deployed.json").read_text(encoding="utf-8"))["IC+CF"]["fgr"]
    fig, ax = plt.subplots(figsize=(6.1, 2.5))
    bp = ax.boxplot(fpr, vert=False, widths=0.45, patch_artist=True, zorder=3,
                    boxprops=dict(facecolor="#9CC3E5", edgecolor=INK, linewidth=0.7),
                    medianprops=dict(color=VERM, linewidth=1.6),
                    whiskerprops=dict(color=INK, linewidth=0.7),
                    capprops=dict(color=INK, linewidth=0.7),
                    flierprops=dict(marker="o", markersize=2.5, markerfacecolor=INK,
                                    markeredgecolor="none", alpha=0.5))
    ax.axvline(hybrid, color=NAVY, linestyle="--", linewidth=1.0, zorder=2)
    ax.text(hybrid, 1.42, f"deployed hybrid ≈ {hybrid:.0f}%", ha="center",
            va="bottom", fontsize=7.5, color=NAVY)
    ax.set_yticks([])
    ax.set_xlabel("per-user false-grant rate (%)", fontsize=9.5)
    ax.set_xlim(-3, 105)
    ax.set_ylim(0.5, 1.62)
    ax.xaxis.grid(True, linestyle="-", linewidth=0.4, color="#D9D9D9", zorder=0)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_linewidth(0.7); sp.set_color(INK)
    med = float(np.median(fpr)); top25 = d["observed"]["top25"]
    ax.text(102, 1.0, f"median {med:.0f}%  ·  top-25% users bear {top25:.0f}% of false grants",
            ha="right", va="center", fontsize=7, color=MUTED)
    return save(fig, "fig-peruser-fpr")


def fig_supp_hist():
    """Supplementary histogram: decisions per user in the Wu corpus."""
    data = json.loads((ROOT / "data" / "processed_dataset.json").read_text(encoding="utf-8"))
    counts = [len(u.get("training", [])) + len(u.get("testing", []))
              for u in data.values()]
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.hist(counts, bins=range(0, max(counts) + 6, 5), color="#9CC3E5",
            edgecolor=INK, linewidth=0.5, zorder=3)
    ax.set_xlabel("decisions per user", fontsize=9.5)
    ax.set_ylabel("users", fontsize=9.5)
    ax.yaxis.grid(True, linestyle="-", linewidth=0.4, color="#D9D9D9", zorder=0)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_linewidth(0.7); sp.set_color(INK)
    m = float(np.mean(counts))
    ax.axvline(m, color=VERM, linestyle="--", linewidth=1.0)
    ax.text(m + 2, ax.get_ylim()[1] * 0.85, f"mean {m:.0f}", fontsize=8, color=VERM)
    return save(fig, "fig-supp-decisions-hist")




def fig_denial_ladder():
    d = json.loads((RESULTS / "table24_denial_ablation.json").read_text(encoding="utf-8"))
    order = ["discard", "bpr", "bce", "signed"]
    labels = ["discarded", "ranking\nnegatives", "calibrated\nlabels", "signed\nchannel"]
    acc = [d[k]["acc"] for k in order]
    perm = [d[k]["permissive"] for k in order]
    restr = [d[k]["restrictive"] for k in order]
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(5.9, 3.5))
    ax.plot(x, perm, color=EXCEPTION, marker="s", markersize=7, linewidth=1.4,
            label=f"permissive refusals ({perm[0]:.1f}–{perm[-1]:.1f}%)", zorder=3)
    ax.plot(x, restr, color=SAFE, marker="o", markersize=7, linewidth=1.4,
            linestyle="--", label="restrictive refusals (−64%)", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("override rate (%)", fontsize=10)
    ax.set_xlabel("denial-representation configuration ladder", fontsize=9.5)
    ax.set_ylim(-1, 38)
    ax.yaxis.grid(True, linestyle="-", linewidth=0.4, color="#D9D9D9", zorder=0)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.7)
        sp.set_color(INK)
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    ax.text(2.35, 20.5, "configuration ladder:", ha="center",
            fontsize=8.5, color="#4B5563")
    ax.text(2.35, 16.2, f"{acc[0]:.1f}% → {acc[3]:.1f}%", ha="center",
            fontsize=9.5, color="#4B5563")
    return save(fig, "fig-denial-ladder")


def _box(ax, x, y, w, h, text, fc, ec=None, fs=7.5, weight="medium"):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                       facecolor=fc, edgecolor=ec or INK, linewidth=0.9, zorder=2)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=INK, fontweight=("bold" if weight == "bold" else "normal"),
            zorder=3, wrap=True)
    return p


def _chip(ax, x, y, w, h, text, fc, ec, tc="#272727", fs=7.5):
    """Rounded decision chip: text is centred; box is sized independently of wrap."""
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.03,rounding_size=0.14",
        facecolor=fc, edgecolor=ec, linewidth=0.9, zorder=3,
    ))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, fontweight="bold", color=tc, zorder=4)


def _v_arrow(ax, x, y0, y1, color, lw=1.15, ms=11):
    ax.add_patch(FancyArrowPatch(
        (x, y0), (x, y1),
        arrowstyle="-|>", mutation_scale=ms, lw=lw,
        color=color, shrinkA=1.5, shrinkB=1.5, zorder=5,
        clip_on=False,
    ))


def fig_erasure_directions():
    """Two-panel schematic. Labels occupy dedicated bands; chips never share a box with titles."""
    prev = {
        "font.family": plt.rcParams["font.family"],
        "font.sans-serif": plt.rcParams["font.sans-serif"],
        "svg.fonttype": plt.rcParams["svg.fonttype"],
        "pdf.fonttype": plt.rcParams["pdf.fonttype"],
        "font.size": plt.rcParams["font.size"],
    }
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 8,
    })

    ink = "#272727"
    muted = "#767676"
    maj_fc, maj_ec = "#DDF3DE", "#5B8F5B"
    exc_fc, exc_ec = "#F6CFCB", "#B64342"
    mod_fc, mod_ec = "#E8E8E8", "#4D4D4D"

    specs = [
        dict(letter="(a)",
             title="Permissive user  ·  majority grant",
             majority="GRANT", exception="DENY", model="GRANT",
             outcome_top="Exception erased",
             outcome_bot="unauthorized release",
             oc="#B64342", fc="#F8E8E6"),
        dict(letter="(b)",
             title="Restrictive user  ·  majority deny",
             majority="DENY", exception="GRANT", model="DENY",
             outcome_top="Exception erased",
             outcome_bot="extra prompt",
             oc="#7A4A2B", fc="#F3EDE4"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(6.7, 4.05))
    fig.subplots_adjust(left=0.03, right=0.99, top=0.90, bottom=0.04, wspace=0.14)

    x0, gap, n = 0.45, 0.16, 4
    chip_w = (10.0 - 2 * x0 - (n - 1) * gap) / n
    chip_h = 1.10

    y_user = 7.70
    y_model = 4.80
    y_out = 0.42
    h_out = 2.88

    for ax, s in zip(axes, specs):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10.2)
        ax.axis("off")
        ax.text(0.00, 1.045, s["letter"], transform=ax.transAxes,
                fontsize=9, fontweight="bold", color=ink, ha="left", va="bottom")
        ax.text(0.12, 1.045, s["title"], transform=ax.transAxes,
                fontsize=8, color=ink, ha="left", va="bottom")

        ax.text(x0, 9.18, "User decision", ha="left", va="bottom",
                fontsize=6.5, color=muted)
        xs = [x0 + i * (chip_w + gap) for i in range(n)]
        labs = [s["majority"]] * 3 + [s["exception"]]
        for x, lab in zip(xs, labs):
            is_exc = lab != s["majority"]
            _chip(ax, x, y_user, chip_w, chip_h, lab,
                  exc_fc if is_exc else maj_fc,
                  exc_ec if is_exc else maj_ec,
                  "#B64342" if is_exc else ink)

        x_exc = xs[-1] + chip_w / 2
        _v_arrow(ax, x_exc, y_user, 6.92, "#B64342", lw=1.2, ms=12)
        ax.text(x_exc, 6.68, "exception", ha="center", va="top",
                fontsize=6.5, color="#B64342", style="italic")

        ax.text(x0, 6.22, "Model output", ha="left", va="bottom",
                fontsize=6.5, color=muted)
        for x in xs:
            _chip(ax, x, y_model, chip_w, chip_h, s["model"],
                  mod_fc, mod_ec, "#4D4D4D")

        x_mid = x0 + (n * chip_w + (n - 1) * gap) / 2
        _v_arrow(ax, x_mid, y_model, y_out + h_out, s["oc"], lw=1.25, ms=13)

        ax.add_patch(FancyBboxPatch(
            (x0, y_out), n * chip_w + (n - 1) * gap, h_out,
            boxstyle="round,pad=0.04,rounding_size=0.16",
            facecolor=s["fc"], edgecolor=s["oc"], linewidth=1.05, zorder=3,
        ))
        ax.text(x_mid, y_out + h_out * 0.62, s["outcome_top"],
                ha="center", va="center", fontsize=8.5, fontweight="bold",
                color=ink, zorder=4)
        ax.text(x_mid, y_out + h_out * 0.32, s["outcome_bot"],
                ha="center", va="center", fontsize=7.5, color=s["oc"], zorder=4)

    out = save(fig, "fig-erasure-directions", pdf=True, png=True, dpi=300)
    plt.rcParams.update(prev)
    return out


def _h_arrow(ax, x0, x1, y, color="#272727", lw=1.1, ms=11):
    ax.add_patch(FancyArrowPatch(
        (x0, y), (x1, y),
        arrowstyle="-|>", mutation_scale=ms, lw=lw,
        color=color, shrinkA=1.2, shrinkB=1.2, zorder=5,
        clip_on=False,
    ))


def _flow_box(ax, x, y, w, h, title, subtitle, fc, ec, ink="#272727"):
    """Pipeline node: title and subtitle are separate glyphs, never wrap=True."""
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.05,rounding_size=0.14",
        facecolor=fc, edgecolor=ec, linewidth=1.0, zorder=2,
    ))
    if subtitle:
        ax.text(x + w / 2, y + h * 0.64, title, ha="center", va="center",
                fontsize=8, fontweight="bold", color=ink, zorder=3)
        ax.text(x + w / 2, y + h * 0.30, subtitle, ha="center", va="center",
                fontsize=6.5, color="#767676", zorder=3)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                fontsize=8, fontweight="bold", color=ink, zorder=3)


def fig_guard_arch():
    prev = {
        "font.family": plt.rcParams["font.family"],
        "font.sans-serif": plt.rcParams["font.sans-serif"],
        "svg.fonttype": plt.rcParams["svg.fonttype"],
        "pdf.fonttype": plt.rcParams["pdf.fonttype"],
        "font.size": plt.rcParams["font.size"],
    }
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 8,
    })

    ink = "#272727"
    muted = "#767676"
    fig, ax = plt.subplots(figsize=(6.7, 3.05))
    ax.set_xlim(0, 20.4)
    ax.set_ylim(0, 8.6)
    ax.axis("off")

    y_pipe, h_pipe, w_pipe = 4.35, 2.70, 4.15
    nodes = [
        (0.25, "Trained scorer", "CF, FM, or hybrid", "#E8F1F8", "#0F4D92"),
        (5.05, "Calibrator", "held-out  →  probability", "#DDF3DE", "#42949E"),
        (9.85, "Decision rule", "per-user τ  +  deferral", "#F3EDE4", "#C47A2C"),
    ]
    for x, title, sub, fc, ec in nodes:
        _flow_box(ax, x, y_pipe, w_pipe, h_pipe, title, sub, fc, ec, ink)

    cy = y_pipe + h_pipe / 2
    _h_arrow(ax, 0.25 + w_pipe, 5.05, cy, ink)
    _h_arrow(ax, 5.05 + w_pipe, 9.85, cy, ink)

    x_out, w_out, h_out = 15.05, 4.95, 1.45
    outcomes = [
        (6.65, "Auto-grant", "#DDF3DE", "#5B8F5B"),
        (4.75, "Auto-deny", "#F6CFCB", "#B64342"),
        (2.85, "Defer to the user", "#E8E8E8", "#4D4D4D"),
    ]
    fork_x = 9.85 + w_pipe
    for y, title, fc, ec in outcomes:
        _flow_box(ax, x_out, y, w_out, h_out, title, None, fc, ec, ink)
        ax.add_patch(FancyArrowPatch(
            (fork_x, cy), (x_out, y + h_out / 2),
            arrowstyle="-|>", mutation_scale=11, lw=1.05,
            color=ink, shrinkA=2.0, shrinkB=2.0, zorder=5, clip_on=False,
        ))

    ax.text(10.2, 1.05,
            "Guard parameters are fitted on a calibration split the evaluation fold never sees.",
            ha="center", va="center", fontsize=6.5, color=muted)
    ax.text(10.2, 0.50,
            "Users below the qualification bar keep the population threshold and deferral band.",
            ha="center", va="center", fontsize=6.5, color=muted)

    out = save(fig, "fig-guard-arch", pdf=True, png=True, dpi=300)
    plt.rcParams.update(prev)
    return out


def main():
    setup()
    if not MS.exists():
        raise SystemExit(f"manuscript dir not found: {MS}")
    pngs = [
        fig_peruser_dist(),
        fig_eor_main(),
        fig_two_priors(),
        fig_denial_ladder(),
        fig_erasure_directions(),
        fig_guard_arch(),
        fig_supp_hist(),
    ]
    print("done:", ", ".join(p.name for p in pngs))


if __name__ == "__main__":
    main()
