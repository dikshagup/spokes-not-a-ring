#!/usr/bin/env python
"""Figure 3: ring position and radius of the two-day uncertainty prompts.

    PYTHONPATH=src:experiments/common:experiments/fig3_uncertainty \\
        python experiments/fig3_uncertainty/plot_fig3.py             # CPU, seconds
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                # noqa: E402
import numpy as np                                             # noqa: E402

from paper_style import helvetica                              # noqa: E402
from ring_coords import (LAYER, capture_path, family_geometry, load_capture,  # noqa: E402
                         load_ruler)
from ring_plates import (BLACK_AXES, FS_LABEL, FS_PANEL, FS_RING, FS_SUB, FS_TICK, FS_LEG,  # noqa: E402
                         G_RULE, G_TEXT, level_colours, ring, scatter_family)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "figures", "fig3_uncertainty")
# cropped width (in) of the other plates, used so type sizes match
REFERENCE_WIDTH_IN = 12.664
KEY = {1.0: ("always", "never"), 0.5: ("usually", "sometimes"),
       0.0: ("sometimes", "sometimes"), -0.5: ("sometimes", "usually"),
       -1.0: ("never", "always")}


def summarise(g):
    """Print radius by level and separation, spoke alignment and favoured side."""
    levels = sorted(set(g["lean"].tolist()))
    print("mean radius (SD) by level, per separation:")
    for s in (1, 2, 3):
        row = []
        for v in levels:
            k = (g["gap"] == s) & (g["lean"] == v)
            row.append(f"{v:+.1f}: {g['r'][k].mean():.2f} ({g['r'][k].std(ddof=1):.2f})")
        conf = (g["gap"] == s) & (np.abs(g["lean"]) == 1)
        even = (g["gap"] == s) & (g["lean"] == 0)
        print(f"  {s} apart  " + "  ".join(row) +
              f"   inward at even split {100 * (1 - g['r'][even].mean() / g['r'][conf].mean()):.0f}%")
    d = np.minimum(np.abs(g["w"]), np.abs(g["w"] - 1))      # arc fraction from nearest spoke
    for s in (2, 3):
        m, e = g["gap"] == s, (g["gap"] == s) & (g["lean"] == 0)
        fa = (g["gap"] == s) & (g["lean"] == 1)
        fb = (g["gap"] == s) & (g["lean"] == -1)
        print(f"  {s} apart: within a quarter-arc of a spoke {100 * (d[m] < .25).mean():.0f}% "
              f"(even split {100 * (d[e] < .25).mean():.0f}%); committed prompts on the "
              f"favoured side: A {100 * (g['w'][fa] < .5).mean():.0f}%, "
              f"B {100 * (g['w'][fb] >= .5).mean():.0f}%")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layer", type=int, default=LAYER)
    ap.add_argument("--capture", default=None, help="capture file (default: the data dir)")
    ap.add_argument("--ruler", default=None)
    ap.add_argument("--out", default=OUT, help="output stem; .png and .pdf are written")
    args = ap.parse_args()

    mu, W, M, prov = load_ruler(args.layer, args.ruler)
    g = family_geometry(load_capture(args.capture or capture_path("frequency", args.layer)),
                        mu, W, M)
    levels = np.array(sorted(set(g["lean"].tolist())))
    col = level_colours(levels)
    summarise(g)

    with helvetica.context(), plt.rc_context(BLACK_AXES):
        fig = draw(g, levels, col)
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        for ext in ("png", "pdf"):
            fig.savefig(f"{args.out}.{ext}", bbox_inches="tight")
        bb = fig.get_tightbbox(fig.canvas.get_renderer()).padded(
            plt.rcParams["savefig.pad_inches"])
        plt.close(fig)
    print(f"include at width={bb.width / REFERENCE_WIDTH_IN:.3f}\\textwidth "
          f"(cropped {bb.width:.2f} x {bb.height:.2f} in)")
    print(f"-> {args.out}.png")


def draw(g, levels, col):
    FW, FH = 9.35, 6.61
    PW, GUT, X0 = 2.564, 0.40, 0.70
    BW, BH = 2.00, 1.55
    fig = plt.figure(figsize=(FW, FH))
    fx = lambda inch: inch / FW                # noqa: E731
    fy = lambda inch: inch / FH                # noqa: E731

    kx = fig.add_axes([fx(X0), fy(5.85), fx(3 * PW + 2 * GUT), fy(0.36)])
    kx.set_xlim(0, 1); kx.set_ylim(0, 1); kx.axis("off")
    for j, v in enumerate(levels[::-1]):
        x0 = j / len(levels)
        kx.plot(x0 + 0.012, 0.5, "o", ms=7.5, mfc=col[v], mec="black", mew=.3, clip_on=False)
        wa, wb = KEY[float(v)]
        kx.text(x0 + 0.030, 0.5, f"{wa} A,\n{wb} B", fontsize=FS_LEG, color=G_TEXT,
                va="center", linespacing=1.0)
    rng = np.random.default_rng(0)
    DH = PW * 2.75 / 2.90
    for i, s in enumerate((1, 2, 3)):
        ax = fig.add_axes([fx(X0 + i * (PW + GUT)), fy(3.40), fx(PW), fy(DH)])
        ring(ax, s)
        scatter_family(ax, g, s, col, rng)
        ax.text(0.5, -0.02, f"{s} day{'s' if s > 1 else ''} apart", fontsize=FS_RING,
                color="black", ha="center", va="top", transform=ax.transAxes)

    sel = {v: g["lean"] == v for v in levels}
    mn = {(s, v): g["r"][(g["gap"] == s) & sel[v]].mean() for s in (1, 2, 3) for v in levels}
    sd = {(s, v): g["r"][(g["gap"] == s) & sel[v]].std(ddof=1) for s in (1, 2, 3) for v in levels}
    lo = min(mn[k] - sd[k] for k in mn)
    hi = max(mn[k] + sd[k] for k in mn)
    pad = 0.05 * (hi - lo)
    curves = []
    for i, s in enumerate((1, 2, 3)):
        ax = fig.add_axes([fx(X0 + i * (PW + GUT) + (PW - BW) / 2), fy(0.95), fx(BW), fy(BH)])
        ys = [mn[(s, v)] for v in levels]
        ax.plot(levels, ys, "-", color=G_RULE, lw=1.6, zorder=3)
        for v, y in zip(levels, ys):
            ax.errorbar(v, y, yerr=sd[(s, v)], fmt="none", ecolor=col[v], elinewidth=1.2,
                        capsize=2.5, capthick=1.2, zorder=2)
            ax.scatter([v], [y], s=42, color=[col[v]], edgecolor="black", lw=.3, zorder=4)
        ax.tick_params(labelsize=FS_TICK)
        ax.set_xlim(-1.15, 1.15)
        ax.set_xticks(levels)
        ax.set_xticklabels([f"{v:g}".replace("-", "−") for v in levels])
        ax.set_ylim(lo - pad, hi + 4.5 * pad)
        ax.set_xlabel("confident ←  uncertain  → confident", fontsize=FS_LABEL,
                      labelpad=17)
        for xx, lab, ha in ((0, "Day B", "left"), (1, "Day A", "right")):
            ax.annotate(lab, xy=(xx, 0), xycoords="axes fraction", xytext=(0, -19),
                        textcoords="offset points", fontsize=FS_TICK, color="black",
                        ha=ha, va="top")
        if i == 0:
            ax.set_ylabel("Radius ($r$)\nmean ± 1 SD", fontsize=FS_LABEL,
                          linespacing=1.15)
        else:
            ax.set_yticklabels([])
        ax.text(0.03, 0.97, f"{s} day{'s' if s > 1 else ''} apart", fontsize=FS_SUB,
                color="black", transform=ax.transAxes, va="top")
        curves.append(ax)

    fig.canvas.draw()
    inv = fig.transFigure.inverted()
    x = min(a.get_tightbbox(fig.canvas.get_renderer()).transformed(inv).x0
            for a in (kx, curves[0]))
    fig.text(x, fy(6.37), "(A)  Uncertain weekday prompts are encoded along weekday spokes",
             fontsize=FS_PANEL, ha="left", va="baseline", color="black")
    fig.text(x, curves[0].get_position().y1 + fy(0.34),
             "(B)  Greater uncertainty moves activations radially inwards",
             fontsize=FS_PANEL, ha="left", va="baseline", color="black")
    return fig


if __name__ == "__main__":
    main()
