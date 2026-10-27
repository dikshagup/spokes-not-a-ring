#!/usr/bin/env python
"""Appendix families plate: the Figure 3A panels for the five other hedge families.

    PYTHONPATH=src:experiments/common:experiments/fig3_uncertainty \\
        python experiments/fig3_uncertainty/plot_families.py        # CPU, seconds
"""
from __future__ import annotations

import argparse
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                # noqa: E402
import numpy as np                                             # noqa: E402

from paper_style import helvetica                              # noqa: E402
from weekday_manifold.manifold.days import DAYS                # noqa: E402
from ring_coords import (LAYER, capture_path, family_geometry, load_capture,  # noqa: E402
                         load_ruler)
from ring_plates import (BLACK_AXES, CMAP, FS_PANEL, FS_RING, G_TEXT,  # noqa: E402
                         level_colours, ring, scatter_family)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "figures", "fig_uncertainty_families")
FS_KEY = 8.5
ROWS = [("either_or", "Either/or wording"), ("certainty", "Certainty words"),
        ("frequency", "Frequency words"), ("odds", "Stated odds"),
        ("percent", "Stated percentages")]
_DAY = re.compile("|".join(DAYS))


def wording(text: str) -> str:
    """The level wording of a prompt, with the days replaced by X and Y."""
    head = text.rstrip(".").rsplit(", ", 1)[0]
    names = iter("XY")
    return _DAY.sub(lambda m: next(names), head).replace(", ", ",\n", 1)


def summarise(fam, g):
    d = np.minimum(np.abs(g["w"]), np.abs(g["w"] - 1))
    top = np.abs(g["lean"]).max()
    out = []
    for s in (2, 3):
        m, e = g["gap"] == s, (g["gap"] == s) & (g["lean"] == 0)
        c = (g["gap"] == s) & (np.abs(g["lean"]) == top)
        fa = (g["gap"] == s) & (g["lean"] == top)
        fb = (g["gap"] == s) & (g["lean"] == -top)
        out.append(f"{s} apart: inward {100 * (1 - g['r'][e].mean() / g['r'][c].mean()):.0f}%, "
                   f"on a spoke {100 * (d[m] < .25).mean():.0f}%, favoured side "
                   f"A {100 * (g['w'][fa] < .5).mean():.0f}% B {100 * (g['w'][fb] >= .5).mean():.0f}%")
    print(f"  {fam:10s} n={len(g['w'])}  " + " | ".join(out))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layer", type=int, default=LAYER)
    ap.add_argument("--capture", default=None)
    ap.add_argument("--ruler", default=None)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    mu, W, M, prov = load_ruler(args.layer, args.ruler)
    cap = load_capture(args.capture or capture_path("hedges", args.layer))
    fams = {}
    for fam, _ in ROWS:
        k = cap["family_name"] == fam
        if not k.any():
            continue
        sub = {key: cap[key][k] for key in cap if key not in ("layer",)}
        fams[fam] = (family_geometry(sub, mu, W, M), sub["text"])
    print("per family:")
    for fam, (g, _) in fams.items():
        summarise(fam, g)

    with helvetica.context(), plt.rc_context(BLACK_AXES):
        fig = draw(fams)
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        for ext in ("png", "pdf"):
            fig.savefig(f"{args.out}.{ext}", bbox_inches="tight")
        plt.close(fig)
    print(f"-> {args.out}.png")


def draw(fams):
    rows = [(fam, title) for fam, title in ROWS if fam in fams]
    PW, GUT, KEYW, X0 = 2.00, 0.32, 2.30, 0.15
    DH = PW * 2.75 / 2.90
    PITCH, TOP, BOT = DH + 0.30, 0.95, 0.10
    FW = X0 + KEYW + 3 * PW + 2 * GUT + 0.35
    FH = TOP + len(rows) * PITCH + BOT
    fig = plt.figure(figsize=(FW, FH))
    fx = lambda inch: inch / FW                # noqa: E731
    fy = lambda inch: inch / FH                # noqa: E731
    XD = X0 + KEYW

    cb = fig.add_axes([fx(X0 + 0.10), fy(FH - 0.50), fx(KEYW - 0.55), fy(0.11)])
    cb.imshow(np.linspace(1, 0, 256)[None, :], cmap=CMAP, aspect="auto", extent=(0, 1, 0, 1))
    cb.set_xticks([]); cb.set_yticks([])
    for sp in cb.spines.values():
        sp.set_visible(False)
    for xx, yy, lab, ha, va in ((0, -0.45, "favours Day A", "left", "top"),
                                (0.5, 1.35, "even", "center", "bottom"),
                                (1, -0.45, "favours Day B", "right", "top")):
        cb.text(xx, yy, lab, fontsize=FS_KEY, color="black", ha=ha, va=va,
                transform=cb.transAxes)
    for i, s in enumerate((1, 2, 3)):
        fig.text(fx(XD + i * (PW + GUT) + PW / 2), fy(FH - 0.62),
                 f"{s} day{'s' if s > 1 else ''} apart", fontsize=FS_RING, color="black",
                 ha="center", va="bottom")

    rng = np.random.default_rng(0)
    for r, (fam, title) in enumerate(rows):
        g, text = fams[fam]
        levels = np.array(sorted(set(g["lean"].tolist())))
        col = level_colours(levels)
        y_row = FH - TOP - (r + 1) * PITCH + 0.22

        fig.text(fx(X0), fy(y_row + DH - 0.02), f"({chr(65 + r)})  {title}",
                 fontsize=FS_PANEL, color="black", ha="left", va="top")
        kx = fig.add_axes([fx(X0), fy(y_row - 0.06), fx(KEYW - 0.10), fy(DH - 0.28)])
        kx.set_xlim(0, 1); kx.set_ylim(0, 1); kx.axis("off")
        mags = sorted({abs(v) for v in levels}, reverse=True)
        for j, a in enumerate(mags):
            yy = 1 - (j + 0.5) / len(mags)
            ex = np.where(g["lean"] == (a if a in col else -a))[0][0]
            for xx, v in ((0.035, a), (0.105, -a)) if a > 0 else ((0.07, 0.0),):
                if v in col:
                    kx.plot(xx, yy, "o", ms=6, mfc=col[v], mec="black", mew=.3, clip_on=False)
            kx.text(0.17, yy, wording(str(text[ex])), fontsize=FS_KEY, color=G_TEXT,
                    va="center", linespacing=1.0)

        for i, s in enumerate((1, 2, 3)):
            ax = fig.add_axes([fx(XD + i * (PW + GUT)), fy(y_row), fx(PW), fy(DH)])
            ring(ax, s, dot_ms=4.5)
            scatter_family(ax, g, s, col, rng, point_s=2.6, point_alpha=.28, mean_s=15,
                           join=False, min_share=0.15)
    return fig


if __name__ == "__main__":
    main()
