"""Draw figure 1, perturbation sensitivity around the weekday ring.

    python experiments/fig1_sensitivity/figure_sensitivity.py \\
        --cascade <data>/ring_walk.npz --field <data>/jacobian_field.npz
writes figures/fig1_sensitivity.png (and .pdf).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
from cascade_row import LETTER_X, TITLE_Y, draw_row  # noqa: E402
from paper_style import helvetica  # noqa: E402
from polar_disc import MEASURES, disc, edges  # noqa: E402
from weekday_manifold.utils import data_dir  # noqa: E402

REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
INK = "#16181d"
STEER_LAYER, COMPARE_LAYER = 2, 28       # steer layers compared in panel A
SCALE_BAND = (0.75, 1.3)                 # radius range used for the disc's colour limits
DISC_FONTSCALE = 1.30


def build(cascade, field):
    zc = np.load(cascade, allow_pickle=True)
    zf = np.load(field, allow_pickle=True)
    meta = json.loads(str(zf["meta"]))
    assert str(zc["model"]) == meta["model"], (
        f"different models: cascade {zc['model']}, field {meta['model']}")
    assert str(zc["formulation"]) == meta["formulation"], (
        f"different prompt families: {zc['formulation']} vs {meta['formulation']}")
    assert int(zc["ro_pos"]) == meta["ro_pos"], "different readout positions"
    assert meta["patch_layer"] == STEER_LAYER, (
        f"the field steers at L{meta['patch_layer']}, the cascade row at L{STEER_LAYER}")

    fig = plt.figure(figsize=(9.8, 3.1))
    gs = GridSpec(1, 3, figure=fig, left=0.098, right=0.940, top=0.820, bottom=0.215,
                  wspace=0.40, width_ratios=[1.0, 1.0, 0.74])

    # ---- A, B: the ring walk --------------------------------------------------------------
    aA = fig.add_subplot(gs[0, 0])
    aB = fig.add_subplot(gs[0, 1])
    draw_row(fig, aA, aB, cascade, STEER_LAYER, COMPARE_LAYER)
    for ax in (aA, aB):
        ax.tick_params(labelsize=10.0)
        for t in ax.texts:
            fsz = t.get_fontsize()
            if abs(fsz - 9.5) < 0.01:
                t.set_fontsize(10.0)
            elif fsz < 9.0:
                t.set_fontsize(fsz * 1.30)
        lg = ax.get_legend()
        if lg is not None:
            for t in lg.get_texts():
                t.set_fontsize(9.2)
        ax.set_xlabel("Steered position on the ring\n(arc length, fraction of loop)",
                      fontsize=10.0, color=INK, linespacing=1.4)
    aA.set_ylabel("Behaviour distance\n(fraction)", fontsize=10.0, color=INK, linespacing=1.4)
    aB.set_ylabel("Readout speed\n$\\|\\Delta$resid$\\|$ / $\\|\\Delta$steer$\\|$",
                  fontsize=10.0, color=INK, linespacing=1.4)
    for j, ax in enumerate((aA, aB)):
        ax.text(LETTER_X, TITLE_Y, "AB"[j], transform=ax.transAxes, fontsize=12.0,
                fontweight="bold", color=INK, ha="left", va="baseline")
    ap_ = aA.get_position()
    title_fig_y = ap_.y0 + TITLE_Y * ap_.height

    # ---- C: the Jacobian norm over the disc ----------------------------------------------
    us, rs = np.asarray(zf["us"]), np.asarray(zf["rs"])
    th_e, r_e = edges(us * 2 * np.pi), edges(rs, lo=0.0)
    keep = (rs >= SCALE_BAND[0]) & (rs <= SCALE_BAND[1])
    key, title, sym, dirn, lab = MEASURES[0]
    a = zf[key].mean(0)
    bb = gs[0, 2].get_position(fig)
    k = 0.98
    w, h = bb.width * k, bb.height * 0.84 * k
    ax = fig.add_axes([bb.x0 + (bb.width - w) / 2, bb.y0 + (bb.height * 0.84 - h), w, h],
                      projection="polar")
    v = a.T[keep]
    disc(ax, th_e, r_e, a.T, "viridis", float(np.percentile(v, 1.0)),
         float(np.percentile(v, 99.0)), lab, dirn=dirn, fs=DISC_FONTSCALE, ring_label=True)
    fig.text(bb.x0 + 0.002, title_fig_y, "C", fontsize=12.0, fontweight="bold", color=INK,
             ha="left", va="baseline")
    t = fig.text(bb.x0 + 0.030, title_fig_y, title, fontsize=10.4, color=INK, ha="left",
                 va="baseline")
    fig.canvas.draw()
    x1 = t.get_window_extent(fig.canvas.get_renderer()).transformed(
        fig.transFigure.inverted()).x1
    fig.text(x1 + 0.010, title_fig_y, sym, fontsize=11.5, color=INK, ha="left",
             va="baseline")
    return fig


def main():
    data = data_dir("fig1_sensitivity", create=False)
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cascade", default=os.path.join(data, "ring_walk.npz"))
    ap.add_argument("--field", default=os.path.join(data, "jacobian_field.npz"))
    ap.add_argument("--out", default=os.path.join(REPO, "figures", "fig1_sensitivity.png"))
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with helvetica.context(), plt.rc_context({"savefig.bbox": None}):
        fig = build(args.cascade, args.field)
        for path in (args.out, os.path.splitext(args.out)[0] + ".pdf"):
            fig.savefig(path, dpi=args.dpi)
            print("[figure] wrote", path)


if __name__ == "__main__":
    main()
