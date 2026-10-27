#!/usr/bin/env python
"""Figure 4: weekday representations in natural text and the effect of context on the ring.

Writes figures/fig4_occupancy.{pdf,png} and experiments/results/fig4_occupancy.json.

    PYTHONPATH=src:experiments/common:experiments/fig4_occupancy \\
        python experiments/fig4_occupancy/figure_occupancy.py   # CPU, ~2 min
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from arc_geometry import spline_op, foot_points, arc_frac
from paper_style import helvetica
from weekday_manifold.corpus.alignment import fit_plane, family_configs
from weekday_manifold.corpus.classes import CONTROL_CLASS, WEEKDAY_CLASSES, resolve_classes
from weekday_manifold.corpus.split import fit_mask, held_out_mask
from weekday_manifold.manifold.days import DAYS
from weekday_manifold.utils import data_dir
from weekday_manifold.utils import dump_json

INK, MID, PALE = "#16181d", "#5b6270", "#aeb4c0"
C_MEAN = "#c02a2a"
DAY_C = ["#CC79A7", "#D55E00", "#E69F00", "#F0E442", "#009E73", "#56B4E9", "#0072B2"]
CONTEXT_TYPES = [
    ("timestamp", r"\b(last updated|updated on|posted on|last edited|published|posted|in sale since)\b"),
    ("opening hours", r"\b(open|opening|hours|closed)\b"),
    ("news attribution", r"\b(said|says|told|announced|reported|testified|stated)\b"),
    ("plain date", r"\bon$"),
]
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def tidy(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_axisbelow(True)


def square(ax, pts, pad_frac=0.04, grow=1.04):
    """Equal-aspect limits that enclose pts with a margin."""
    lo, hi = pts.min(0), pts.max(0)
    pad = pad_frac * (hi - lo).max()
    span = grow * ((hi - lo).max() + 2 * pad)
    mid = (lo + hi) / 2
    ax.set_xlim(mid[0] - span / 2, mid[0] + span / 2)
    ax.set_ylim(mid[1] - span / 2, mid[1] + span / 2)


def panel_titles(fig, head, fs=9.5, dy=0.030, gap=0.012):
    """Place '(A)  Title' above each panel, shrinking the font so no title overlaps the next panel."""
    fig.canvas.draw()
    r, inv = fig.canvas.get_renderer(), fig.transFigure.inverted()
    Wpx, Hpx = fig.get_figwidth() * fig.dpi, fig.get_figheight() * fig.dpi
    from matplotlib.transforms import Bbox
    boxes = [ax.get_tightbbox(r).transformed(inv) for _, _, ax in head]
    off = min(b.x0 - ax.get_position().x0 for b, (_, _, ax) in zip(boxes, head))
    boxes = [Bbox.from_extents(ax.get_position().x0 + off, b.y0, b.x1, b.y1)
             for b, (_, _, ax) in zip(boxes, head)]
    y = max(b.y1 for b in boxes) + dy
    room = [boxes[i + 1].x0 for i in range(len(head) - 1)] + [0.999]
    need = []
    for (letter, title, _), bb, x1 in zip(head, boxes, room):
        t = fig.text(0, 0, f"({letter})  {max(title.split(chr(10)), key=len)}", fontsize=fs)
        need.append((x1 - gap - bb.x0) / (t.get_window_extent(r).width / Wpx))
        t.remove()
    fs = fs * min(1.0, min(need))
    placed = []
    for (letter, title, _), bb in zip(head, boxes):
        lt = fig.text(bb.x0, y, f"({letter})", ha="left", va="bottom", fontsize=fs,
                      color=INK, fontweight="bold")
        lw = lt.get_window_extent(r).width / Wpx
        tt = fig.text(bb.x0 + lw + 0.55 * fs / 72 / fig.get_figwidth(), y, title,
                      ha="left", va="bottom", fontsize=fs, color=INK, linespacing=1.18)
        placed.append((lt, tt))
    top = max(tt.get_window_extent(r).y1 for _, tt in placed) / Hpx
    for lt, tt in placed:
        for t in (lt, tt):
            t.set_verticalalignment("top"); t.set_y(top)


def example_contexts(S, sel, shift, list_path, model):
    """Per context type, the held-out context with the largest shift; [] without a tokeniser."""
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model)
    except Exception as e:                                   # noqa: BLE001
        print(f"[examples] skipped ({type(e).__name__}: {e})")
        return [], None
    IDS = np.load(list_path, allow_pickle=True)["ids"]
    rows = S["fam_row"][sel]; fday = np.asarray(S["fam_day"])[sel].astype(str)
    n = int(S["n"])
    ctx = [tok.decode(IDS[r][-n:-1]).lower() for r in rows]
    picks = []
    for _, pat in CONTEXT_TYPES:
        hit = np.array([bool(re.search(pat, c)) for c in ctx])
        if hit.any():
            cand = np.where(hit)[0]
            picks.append(int(cand[np.argmax(shift[cand])]))

    def text(i):
        r = rows[i]; d = fday[i]
        return tok.decode(IDS[r][-n:-1]) + tok.decode([IDS[r][-1]]).replace(d, "{" + d + "}")
    return picks, text


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", default=data_dir("fig4_occupancy"))
    ap.add_argument("--layer", type=int, default=28)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--classes", default="clean,temporal",
                    help="weekday classes analysed (weekday_manifold.corpus.classes)")
    ap.add_argument("--max-points", type=int, default=4000,
                    help="panel A draws this many held-out weekday windows and this many "
                         "controls, sampled, so neither cloud buries the other")
    ap.add_argument("--max-contexts", type=int, default=2200, help="contexts drawn in panel C")
    ap.add_argument("--grid", type=int, default=7000, help="samples along the spline")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(REPO, "figures", "fig4_occupancy.pdf"))
    ap.add_argument("--results", default=os.path.join(REPO, "experiments", "results",
                                                      "fig4_occupancy.json"))
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    classes = resolve_classes(args.classes)

    R = np.load(os.path.join(args.inp, f"raw_L{args.layer}_n{args.n}.npz"), allow_pickle=True)
    cls = np.asarray(R["cls"]).astype(str); day = np.asarray(R["day"]).astype(str)
    doc = np.asarray(R["doc"]).astype(str)
    fitm = fit_mask(cls, doc, classes, seed=args.seed)
    held = held_out_mask(cls, doc, classes, seed=args.seed)
    G = fit_plane(R, fitm)
    ctrl = cls == CONTROL_CLASS
    print(f"[fit] {int(fitm.sum())} fit / {int(held.sum())} held-out weekday windows "
          f"({', '.join(classes)}); {int(ctrl.sum())} controls")

    A = R["acts"]
    N = len(A)
    xy = np.empty((N, 2)); share = np.empty(N)
    for s in range(0, N, 20000):
        C = A[s:s + 20000].astype(np.float64)
        xy[s:s + 20000] = (C - G["mean_c"]) @ G["plane"].T
        d = C - G["mu"]
        share[s:s + 20000] = ((d @ G["basis6"].T) ** 2).sum(1) / np.maximum((d ** 2).sum(1), 1e-30)

    K6 = (G["cents"] - G["mean_c"]) @ G["hull6"].T
    curve_full = spline_op(args.grid) @ K6
    s_full = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(curve_full, axis=0), axis=1))]
    L = float(s_full[-1]); curve, s_grid = curve_full[:-1], s_full[:-1]
    s_knot = s_full[np.arange(7) * (args.grid // 7)]
    z_held = (A[held].astype(np.float64) - G["mean_c"]) @ G["hull6"].T
    s_pos, _, _ = foot_points(z_held, curve, s_grid, L, sep=L / 14.0)
    dk = np.abs(s_pos[:, None] - s_knot[None, :]); dk = np.minimum(dk, L - dk)
    pday = np.array([DAYS.index(d) for d in day[held]])
    own_centroid = float((dk.argmin(1) == pday).mean())
    p_arc, _ = arc_frac(s_pos, s_knot, L)
    arc_interior = float(((p_arc > 1 / 3) & (p_arc < 2 / 3)).mean())

    S = np.load(os.path.join(args.inp, f"swap_L{args.layer}_n{args.n}.npz"), allow_pickle=True)
    Z = family_configs(S, G["plane"], G["mean_c"])            # [F, 7, 2]
    fcls = np.asarray(S["fam_cls"]).astype(str); fdoc = np.asarray(S["fam_doc"]).astype(str)
    fheld = held_out_mask(fcls, fdoc, classes, seed=args.seed)
    Zh = Z[fheld]
    offs = Zh.mean(1)
    Dm = Zh - offs[:, None, :]
    T = Dm.mean(0)                                             # the mean ring, as 7 knots
    shift = np.linalg.norm(offs, axis=1) / G["ring_R"]
    W = spline_op(600)
    ck, cxy = (G["cents"] - G["mean_c"]) @ G["plane"].T, None
    cxy = W @ ck
    tc = W @ T
    picks, text = example_contexts(S, fheld, shift, os.path.join(args.inp, "capture_list_n256.npz"),
                                   args.model)

    sn = rng.choice(np.where(ctrl)[0], min(args.max_points, int(ctrl.sum())), replace=False)
    sp = np.where(held)[0]
    if len(sp) > args.max_points:
        sp = np.sort(rng.choice(sp, args.max_points, replace=False))
    take = rng.choice(len(Dm), min(args.max_contexts, len(Dm)), replace=False)
    with helvetica.context():
        Rc = plt.rcParams
        fs_lab, fs_leg, c_lab = Rc["axes.labelsize"], Rc["legend.fontsize"], Rc["axes.labelcolor"]
        fs_day, K = fs_leg, 0.85
        # the plate is 7 in wide so the style's point sizes print at their set size
        fig = plt.figure(figsize=(7.0, 3.05))
        gs = fig.add_gridspec(1, 3, left=0.065, right=0.995, top=0.760, bottom=0.150, wspace=0.34)
        axA, axB, axC = (fig.add_subplot(gs[0, j]) for j in range(3))
        for ax in (axA, axB, axC):
            tidy(ax); ax.set_aspect("equal")
            ax.set_xlabel("PC1", fontsize=fs_lab, color=c_lab)
            ax.set_ylabel("PC2", fontsize=fs_lab, color=c_lab)

        axA.plot(xy[sn, 0], xy[sn, 1], ".", ms=0.9 * K, color=PALE, alpha=0.45, zorder=2)
        for k in range(7):
            kk = sp[day[sp] == DAYS[k]]
            axA.plot(xy[kk, 0], xy[kk, 1], "o", ms=1.3 * K, color=DAY_C[k], alpha=0.5,
                     markeredgecolor="none", zorder=3)
        axA.plot(cxy[:, 0], cxy[:, 1], "-", color="white", lw=2.6, zorder=4)
        axA.plot(cxy[:, 0], cxy[:, 1], "-", color=INK, lw=1.0, zorder=5)
        for k in range(7):
            axA.plot(ck[k, 0], ck[k, 1], "o", ms=5.5 * K, color=DAY_C[k], markeredgecolor=INK,
                     markeredgewidth=0.7, zorder=6)
            axA.annotate(DAYS[k][:3], (ck[k, 0] * 1.45, ck[k, 1] * 1.45), fontsize=fs_day,
                         color=INK, ha="center", va="center", zorder=7, fontweight="bold")
        axA.legend(handles=[Line2D([], [], marker="o", linestyle="none", markersize=5.0,
                                   color=PALE, label="Other capitalised words (control)")],
                   frameon=False, fontsize=fs_leg, loc="upper left", handletextpad=0.4,
                   borderaxespad=0.1)
        square(axA, np.vstack([xy[sp], ck * 1.6]))

        LS = [(0, (8, 2)), (0, (4, 2)), (0, (2, 1.4)), (0, (8, 2, 2, 2))]
        rings = []
        for j, i in enumerate(picks):
            Kp = Dm[i] + offs[i]; c = W @ Kp; rings.append(c)
            axB.plot(c[:, 0], c[:, 1], linestyle=LS[j], color=MID, lw=1.0, zorder=4,
                     label="Example contexts" if j == 0 else None)
            for k in range(7):
                axB.plot(Kp[k, 0], Kp[k, 1], "o", ms=4.4 * K, color=DAY_C[k],
                         markeredgecolor=MID, markeredgewidth=0.6, zorder=5)
        axB.plot(tc[:, 0], tc[:, 1], "-", color=C_MEAN, lw=1.9, alpha=0.9, zorder=3,
                 label="Mean over all contexts")
        axB.legend(frameon=False, fontsize=fs_leg, loc="upper left", handletextpad=0.5,
                   borderaxespad=0.1, labelspacing=0.28)

        P = Dm[take]
        for k in range(7):
            axC.plot(P[:, k, 0], P[:, k, 1], "o", ms=1.6 * K, color=DAY_C[k], alpha=0.5,
                     markeredgecolor="none", zorder=3)
        axC.plot(tc[:, 0], tc[:, 1], "-", color="white", lw=2.6, zorder=5)
        axC.plot(tc[:, 0], tc[:, 1], "-", color=C_MEAN, lw=1.6, zorder=6)
        for k in range(7):
            axC.plot(T[k, 0], T[k, 1], "o", ms=5.5 * K, color=DAY_C[k], markeredgecolor=INK,
                     markeredgewidth=0.7, zorder=7)
            axC.annotate(DAYS[k][:3], (T[k, 0] * 2.10, T[k, 1] * 2.10), fontsize=fs_day,
                         color=INK, ha="center", va="center", zorder=8, fontweight="bold")
        # B and C use the same frame and scale
        ALL = np.concatenate([np.r_[tc, T * 1.5], P.reshape(-1, 2)] + rings)
        for ax in (axB, axC):
            square(ax, ALL, grow=1.46)

        panel_titles(fig, [("A", "Weekday representations\nin natural text", axA),
                           ("B", "Across contexts, the ring is scaled\nand translated but not rotated", axB),
                           ("C", "Translation-corrected points\nshow little remaining scatter", axC)])
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        fig.savefig(args.out, bbox_inches="tight", pad_inches=0.015)
        fig.savefig(os.path.splitext(args.out)[0] + ".png", dpi=300, bbox_inches="tight",
                    pad_inches=0.015)
        plt.close(fig)
        print(f"[saved] {args.out} (+ .png)")

    shares = {"weekday": float(np.median(share[held])),
              "near_miss": float(np.median(share[cls == "near_miss"])),
              "capitalised_word": float(np.median(share[ctrl])),
              "any_token": float(np.median(share[cls == "floor"]))}
    print(f"plane: PC1 = {100 * G['pc_var'][0]:.1f}% of centroid variance, PC2 = {100 * G['pc_var'][1]:.1f}%")
    print(f"held-out weekday windows nearest their own day along the ring: {100 * own_centroid:.1f}%; "
          f"in the middle third of an arc: {100 * arc_interior:.1f}%")
    print("median energy share in the weekday subspace: " +
          ", ".join(f"{k} {100 * v:.2f}%" for k, v in shares.items()))
    print("example contexts (panel B), weekday the text used in braces:")
    for j, i in enumerate(picks):
        print(f"  {j + 1}. [{CONTEXT_TYPES[j][0]}] shift {shift[i]:.2f} R  {text(i)!r}")
    out = dict(classes=classes, n_fit=int(fitm.sum()), n_held=int(held.sum()),
               n_controls=int(ctrl.sum()), pc_var=[float(v) for v in G["pc_var"]],
               own_centroid_rate=own_centroid, arc_interior_rate=arc_interior,
               median_energy_share=shares,
               example_contexts=[dict(type=CONTEXT_TYPES[j][0], shift_ring_radii=float(shift[i]),
                                      text=text(i)) for j, i in enumerate(picks)])
    os.makedirs(os.path.dirname(args.results), exist_ok=True)
    dump_json(out, args.results, indent=1)
    print(f"[saved] {args.results}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
