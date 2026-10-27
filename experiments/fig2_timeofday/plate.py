#!/usr/bin/env python
"""Draw figure 2 to figures/fig2_time_of_day.pdf (and .png with --also-png).

    python experiments/fig2_timeofday/plate.py --also-png
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
from scipy.interpolate import CubicSpline

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "src")))
from weekday_manifold.manifold.days import DAYS  # noqa: E402
from weekday_manifold.utils import data_dir  # noqa: E402
from paper_style import helvetica  # noqa: E402
from analysis import DAY_C, load_capture, plate_geometry  # noqa: E402

INK, MID, PALE = "#16181d", "#5b6270", "#aeb4c0"
FIG_OUT = os.path.abspath(os.path.join(_HERE, "..", "..", "figures", "fig2_time_of_day.pdf"))


def titles(fig, head, fs, dy=0.030, gap=0.012, right=0.999):
    """Place '(A)  Title' headers at each panel's left edge, at the largest common font size <= fs that fits."""
    fig.canvas.draw()
    r, inv = fig.canvas.get_renderer(), fig.transFigure.inverted()
    Wpx, Hpx = fig.get_figwidth() * fig.dpi, fig.get_figheight() * fig.dpi
    tb = [ax.get_tightbbox(r).transformed(inv) for _, _, ax in head]
    x0 = []
    for b, (_, _, ax) in zip(tb, head):
        if ax.name != "3d":
            x0.append(max(0.002, b.x0))
        else:
            # for 3-D axes use the axis lines, since the bounding box is larger than the drawing
            xs = []
            for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                bb = axis.line.get_window_extent(r)
                if np.isfinite(bb.x0):
                    xs.append(inv.transform((bb.x0, bb.y0))[0])
            x0.append(min(xs) if xs else 0.002)
    y = max(b.y1 for b in tb) + dy
    room = x0[1:] + [right]
    need = []
    for (letter, title, _), a, b in zip(head, x0, room):
        wide = max(title.split("\n"), key=len)
        t = fig.text(0, 0, f"({letter})  {wide}", fontsize=fs)
        need.append((b - gap - a) / (t.get_window_extent(r).width / Wpx)); t.remove()
    fs = fs * min(1.0, min(need))
    placed = []
    for (letter, title, _), a in zip(head, x0):
        lt = fig.text(a, y, f"({letter})", ha="left", va="bottom", fontsize=fs, color=INK,
                      fontweight="bold")
        lw = lt.get_window_extent(r).width / Wpx
        tt = fig.text(a + lw + 0.55 * fs / 72 / fig.get_figwidth(), y, title, ha="left",
                      va="bottom", fontsize=fs, color=INK, linespacing=1.18)
        placed.append((lt, tt))
    top = max(tt.get_window_extent(r).y1 for _, tt in placed) / Hpx
    for lt, tt in placed:
        for t in (lt, tt):
            t.set_verticalalignment("top"); t.set_y(top)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=data_dir("fig2_timeofday", create=False))
    ap.add_argument("--layer", type=int, default=28)
    ap.add_argument("--width", type=float, default=7.0, help="plate width, inches")
    ap.add_argument("--title-size", type=float, default=9.5)
    ap.add_argument("--out", default=FIG_OUT)
    ap.add_argument("--also-png", action="store_true")
    args = ap.parse_args()

    import pandas as pd
    cap = load_capture(os.path.join(args.data_dir, f"timeofday_L{args.layer}.npz"))
    A, day, hour = cap["A"], cap["day"], cap["hour"]
    G = plate_geometry(cap)
    plane, e3, pcs, mu, cl = G["plane"], G["e3"], G["pcs"], G["mu"], G["cl"]
    rows, stat = G["rows"], G["stat"]
    STEP_MIN, STEP_MAX = G["step_min"], G["step_max"]
    LS = pd.read_csv(os.path.join(args.data_dir, "ladder", "summary.csv"))

    with helvetica.context():
        R = plt.rcParams
        fs_lab, fs_leg = R["axes.labelsize"], R["legend.fontsize"]
        c_lab = R["axes.labelcolor"]
        cyc, nrm = plt.get_cmap("twilight"), Normalize(vmin=0, vmax=24)

        fig = plt.figure(figsize=(args.width, 3.60))
        axA = fig.add_axes([-0.035, 0.175, 0.470, 0.630], projection="3d")
        axB = fig.add_axes([0.505, 0.175, 0.190, 0.570])
        axC = fig.add_axes([0.805, 0.175, 0.190, 0.470])

        # ===== A: the weekday ring and the hour axis ======================================
        hrs = np.array(sorted(set(hour[cl].tolist())))
        Chour = np.stack([A[cl & (hour == h)].mean(0) for h in hrs])
        if np.corrcoef(hrs, (Chour - Chour.mean(0)) @ e3)[0, 1] > 0:
            e3 = -e3
        B3 = np.column_stack([plane.T[:, 0], plane.T[:, 1], e3])
        X = (A - mu) @ B3
        KD = (np.stack([A[cl & (day == d)].mean(0) for d in range(7)]) - mu) @ B3
        ring = CubicSpline(np.arange(8.0), np.vstack([KD, KD[:1]]), bc_type="periodic",
                           axis=0)(np.linspace(0, 7, 400))
        axA.plot(ring[:, 0], ring[:, 1], ring[:, 2], color=INK, lw=0.6, zorder=2)
        RING_C = KD[:, :2].mean(0)
        # per-day label nudges in data units (plane x, plane y, hour axis)
        NUDGE = {0: (0.0, -1.2, -1.6), 1: (0.0, 0.0, -0.6), 2: (0.6, 0.0, 0.0),
                 3: (1.8, 0.0, 1.9), 4: (0.0, 0.0, 1.6), 5: (-0.4, 0.0, 1.4),
                 6: (-0.6, 0.0, -0.3)}
        for d in range(7):
            rd = np.array([np.where(cl & (day == d) & (hour == h))[0][0] for h in hrs])
            Pd = X[rd]
            S = CubicSpline(hrs, Pd, bc_type="natural", axis=0)(np.linspace(hrs[0], hrs[-1], 240))
            axA.plot(S[:, 0], S[:, 1], S[:, 2], color=MID, lw=0.7, alpha=0.75, zorder=3)
            axA.scatter(Pd[:, 0], Pd[:, 1], Pd[:, 2], s=7.0, c=[cyc(nrm(h)) for h in hrs],
                        depthshade=False, edgecolors="none", zorder=4)
            axA.scatter([KD[d, 0]], [KD[d, 1]], [KD[d, 2]], s=17, color=DAY_C[d],
                        depthshade=False, edgecolors=INK, linewidths=0.6, zorder=6)
            v = KD[d, :2] - RING_C
            lx, ly = RING_C + v * (1.0 + 3.0 / max(np.hypot(*v), 1e-9))
            nx, ny, nz = NUDGE[d]
            axA.text(lx + nx, ly + ny, KD[d, 2] + nz, DAYS[d][:3], fontsize=fs_leg,
                     color=INK, ha="center", va="center", zorder=7)
        drawn = np.vstack([X[cl], KD])
        ctr = (drawn.max(0) + drawn.min(0)) / 2
        half = (drawn.max(0) - drawn.min(0)).max() / 2 * 1.06
        for setter, c in ((axA.set_xlim, ctr[0]), (axA.set_ylim, ctr[1]), (axA.set_zlim, ctr[2])):
            setter(c - half, c + half)
        axA.set_box_aspect((1, 1, 1), zoom=1.15)
        axA.view_init(elev=24.0, azim=-64.0)
        axA.set_xticks([]); axA.set_yticks([]); axA.set_zticks([])
        for axis in (axA.xaxis, axA.yaxis, axA.zaxis):
            axis.pane.set_visible(False); axis.line.set_color(PALE)
        i, j = pcs["plane_idx"]
        axA.set_xlabel(f"PC{i + 1}", fontsize=fs_leg, color=c_lab, labelpad=-15)
        axA.set_ylabel(f"PC{j + 1}", fontsize=fs_leg, color=c_lab, labelpad=-15)
        axA.set_zlabel(f"PC{pcs['e3_idx'] + 1}", fontsize=fs_leg, color=c_lab, labelpad=-13)
        cax = fig.add_axes([0.085, 0.060, 0.245, 0.017])
        sm = plt.cm.ScalarMappable(norm=Normalize(vmin=1, vmax=23), cmap=cyc)
        cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
        cb.set_ticks([1, 6, 12, 18, 23]); cb.set_ticklabels(["1am", "6am", "12pm", "6pm", "11pm"])
        cb.ax.tick_params(labelsize=fs_leg - 1.0, length=2, pad=1.5); cb.outline.set_visible(False)

        # ===== B: in-plane rotation of the weekday by each modifier =======================
        for sp in ("top", "right"):
            axB.spines[sp].set_visible(False)
        pos_y = np.arange(len(rows))[::-1]
        for sgn, side in ((-1, "Previous day"), (+1, "Next day")):
            lo, hi = sorted((sgn * STEP_MIN, sgn * STEP_MAX))
            axB.axvspan(lo, hi, color=INK, alpha=0.10, lw=0, zorder=1)
            axB.text(sgn * (STEP_MIN + STEP_MAX) / 2, (len(rows) - 1) / 2, side, rotation=90,
                     ha="center", va="center", fontsize=fs_leg - 1.0, color=INK, style="italic",
                     zorder=5)
        axB.axvline(0, color=PALE, lw=0.8, zorder=2)
        for p, (key, _, f_, hr_) in zip(pos_y, rows):
            m_, se_ = stat[key]
            c = cyc(nrm(hr_)) if f_ in ("clock", "word") else MID
            axB.errorbar(m_, p, xerr=se_, fmt="o", ms=3.4, color=c, ecolor=c, elinewidth=0.9,
                         capsize=1.6, markeredgewidth=0.4,
                         markeredgecolor=INK if f_ in ("clock", "word") else "none", zorder=4)
        axB.set_yticks(pos_y)
        axB.set_yticklabels([d for _, d, _, _ in rows], fontsize=fs_leg - 1.0, color=INK)
        axB.tick_params(axis="y", length=0, pad=2)
        seen = None
        for k, (_, _, f_, _) in enumerate(rows):
            if seen is not None and f_ != seen:
                axB.axhline(len(rows) - k - 0.5, color=PALE, lw=0.6, zorder=2)
            seen = f_
        axB.set_ylim(-0.6, len(rows) - 0.4)
        axB.set_xlim(-STEP_MAX * 1.12, STEP_MAX * 1.12)
        axB.set_xticks([-50, 0, 50])
        axB.set_xlabel("Rotation within the ring plane\nfrom the day's centroid (degrees)",
                       fontsize=fs_lab, color=c_lab)

        # ===== C: dose-response for the on-ring and off-ring arms ========================
        for sp in ("top", "right"):
            axC.spines[sp].set_visible(False)
        for arm, lab, mk in (("rand_perp", "Perpendicular, random direction (control)", "^"),
                             ("manifold", "Along the ring", "o"),
                             ("late_perp", "Perpendicular, late−early direction", "s")):
            g = LS[LS.arm == arm].sort_values("alpha")
            if arm == "late_perp":
                C_LINE, C_MARK = cyc(nrm(19.5)), cyc(nrm(12.0))
                axC.fill_between(g.alpha, g.lo, g.hi, color=C_LINE, alpha=0.20, lw=0)
                axC.plot(g.alpha, g["mean"], "-", color=C_LINE, lw=1.6, zorder=4)
                axC.plot(g.alpha, g["mean"], linestyle="none", marker=mk, ms=3.2 * 0.88,
                         mfc=C_MARK, mec="white", mew=0.5, zorder=5)
                axC.plot([], [], "-", color=C_LINE, marker=mk, ms=3.2 * 0.88, mfc=C_MARK,
                         mec="white", mew=0.5, lw=1.6, label=lab)
            else:
                col, z = (INK, 6) if arm == "manifold" else ("#009988", 2)
                axC.fill_between(g.alpha, g.lo, g.hi, color=col, alpha=0.15, lw=0, zorder=z - 1)
                if arm == "manifold":
                    axC.plot(g.alpha, g["mean"], linestyle=(0, (2.2, 1.6)), color=col, lw=1.2,
                             marker=mk, ms=3.4, mfc="white", mec=col, mew=0.8, label=lab, zorder=z)
                else:
                    axC.plot(g.alpha, g["mean"], "-", color=col, lw=1.9, marker=mk, ms=4.4,
                             mec="white", mew=0.5, label=lab, zorder=z)
        axC.axhline(0, color=PALE, lw=0.7, zorder=0)
        axC.axvline(0, color=PALE, lw=0.7, zorder=0)
        axC.set_xlim(-0.55, 0.55)
        ymax = float(max(LS.hi.abs().max(), LS.lo.abs().max())) * 1.08
        axC.set_ylim(-ymax * 1.22, ymax)
        axC.text(-0.53, -ymax * 1.20, "← toward\nprevious day", fontsize=fs_leg - 1.5,
                 color=MID, ha="left", va="bottom", linespacing=1.1)
        axC.text(0.53, -ymax * 1.20, "toward →\nnext day", fontsize=fs_leg - 1.5,
                 color=MID, ha="right", va="bottom", linespacing=1.1)
        axC.set_xticks([-0.5, 0, 0.5])
        axC.set_xticklabels(["−0.5", "0", "0.5"])
        axC.set_xlabel("Steering amount\n0.5 units = halfway to\nadjacent day",
                       fontsize=fs_lab, color=c_lab, linespacing=1.1)
        axC.set_ylabel("Shift in predicted hour (h)", fontsize=fs_lab, color=c_lab)
        hs, ls = axC.get_legend_handles_labels()
        order = [ls.index(l) for l in ("Along the ring", "Perpendicular, late−early direction",
                                       "Perpendicular, random direction (control)")]
        axC.legend([hs[i] for i in order], [ls[i] for i in order], loc="lower right",
                   bbox_to_anchor=(1.0, 1.02), ncol=1, fontsize=fs_leg - 1.0, frameon=False,
                   handlelength=1.3, handletextpad=0.4, labelspacing=0.25, borderaxespad=0.0)

        titles(fig, [("A", "Clock time is represented\nlargely orthogonal to\nthe weekday ring", axA),
                     ("B", "In-plane temporal shifts\nare directionally\nconsistent but small", axB),
                     ("C", "Steering along the\nring barely changes\nthe predicted time", axC)],
               fs=args.title_size)

        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        fig.savefig(args.out, bbox_inches="tight", pad_inches=0.015)
        print(f"[saved] {args.out}")
        if args.also_png:
            png = os.path.splitext(args.out)[0] + ".png"
            fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.015)
            print(f"[saved] {png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
