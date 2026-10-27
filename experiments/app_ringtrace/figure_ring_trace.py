#!/usr/bin/env python
"""Draw the ring-trace appendix figure and write experiments/results/fig_ring_trace_numbers.json.

  python experiments/app_ringtrace/figure_ring_trace.py --trace <corpus dir>/trace_n16.npz --also-png
writes figures/fig_ring_trace.pdf (and .png).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_COMMON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common")
sys.path.insert(0, _COMMON)
from arc_geometry import spline_op                                        # noqa: E402
from paper_style import helvetica                                         # noqa: E402
from weekday_manifold.utils import dump_json

INK, MID, PALE = "#16181d", "#5b6270", "#aeb4c0"
DAY_C = ["#CC79A7", "#D55E00", "#E69F00", "#F0E442", "#009E73", "#56B4E9", "#0072B2"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
CTRL = ("matched_negative", "matched_negative_far", "floor")
PERIOD_SLOT = 15
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def tidy(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_axisbelow(True)


def square(ax, pts, pad_frac=0.04, grow=1.04):
    """Set equal-aspect axis limits that contain every row of pts, with a margin."""
    lo, hi = pts.min(0), pts.max(0)
    pad = pad_frac * (hi - lo).max()
    span = grow * ((hi - lo).max() + 2 * pad)
    mid = (lo + hi) / 2
    ax.set_xlim(mid[0] - span / 2, mid[0] + span / 2)
    ax.set_ylim(mid[1] - span / 2, mid[1] + span / 2)


def titles(fig, head, fs=10.5, dy=0.030, gap=0.012, right=0.999):
    """Place '(A) Title' headers at each panel's left edge, shrinking all fonts if one would not fit."""
    fig.canvas.draw()
    r, inv = fig.canvas.get_renderer(), fig.transFigure.inverted()
    Wpx = fig.get_figwidth() * fig.dpi
    x0 = [ax.get_tightbbox(r).transformed(inv).x0 for _, _, ax in head]
    ys = [ax.get_tightbbox(r).transformed(inv).y1 for _, _, ax in head]
    room = x0[1:] + [right]
    need = []
    for (letter, title, _), a, b in zip(head, x0, room):
        wide = max(title.split("\n"), key=len)
        t = fig.text(0, 0, f"({letter})  {wide}", fontsize=fs)
        need.append((b - gap - a) / (t.get_window_extent(r).width / Wpx)); t.remove()
    fs = fs * min(1.0, min(need))
    y = max(ys) + dy
    for (letter, title, _), a in zip(head, x0):
        lt = fig.text(a, y, f"({letter})", ha="left", va="bottom", fontsize=fs, color=INK,
                      fontweight="bold")
        lw = lt.get_window_extent(r).width / Wpx
        fig.text(a + lw + 0.55 * fs / 72 / fig.get_figwidth(), y, title, ha="left",
                 va="bottom", fontsize=fs, color=INK, linespacing=1.18)
    return fs


def slot_label(s):
    return "day" if s == 0 else ("period" if s == PERIOD_SLOT else f"+{s}")


def cyclic(cents):
    """True if the seven centroids are in weekday order around their top-2 principal plane."""
    C = cents - cents.mean(0)
    P = np.linalg.svd(C, full_matrices=False)[2][:2]
    xy = C @ P.T
    order = np.argsort(np.arctan2(xy[:, 1], xy[:, 0]))
    seq = list(order)
    for rev in (seq, seq[::-1]):
        for r in range(7):
            if rev[r:] + rev[:r] == list(range(7)):
                return True
    return False


def plane(cents):
    mean_c = cents.mean(0)
    P = np.linalg.svd(cents - mean_c, full_matrices=False)[2][:2]
    return mean_c, P


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trace", required=True, help="trace_n16.npz from capture_trace.py")
    ap.add_argument("--out", default=os.path.join(REPO, "figures", "fig_ring_trace.pdf"))
    ap.add_argument("--also-png", action="store_true")
    ap.add_argument("--layer", type=int, default=28, help="layer of the ring strip (C)")
    ap.add_argument("--max-offset", type=int, default=8)
    ap.add_argument("--strip", default="0,1,2,3,4,6,15", help="slots shown in (C)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    T = np.load(args.trace, allow_pickle=True)
    z_own, z_tr = T["z_own"].astype(np.float32), T["z_tr"].astype(np.float32)
    n_own, n_tr = T["norm_own"].astype(np.float32), T["norm_tr"].astype(np.float32)
    valid, cell_ok = T["valid"].astype(bool), T["cell_ok"].astype(bool)
    c_own, c_tr = T["cents_own"], T["cents_tr"]
    layers = T["layers"].tolist()
    cls = T["cls"].astype(str); day = T["day"].astype(str)
    fit_doc = T["fit_doc"].astype(bool); is_fit = T["is_fit"].astype(bool)
    dix = np.array([DAYS.index(d) if d in DAYS else -1 for d in day])
    held = (cls == "positive_clean") & ~fit_doc
    ctrl = np.isin(cls, CTRL)
    L, S = len(layers), z_own.shape[2]
    cols = list(range(0, args.max_offset + 1)) + [PERIOD_SLOT]
    print(f"[trace] {len(cls)} rows, {L} layers, {S} slots; fit {int(is_fit.sum())}, "
          f"held-out {int(held.sum())}, controls {int(ctrl.sum())}")
    print(f"[trace] fit rows per slot {T['fit_counts'].sum(0).tolist()}")

    def decode(z, cents, m, li, si):
        zz = z[m, li, si]
        pred = np.argmin(((zz[:, None, :] - cents[li, si][None]) ** 2).sum(-1), 1)
        return float((pred == dix[m]).mean()) if m.any() else np.nan

    def share(z, n, m, li, si):
        e = (z[m, li, si] ** 2).sum(1)
        return float(np.median(e / np.maximum(n[m, li, si] ** 2, 1e-9))) if m.any() else np.nan

    dec = np.full((2, L, S), np.nan); shp = np.full((2, L, S), np.nan)
    shn = np.full((2, L, S), np.nan); cyc = np.zeros((2, L, S), bool)
    n_held = np.zeros(S, int); n_ctrl = np.zeros(S, int)
    for si in range(S):
        hv, cv = held & valid[:, si], ctrl & valid[:, si]
        n_held[si], n_ctrl[si] = int(hv.sum()), int(cv.sum())
        for li in range(L):
            for k, (z, n, c) in enumerate(((z_own, n_own, c_own), (z_tr, n_tr, c_tr))):
                dec[k, li, si] = decode(z, c, hv, li, si)
                shp[k, li, si] = share(z, n, hv, li, si)
                shn[k, li, si] = share(z, n, cv, li, si)
                cyc[k, li, si] = cyclic(c[li, si])
    print(f"[n] held-out per slot {n_held.tolist()}\n[n] controls per slot {n_ctrl.tolist()}")
    hdr = "L   | " + " ".join(f"{slot_label(s):>6}" for s in cols)
    for k, name in enumerate(("fresh fit", "transfer")):
        print(f"\n[decode, {name}]\n{hdr}")
        for li, l in enumerate(layers):
            if l % 4 == 0 or l == args.layer or l == layers[-1]:
                print(f"{l:3d} | " + " ".join(
                    f"{100 * dec[k, li, s]:5.0f}{'*' if cyc[k, li, s] else ' '}" for s in cols))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    strip = [int(s) for s in args.strip.split(",")]
    li0 = layers.index(args.layer)

    with helvetica.context():
        Rc = plt.rcParams
        fs = dict(lab=Rc["axes.labelsize"], leg=Rc["legend.fontsize"], tick=Rc["xtick.labelsize"],
                  c_lab=Rc["axes.labelcolor"])
        fig = plt.figure(figsize=(7.0, 6.6))
        gs_top = fig.add_gridspec(1, 2, left=0.075, right=0.905, top=0.93, bottom=0.575,
                                  wspace=0.16)
        gs_bot = fig.add_gridspec(1, len(strip), left=0.04, right=0.995, top=0.435,
                                  bottom=0.285, wspace=0.10)
        gs_zoom = fig.add_gridspec(1, len(strip), left=0.04, right=0.995, top=0.195,
                                  bottom=0.045, wspace=0.10)
        cmap = plt.get_cmap("viridis").copy(); cmap.set_bad("#e6e8ec")
        head, ims = [], []
        for k, (letter, ttl) in enumerate((("A", "Day decode, subspace fitted at each position"),
                                           ("B", "Day decode, weekday-token subspace transferred"))):
            ax = fig.add_subplot(gs_top[0, k])
            M = dec[k][:, cols].copy()
            M[:, [not cell_ok[s] for s in cols]] = np.nan
            im = ax.imshow(M, origin="lower", aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0,
                           interpolation="nearest")
            ims.append(im)
            for j, s in enumerate(cols):
                if not cell_ok[s]:
                    ax.add_patch(plt.Rectangle((j - 0.5, -0.5), 1, L, fill=False,
                                               hatch="////", edgecolor="#c8ccd4", lw=0))
                for li in range(L):
                    if cyc[k, li, s] and cell_ok[s]:
                        ax.plot(j, li, ".", ms=2.6, color="white", zorder=4)
            ax.set_xticks(range(len(cols)))
            ax.set_xticklabels([slot_label(s) for s in cols], fontsize=fs["tick"] - 1)
            ax.set_yticks(range(0, L, 4))
            ax.tick_params(length=0)
            ax.set_xlabel("Read position (tokens after the weekday)", fontsize=fs["lab"])
            if k == 0:
                ax.set_ylabel("Layer", fontsize=fs["lab"])
            else:
                ax.tick_params(labelleft=False)
            for sp in ax.spines.values(): sp.set_visible(False)
            head.append((letter, ttl, ax))
        pos1 = head[1][2].get_position()
        cax = fig.add_axes([pos1.x1 + 0.012, pos1.y0, 0.012, pos1.y1 - pos1.y0])
        cb = fig.colorbar(ims[1], cax=cax)
        cb.set_label("Held-out day decode (7-way)", fontsize=fs["lab"])
        cb.outline.set_visible(False); cb.ax.tick_params(length=0)

        # ---- (C)/(D) each slot centred on its own mean and projected onto the weekday-token plane
        mean0, P0 = plane(c_tr[li0, 0])
        R0 = float(np.linalg.norm(c_tr[li0, 0] - c_tr[li0, 0].mean(0), axis=1).mean())
        strips = [("C", gs_bot, f"Layer {args.layer}: the weekday-token ring plane at each "
                                 f"read position (shared scale; held-out own-knot rate below)",
                   False),
                  ("D", gs_zoom, "Same panels, each zoomed to its own knots "
                                 "(ring radius relative to the weekday token below)", True)]
        lim_pts = None
        for letter, gsr, ttl, zoom in strips:
            axes_c = []
            for j, s_ in enumerate(strip):
                ax = fig.add_subplot(gsr[0, j]); tidy(ax); ax.set_aspect("equal")
                hv, cv = held & valid[:, s_], ctrl & valid[:, s_]
                mc = c_tr[li0, s_].mean(0)
                ih = np.where(hv)[0]
                if len(ih) > 1500:
                    ih = rng.choice(ih, 1500, replace=False)
                ic = np.where(cv)[0]
                if len(ic) > 1500:
                    ic = rng.choice(ic, 1500, replace=False)
                xy = (z_tr[ih, li0, s_] - mc) @ P0.T
                xyn = (z_tr[ic, li0, s_] - mc) @ P0.T
                ck = (c_tr[li0, s_] - mc) @ P0.T
                curve = spline_op(600) @ ck
                Rs = float(np.linalg.norm(c_tr[li0, s_] - mc, axis=1).mean())
                ax.plot(xyn[:, 0], xyn[:, 1], ".", ms=0.8, color=PALE, alpha=0.4, zorder=2)
                for d in range(7):
                    m = dix[ih] == d
                    ax.plot(xy[m, 0], xy[m, 1], "o", ms=1.2, color=DAY_C[d],
                            alpha=0.25 if zoom else 0.45, markeredgecolor="none", zorder=3)
                ax.plot(curve[:, 0], curve[:, 1], "-", color="white", lw=1.8, zorder=4)
                ax.plot(curve[:, 0], curve[:, 1], "-", color=INK, lw=0.8, zorder=5)
                for d in range(7):
                    ax.plot(ck[d, 0], ck[d, 1], "o", ms=3.6, color=DAY_C[d],
                            markeredgecolor=INK, markeredgewidth=0.5, zorder=6)
                    if j == 0:
                        ax.annotate(DAYS[d][:2], (ck[d, 0] * 1.5, ck[d, 1] * 1.5),
                                    fontsize=fs["leg"] - 1.5, color=INK, ha="center",
                                    va="center", zorder=7)
                if zoom:
                    square(ax, ck * 1.9)
                else:
                    if lim_pts is None:
                        lim_pts = np.vstack([xy, ck * 1.7])
                    square(ax, lim_pts)
                ax.set_xticks([]); ax.set_yticks([])
                ax.spines["left"].set_visible(False); ax.spines["bottom"].set_visible(False)
                if not zoom:
                    ax.set_title(("weekday token" if s_ == 0 else
                                  ("period" if s_ == PERIOD_SLOT else
                                   f"+{s_} token" + ("s" if s_ > 1 else ""))),
                                 fontsize=fs["lab"], color=INK, pad=3)
                    note = f"{100 * dec[1, li0, s_]:.0f}%  ·  n={n_held[s_]}"
                else:
                    note = (f"radius {Rs:.2f}  ({Rs / R0:.2f}×)" if s_ else
                            f"radius {Rs:.2f}  (1×)")
                ax.text(0.5, -0.03, note, transform=ax.transAxes, ha="center", va="top",
                        fontsize=fs["leg"] - 1, color=MID)
                axes_c.append(ax)
            fig.canvas.draw()
            r, inv = fig.canvas.get_renderer(), fig.transFigure.inverted()
            b0 = axes_c[0].get_tightbbox(r).transformed(inv)
            yC = max(a.get_tightbbox(r).transformed(inv).y1 for a in axes_c) + 0.010
            lt = fig.text(b0.x0, yC, f"({letter})", ha="left", va="bottom", fontsize=9.5,
                          color=INK, fontweight="bold")
            lw = lt.get_window_extent(r).width / (fig.get_figwidth() * fig.dpi)
            fig.text(b0.x0 + lw + 0.55 * 9.5 / 72 / 7.0, yC, ttl, ha="left", va="bottom",
                     fontsize=9.5, color=INK)
        titles(fig, head, fs=9.5, dy=0.012)

        p = args.out
        fig.savefig(p, bbox_inches="tight", pad_inches=0.015); print(f"[saved] {p}")
        if args.also_png:
            fig.savefig(p[:-4] + ".png", dpi=300, bbox_inches="tight", pad_inches=0.015)
            print(f"[saved] {p[:-4]}.png")
        plt.close(fig)

    numbers = dict(layers=layers, cols=cols, n_fit=int(is_fit.sum()), n_held=n_held.tolist(),
                   n_ctrl=n_ctrl.tolist(), cell_ok=cell_ok.tolist(),
                   decode_own=np.nan_to_num(dec[0], nan=-1).round(4).tolist(),
                   decode_tr=np.nan_to_num(dec[1], nan=-1).round(4).tolist(),
                   share_pos_own=np.nan_to_num(shp[0], nan=-1).round(4).tolist(),
                   share_neg_own=np.nan_to_num(shn[0], nan=-1).round(4).tolist(),
                   share_pos_tr=np.nan_to_num(shp[1], nan=-1).round(4).tolist(),
                   share_neg_tr=np.nan_to_num(shn[1], nan=-1).round(4).tolist(),
                   cyclic_own=cyc[0].tolist(), cyclic_tr=cyc[1].tolist())
    jp = os.path.join(REPO, "experiments", "results", "fig_ring_trace_numbers.json")
    os.makedirs(os.path.dirname(jp), exist_ok=True)
    dump_json(numbers, jp)
    print(f"[saved] {jp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
