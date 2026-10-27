"""Appendix "Non-uniform sensitivity with weekday arithmetic" (fig6_arithmetic): the plate for the interrogative prompts.

    python experiments/fig6_arithmetic/figure_arithmetic.py --rings <data>/rings.npz \\
        --walk <data>/ring_walk.npz --field <data>/jacobian_field.npz
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import NullFormatter  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
from paper_style import helvetica  # noqa: E402
from polar_disc import MEASURES, disc, edges, ratios  # noqa: E402
from weekday_manifold.manifold.days import DAYS, N_DAYS  # noqa: E402
from weekday_manifold.manifold.steering import fit_steer_spline  # noqa: E402
from weekday_manifold.utils import data_dir  # noqa: E402

REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
INK, MID, PALE = "#16181d", "#5b6270", "#aeb4c0"
IN_C, ANS_C, OUT_C = "#2b5d9e", "#b3322c", "#1f7a4d"
DAY_C = ["#CC79A7", "#D55E00", "#E69F00", "#F0E442", "#009E73", "#56B4E9", "#0072B2"]
BLUES, REDS = plt.get_cmap("Blues"), plt.get_cmap("Reds")
WORD = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7}
MAX_OFFSET = 1
RELAY_MIN_ACC = 0.5    # panel D draws a layer only if its best site reaches this accuracy in panel B
SCALE_BAND = (0.75, 1.3)


def shade(cmap, i, n):
    """Colour for position i of n on a sequential colormap, skipping its palest end."""
    return cmap(0.32 + 0.62 * (i / max(n - 1, 1)))


def load(path):
    z = np.load(path, allow_pickle=True)
    layers = np.asarray(z["layers"])
    order = np.argsort(layers)
    out = {"layers": layers[order], "us": z["us"], "sites": [str(s) for s in z["sites"]],
           "readout_layer": int(z["readout_layer"]),
           "prompt_texts": [str(t) for t in z["prompt_texts"]],
           "prompts": np.asarray(z["prompts"]),
           "dist_unpatched": np.asarray(z["dist_unpatched"]),
           "model": str(z["model"]), "formulation": str(z["formulation"]),
           "ro_pos": int(z["ro_pos"])}
    for k in ("dists", "stride_in"):
        out[k] = np.asarray(z[k])[:, order]
    return out


def knot_index(us, k):
    """Grid index of day `k` in the sweep."""
    return int(np.argmin(np.abs(np.asarray(us) - k / N_DAYS)))


def sources(d):
    """Index of the day each prompt mentions; steering to that knot changes nothing."""
    return np.array([DAYS.index(re.search(r"after (\w+)", t).group(1))
                     for t in d["prompt_texts"]])


def offsets(d):
    return np.array([WORD[re.search(r"is (\w+) days", t).group(1)] for t in d["prompt_texts"]])


def clean_accuracy(d, off, max_offset=N_DAYS):
    """Unpatched accuracy on the prompts the sweep steers, with offset at most max_offset."""
    d0, sel, off = d["dist_unpatched"], d["prompts"], np.asarray(off)
    ok = d0[sel][:, :N_DAYS].argmax(1) == (sources(d) + off) % N_DAYS
    return float(ok[off <= max_offset].mean())


def accuracy(d, off, src, max_offset=1):
    """Fraction of the steers, over the kept prompts, that yield the correct answer."""
    D, us = d["dists"], d["us"]
    acc = np.zeros((2, len(d["layers"])))
    for si in range(2):
        for li in range(len(d["layers"])):
            hit = tot = 0
            for pi, o in enumerate(off):
                if o > max_offset:
                    continue
                for k in range(N_DAYS):
                    if k == src[pi]:                      # skip the mentioned day's knot
                        continue
                    i = knot_index(us, k)
                    hit += int(np.argmax(D[si, li, pi, i, :N_DAYS]) == (k + o) % N_DAYS)
                    tot += 1
            acc[si, li] = hit / tot
    return acc


def hell_speed(dists, stride):
    """Behaviour distance moved per unit of activation distance, per step."""
    y = np.sqrt(np.clip(dists, 0, None))
    return np.linalg.norm(np.diff(y, axis=-2), axis=-1) / np.sqrt(2.0) / stride


def tidy(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_linewidth(0.6)
        ax.spines[sp].set_color(PALE)
    ax.tick_params(labelsize=8.3, length=2.5, width=0.6, colors=MID)
    ax.grid(True, lw=0.4, color="#eceef2", zorder=0)
    ax.set_axisbelow(True)


def pca3(C):
    """Projector onto the top three principal directions of a set of centroids."""
    ctr = C.mean(0)
    _, _, Vt = np.linalg.svd(C - ctr, full_matrices=False)
    basis = Vt[:3]
    return lambda X: (np.atleast_2d(X) - ctr) @ basis.T


def align(Y, X):
    """Rotation (reflections allowed) carrying Y onto X, by Procrustes on the day pairing."""
    U, _, Vt = np.linalg.svd(Y.T @ X)
    return U @ Vt


def camera(P, tilt=np.deg2rad(28.0)):
    """Orthographic projection with a fixed tilt, used for every ring."""
    ct, st = np.cos(tilt), np.sin(tilt)
    return np.stack([P[:, 0], P[:, 1] * ct + P[:, 2] * st], 1)


def behavior_ring(d):
    """Mean output distribution per answer day in Hellinger coordinates, and nearest-mean accuracy."""
    d0, sel = d["dist_unpatched"], d["prompts"]
    z = (sources(d) + offsets(d)) % N_DAYS
    P = d0[sel]
    C = np.stack([P[z == k].mean(0) for k in range(N_DAYS)])
    Y, E = np.sqrt(np.clip(C, 0, None)), np.sqrt(np.clip(P, 0, None))
    pred = np.argmin(((E[:, None, :] - Y[None, :, :]) ** 2).sum(-1), axis=1)
    return Y, float((pred == z).mean())


def panel_stack(ax, rings, beh, header_fs=10.3):
    """Draw the rings at every layer and the behaviour manifold."""
    L = [int(v) for v in rings["layers"]]
    keys = [str(k) for k in rings["keys"]]
    which = [str(w) for w in rings["which"]]
    C_all = rings["centroids"]
    idx = {(int(k.split("@")[0][1:]), w): i for i, (k, w) in enumerate(zip(keys, which))}
    rnorm = np.asarray(rings["resid_norm"])

    proj, ref = {}, {}
    for w in ("input", "answer"):
        for layer in L:
            C = C_all[idx[(layer, w)]].astype(np.float64)
            f = pca3(C)
            K = f(C)
            R = align(K, ref[w]) if w in ref else np.eye(3)
            K = K @ R
            ref[w] = K
            S = f(np.asarray(fit_steer_spline(C).forward(np.linspace(0, 1, 241)))) @ R
            g = 1.0 / rnorm[idx[(layer, w)]]
            proj[(layer, w)] = (camera(K) * g, camera(S) * g)

    span = max(np.abs(v[1]).max() for v in proj.values())
    # each ring is divided by its site's residual norm
    step, half = 1.02, 0.42 / span
    n = len(L)
    XI, XA = 0.30, 2.15
    Y0 = 1.30
    ytop = Y0 + (n - 1) * step

    def yof(li):                 # L2 at the top, L31 at the foot, behaviour below them all
        return Y0 + (n - 1 - li) * step

    ax.set_aspect("equal")
    ax.set_xlim(-0.62, 4.60)
    ax.set_ylim(-2.05, ytop + 1.15)
    ax.axis("off")

    for x, c, lab in ((XI, IN_C, "input-day ring\n(weekday token)"),
                      (XA, ANS_C, "answer-day ring\n(answer slot)")):
        ax.text(x, ytop + 0.30, lab, ha="center", va="bottom", fontsize=header_fs, color=c,
                linespacing=1.35)

    for li, layer in enumerate(L):
        y0 = yof(li)
        ax.text(-0.60, y0, f"L{layer}", ha="left", va="center", fontsize=10.5, color=MID)
        for x0, w, c in ((XI, "input", IN_C), (XA, "answer", ANS_C)):
            K, S = proj[(layer, w)]
            ax.plot(x0 + S[:, 0] * half, y0 + S[:, 1] * half, color=c, lw=1.0, alpha=0.55,
                    zorder=3, solid_capstyle="round")
            ax.scatter(x0 + K[:, 0] * half, y0 + K[:, 1] * half, s=9.0, c=DAY_C, zorder=4,
                       linewidths=0)

    Yb, _ = beh
    fb = pca3(Yb)
    Kb = camera(fb(Yb))
    Sb = camera(fb(np.asarray(fit_steer_spline(Yb).forward(np.linspace(0, 1, 241)))))
    gb = 0.33 / np.abs(Sb).max()        # separate scale for the behaviour ring
    xb = 0.5 * (XI + XA)
    ax.plot(xb + Sb[:, 0] * gb, Sb[:, 1] * gb, color=OUT_C, lw=1.0, alpha=0.55, zorder=3)
    ax.scatter(xb + Kb[:, 0] * gb, Kb[:, 1] * gb, s=9.0, c=DAY_C, zorder=4, linewidths=0)
    ax.text(-0.60, 0.0, "output", ha="left", va="center", fontsize=10.5, color=MID)
    ax.text(xb, -0.44, "behavior manifold", ha="center", va="top", fontsize=10.5, color=OUT_C)
    for x in (XI, XA):
        ax.annotate("", xy=(xb + 0.30 * np.sign(x - xb), 0.24),
                    xytext=(x, yof(n - 1) - 0.26),
                    arrowprops=dict(arrowstyle="-|>", color=PALE, lw=0.8))

    ax.legend(handles=[Line2D([], [], marker="o", ls="", ms=3.6, color=DAY_C[k],
                              label=DAYS[k][:3]) for k in range(N_DAYS)],
              loc="upper center", bbox_to_anchor=(0.33, 0.055), ncol=4, fontsize=9.7,
              frameon=False, handletextpad=0.15, columnspacing=0.75, labelspacing=0.28)


def panel_acc(ax, d, acc, clean, max_offset, kept=()):
    """Plot steering accuracy against steered layer, one curve per site."""
    L = d["layers"]
    # the mentioned day's knot is excluded, so ignoring the patch scores 0 and guessing 1/7
    ax.axhspan(0, 1.0 / N_DAYS, color="#f1f2f5", zorder=0)
    ax.axhline(clean, color=PALE, lw=1.0, zorder=1)
    ax.text(L[-1], clean + 0.015, "unsteered", fontsize=7.6, color=MID, va="bottom",
            ha="right")
    for si, (nm, c) in enumerate((("weekday token", IN_C), ("answer slot", ANS_C))):
        ax.plot(L, acc[si], "o-", color=c, lw=1.9, ms=4.2, label=nm, zorder=3)
    ax.set_xlabel("Steered layer", fontsize=9.0, color=INK)
    ktxt = (f"= {kept[0]}" if len(kept) == 1 else
            f"in {{{', '.join(str(v) for v in kept)}}}" if kept else f"≤ {max_offset}")
    ax.set_ylabel("Steers giving the correct answer\n"
                  f"(prompts with offset {ktxt})", fontsize=9.0, color=INK)
    ax.set_ylim(0, 1.08)
    ax.legend(fontsize=8.2, frameon=False, loc="center left")
    tidy(ax)


def in_control(d, acc, min_acc=0.5):
    """Layers whose best site reaches min_acc, and the best site of every layer."""
    L = list(d["layers"])
    pick = {lay: int(np.argmax(acc[:, i])) for i, lay in enumerate(L)}
    keep = [lay for i, lay in enumerate(L) if acc[pick[lay], i] >= min_acc]
    return keep, pick


def panel_ladder(axes, d, conds=((2, 0), (28, 1))):
    """Plot behaviour distance against activation distance, one (layer, site) per axis."""
    L = list(d["layers"])
    for ai, (ax, (layer, si)) in enumerate(zip(axes, conds)):
        li = L.index(layer)
        c = IN_C if si == 0 else ANS_C
        x = np.concatenate([[0.0], np.cumsum(d["stride_in"][si, li])])
        x = x / x[-1]
        dd = np.sqrt(np.clip(d["dists"][si, li], 0, None))
        Y = np.concatenate([np.zeros((dd.shape[0], 1)),
                            np.cumsum(np.linalg.norm(np.diff(dd, axis=1), axis=2)
                                      / np.sqrt(2.0), axis=1)], axis=1)
        for y in Y:
            ax.plot(x, y / y[-1], color=c, lw=0.7, alpha=0.16, zorder=3)
        M = Y.mean(0) / Y.mean(0)[-1]
        ax.plot(x, M, color=c, lw=2.2, zorder=4)
        ki = [knot_index(d["us"], k) for k in range(N_DAYS)]
        kx = [x[i] for i in ki]
        for v in kx:
            ax.axvline(v, color=PALE, ls=":", lw=0.8, zorder=0)
        ax.scatter(kx, M[ki], s=17, c=DAY_C, zorder=6, edgecolors="white", linewidths=0.6)
        # no Mon label: its knot is at x = 0
        ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])
        for kk, vx in enumerate(kx):
            if kk == 0:
                continue
            ax.text(vx, 0.985, DAYS[kk][:3], rotation=90, fontsize=6.0, color=MID,
                    ha="right", va="top", zorder=7)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.5, 1])
        ax.text(0.965, 0.045,
                f"steered at {'answer slot' if si else 'weekday token'}, L{layer}",
                transform=ax.transAxes, fontsize=8.0, color=c, va="bottom", ha="right")
        if ai == 0:
            ax.set_ylabel("Behavior distance\n(fraction)", fontsize=8.4, color=INK)
        tidy(ax)
        ax.grid(False)
    lo = min(ax.get_ylim()[0] for ax in axes)
    hi = max(ax.get_ylim()[1] for ax in axes)
    for ax in axes:
        ax.set_ylim(lo * 0.22, hi)
        ax.set_xlabel("Activation distance (fraction of ring)", fontsize=8.0, color=INK)


def panel_relay(ax, d, acc, min_acc=0.5):
    """Plot behaviour gain around the ring for every layer kept by in_control."""
    us, L = d["us"], list(d["layers"])
    keep, pick = in_control(d, acc, min_acc)
    dropped = [l for l in L if l not in keep]
    blues = [lay for lay in keep if pick[lay] == 0]
    reds = [lay for lay in keep if pick[lay] == 1]
    for lay in keep:
        li, si = L.index(lay), pick[lay]
        fam = blues if si == 0 else reds
        c = shade(BLUES if si == 0 else REDS, fam.index(lay), len(fam))
        y = hell_speed(d["dists"][si, li], d["stride_in"][si, li]).mean(0)
        xa = np.concatenate([[0.0], np.cumsum(d["stride_in"][si, li])])
        xa = xa / xa[-1]
        xm = 0.5 * (xa[1:] + xa[:-1])                    # step midpoints, in arc length
        ax.plot(xm, y / y.mean(), color=c, lw=1.5, zorder=3)
        if lay == keep[0]:
            rule_x = [xa[knot_index(us, k)] for k in range(N_DAYS)]
    for v in rule_x:
        ax.axvline(v, color=PALE, ls=":", lw=0.7, zorder=0)
    ax.axhline(1.0, color=PALE, lw=0.9, zorder=1)
    # ticks at the days' arc-length positions
    ax.set_xticks(rule_x)
    ax.set_xticklabels([nm[:3] for nm in DAYS], fontsize=7.2, color=MID)
    ax.set_xlim(0.0, 1.0)
    ax.set_yscale("log")
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Steered coordinate", fontsize=8.4, color=INK)
    ax.set_ylabel("Behavior gain  " r"$d_{\mathcal{M}_y}/d_{\mathcal{M}_h}$" "\n"
                  "relative to this curve's ring average", fontsize=8.0, color=INK)
    hs = [Line2D([], [], color=shade(BLUES, i, len(blues)), lw=1.6, label=f"L{lay}")
          for i, lay in enumerate(blues)]
    hs += [Line2D([], [], color=shade(REDS, i, len(reds)), lw=1.6, label=f"L{lay}")
           for i, lay in enumerate(reds)]
    ax.legend(handles=hs, fontsize=7.1, frameon=False, ncol=2, loc="lower left",
              handletextpad=0.5, columnspacing=0.8, labelspacing=0.25,
              title=f"steering accuracy > {min_acc:.2f}\nweekday   |   answer",
              title_fontsize=7.1)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo * 0.30, hi)
    if dropped:
        print(f"[fig6] panel D excludes (steering accuracy < {min_acc:.2f}): "
              + ", ".join(f"L{l}" for l in dropped))
    tidy(ax)


def build(rings_path, walk_path, field_path):
    d = load(walk_path)
    rings = np.load(rings_path, allow_pickle=True)
    zf = np.load(field_path, allow_pickle=True)
    meta = json.loads(str(zf["meta"]))
    assert d["model"] == meta["model"], f"different models: {d['model']} vs {meta['model']}"
    assert d["formulation"] == meta["formulation"], (
        f"different prompt families: {d['formulation']} vs {meta['formulation']}")
    assert d["readout_layer"] == meta["readout_layer"], (
        f"walk reads L{d['readout_layer']}, field reads L{meta['readout_layer']}")
    assert d["ro_pos"] == meta["ro_pos"], "different readout positions"
    assert str(rings["model"]) == d["model"] and int(rings["ro_pos"]) == d["ro_pos"], (
        "the ring dump is not from the walk's setup")
    assert "gain_off" in zf.files, (
        f"{field_path} has no gain_off; without the random-direction control the other "
        f"three discs have no baseline")

    off = offsets(d)
    clean = clean_accuracy(d, off, MAX_OFFSET)
    acc = accuracy(d, off, sources(d), MAX_OFFSET)
    print(f"[fig6] {d['formulation']} / {d['model']}, steer L{meta['patch_layer']} -> "
          f"read L{meta['readout_layer']}; walk layers {d['layers'].tolist()}; field over "
          f"{zf['fro'].shape[0]} backgrounds")
    print(f"[fig6] unpatched accuracy on the offset-{MAX_OFFSET} prompts: {clean:.3f}")
    for si, nm in enumerate(("weekday token", "answer slot")):
        print(f"[fig6] steering accuracy, {nm:13s}: "
              + " ".join(f"L{l}={v:.2f}" for l, v in zip(d["layers"], acc[si])))

    fig = plt.figure(figsize=(9.4, 12.2))
    gs = GridSpec(3, 2, figure=fig, hspace=0.20, wspace=0.06, width_ratios=[1.10, 1.02],
                  height_ratios=[0.44, 0.42, 0.54],
                  left=0.035, right=0.930, top=0.966, bottom=0.340)
    a = fig.add_subplot(gs[:, 0])
    b = fig.add_subplot(gs[0, 1].subgridspec(20, 1)[0:17])
    csub = gs[1, 1].subgridspec(20, 1)[0:15].subgridspec(1, 2, wspace=0.28)
    dax = [fig.add_subplot(csub[0]), fig.add_subplot(csub[1])]
    eax = fig.add_subplot(gs[2, 1].subgridspec(20, 1)[0:16])

    panel_stack(a, rings, behavior_ring(d), header_fs=8.2)
    panel_acc(b, d, acc, clean, MAX_OFFSET,
              kept=sorted(set(off[off <= MAX_OFFSET].tolist())))
    panel_ladder(dax, d)
    panel_relay(eax, d, acc, min_acc=RELAY_MIN_ACC)

    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    for ax, letter, title in ((a, "A", "Weekday representations in the model"),
                              (b, "B", "Layerwise accuracy of steering"),
                              (dax[0], "C", "Steering snaps to weekdays"),
                              (eax, "D", "Path speed is lowest at the weekdays")):
        bb = ax.get_position()
        x = bb.x0 - 0.042
        lab = ax.yaxis.label
        if lab.get_text():
            x = (lab.get_window_extent(rend).transformed(fig.transFigure.inverted())).x0
        t = fig.text(x, bb.y1 + 0.012, letter, fontsize=13.4, fontweight="bold",
                     color=INK, ha="left", va="bottom")
        fig.canvas.draw()
        w_ = (t.get_window_extent(rend).transformed(fig.transFigure.inverted())).width
        fig.text(x + w_ + 0.006, bb.y1 + 0.014, title, fontsize=10.5, color=INK, ha="left",
                 va="bottom")

    us, rs = np.asarray(zf["us"]), np.asarray(zf["rs"])
    th_e, r_e = edges(us * 2 * np.pi), edges(rs, lo=0.0)
    keep = (rs >= SCALE_BAND[0]) & (rs <= SCALE_BAND[1])
    R = ratios(field_path)
    print("[fig6] between-days / at-days at r = 1: "
          + "  ".join(f"{k} {R[k][0]:.2f}x [{R[k][1]:.2f}, {R[k][2]:.2f}]"
                      for k, *_ in MEASURES))
    TITLE_Y, SYM_Y = 0.316, 0.262

    gsb = GridSpec(1, 4, figure=fig, left=0.035, right=0.930, top=0.240, bottom=0.052,
                   wspace=0.14)
    aspect = fig.get_figwidth() / fig.get_figheight()
    for j, (key, title, sym, dirn, _) in enumerate(MEASURES):
        grid = zf[key].mean(0).T
        bb = gsb[0, j].get_position(fig)
        # set height first and derive width so the polar axes stay circular
        h = min(bb.height, bb.width * 0.62 * aspect)
        w = h / aspect
        existing = set(map(id, fig.axes))
        ax = fig.add_axes([bb.x0 + (bb.width - w) / 2, bb.y1 - h, w, h], projection="polar")
        v = grid[keep]
        disc(ax, th_e, r_e, grid, "viridis",
             float(np.percentile(v, 1.0)), float(np.percentile(v, 99.0)), "",
             dirn=dirn, fs=0.95, ring_label=False)
        for cax in [q for q in fig.axes if id(q) not in existing and q is not ax]:
            cp, apos = cax.get_position(), ax.get_position()
            cax.set_position([cp.x0 + cp.width * 0.06, apos.y0 - 0.044,
                              cp.width * 0.88, 0.0072])
            cax.tick_params(labelsize=8.0, length=2, width=0.4)
        if j == 0:
            ax.annotate("r = 1", xy=(2 * np.pi * 1.5 / 7, 1.0),
                        xytext=(2 * np.pi * 1.5 / 7, r_e[-1] * 1.52),
                        fontsize=9.0, color=INK, ha="center", va="center",
                        bbox=dict(fc="white", ec="none", pad=1.2),
                        arrowprops=dict(arrowstyle="-", lw=0.8, color=INK,
                                        shrinkA=2, shrinkB=1), zorder=8)
        m, lo, hi, _, _ = R[key]
        ax.text(0.5, -0.075, f"{m:.2f}×", transform=ax.transAxes, ha="center",
                va="top", fontsize=11.0, color=INK)
        ax.text(0.5, -0.205, f"[{lo:.2f}, {hi:.2f}]", transform=ax.transAxes,
                ha="center", va="top", fontsize=8.6, color=MID)
        t = title if len(title) < 20 else title.rsplit(" ", 1)[0] + "\n" + \
            title.rsplit(" ", 1)[1]
        fig.text(bb.x0 - 0.008, TITLE_Y, "EFGH"[j], fontsize=13.4, fontweight="bold",
                 color=INK, ha="left", va="top")
        fig.text(bb.x0 + bb.width / 2, TITLE_Y, t, fontsize=9.8, color=INK,
                 ha="center", va="top", linespacing=1.5)
        fig.text(bb.x0 + bb.width / 2, SYM_Y, sym, fontsize=11.0, color=INK,
                 ha="center", va="baseline")
    return fig


def main():
    data = data_dir("fig6_arithmetic", create=False)
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rings", default=os.path.join(data, "rings.npz"),
                    help="day centroids at every (layer, site): ring_walk.py --dump-rings")
    ap.add_argument("--walk", default=os.path.join(data, "ring_walk.npz"))
    ap.add_argument("--field", default=os.path.join(data, "jacobian_field.npz"))
    ap.add_argument("--out", default=os.path.join(REPO, "figures", "fig6_arithmetic.png"))
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with helvetica.context():
        fig = build(args.rings, args.walk, args.field)
        for path in (args.out, os.path.splitext(args.out)[0] + ".pdf"):
            fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
            print("[figure] wrote", path)


if __name__ == "__main__":
    main()
