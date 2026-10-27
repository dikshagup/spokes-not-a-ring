"""Polar heat map of one Jacobian measure over the day disc (figures 1, 5, 6).

Field npz (from jacobian_field.py): ``us``, ``rs``, and one [n_backgrounds, n_u, n_r] array per measure.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import numpy as np  # noqa: E402

from weekday_manifold.manifold.days import DAYS, N_DAYS  # noqa: E402

INK, MID, PALE = "#16181d", "#5b6270", "#aeb4c0"

# (npz key, title, symbol, inset direction, colourbar note)
MEASURES = [("fro", "Jacobian norm", r"$\|J\|_F$", "all", "estimated (Hutchinson)"),
            ("gain_t", "Jacobian along ring tangent", r"$\|J\hat{t}\|$", "tangent", ""),
            ("gain_r", "Jacobian along ring radius", r"$\|J\hat{r}\|$", "radial", ""),
            ("gain_off", "Jacobian along a random direction",
             r"$\|J\hat{v}_{\perp}\|$", "off", "")]


def ratios(path, seed=0, n=6000):
    """Mid-arc / knot ratio at r = 1 per background, with a bootstrap 95% interval.

    Returns {key: (mean, lo, hi, mean field value, n_backgrounds)}.
    """
    z = np.load(path, allow_pickle=True)
    us, rs = z["us"], z["rs"]
    i1 = int(np.argmin(np.abs(rs - 1.0)))
    kn = np.array([int(round(v * len(us))) % len(us) for v in np.arange(N_DAYS) / N_DAYS])
    md = (kn + len(us) // (2 * N_DAYS)) % len(us)
    rng = np.random.default_rng(seed)
    out = {}
    for k, *_ in MEASURES:
        if k not in z.files:
            continue
        P = z[k][:, :, i1].astype(float)
        r = P[:, md].mean(1) / P[:, kn].mean(1)
        b = rng.choice(r, size=(n, len(r)), replace=True).mean(1)
        out[k] = (float(r.mean()), float(np.percentile(b, 2.5)),
                  float(np.percentile(b, 97.5)), float(P.mean()), len(r))
    return out


def edges(centres, lo=None):
    """Cell edges for pcolormesh from cell centres (``lo`` pins the first edge)."""
    c = np.asarray(centres, float)
    mid = (c[1:] + c[:-1]) / 2.0
    first = c[0] - (mid[0] - c[0]) if lo is None else lo
    return np.concatenate([[first], mid, [c[-1] + (c[-1] - mid[-1])]])


def measurement_inset(ax, kind, fs=1.0):
    """Draw an inset showing the direction each Jacobian measure probes."""
    ins = ax.inset_axes([0.97, 0.97, 0.23, 0.27])
    ins.set_xlim(-1.7, 1.7)
    ins.set_ylim(-1.7, 2.7)
    ins.set_aspect("equal")
    ins.axis("off")
    th = np.linspace(0, 2 * np.pi, 200)
    ins.plot(np.cos(th), np.sin(th), color="0.75", lw=0.8, zorder=1)
    a = np.pi / 3.2
    px, py = np.cos(a), np.sin(a)
    if kind == "all":                       # ||J||_F: random probes, every direction
        for t in np.linspace(0, 2 * np.pi, 8, endpoint=False) + 0.35:
            ins.annotate("", xy=(px + 0.80 * np.cos(t), py + 0.80 * np.sin(t)),
                         xytext=(px, py),
                         arrowprops=dict(arrowstyle="-|>", color="0.45", lw=0.40,
                                         mutation_scale=2.8, shrinkA=0, shrinkB=0), zorder=3)
        lab = "every direction"
    elif kind == "tangent":                 # ||J t||: along the loop
        tx, ty = -np.sin(a), np.cos(a)
        ins.annotate("", xy=(px + 1.0 * tx, py + 1.0 * ty), xytext=(px, py),
                     arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.7, mutation_scale=4.5,
                                     shrinkA=0, shrinkB=0), zorder=3)
        lab = "$\\hat{t}$, direction of $u$"
    elif kind == "radial":                  # ||J r||: radial
        ins.annotate("", xy=(px * 2.0, py * 2.0), xytext=(px, py),
                     arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.7, mutation_scale=4.5,
                                     shrinkA=0, shrinkB=0), zorder=3)
        lab = "$\\hat{r}$, direction of $r$"
    else:                                   # ||J v||: orthogonal to the ring
        ins.annotate("", xy=(px + 0.30, py + 0.95), xytext=(px, py),
                     arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.7, mutation_scale=4.5,
                                     shrinkA=0, shrinkB=0, linestyle=(0, (1.6, 1.2))),
                     zorder=3)
        lab = "$\\hat{v}_{\\perp}$, orthogonal to the ring"
    ins.plot([px], [py], marker="o", ms=2.0, color="0.1", zorder=4)
    ins.text(0, 1.75, lab, ha="center", va="bottom", fontsize=6.0 * fs, color=MID)


def disc(ax, th_e, r_e, grid, cmap, vmin, vmax, label, ring_r=1.0, dirn=None,
         fs=1.0, ring_label=False):
    """Draw one polar field panel with its colourbar; ``dirn`` adds the direction inset."""
    m = ax.pcolormesh(th_e, r_e, grid, cmap=cmap, vmin=vmin, vmax=vmax, shading="flat",
                      rasterized=True)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rlim(0, r_e[-1])
    ax.grid(False)
    ax.set_yticks([])
    ax.spines["polar"].set_visible(False)
    ax.set_xticks([2 * np.pi * k / N_DAYS for k in range(N_DAYS)])
    ax.set_xticklabels([DAYS[k][:3] for k in range(N_DAYS)], fontsize=7.6 * fs,
                       color="0.2")
    ax.tick_params(axis="x", pad=-2.0)

    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(th, np.full_like(th, ring_r), color="white", lw=2.2, zorder=4)
    ax.plot(th, np.full_like(th, ring_r), color="0.1", lw=0.8, zorder=5)
    if ring_label:
        ax.text(np.pi * 0.5, ring_r, "  r = 1", fontsize=7.0 * fs, color="0.1",
                ha="left", va="bottom", zorder=6)
    for k in range(N_DAYS):
        a = 2 * np.pi * k / N_DAYS
        ax.plot([a, a], [ring_r - 0.11, ring_r + 0.11], color="white", lw=3.0, zorder=4)
        ax.plot([a, a], [ring_r - 0.11, ring_r + 0.11], color="0.1", lw=1.3, zorder=5)
    ax.plot([0], [0], marker="o", ms=2.6, color="0.1", zorder=6,
            markeredgecolor="white", markeredgewidth=0.6)
    if dirn is not None:
        measurement_inset(ax, dirn, fs=fs)

    bb = ax.get_position()
    cw = bb.width * 0.88
    cax = ax.figure.add_axes([bb.x0 + (bb.width - cw) / 2, bb.y0 - 0.078, cw, 0.017])
    cb = ax.figure.colorbar(m, cax=cax, orientation="horizontal")
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(labelsize=7.4 * fs, length=2, width=0.4)
    cb.set_label(label, fontsize=7.6 * fs, labelpad=2)
