"""Drawing shared by the two uncertainty plates: the ring panel, its colours and sizes."""
from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from weekday_manifold.manifold.days import N_DAYS

# sampled at (lean + 1) / 2: lean -1 (day B) brown, 0 green, +1 (day A) blue
CMAP = LinearSegmentedColormap.from_list(
    "lean", ["#803200", "#cf8d61", "#2e7d5b", "#8d99dd", "#3c4292"])
G_TEXT, G_ANNOT, G_RULE, G_SPINE, G_BACK = "0.2", "0.35", "0.45", "0.6", "0.88"
FS_PANEL, FS_LABEL, FS_TICK, FS_SUB, FS_ANNOT, FS_LEG, FS_RING = 12.5, 10, 10, 10, 7.5, 10, 11

BLACK_AXES = {"axes.edgecolor": "black", "axes.labelcolor": "black",
              "xtick.color": "black", "ytick.color": "black"}


def colour_for(lean: float):
    return CMAP((lean + 1) / 2)


def level_colours(levels: Iterable[float]) -> Dict[float, tuple]:
    return {float(v): colour_for(float(v)) for v in levels}


def ring(ax, sep: int, dot_ms: float = 7) -> None:
    """Draw the ring (r = 1), radial guides, and days A (left of top) and B (right of top), sep days apart."""
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(th), np.sin(th), color="0.4", lw=.6, zorder=1)
    for rr in (0.25, 0.50, 0.75):
        ax.plot(rr * np.cos(th), rr * np.sin(th), color=G_BACK, lw=.8, zorder=0)
        ax.text(0.06, -rr, f"{rr:.2f}", fontsize=FS_ANNOT, color=G_ANNOT, ha="left",
                va="center")
    ax.text(0.06, -1.09, "r = 1", fontsize=FS_ANNOT, color=G_ANNOT, ha="left", va="center")
    for t, lab in ((spoke_angle(0, sep), "Day A"), (spoke_angle(1, sep), "Day B")):
        ax.plot(np.cos(t), np.sin(t), "o", ms=dot_ms, color="black", zorder=4)
        ax.text(1.10 * np.cos(t), 1.10 * np.sin(t), lab, fontsize=FS_LABEL, color="black",
                ha="right" if np.cos(t) < 0 else "left", va="bottom", zorder=4)
    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-1.20, 1.55)
    ax.set_aspect("equal")
    ax.axis("off")


def spoke_angle(w, sep: int):
    """Angle of pair position w (0 = day A, 1 = day B) for a pair sep days apart."""
    return np.pi / 2 + sep * np.pi / N_DAYS - np.asarray(w) * sep * (2 * np.pi / N_DAYS)


def xy(w, r, sep: int) -> Tuple[np.ndarray, np.ndarray]:
    t = spoke_angle(w, sep)
    return np.asarray(r) * np.cos(t), np.asarray(r) * np.sin(t)


def lobe_means(X, Y, w, lean, levels, min_share: float = 0.0, min_n: int = 3):
    """One mean per level per lobe (day A: w < 0.5; day B: w >= 0.5), on the plotted x, y."""
    out = []
    for side in (w < 0.5, w >= 0.5):
        pts = []
        for v in levels:
            k = side & (lean == v)
            if k.sum() >= max(min_n, min_share * (lean == v).sum()):
                pts.append((v, X[k].mean(), Y[k].mean()))
        out.append(pts)
    return out


def scatter_family(ax, g: Dict[str, np.ndarray], sep: int, col, rng, point_s=2.2,
                   point_alpha=.20, mean_s=20, join=True, min_share=0.0) -> None:
    """Draw one ring panel: every prompt of the family at this separation, plus lobe means."""
    m = g["gap"] == sep
    o = rng.permutation(int(m.sum()))
    w, r, lean = g["w"][m][o], g["r"][m][o], g["lean"][m][o]
    X, Y = xy(w, r, sep)
    ax.scatter(X, Y, s=point_s, lw=0, alpha=point_alpha, zorder=2,
               c=[col[v] for v in lean], rasterized=True)
    for pts in lobe_means(X, Y, w, lean, sorted(col), min_share=min_share):
        if not pts:
            continue
        if join:
            ax.plot([p[1] for p in pts], [p[2] for p in pts], "-", color=G_RULE, lw=1,
                    zorder=3)
        ax.scatter([p[1] for p in pts], [p[2] for p in pts], s=mean_s,
                   color=[col[p[0]] for p in pts], edgecolor="black", lw=.5, zorder=4)
