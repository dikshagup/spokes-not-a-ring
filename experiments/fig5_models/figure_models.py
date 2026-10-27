"""Appendix "Reproducing sensitivity results with other models" (fig5_models): the Jacobian field in three models.

Layers are matched by fraction of depth: 87.5% is L28 of 32 and L42 of 48.

    python experiments/fig5_models/figure_models.py \\
        --llama <fig1 data>/jacobian_field.npz --mistral <data>/jacobian_field_mistral.npz \\
        --gpt2xl <data>/jacobian_field_gpt2xl.npz
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
from paper_style import helvetica  # noqa: E402
from polar_disc import MEASURES, disc, edges, ratios  # noqa: E402
from weekday_manifold.utils import data_dir  # noqa: E402

REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
INK, MID = "#16181d", "#5b6270"
SCALE_BAND = (0.75, 1.3)     # radii the colour limits are taken over (panels are read at r = 1)
ROWS = [("Llama-3.1-8B", "llama"), ("Mistral-7B", "mistral"), ("GPT-2 XL", "gpt2xl")]


def build(fields):
    """``fields``: [(row label, path)] in row order."""
    metas = {nm: json.loads(str(np.load(p, allow_pickle=True)["meta"])) for nm, p in fields}
    ref = metas[fields[0][0]]
    for nm, m in metas.items():
        for k in ("formulation", "patch_layer", "ro_pos", "n_hutch"):
            assert m[k] == ref[k], (f"{nm} differs on {k}: {m[k]} vs {ref[k]} -- these "
                                    f"fields are not measuring the same intervention")

    nrow, ncol = len(fields), len(MEASURES)
    fig = plt.figure(figsize=(2.55 * ncol + 1.15, 3.55 * nrow))
    gs = GridSpec(nrow, ncol, figure=fig, left=0.100, right=0.955,
                  top=0.855, bottom=0.105, wspace=0.14, hspace=1.00)
    aspect = fig.get_figwidth() / fig.get_figheight()

    for i, (nm, path) in enumerate(fields):
        z = np.load(path, allow_pickle=True)
        meta = metas[nm]
        us, rs = np.asarray(z["us"]), np.asarray(z["rs"])
        th_e, r_e = edges(us * 2 * np.pi), edges(rs, lo=0.0)
        keep = (rs >= SCALE_BAND[0]) & (rs <= SCALE_BAND[1])
        R = ratios(path)
        print(f"[fig5] {nm} (L{meta['patch_layer']} -> L{meta['readout_layer']}): "
              + "  ".join(f"{k} {R[k][0]:.2f}x [{R[k][1]:.2f}, {R[k][2]:.2f}]"
                          for k, *_ in MEASURES if k in R))

        for j, (key, title, sym, dirn, _) in enumerate(MEASURES):
            a = z[key].mean(0).T                      # (n_r, n_u)
            bb = gs[i, j].get_position(fig)
            h = min(bb.height, bb.width * 0.60 * aspect)
            w = h / aspect
            existing = set(map(id, fig.axes))
            ax = fig.add_axes([bb.x0 + (bb.width - w) / 2, bb.y1 - h, w, h],
                              projection="polar")
            v = a[keep]
            disc(ax, th_e, r_e, a, "viridis", float(np.percentile(v, 1.0)),
                 float(np.percentile(v, 99.0)), "", dirn=dirn if i == 0 else None,
                 ring_label=False)
            for cax in [q for q in fig.axes if id(q) not in existing and q is not ax]:
                cax.tick_params(labelsize=9.6)

            if i == 0 and j == 0:
                ax.annotate("r = 1", xy=(2 * np.pi * 1.5 / 7, 1.0),
                            xytext=(2 * np.pi * 1.5 / 7, r_e[-1] * 1.46),
                            fontsize=10.5, color=INK, ha="center", va="center",
                            bbox=dict(fc="white", ec="none", pad=1.4),
                            arrowprops=dict(arrowstyle="-", lw=0.9, color=INK,
                                            shrinkA=2, shrinkB=1), zorder=8)
            m, lo, hi, _, _ = R[key]
            ax.text(0.5, -0.065, f"{m:.2f}×", transform=ax.transAxes,
                    ha="center", va="top", fontsize=10.6, color=INK)
            ax.text(0.5, -0.170, f"[{lo:.2f}, {hi:.2f}]", transform=ax.transAxes,
                    ha="center", va="top", fontsize=9.0, color=MID)

            if i == 0:
                t = title if len(title) < 20 else title.rsplit(" ", 1)[0] + "\n" + \
                    title.rsplit(" ", 1)[1]
                fig.text(bb.x0 + bb.width / 2, 0.985, t + "\n" + sym,
                         fontsize=13.0, color=INK, ha="center", va="top", linespacing=1.55)

        cell = gs[i, 0].get_position(fig)
        dh = min(cell.height, cell.width * 0.60 * aspect)
        fig.text(0.008, cell.y1 - dh / 2,
                 f"{nm}\nL{meta['patch_layer']} → L{meta['readout_layer']}",
                 fontsize=13.0, color=INK, ha="left", va="center", fontweight="bold",
                 linespacing=1.6)
    return fig


def main():
    data = data_dir("fig5_models", create=False)
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--llama", default=os.path.join(data_dir("fig1_sensitivity", create=False),
                                                    "jacobian_field.npz"),
                    help="figure 1's field: the same file")
    ap.add_argument("--mistral", default=os.path.join(data, "jacobian_field_mistral.npz"))
    ap.add_argument("--gpt2xl", default=os.path.join(data, "jacobian_field_gpt2xl.npz"))
    ap.add_argument("--out", default=os.path.join(REPO, "figures", "fig5_models.png"))
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    fields = [(nm, getattr(args, key)) for nm, key in ROWS]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    # save with the GridSpec margins, without tight cropping
    with helvetica.context(), plt.rc_context({"savefig.bbox": None}):
        fig = build(fields)
        for path in (args.out, os.path.splitext(args.out)[0] + ".pdf"):
            fig.savefig(path, dpi=args.dpi)
            print("[figure] wrote", path)


if __name__ == "__main__":
    main()
