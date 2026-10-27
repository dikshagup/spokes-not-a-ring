"""Appendix "Activation steering" (fig4_schematic): the steering construction and sensitivity analysis.

    python experiments/methods_schematic/figure_schematic.py --field <data>/jacobian_field.npz
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
from method_schematic import panel_headers, panel_pipeline, panel_traces  # noqa: E402
from paper_style import helvetica  # noqa: E402
from weekday_manifold.manifold.days import DAYS  # noqa: E402
from weekday_manifold.manifold.steering import fit_steer_spline  # noqa: E402
from weekday_manifold.utils import data_dir  # noqa: E402

REPO = os.path.abspath(os.path.join(HERE, "..", ".."))


def build(field):
    z = np.load(field, allow_pickle=True)
    C, mu = z["centroids"], z["mu"]
    meta = json.loads(str(z["meta"]))
    sp = fit_steer_spline(C)
    _, _, Vt = np.linalg.svd(C - mu, full_matrices=False)

    day_pos, ro_pos = meta["day_pos"], meta["ro_pos"]
    toks = ["<bos>"] + [t.replace("\n", "\\n") for t in z["tokens"][1:]]
    assert toks[day_pos].strip() in DAYS, (
        "patch site is %r, not a weekday -- token list is out of step" % toks[day_pos])
    assert 0 <= ro_pos < len(toks), "readout position is outside the token strip"
    toks = toks[:ro_pos + 1]

    measures = "$\\|J\\|_F$ · $\\|J\\hat t\\|$ · $\\|J\\hat r\\|$ · $\\|J\\hat v_{\\perp}\\|$"
    fig = plt.figure(figsize=(8.40, 2.85))
    gs = GridSpec(1, 4, figure=fig, hspace=0.20, wspace=0.04,
                  left=0.006, right=0.994, top=0.865, bottom=0.04)
    ba = gs[0, 0].get_position(fig)
    ax_a = fig.add_axes([ba.x0 + ba.width * 0.16, ba.y0, ba.width, ba.height])
    bb = gs[0, 1:].get_position(fig)
    w = bb.width * 0.80
    ax_b = fig.add_axes([bb.x0 + (bb.width - w) / 2, bb.y0, w, bb.height])

    panel_traces(ax_a, sp, mu, Vt)
    panel_pipeline(ax_b, day_pos, ro_pos, toks,
                   meta["patch_layer"], meta["readout_layer"], measures)
    panel_headers(fig, [("A", "Measuring sensitivity to perturbations along and across "
                             "the ring", gs[0, :].get_position(fig))], dy=0.014)
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--field", default=os.path.join(data_dir("fig1_sensitivity", create=False),
                                                    "jacobian_field.npz"),
                    help="figure 1's Jacobian field; read for the centroids, mu, the token "
                         "strip and the two positions")
    ap.add_argument("--out", default=os.path.join(REPO, "figures", "fig4_schematic.png"))
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with helvetica.context():
        fig = build(args.field)
        for path in (args.out, os.path.splitext(args.out)[0] + ".pdf"):
            fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
            print("[figure] wrote", path)


if __name__ == "__main__":
    main()
