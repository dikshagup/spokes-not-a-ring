#!/usr/bin/env python
"""Figure 4 alignment numbers: the translation, scale and rotation each context applies to the ring.

Writes experiments/results/fig4_alignment.json.

    PYTHONPATH=src:experiments/common:experiments/fig4_occupancy \\
        python experiments/fig4_occupancy/alignment_numbers.py   # CPU, ~3 min
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

from figure_occupancy import CONTEXT_TYPES, example_contexts
from weekday_manifold.corpus.alignment import (fit_plane, family_configs, ladder, noise_null,
                                               spoke_distances, stage_distances)
from weekday_manifold.corpus.classes import resolve_classes
from weekday_manifold.corpus.split import fit_mask, held_out_mask
from weekday_manifold.manifold.days import DAYS
from weekday_manifold.utils import data_dir
from weekday_manifold.utils import dump_json

STAGES = ["raw", "T", "T+S", "T+S+R"]
STAGE_LABEL = {"raw": "as captured", "T": "+translation", "T+S": "+scale", "T+S+R": "+rotation"}
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def dispersion(P, day_ix, R):
    """Within-weekday RMS radius (in ring radii) and the in-plane variance share weekday explains."""
    within = [np.sqrt(((P[day_ix == d] - P[day_ix == d].mean(0)) ** 2).sum(1).mean()) for d in range(7)]
    tot = ((P - P.mean(0)) ** 2).sum()
    w_ss = sum(((P[day_ix == d] - P[day_ix == d].mean(0)) ** 2).sum() for d in range(7))
    return float(np.mean(within) / R), float(1 - w_ss / tot)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", default=data_dir("fig4_occupancy"))
    ap.add_argument("--layer", type=int, default=28)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--classes", default="clean,temporal")
    ap.add_argument("--n-null", type=int, default=200, help="noise draws per context")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--results", default=os.path.join(REPO, "experiments", "results",
                                                      "fig4_alignment.json"))
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    classes = resolve_classes(args.classes)

    R = np.load(os.path.join(args.inp, f"raw_L{args.layer}_n{args.n}.npz"), allow_pickle=True)
    cls, doc = np.asarray(R["cls"]).astype(str), np.asarray(R["doc"]).astype(str)
    G = fit_plane(R, fit_mask(cls, doc, classes, seed=args.seed))
    S = np.load(os.path.join(args.inp, f"swap_L{args.layer}_n{args.n}.npz"), allow_pickle=True)
    fcls, fdoc = np.asarray(S["fam_cls"]).astype(str), np.asarray(S["fam_doc"]).astype(str)
    fit = fit_mask(fcls, fdoc, classes, seed=args.seed)
    held = held_out_mask(fcls, fdoc, classes, seed=args.seed)
    Z = family_configs(S, G["hull6"], G["mean_c"])            # [F, 7, 6]; the plane is [:, :, :2]
    Dm = Z - Z.mean(1, keepdims=True)
    k6 = Dm[fit].mean(0); k2 = k6[:, :2]                      # the reference ring
    R2 = float(np.linalg.norm(k2, axis=1).mean()); R6 = float(np.linalg.norm(k6, axis=1).mean())
    print(f"[fit] reference ring from {int(fit.sum())} fit contexts; {int(held.sum())} held-out "
          f"contexts; ring radius {R2:.2f} (plane), {R6:.2f} (6-D)")

    # report only the weekday each text actually used
    orig = np.array([DAYS.index(d) for d in np.asarray(S["fam_day"]).astype(str)[held]])
    F = int(held.sum()); rows = np.arange(F)

    res, null = {}, {}
    for name, k, x in (("2D", k2, Z[held][:, :, :2]), ("6D", k6, Z[held])):
        L = ladder(x, k)
        gain, theta_null = noise_null(k, L["s"], L["sseTS"], args.n_null, rng, per_point=True)
        d, _ = stage_distances(L, k)
        sse = {st: float((d[st][rows, orig] ** 2).sum()) for st in STAGES}
        tot = sse["raw"]
        res[name] = dict(L=L, share=dict(translation=(sse["raw"] - sse["T"]) / tot,
                                         scale=(sse["T"] - sse["T+S"]) / tot,
                                         rotation=(sse["T+S"] - sse["T+S+R"]) / tot,
                                         remaining=sse["T+S+R"] / tot))
        null[name] = dict(share=gain[:, rows, orig].sum(1) / tot, theta=theta_null)
    L2 = res["2D"]["L"]
    theta = L2["theta_deg"]; theta_null = null["2D"]["theta"].ravel()
    tnorm = np.linalg.norm(L2["t"], axis=1) / R2
    rho, pval = spearmanr(tnorm, np.abs(theta))
    sd, sh = spoke_distances(L2, k2); cd, ch = stage_distances(L2, k2)
    spoke = {st: dict(median_dist_R=float(np.median(sd[st][rows, orig]) / R2),
                      own_rate=float(sh[st][rows, orig].mean())) for st in STAGES}
    centroid = {st: dict(median_dist_R=float(np.median(cd[st][rows, orig]) / R2),
                         own_rate=float(ch[st][rows, orig].mean())) for st in STAGES}
    x_raw = (L2["xt"] + L2["t"][:, None, :])[rows, orig]
    disp_raw, ve_raw = dispersion(x_raw, orig, R2)
    disp_c, ve_c = dispersion(L2["xt"][rows, orig], orig, R2)
    spacing = float(np.linalg.norm(k2 - np.roll(k2, 1, 0), axis=1).mean() / R2)
    shift = np.linalg.norm(Z[held].mean(1)[:, :2], axis=1) / G["ring_R"]
    picks, text = example_contexts(S, held, shift, os.path.join(args.inp, "capture_list_n256.npz"),
                                   args.model)

    q = lambda v, p: float(np.percentile(v, p))
    print("\nshare of between-context misfit removed   translation   scale   rotation   left   | rotation on noise")
    for name in ("2D", "6D"):
        s = res[name]["share"]; nl = null[name]["share"]
        print(f"  {name:3s}  {100 * s['translation']:30.1f}%  {100 * s['scale']:5.1f}%  "
              f"{100 * s['rotation']:7.2f}%  {100 * s['remaining']:5.1f}%  | "
              f"{100 * nl.mean():.2f}% [{100 * q(nl, 2.5):.2f}, {100 * q(nl, 97.5):.2f}]")
    print(f"\nshift |t|/R: median {np.median(tnorm):.2f}, IQR {q(tnorm, 25):.2f}-{q(tnorm, 75):.2f}, "
          f"95th {q(tnorm, 95):.2f}")
    print(f"scale s:     median {np.median(L2['s']):.2f}, IQR {q(L2['s'], 25):.2f}-{q(L2['s'], 75):.2f}, "
          f"90% within {q(L2['s'], 5):.2f}-{q(L2['s'], 95):.2f}")
    print(f"rotation:    median |theta| {np.median(np.abs(theta)):.2f} deg (noise null "
          f"{np.median(np.abs(theta_null)):.2f}); 90% of contexts within +-{q(np.abs(theta), 90):.1f} deg; "
          f"reflections {int(L2['reflect'].sum())}/{F}; Spearman rho(|t|,|theta|) = {rho:.3f}")
    print(f"adjacent-centroid spacing {spacing:.2f} R")
    print(f"within-weekday dispersion: {disp_raw:.2f} R as captured -> {disp_c:.2f} R after centering; "
          f"weekday explains {100 * ve_raw:.0f}% -> {100 * ve_c:.0f}% of in-plane variance")
    print("\nstage          nearest own centroid   median dist   nearest own spoke   median dist")
    for st in STAGES:
        print(f"  {STAGE_LABEL[st]:13s}{100 * centroid[st]['own_rate']:11.1f}%  "
              f"{centroid[st]['median_dist_R']:11.3f} R  {100 * spoke[st]['own_rate']:11.1f}%  "
              f"{spoke[st]['median_dist_R']:11.3f} R")
    if picks:
        print("\nexample contexts of panel B:")
        for j, i in enumerate(picks):
            print(f"  {CONTEXT_TYPES[j][0]:17s} |t|/R {tnorm[i]:.2f}  s {L2['s'][i]:.2f}  "
                  f"theta {theta[i]:+.1f} deg   {text(i)!r}")

    out = dict(classes=classes, n_fit=int(fit.sum()), n_held=F, n_null=args.n_null,
               ring_radius_plane=R2, ring_radius_6d=R6, adjacent_spacing_R=spacing,
               share={n: {k: float(v) for k, v in res[n]["share"].items()} for n in res},
               rotation_null_share={n: dict(mean=float(null[n]["share"].mean()),
                                            q025=q(null[n]["share"], 2.5),
                                            q975=q(null[n]["share"], 97.5)) for n in null},
               shift_R=dict(median=float(np.median(tnorm)), q25=q(tnorm, 25), q75=q(tnorm, 75),
                            q95=q(tnorm, 95)),
               scale=dict(median=float(np.median(L2["s"])), q05=q(L2["s"], 5), q25=q(L2["s"], 25),
                          q75=q(L2["s"], 75), q95=q(L2["s"], 95)),
               rotation_deg=dict(median_abs=float(np.median(np.abs(theta))),
                                 null_median_abs=float(np.median(np.abs(theta_null))),
                                 p90_abs=q(np.abs(theta), 90), reflections=int(L2["reflect"].sum()),
                                 spearman_shift_abs_theta=dict(rho=float(rho), p=float(pval))),
               dispersion=dict(within_day_R_captured=disp_raw, within_day_R_centered=disp_c,
                               variance_explained_captured=ve_raw, variance_explained_centered=ve_c),
               spoke=spoke, centroid=centroid,
               example_contexts=[dict(type=CONTEXT_TYPES[j][0], shift_R=float(tnorm[i]),
                                      scale=float(L2["s"][i]), rotation_deg=float(theta[i]),
                                      text=text(i)) for j, i in enumerate(picks)])
    os.makedirs(os.path.dirname(args.results), exist_ok=True)
    dump_json(out, args.results, indent=1)
    print(f"\n[saved] {args.results}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
