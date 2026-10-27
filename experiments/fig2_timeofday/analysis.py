#!/usr/bin/env python
"""Geometry for figure 2 (used by plate.py); as a script, writes experiments/results/fig2_timeofday_numbers.json.

    python experiments/fig2_timeofday/analysis.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "src")))
from weekday_manifold.manifold.days import DAYS  # noqa: E402
from weekday_manifold.utils import data_dir  # noqa: E402
from capture_prompts import clock_word, WORDS  # noqa: E402
from weekday_manifold.utils import dump_json

RESULTS_JSON = os.path.abspath(os.path.join(_HERE, "..", "results", "fig2_timeofday_numbers.json"))

DAY_C = ["#CC79A7", "#D55E00", "#E69F00", "#F0E442", "#009E73", "#56B4E9", "#0072B2"]
# modifier rows shown in panel B
KEEP_CLOCK = (3, 7, 11, 15, 19, 23)
KEEP_WORDS = tuple(w for w, _ in WORDS)
WORD_HOURS = dict(WORDS)
KEEP_PLACEBO = ("very rainy", "very sunny", "notably dusty")
TIME_FAMILIES = ("clock", "clock_midnight", "word")


def wrap(a):
    return (a + 180.0) % 360.0 - 180.0


def load_capture(path):
    z = np.load(path, allow_pickle=True)
    return dict(A=z["A"].astype(np.float64), day=z["day"], fam=z["fam"].astype(str),
                lab=z["lab"].astype(str), hour=z["hour"])


def _between_ss(x, g):
    m = x.mean()
    return float(sum(((x[g == u].mean() - m) ** 2) * (g == u).sum() for u in np.unique(g)))


def raw_pca_axes(A, day, hour, cl, fit, n_scan=30):
    """PCA of the `fit` rows without labels. Returns (weekday plane [2, d], hour axis [d], table, mean)."""
    mu = A[fit].mean(0)
    _, S, V = np.linalg.svd(A[fit] - mu, full_matrices=False)
    n_scan = min(n_scan, V.shape[0])
    Z = (A - mu) @ V[:n_scan].T
    varf = S[:n_scan] ** 2 / float((S ** 2).sum())
    d_ss = np.array([_between_ss(Z[fit, k], day[fit]) for k in range(n_scan)])
    h_ss = np.array([_between_ss(Z[cl, k], hour[cl]) for k in range(n_scan)])
    d_tot = ((Z[fit] - Z[fit].mean(0)) ** 2).sum(0)
    h_tot = ((Z[cl] - Z[cl].mean(0)) ** 2).sum(0)
    d_abs = d_ss / ((A[fit] - mu) ** 2).sum()
    h_abs = h_ss / ((A[cl] - A[cl].mean(0)) ** 2).sum()
    p = np.argsort(-d_abs)[:2]
    e3i = int(next(k for k in np.argsort(-h_abs) if k not in set(p.tolist())))
    table = dict(varf=varf, d_eta=d_ss / d_tot, h_eta=h_ss / h_tot, d_abs=d_abs, h_abs=h_abs,
                 plane_idx=tuple(int(i) for i in p), e3_idx=e3i, n_fit=int(fit.sum()))
    return V[list(p)], V[e3i], table, mu


def ring_handedness(plane, Cday, mu):
    """+1 anticlockwise, -1 clockwise, 0 if the centroids are not in cyclic order."""
    xyC = (Cday - mu) @ plane.T
    th = np.degrees(np.arctan2(xyC[:, 1], xyC[:, 0]))
    step = wrap(th[(np.arange(7) + 1) % 7] - th)
    return 1 if (step > 0).all() else (-1 if (step < 0).all() else 0)


def orient_plane(plane, Cday, mu):
    """Flip the second axis if needed so the days run anticlockwise (a positive shift is toward the next day)."""
    hand = ring_handedness(plane, Cday, mu)
    if hand == 0:
        raise SystemExit("the seven day centroids are not in cyclic order in this plane")
    return plane if hand == +1 else np.stack([plane[0], -plane[1]])


def plate_geometry(cap):
    A, day, fam, lab, hour = cap["A"], cap["day"], cap["fam"], cap["lab"], cap["hour"]
    cl = fam == "clock"
    fit = np.isin(fam, TIME_FAMILIES)
    plane, e3, pcs, mu = raw_pca_axes(A, day, hour, cl, fit=fit)
    Cday = np.stack([A[day == d].mean(0) for d in range(7)])
    plane = orient_plane(plane, Cday, mu)
    xy, xyC = (A - mu) @ plane.T, (Cday - mu) @ plane.T
    thC = np.degrees(np.arctan2(xyC[:, 1], xyC[:, 0]))
    dth = wrap(np.degrees(np.arctan2(xy[:, 1], xy[:, 0])) - thC[day])
    order = np.argsort(wrap(thC - thC[0]) % 360.0)
    steps = np.abs(wrap(np.diff(np.append(thC[order], thC[order][0]))))
    rows = [(f"{h:02d}:00", clock_word(h), "clock", float(h)) for h in KEEP_CLOCK]
    rows += [(w, w, "word", WORD_HOURS[w]) for w in KEEP_WORDS]
    rows += [(p, p, "placebo", np.nan) for p in KEEP_PLACEBO]
    stat = {}
    for key, _, _, _ in rows:
        per_day = np.array([dth[(lab == key) & (day == d)].mean() for d in range(7)])
        stat[key] = (float(per_day.mean()), float(per_day.std(ddof=1) / np.sqrt(7)))
    return dict(plane=plane, e3=e3, pcs=pcs, mu=mu, cl=cl, dth=dth, rows=rows, stat=stat,
                step_min=float(steps.min()), step_max=float(steps.max()))


def pca_table(pcs, n=5):
    return [dict(pc=k + 1, var_share=float(pcs["varf"][k]), hour_eta2=float(pcs["h_eta"][k]),
                 weekday_eta2=float(pcs["d_eta"][k]),
                 role=("weekday plane" if k in pcs["plane_idx"] else
                       "hour axis" if k == pcs["e3_idx"] else ""))
            for k in range(n)]


def orthogonality(cap):
    """Share of hour-centroid variation in the weekday plane, and the hour axis's angle to it."""
    A, day, fam, hour = cap["A"], cap["day"], cap["fam"], cap["hour"]
    cl = fam == "clock"
    Cday = np.stack([A[day == d].mean(0) for d in range(7)])
    P = np.linalg.svd(Cday - Cday.mean(0), full_matrices=False)[2][:2]
    hrs = np.unique(hour[cl])
    Chour = np.stack([A[cl & (hour == h)].mean(0) for h in hrs])
    Chour_c = Chour - Chour.mean(0)
    tot = (Chour_c ** 2).sum()
    in_plane = ((Chour_c @ P.T) ** 2).sum()
    e_h = np.linalg.svd(Chour_c, full_matrices=False)[2][0]
    cos = float(np.linalg.norm(P @ e_h))
    Cd = Cday - Cday.mean(0)
    return dict(hour_variation_in_weekday_plane=float(in_plane / tot),
                hour_axis_angle_to_plane_deg=float(np.degrees(np.arccos(min(cos, 1.0)))),
                weekday_variation_along_hour_axis=float(((Cd @ e_h) ** 2).sum() / (Cd ** 2).sum()),
                ring_radius_days=float(np.linalg.norm(Cd, axis=1).mean()),
                ring_radius_hours=float(np.linalg.norm(Chour_c, axis=1).mean()))


def inplane_shifts(cap):
    """In-plane shift of each prompt from its day's centroid, in day steps (360/7 deg), positive toward the next day."""
    from scipy.stats import spearmanr
    A, day, fam, lab, hour = cap["A"], cap["day"], cap["fam"], cap["lab"], cap["hour"]
    Cday = np.stack([A[day == d].mean(0) for d in range(7)])
    mu = A.mean(0)
    P = np.linalg.svd(Cday - Cday.mean(0), full_matrices=False)[2][:2]
    P = orient_plane(P, Cday, mu)
    xyC = (Cday - mu) @ P.T
    thC = np.degrees(np.arctan2(xyC[:, 1], xyC[:, 0]))
    xy = (A - mu) @ P.T
    shift = wrap(np.degrees(np.arctan2(xy[:, 1], xy[:, 0])) - thC[day]) / (360.0 / 7)
    clock = np.isin(fam, ["clock", "clock_midnight"])
    per_hour = {int(h): dict(mean_day_steps=float(shift[clock & (hour == h)].mean()),
                             sd_over_days=float(shift[clock & (hour == h)].std()))
                for h in np.unique(hour[clock])}
    words = {w: float(shift[lab == w].mean()) for w in KEEP_WORDS}
    early = clock & (hour >= 1) & (hour <= 6)
    late = clock & (hour >= 18) & (hour <= 23)
    rho, p = spearmanr(hour[clock], shift[clock])
    per_day = [float(shift[late & (day == d)].mean() - shift[early & (day == d)].mean())
               for d in range(7)]
    return dict(per_hour=per_hour, words=words,
                early_1_6am_mean=float(shift[early].mean()), late_6_11pm_mean=float(shift[late].mean()),
                late_minus_early_day_steps=float(shift[late].mean() - shift[early].mean()),
                late_minus_early_degrees=float((shift[late].mean() - shift[early].mean()) * 360 / 7),
                spearman_hour_vs_shift=dict(rho=float(rho), p=float(p)),
                late_minus_early_per_day=per_day,
                radius_clock_over_centroids=float(np.linalg.norm(xy[clock], axis=1).mean()
                                                  / np.linalg.norm(xyC, axis=1).mean()))


def ladder_numbers(ladder_dir):
    import pandas as pd
    S = pd.read_csv(os.path.join(ladder_dir, "summary.csv"))
    R = pd.read_csv(os.path.join(ladder_dir, "rows.csv"))
    table = [dict(arm=r.arm, alpha=float(r.alpha), mean_h=float(r["mean"]), lo=float(r.lo),
                  hi=float(r.hi), displacement=float(r.disp), size_err=float(r.size_err))
             for _, r in S.sort_values(["arm", "alpha"]).iterrows()]
    cons = {}
    for arm in ("manifold", "late_perp", "rand_perp"):
        for a in (-0.5, 0.5):
            s = R[(R.arm == arm) & np.isclose(R.alpha, a)].groupby("carrier").d_cm.mean()
            cons[f"{arm}@{a:+.1f}"] = dict(carriers_positive=int((s > 0).sum()), n=int(len(s)),
                                          min_h=float(s.min()), max_h=float(s.max()))
    return dict(table=table, per_carrier=cons, n_carriers=int(R.carrier.nunique()),
                baseline_hour_mean=float(R.cm0.mean()), baseline_hour_min=float(R.cm0.min()),
                baseline_hour_max=float(R.cm0.max()))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=data_dir("fig2_timeofday", create=False))
    ap.add_argument("--layer", type=int, default=28)
    ap.add_argument("--out", default=RESULTS_JSON)
    args = ap.parse_args()

    cap = load_capture(os.path.join(args.data_dir, f"timeofday_L{args.layer}.npz"))
    G = plate_geometry(cap)
    out = dict(n_prompts=int(len(cap["A"])), n_time_prompts=int(G["pcs"]["n_fit"]),
               pca=pca_table(G["pcs"]), orthogonality=orthogonality(cap),
               inplane=inplane_shifts(cap),
               panel_b=dict(adjacent_day_step_deg=[G["step_min"], G["step_max"]],
                            rows={k: dict(mean_deg=m, se_deg=s) for k, (m, s) in G["stat"].items()}))
    ladder_dir = os.path.join(args.data_dir, "ladder")
    if os.path.exists(os.path.join(ladder_dir, "summary.csv")):
        out["ladder"] = ladder_numbers(ladder_dir)

    print(f"PCA on the {out['n_time_prompts']} time prompts (variance share; eta^2 by hour / weekday)")
    for r in out["pca"]:
        print(f"  PC{r['pc']}  {100 * r['var_share']:5.1f}%   hour {r['hour_eta2']:.2f}   "
              f"weekday {r['weekday_eta2']:.2f}   {r['role']}")
    o = out["orthogonality"]
    print(f"hour-centroid variation in the weekday plane: {100 * o['hour_variation_in_weekday_plane']:.1f}%; "
          f"leading hour axis is {o['hour_axis_angle_to_plane_deg']:.1f} deg from the plane")
    i = out["inplane"]
    print(f"in-plane shift (day steps): 1-6am {i['early_1_6am_mean']:+.3f}, 6-11pm "
          f"{i['late_6_11pm_mean']:+.3f}, late - early {i['late_minus_early_day_steps']:.3f} "
          f"({i['late_minus_early_degrees']:.1f} deg); Spearman rho {i['spearman_hour_vs_shift']['rho']:.2f}")
    print("  words: " + ", ".join(f"{w} {v:+.3f}" for w, v in i["words"].items()))
    print(f"panel B adjacent-day band {G['step_min']:.1f}-{G['step_max']:.1f} deg; largest "
          f"modifier rotation {max(abs(m) for m, _ in G['stat'].values()):.1f} deg")
    if "ladder" in out:
        print("ladder (mean shift in predicted hour, 95% CI over carriers):")
        for r in out["ladder"]["table"]:
            print(f"  {r['arm']:<10} {r['alpha']:+.2f}  {r['mean_h']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]"
                  f"  ||D|| {r['displacement']:.1f}  size err {100 * r['size_err']:.1f}%")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    dump_json(out, args.out, indent=1)
    print(f"[saved] {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
