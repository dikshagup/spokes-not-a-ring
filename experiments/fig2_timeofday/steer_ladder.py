#!/usr/bin/env python
"""Figure 2C stage 3: dose-response of the predicted hour to steering, for three size-matched arms.

Writes <data>/ladder/rows.csv and <data>/ladder/summary.csv.

    python experiments/fig2_timeofday/steer_ladder.py            # ~2.5 h on a GPU
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))
from weekday_manifold.model import load_model  # noqa: E402
from weekday_manifold.manifold.days import DAYS, N_DAYS  # noqa: E402
from weekday_manifold.manifold.steering import fit_steer_spline, manifold_velocity  # noqa: E402
from weekday_manifold.utils import data_dir  # noqa: E402
from carrier_screen import prompt  # noqa: E402
from hour_readout import TimeReadout, circ_diff  # noqa: E402
from ring_frames import fit_pos_frames, clean_state, spline_set_hook, clamp_dir_hook  # noqa: E402

ARMS = ("manifold", "late_perp", "rand_perp")
DOSES = (-0.5, -0.3, -0.15, 0.0, 0.15, 0.3, 0.5)


def summarise(rows_csv, summary_csv, n_boot=2000, seed=0):
    """Mean shift per (arm, dose) with a 95% bootstrap interval over carriers."""
    import pandas as pd
    R = pd.read_csv(rows_csv)
    brng = np.random.default_rng(seed)
    summ = []
    for (arm, alpha), g in R.groupby(["arm", "alpha"]):
        cm = g.groupby("carrier").d_cm.mean().values
        b = np.array([brng.choice(cm, len(cm), True).mean() for _ in range(n_boot)])
        summ.append(dict(arm=arm, alpha=alpha, n=len(g), mean=g.d_cm.mean(),
                         lo=np.percentile(b, 2.5), hi=np.percentile(b, 97.5),
                         disp=g.displacement.mean(),
                         size_err=(g.displacement / g.target - 1).abs().max()))
    S = pd.DataFrame(summ).sort_values(["alpha", "arm"])
    S.to_csv(summary_csv, index=False)
    return S


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--start-layer", type=int, default=2)
    ap.add_argument("--end-layer", type=int, default=28)
    ap.add_argument("--alphas", type=float, nargs="+", default=list(DOSES),
                    help="signed fractions of a day step; the off-ring arms take the sign")
    ap.add_argument("--n-cells", type=int, default=0, help="stop after this many cells")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-dir", default=data_dir("fig2_timeofday"))
    args = ap.parse_args()

    meta = json.load(open(os.path.join(args.data_dir, "meta.json")))
    keep = meta["carriers_kept"]
    LAYERS = list(range(args.start_layer, args.end_layer + 1))
    OFF = [0]
    rng = np.random.default_rng(args.seed)

    model = load_model(args.model, device="cuda", fold_ln=True, center_writing_weights=True,
                       center_unembed=True, dtype="bfloat16")
    dev = model.cfg.device
    frames, acts, days = fit_pos_frames(model, prompt, keep, LAYERS, OFF)

    # steer only layers whose seven neutral day centroids are in cyclic order
    SITES = [(L, o) for L in LAYERS for o in OFF
             if frames[(L, o)].heptagon_check()["in_cyclic_order"]]
    print(f"[gate] {len(SITES)} of {len(LAYERS) * len(OFF)} sites in cyclic order", flush=True)
    SPL = {k: fit_steer_spline(frames[k].centroids) for k in SITES}
    D_LATE = {}
    for k in SITES:
        A_e, A_l = acts[("early",) + k], acts[("late",) + k]
        for d in range(N_DAYS):
            D_LATE[k + (d,)] = A_l[days == d].mean(0) - A_e[days == d].mean(0)

    TR = TimeReadout(model)
    cells = [(c, d) for c in keep for d in range(N_DAYS)]
    if args.n_cells:
        cells = cells[:args.n_cells]

    def wrap(base, rec, key, pos):
        def fn(resid, hook):
            before = resid[0, pos, :].detach().float().clone()
            out = base(resid, hook)
            rec[key] = float((out[0, pos].float() - before).norm())
            return out
        return fn

    def ring_hooks(u_tgt, P, rec):
        return [(f"blocks.{L}.hook_resid_post",
                 wrap(spline_set_hook(SPL[(L, o)], u_tgt[(L, o)], [P + o], dev), rec, (L, o), P + o))
                for L, o in SITES]

    def dir_hooks(U, dose, ref, P, rec):
        return [(f"blocks.{L}.hook_resid_post",
                 wrap(clamp_dir_hook(U[(L, o)], float(ref[(L, o)] @ U[(L, o)]) + dose[(L, o)],
                                     [P + o], dev), rec, (L, o), P + o))
                for L, o in SITES]

    rows, t0 = [], time.time()
    print(f"[ladder] {len(cells)} cells x {len(args.alphas)} doses x 3 arms on {len(SITES)} sites",
          flush=True)
    for i, (c, d) in enumerate(cells):
        text = prompt(c, "", DAYS[d])
        ref, P = clean_state(model, text, DAYS[d], LAYERS, OFF)
        s0 = TR.score(text)
        nxt = (d + 1) % N_DAYS

        u_late, u_rand = {}, {}
        for k in SITES:
            tng = manifold_velocity(SPL[k], d, nxt, 0.0, param="arclength")
            tng = tng / np.linalg.norm(tng)
            v = D_LATE[k + (d,)]
            v = v - (v @ tng) * tng
            u_late[k] = v / np.linalg.norm(v)
            r = rng.standard_normal(v.shape)
            r = r - (r @ tng) * tng
            u_rand[k] = r / np.linalg.norm(r)

        for alpha in args.alphas:
            # target: the foot point moved |alpha| of a day step along the ring, by arc length
            u_tgt = {}
            for k in SITES:
                spl = SPL[k]
                u0 = float(spl.inverse(ref[k]))
                step = frames[k].orient * (1.0 if alpha >= 0 else -1.0) / N_DAYS
                us = u0 + np.linspace(0.0, 1.0, 600) * step
                pts = spl.forward(us)
                s_ = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
                u_tgt[k] = float(np.interp(abs(alpha) * s_[-1], s_, us))
            rec = {}
            s_walk = TR.score(text, ring_hooks(u_tgt, P, rec))
            D_t = float(sum(rec.values()))
            out = {"manifold": (s_walk, D_t)}
            # off-ring arms use the ring walk's displacement size at each site
            sgn = 1.0 if alpha >= 0 else -1.0
            for name, U in (("late_perp", u_late), ("rand_perp", u_rand)):
                dose = {k: sgn * rec[k] for k in SITES}
                rec2 = {}
                s_arm = TR.score(text, dir_hooks(U, dose, ref, P, rec2))
                out[name] = (s_arm, float(sum(rec2.values())))
            for arm, (s, D) in out.items():
                rows.append(dict(carrier=c, day=d, alpha=alpha, arm=arm,
                                 d_cm=circ_diff(s["circ_mean"], s0["circ_mean"]),
                                 cm=s["circ_mean"], cm0=s0["circ_mean"], conc=s["conc"],
                                 captured=s["captured"], ambiguous=s["ambiguous"],
                                 displacement=D, target=D_t,
                                 cm_am=TR.bounds(s)["am"], cm_pm=TR.bounds(s)["pm"]))
        if (i + 1) % 7 == 0:
            el = time.time() - t0
            print(f"   {i + 1}/{len(cells)} cells  {el / (i + 1):.1f}s/cell  "
                  f"eta {(len(cells) - i - 1) * el / (i + 1) / 60:.1f} min", flush=True)

    import pandas as pd
    outdir = os.path.join(args.data_dir, "ladder")
    os.makedirs(outdir, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(outdir, "rows.csv"), index=False)
    S = summarise(os.path.join(outdir, "rows.csv"), os.path.join(outdir, "summary.csv"))
    print(f"\n{'alpha':>6} {'||D||':>7}   " + "   ".join(f"{a:>24}" for a in ARMS))
    for alpha in args.alphas:
        r = {a: S[(S.arm == a) & (S.alpha == alpha)].iloc[0] for a in ARMS}
        print(f"{alpha:6.2f} {r['manifold'].disp:7.1f}   " + "   ".join(
            f"{r[a]['mean']:+.3f} [{r[a].lo:+.3f},{r[a].hi:+.3f}]" for a in ARMS))
    print(f"\n[saved] {outdir}   ({(time.time() - t0) / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
