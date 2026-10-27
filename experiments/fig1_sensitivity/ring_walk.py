"""Walk the weekday ring by finite differences and record how far the readout moves.

    python experiments/fig1_sensitivity/ring_walk.py --formulation mention_early \\
        --readout-pos 8 --out <data>/ring_walk.npz                  # figure 1, ~2-3 h GPU
    python experiments/fig1_sensitivity/ring_walk.py --formulation interrogative \\
        --out <data>/ring_walk.npz                                  # figure 6, ~3-4 h GPU

Requires float16; in bfloat16 the rounding error is larger than the step size.
"""
from __future__ import annotations

import argparse
import collections
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from weekday_manifold.manifold.behavior import hellinger_distance, restrict_to_concept  # noqa: E402
from weekday_manifold.manifold.config import ManifoldConfig  # noqa: E402
from weekday_manifold.manifold.days import DAYS, N_DAYS, build_prompts, days_token_ids  # noqa: E402
from weekday_manifold.manifold.probes import build_mention_early  # noqa: E402
from weekday_manifold.manifold.steering import (fit_steer_spline, ring_report,  # noqa: E402
                                                template_demeaned_centroids)
from weekday_manifold.plateau.model import load_plateau_model  # noqa: E402
from weekday_manifold.utils import set_seed  # noqa: E402

FAMILIES = {
    "mention_early": dict(build=lambda: build_mention_early("thisday"), sites=["weekday"],
                          readout_group="input", n_prompts=70),
    "interrogative": dict(build=lambda: build_prompts("interrogative"),
                          sites=["weekday", "answer"], readout_group="answer", n_prompts=0),
}


def global_arclength(spline, n_u, n_days=N_DAYS, dense=4000):
    """Return a u grid evenly spaced in arc length around the loop, with each day knot on it, and the knot indices."""
    ug = np.linspace(0.0, 1.0, dense, endpoint=False)
    P = spline.forward(ug)
    seg = np.linalg.norm(np.diff(np.r_[P, P[:1]], axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    ki = [int(np.argmin(np.abs(ug - k / n_days))) for k in range(n_days)]
    arc = np.array([(cum[ki[j + 1]] if j < n_days - 1 else cum[dense]) - cum[ki[j]]
                    for j in range(n_days)])
    total = n_u - 1
    exact = arc / arc.sum() * total
    cnt = np.floor(exact).astype(int)
    for j in np.argsort(-(exact - cnt))[:total - int(cnt.sum())]:
        cnt[j] += 1
    assert (cnt > 0).all() and cnt.sum() == total, "arc allocation is degenerate"
    uu, kidx = [], []
    for j in range(n_days):
        kidx.append(len(uu))
        targets = cum[ki[j]] + np.arange(cnt[j]) * arc[j] / cnt[j]
        got = np.interp(targets, cum, np.r_[ug, 1.0])
        # first sample exactly on the knot; the alpha = 0 identity check uses it
        got[0] = j / n_days
        uu.extend(got)
    # last sample at u = 1 closes the loop
    uu = np.array(uu + [1.0])
    assert len(uu) == n_u, f"grid is {len(uu)} long, expected {n_u}"
    assert np.allclose(uu[kidx], np.arange(n_days) / n_days, atol=1e-6), (
        "a day knot is not sampled exactly; the identity write would be off the grid")
    return uu, np.array(kidx)


def knot_arclength(spline, n=20001):
    """Arc-length fraction around the loop at each day knot."""
    u = np.linspace(0.0, 1.0, n)
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(spline.forward(u), axis=0),
                                                        axis=1))])
    return np.interp(np.arange(N_DAYS) / N_DAYS, u, s / s[-1])


def spread_pos(A_layer, day_ids, template_ids):
    """Token position where the day centroids are most spread out (the weekday-mention position)."""
    spreads = []
    for p in range(A_layer.shape[1]):
        Cp = template_demeaned_centroids(A_layer[:, p, :], day_ids, template_ids, N_DAYS)
        spreads.append(float(np.linalg.norm(Cp - Cp.mean(0), axis=1).mean()))
    return int(np.argmax(spreads))


def balanced_prompts(z_in, n):
    """Prompt indices taken round-robin over the days; n <= 0 takes every prompt."""
    if n <= 0:
        return list(range(len(z_in)))
    by_day = {k: [i for i in range(len(z_in)) if int(z_in[i]) == k] for k in range(N_DAYS)}
    out, rank = [], 0
    while len(out) < n:
        grew = False
        for k in range(N_DAYS):
            if rank < len(by_day[k]) and len(out) < n:
                out.append(by_day[k][rank]); grew = True
        if not grew:
            break
        rank += 1
    assert len(out) == min(n, len(z_in)), f"asked for {n} prompts, the family has {len(z_in)}"
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--formulation", choices=sorted(FAMILIES), default="mention_early")
    ap.add_argument("--layers", default="2,6,10,14,18,22,26,28,30,31",
                    help="the steer layers")
    ap.add_argument("--readout-layer", type=int, default=28)
    ap.add_argument("--readout-pos", type=int, default=-1,
                    help="8 (the full stop) for the mention family; -1 (the final ':') "
                         "for the interrogative family")
    ap.add_argument("--n-u", type=int, default=141,
                    help="ring samples; 7m+1 so the day knots are on the grid")
    ap.add_argument("--chunk", type=int, default=48, help="forward-pass batch size")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dump-rings", default=None,
                    help="write the day CENTROIDS at every (layer, site) to this npz and "
                         "exit before the sweep (figure 6, panel A)")
    args = ap.parse_args()

    fam = FAMILIES[args.formulation]
    sites, readout_group = fam["sites"], fam["readout_group"]
    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    RO = args.readout_layer
    # for layers >= RO only the output is recorded; residual columns are NaN
    late = [L for L in layers if L >= RO]
    if late:
        print(f"[walk] layers {late} are at or after the readout (L{RO}): OUTPUT ONLY -- "
              f"their residual columns are NaN by construction, not by measurement",
              flush=True)

    cfg = ManifoldConfig(model_name=args.model, dtype=args.dtype,
                         formulation=args.formulation, layer_index=RO)
    set_seed(cfg.seed)
    model = load_plateau_model(cfg)
    assert all(0 <= L < model.cfg.n_layers for L in layers), (
        f"patch layers must be valid blocks 0..{model.cfg.n_layers - 1}; got {layers}")
    dev, dt = model.cfg.device, model.W_E.dtype

    mention = args.formulation == "mention_early"
    specs = fam["build"]()
    texts = [sp.text for sp in specs]
    tn = sorted({sp.meta["template"] for sp in specs})
    ti = {t: i for i, t in enumerate(tn)}
    template_ids = np.array([ti[sp.meta["template"]] for sp in specs])
    z_in = np.array([int(sp.meta["z"]) for sp in specs])
    z_ans = z_in if mention else np.array([int(sp.answer_day) for sp in specs])

    def labels_at(pos, dpos):
        return z_in if pos == dpos else z_ans

    toks_all = torch.cat([model.to_tokens(t, prepend_bos=cfg.prepend_bos) for t in texts], 0)
    seq_len = toks_all.shape[1]
    ro_pos = args.readout_pos % seq_len

    dtoks = days_token_ids(model)
    multi = [DAYS[k] for k in range(N_DAYS) if len(dtoks[k]) != 1]
    assert not multi, (f"weekdays are not single tokens for this tokenizer: {multi} -- the "
                       f"first-token restriction would misattribute their tails to 'other'")
    concept_ids = [int(dtoks[k][0]) for k in range(N_DAYS)]

    # ------------------------------------------------------------ unsteered capture pass
    cap_layers = sorted(set(layers) | {RO})
    store = {}

    def mk(L):
        def fn(act, hook):
            store[L] = act.detach().float().cpu().numpy()
            return act
        return fn
    with torch.no_grad():
        model.run_with_hooks(toks_all, return_type=None,
                             fwd_hooks=[(f"blocks.{L}.hook_resid_post", mk(L))
                                        for L in cap_layers])
    A = {L: store[L] for L in cap_layers}                        # [N, seq, d] each
    d = A[RO].shape[2]

    # unsteered reference at the sweep's batch size, since half-precision results depend on batch shape
    d0 = []
    for c0 in range(0, toks_all.shape[0], args.chunk):
        ch = toks_all[c0:c0 + args.chunk]
        pad = np.minimum(np.arange(args.chunk), ch.shape[0] - 1)
        with torch.no_grad():
            lg0 = model(ch[pad], return_type="logits")[:, ro_pos, :]
            probs0 = torch.softmax(lg0.float(), dim=-1).cpu().numpy()
        d0.append(restrict_to_concept(probs0, concept_ids)[:ch.shape[0]])
    dist0 = np.concatenate(d0, 0)                                # [N, 8]
    acc0 = float((dist0[:, :N_DAYS].argmax(1) == z_ans).mean())
    print(f"[walk] unpatched: argmax-over-weekdays accuracy={acc0:.3f} (chance "
          f"{1 / N_DAYS:.3f}); mass on the seven days="
          f"{dist0[:, :N_DAYS].sum(1).mean():.3f}", flush=True)

    # ---------------------------------------------------------------------- fit the rings
    day_pos = {L: spread_pos(A[L], z_in, template_ids) for L in layers}
    print(f"[walk] weekday-mention position by layer: {day_pos}; readout position {ro_pos} "
          f"of {seq_len}", flush=True)

    scale_store = {}

    def _scale_hook(L_):
        def fn(act, hook):
            a = act[:, [day_pos[layers[0]], ro_pos], :].detach().float().cpu().numpy()
            scale_store[L_] = np.linalg.norm(a, axis=2).mean(0)      # [2]
            return act
        return fn
    with torch.no_grad():
        model.run_with_hooks(toks_all, return_type=None, fwd_hooks=[
            (f"blocks.{L_}.hook_resid_post", _scale_hook(L_))
            for L_ in range(model.cfg.n_layers)])

    rings = {}                       # (layer, pos) -> (C, spline, diagnostics, uu, kidx)
    for L in layers:
        for pos in sorted({day_pos[L], ro_pos}):
            lab = labels_at(pos, day_pos[L])
            which = "input" if lab is z_in else "answer"
            C = template_demeaned_centroids(A[L][:, pos, :], lab, template_ids, N_DAYS)
            sp = fit_steer_spline(C)
            dg = ring_report(f"patch L{L}@pos{pos} ({which} day)", C, A[L][:, pos, :],
                             lab, template_ids, sp)
            uu, kidx = global_arclength(sp, args.n_u)
            st = np.linalg.norm(np.diff(sp.forward(uu), axis=0), axis=1)
            print(f"[walk] ring L{L}@pos{pos}: input stride min={st.min():.4f} "
                  f"max={st.max():.4f} ratio={st.max() / st.min():.2f} | day knots at "
                  f"arc-length fraction {np.round(knot_arclength(sp), 3).tolist()}",
                  flush=True)
            rings[(L, pos)] = (C, sp, dg, uu, kidx)

    ro_labels = z_ans if readout_group == "answer" else z_in
    C_ro = template_demeaned_centroids(A[RO][:, ro_pos, :], ro_labels, template_ids, N_DAYS)
    dg_ro = ring_report(f"readout L{RO}@pos{ro_pos} ({readout_group})", C_ro,
                        A[RO][:, ro_pos, :], ro_labels, template_ids, fit_steer_spline(C_ro))

    if args.dump_rings:
        os.makedirs(os.path.dirname(args.dump_rings) or ".", exist_ok=True)
        keys = [(L, pos) for L in layers for pos in sorted({day_pos[L], ro_pos})]
        np.savez(args.dump_rings,
                 layers=np.array(layers), readout_layer=RO, ro_pos=ro_pos,
                 keys=np.array([f"L{L}@{p}" for L, p in keys]),
                 which=np.array(["input" if labels_at(p, day_pos[L]) is z_in else "answer"
                                 for L, p in keys]),
                 centroids=np.stack([rings[k][0] for k in keys]).astype(np.float32),
                 recovery=np.array([rings[k][2]["recovery"] for k in keys]),
                 radius=np.stack([rings[k][2]["radius"] for k in keys]),
                 centroids_ro=C_ro.astype(np.float32), recovery_ro=dg_ro["recovery"],
                 radius_ro=dg_ro["radius"], model=cfg.model_name,
                 resid_norm=np.array([float(np.linalg.norm(A[L][:, pos, :], axis=1).mean())
                                      for L, pos in keys]))
        print(f"[walk] wrote rings -> {args.dump_rings}; exiting before the sweep", flush=True)
        return

    # ------------------------------------------------------------------------------ sweep
    us = np.linspace(0.0, 1.0, args.n_u)
    prompts = balanced_prompts(z_in, fam["n_prompts"])
    P = len(prompts)

    def identity_index(L, pos, pidx):
        """Grid index of the prompt's own day knot on the patched ring (the alpha = 0 point)."""
        return int(rings[(L, pos)][4][int(z_in[pidx])])

    def sweep(tokens, patch_vals, Lp):
        """Run forward passes with `patch_vals` (position -> [n_u, d]) written in at layer `Lp`."""
        n = next(iter(patch_vals.values())).shape[0]
        res, dists = [], []
        for c0 in range(0, n, args.chunk):
            # pad every chunk to the same batch size so the alpha = 0 identity check holds
            take = min(args.chunk, n - c0)
            pad = np.minimum(np.arange(args.chunk), take - 1)     # repeat the last row
            vals = {p: torch.tensor(v[c0:c0 + take][pad], device=dev, dtype=dt)
                    for p, v in patch_vals.items()}
            b = args.chunk
            rec = {}

            def patch(resid, hook, vals=vals):
                resid = resid.clone()
                for p, v in vals.items():
                    resid[:, p, :] = v
                return resid

            def read(act, hook, rec=rec):
                li_ = int(hook.name.split(".")[1])
                rec[li_] = act[:, [day_pos[Lp], ro_pos], :].detach().float().cpu().numpy()
                return act
            with torch.no_grad():
                lg = model.run_with_hooks(
                    tokens.expand(b, -1), return_type="logits",
                    fwd_hooks=[(f"blocks.{Lp}.hook_resid_post", patch)]
                    + [(f"blocks.{L_}.hook_resid_post", read)
                       for L_ in range(model.cfg.n_layers)])[:, ro_pos, :]
                pr = torch.softmax(lg.float(), dim=-1).cpu().numpy()
            res.append(np.stack([rec[L_][:take] for L_ in range(model.cfg.n_layers)], 1))
            dists.append(restrict_to_concept(pr, concept_ids)[:take])
        # [n_u, n_layers, 2, d]; position 0 is the weekday token, 1 the readout
        return np.concatenate(res, 0), np.concatenate(dists, 0)

    S, Lc, U, nL = len(sites), len(layers), args.n_u, model.cfg.n_layers
    dists_all = np.zeros((S, Lc, P, U, N_DAYS + 1), dtype=np.float32)
    stride = np.zeros((S, Lc, U - 1))       # input step length; same for every prompt
    # [site, steer layer, prompt, u, depth, position]
    d_layer = np.zeros((S, Lc, P, U, nL, 2), dtype=np.float32)
    step_layer = np.zeros((S, Lc, P, U - 1, nL, 2), dtype=np.float32)
    resid_scale = np.stack([scale_store[L_] for L_ in range(nL)]).astype(np.float32)
    h_ident = np.zeros((S, Lc, P))          # Hellinger(sweep alpha=0, unpatched forward)
    w_ident = np.zeros((S, Lc, P))          # max |written - recorded| at alpha=0
    t0 = time.time()
    for si, site in enumerate(sites):
        for li, L in enumerate(layers):
            pos_list = {day_pos[L]} if site == "weekday" else {ro_pos}
            for pi, pidx in enumerate(prompts):
                tokens = model.to_tokens(texts[pidx], prepend_bos=cfg.prepend_bos)
                s, s_ans = int(z_in[pidx]), int(z_ans[pidx])
                # each position uses its own day and ring; the answer ring is shifted by the day offset
                du = ((s_ans - s) % N_DAYS) / N_DAYS
                patch_vals = {}
                for pos in sorted(pos_list):
                    C, sp, _, uu, _ = rings[(L, pos)]
                    is_answer_slot = (pos == ro_pos and pos != day_pos[L])
                    ref = s_ans if is_answer_slot else s
                    shift = du if is_answer_slot else 0.0
                    patch_vals[pos] = ((A[L][pidx, pos] - C[ref])[None, :]
                                       + sp.forward((uu + shift) % 1.0))
                res_all, dd = sweep(tokens, patch_vals, L)      # [U, nL, 2, d]

                joint_all = np.concatenate([patch_vals[p] for p in sorted(patch_vals)],
                                           axis=1)
                stride[si, li] = np.linalg.norm(np.diff(joint_all, axis=0), axis=1)
                i0 = identity_index(L, sorted(pos_list)[0], pidx)
                dists_all[si, li, pi] = dd
                d_layer[si, li, pi] = np.linalg.norm(res_all - res_all[i0][None], axis=3)
                step_layer[si, li, pi] = np.linalg.norm(np.diff(res_all, axis=0), axis=3)
                if L >= RO:
                    h_ident[si, li, pi] = np.nan
                    w_ident[si, li, pi] = np.nan
                    continue

                # at the other position, depths <= L must equal the unpatched run exactly
                if ro_pos not in pos_list:
                    assert np.abs(d_layer[si, li, pi, :, :L + 1, 1]).max() == 0.0, (
                        f"steer at L{L} moved the readout at or before its own depth")

                # --- alpha = 0 identity checks ------------------------------------------
                # (1) the value written at the own knot matches the recorded activation within one ulp
                w = max(float((torch.tensor(v[i0], dtype=dt)
                               - torch.tensor(A[L][pidx, p], dtype=dt)).abs().max())
                        for p, v in patch_vals.items())
                tol = 2.0 ** -7 * max(float(np.abs(A[L][pidx, p]).max()) for p in patch_vals)
                # (2) the output distribution matches the unpatched forward
                h = hellinger_distance(dd[i0], dist0[pidx])
                w_ident[si, li, pi], h_ident[si, li, pi] = w, h
                assert w <= tol, (f"{site} L{L} {DAYS[s]}: the alpha=0 write differs from the "
                                  f"recorded activation by {w:g} (> one ulp {tol:g})")
                assert h < 0.05, (f"{site} L{L} {DAYS[s]}: the alpha=0 output distribution is "
                                  f"{h:g} from an unpatched forward, not 0")
            print(f"[walk] {site:8s} L{L:<3d} done  ({time.time() - t0:.0f}s)  "
                  f"identity: write max|diff|={np.nanmax(w_ident[si, li]):g} "
                  f"H(alpha=0, unpatched)={np.nanmax(h_ident[si, li]):.2e}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    ring_keys = [(L, pos) for L in layers for pos in sorted({day_pos[L], ro_pos})]
    np.savez(
        args.out, us=us, layers=np.array(layers), sites=np.array(sites),
        dists=dists_all, stride_in=stride, d_layer=d_layer, step_layer=step_layer,
        resid_scale=resid_scale,
        knot_index=np.stack([rings[(L, day_pos[L])][4] for L in layers]),
        h_identity=h_ident, w_identity=w_ident,
        prompts=np.array(prompts), prompt_days=np.array([int(z_in[p]) for p in prompts]),
        prompt_texts=np.array([texts[p] for p in prompts]),
        day_pos=np.array([day_pos[L] for L in layers]), ro_pos=ro_pos, seq_len=seq_len,
        readout_layer=RO, d_model=d, n_prompts=P, concept_ids=np.array(concept_ids),
        clean_acc=acc0, dist_unpatched=dist0.astype(np.float32),
        formulation=args.formulation, readout_group=readout_group, model=cfg.model_name,
        ring_sites=np.array([f"L{L}@{pos}" for L, pos in ring_keys]),
        ring_radius=np.array([rings[k][2]["radius"] for k in ring_keys]),
        ring_recovery=np.array([rings[k][2]["recovery"] for k in ring_keys]),
        ring_knot_alpha=np.array([knot_arclength(rings[k][1]) for k in ring_keys]),
        ring_radius_ro=dg_ro["radius"], ring_recovery_ro=dg_ro["recovery"])
    print(f"[walk] wrote {args.out}  ({time.time() - t0:.0f}s of forward passes)", flush=True)


if __name__ == "__main__":
    main()
