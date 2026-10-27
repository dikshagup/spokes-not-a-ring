"""Compute the readout Jacobian over the day disc with forward-mode autodiff (figures 1, 5, 6).

    python experiments/fig1_sensitivity/jacobian_field.py --out <data>/jacobian_field.npz
        [--model mistralai/Mistral-7B-v0.1 --require-seq-len 12]            # figure 5
        [--model gpt2-xl --dtype float32 --readout-layer 42]                # figure 5
        [--formulation interrogative --readout-pos -1 --n-prompts 49]       # figure 6
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from weekday_manifold.manifold.config import ManifoldConfig  # noqa: E402
from weekday_manifold.manifold.days import DAYS, N_DAYS, build_prompts  # noqa: E402
from weekday_manifold.manifold.probes import build_mention_early  # noqa: E402
from weekday_manifold.manifold.steering import fit_steer_spline, template_demeaned_centroids  # noqa: E402
from weekday_manifold.plateau.model import load_plateau_model  # noqa: E402
from weekday_manifold.utils import set_seed  # noqa: E402


def polar_point(spline, mu, u, r):
    """Return p(u, r) = mu + r * (sigma(u) - mu) and the unit tangent and radial directions."""
    P = spline.forward(u)                      # [n, d]
    T = spline.derivative(u)                   # [n, d]
    rad = P - mu[None, :]
    pt = mu[None, :] + r[:, None] * rad
    vt = T / np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-30)
    vr = rad / np.maximum(np.linalg.norm(rad, axis=1, keepdims=True), 1e-30)
    return pt, vt, vr


def balanced(day_ids, n):
    """Indices of n background prompts, taken round-robin over the days."""
    by_day = {k: [i for i in range(len(day_ids)) if day_ids[i] == k] for k in range(N_DAYS)}
    sel, rank = [], 0
    while len(sel) < n:
        grew = False
        for k in range(N_DAYS):
            if rank < len(by_day[k]) and len(sel) < n:
                sel.append(by_day[k][rank]); grew = True
        if not grew:
            break
        rank += 1
    return sel


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--formulation", choices=("mention_early", "interrogative"),
                    default="mention_early")
    ap.add_argument("--patch-layer", type=int, default=2)
    ap.add_argument("--readout-layer", type=int, default=28,
                    help="87.5% of the stack: 28 of 32 for Llama and Mistral, 42 of 48 "
                         "for GPT-2 XL")
    ap.add_argument("--readout-pos", type=int, default=8,
                    help="the full stop of the mention prompts; pass -1 (the final ':') "
                         "for the interrogative family")
    ap.add_argument("--n-u", type=int, default=56, help="samples around the ring")
    ap.add_argument("--n-r", type=int, default=17, help="radial samples")
    ap.add_argument("--r-max", type=float, default=1.6, help="1.0 = the ring itself")
    ap.add_argument("--require-seq-len", type=int, default=0,
                    help="keep only prompts of exactly this tokenised length, dropping "
                         "whole templates so the days stay balanced. The field addresses "
                         "the steer and readout sites by INDEX; Llama's BPE gives every "
                         "mention template 12 tokens with the full stop at 8, Mistral's "
                         "SentencePiece gives 12/13/14/15, so pass 12 there.")
    ap.add_argument("--n-prompts", type=int, default=70,
                    help="background prompts: 70 (10 templates x 7 days) on the mention "
                         "family, all 49 on the interrogative family")
    ap.add_argument("--n-hutch", type=int, default=6, help="Hutchinson probes per background")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = ManifoldConfig(model_name=args.model, dtype=args.dtype,
                         formulation=args.formulation, layer_index=args.readout_layer)
    set_seed(cfg.seed)
    model = load_plateau_model(cfg)
    model.requires_grad_(False)
    dev, dt = model.cfg.device, model.W_E.dtype
    Lp, RO = args.patch_layer, args.readout_layer

    if args.formulation == "mention_early":
        specs = build_mention_early("thisday")
    else:
        specs = build_prompts("interrogative")
    day_ids = np.array([int(sp.meta["z"]) for sp in specs])
    if args.require_seq_len:
        ok = np.array([len(model.to_tokens(sp.text, prepend_bos=cfg.prepend_bos)[0])
                       == args.require_seq_len for sp in specs])
        assert ok.any(), f"no prompt tokenises to {args.require_seq_len} tokens"
        cnt = collections.Counter(int(sp.meta["z"]) for sp, k in zip(specs, ok) if k)
        assert len(set(cnt.values())) == 1, f"length filter unbalanced the days: {dict(cnt)}"
        print(f"[field] length filter: kept {int(ok.sum())}/{len(specs)} prompts at "
              f"{args.require_seq_len} tokens, {cnt[0]} per day", flush=True)
        specs = [sp for sp, k in zip(specs, ok) if k]
        day_ids = day_ids[ok]
    texts = [sp.text for sp in specs]
    template_ids = np.array([int(sp.meta.get("content_id", 0)) for sp in specs])

    # torch.cat raises if the prompts tokenise to different lengths
    toks_all = torch.cat([model.to_tokens(t, prepend_bos=cfg.prepend_bos) for t in texts], 0)
    seq_len = toks_all.shape[1]
    _store = {}

    def _rec_all(act, hook):
        _store["x"] = act.detach().float().cpu().numpy()
        return act
    with torch.no_grad():
        model.run_with_hooks(toks_all, return_type=None,
                             fwd_hooks=[(f"blocks.{Lp}.hook_resid_post", _rec_all)])
    A_all = _store["x"]

    spreads = [float(np.linalg.norm(
        (Cp := template_demeaned_centroids(A_all[:, p, :], day_ids, template_ids, N_DAYS))
        - Cp.mean(0), axis=1).mean()) for p in range(seq_len)]
    day_pos = int(np.argmax(spreads))
    ro_pos = args.readout_pos % seq_len
    tokens = model.to_str_tokens(texts[0], prepend_bos=cfg.prepend_bos)
    print("[field] day-ring radius by position: "
          + ", ".join(f"{p}({tokens[p]!r}):{s:.2f}" for p, s in enumerate(spreads)), flush=True)
    assert tokens[day_pos].strip() in DAYS, f"steer site {tokens[day_pos]!r} is not a weekday"

    C = template_demeaned_centroids(A_all[:, day_pos, :], day_ids, template_ids, N_DAYS)
    spline = fit_steer_spline(C)
    mu = C.mean(0)
    d = C.shape[1]
    ring_r = float(np.linalg.norm(C - mu, axis=1).mean())
    print(f"[field] {args.formulation}: patch L{Lp}@pos{day_pos} -> read L{RO}@pos{ro_pos}; "
          f"ring radius {ring_r:.2f}", flush=True)

    def make_F_batched(tokens):
        pname, rname = f"blocks.{Lp}.hook_resid_post", f"blocks.{RO}.hook_resid_post"

        def F(X):
            store = {}
            B = X.shape[0]

            def patch(resid, hook):
                return torch.cat([resid[:, :day_pos, :], X.view(B, 1, -1),
                                  resid[:, day_pos + 1:, :]], dim=1)

            def rec(act, hook):
                store["x"] = act[:, ro_pos, :]
                return act
            model.run_with_hooks(tokens.expand(B, -1), return_type=None,
                                 fwd_hooks=[(pname, patch), (rname, rec)])
            return store["x"]
        return F

    def jvp_b(F, X_np, U_np):
        x = torch.as_tensor(np.ascontiguousarray(X_np), device=dev, dtype=dt)
        u = torch.as_tensor(np.ascontiguousarray(U_np), device=dev, dtype=dt)
        _, tang = torch.func.jvp(F, (x,), (u,))
        return tang.detach().float().cpu().numpy()

    us = np.linspace(0.0, 1.0, args.n_u, endpoint=False)
    rs = np.linspace(0.0, args.r_max, args.n_r)
    UU, RR = np.meshgrid(us, rs, indexing="ij")            # [n_u, n_r]
    flat_u, flat_r = UU.ravel(), RR.ravel()
    P, VT, VR = polar_point(spline, mu, flat_u, flat_r)     # [M, d] each
    M = len(flat_u)

    rng = np.random.default_rng(cfg.seed)
    sel = balanced(day_ids, args.n_prompts)
    print(f"[field] {len(sel)} backgrounds, grid {args.n_u}x{args.n_r} = {M} points, "
          f"r up to {args.r_max}, {args.n_hutch} probes/background", flush=True)

    # control: one random unit direction orthogonal to the ring's span, per background
    SPAN = np.linalg.svd(C - mu, full_matrices=False)[2][:N_DAYS - 1]
    assert np.allclose(SPAN @ SPAN.T, np.eye(SPAN.shape[0]), atol=1e-6), "span not orthonormal"

    FRO = np.zeros((len(sel), M)); GT = np.zeros((len(sel), M)); GR = np.zeros((len(sel), M))
    GO = np.zeros((len(sel), M))
    VOFF = np.zeros((len(sel), d), np.float32)           # random direction used per background
    PSQ = np.zeros((len(sel), M, args.n_hutch))          # per-probe ||J u||^2
    for n, pidx in enumerate(sel):
        Fb = make_F_batched(model.to_tokens(texts[pidx], prepend_bos=cfg.prepend_bos))
        s = int(day_ids[pidx])
        base_off = A_all[pidx, day_pos] - C[s]
        X = base_off[None, :] + P
        U = rng.normal(size=(args.n_hutch, d))
        v = rng.normal(size=d)
        v -= SPAN.T @ (SPAN @ v)
        v /= np.linalg.norm(v)
        assert np.abs(SPAN @ v).max() < 1e-6, "control direction is not orthogonal to the span"
        VOFF[n] = v
        for c0 in range(0, M, args.batch):
            idx = np.arange(c0, min(c0 + args.batch, M))
            Xb = X[idx]
            GT[n, idx] = np.linalg.norm(jvp_b(Fb, Xb, VT[idx]), axis=1)
            GR[n, idx] = np.linalg.norm(jvp_b(Fb, Xb, VR[idx]), axis=1)
            GO[n, idx] = np.linalg.norm(
                jvp_b(Fb, Xb, np.broadcast_to(v.astype(np.float32), Xb.shape)), axis=1)
            for q, u in enumerate(U):
                PSQ[n, idx, q] = np.sum(
                    jvp_b(Fb, Xb, np.broadcast_to(u, Xb.shape)) ** 2, axis=1)
            FRO[n, idx] = np.sqrt(PSQ[n, idx].mean(axis=1))
        print(f"[field] {n + 1}/{len(sel)} {DAYS[s]} done", flush=True)

    shape = (args.n_u, args.n_r)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out, us=us, rs=rs, fro=FRO.reshape((len(sel),) + shape),
             gain_t=GT.reshape((len(sel),) + shape), gain_r=GR.reshape((len(sel),) + shape),
             gain_off=GO.reshape((len(sel),) + shape),
             psq=PSQ.reshape((len(sel),) + shape + (args.n_hutch,)),
             span_basis=SPAN, v_off=VOFF,
             centroids=C, mu=mu, ring_radius=ring_r, days=np.array(DAYS, dtype=object),
             prompts=np.array([texts[i] for i in sel], dtype=object),
             src_day=np.array([int(day_ids[i]) for i in sel]),
             tokens=np.array(tokens, dtype=object),
             meta=np.array(json.dumps({
                 "formulation": args.formulation, "patch_layer": Lp, "readout_layer": RO,
                 "day_pos": day_pos, "ro_pos": ro_pos, "n_hutch": args.n_hutch,
                 "model": cfg.model_name}), dtype=object))
    print(f"[field] wrote {args.out}", flush=True)

    fro, gt, gr = (a.mean(0).reshape(shape) for a in (FRO, GT, GR))
    knot = np.array([int(round(k * args.n_u / N_DAYS)) % args.n_u for k in range(N_DAYS)])
    gapi = np.array([int(round((k + 0.5) * args.n_u / N_DAYS)) % args.n_u for k in range(N_DAYS)])
    ir = int(np.argmin(np.abs(rs - 1.0)))
    print(f"[field] at r = 1: between-days / at-days  fro "
          f"{fro[gapi, ir].mean() / fro[knot, ir].mean():.3f}   tangential "
          f"{gt[gapi, ir].mean() / gt[knot, ir].mean():.3f}   radial "
          f"{gr[gapi, ir].mean() / gr[knot, ir].mean():.3f}")


if __name__ == "__main__":
    main()
