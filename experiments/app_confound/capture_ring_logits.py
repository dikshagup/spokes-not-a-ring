"""Steer along the layer-2 weekday ring and save the tangential gain and output logits (confound appendix).

  PYTHONPATH=src:experiments/common:experiments/app_confound \\
      python experiments/app_confound/capture_ring_logits.py     # GPU, about 10 min, 16 GB
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from weekday_manifold.manifold.config import ManifoldConfig
from weekday_manifold.manifold.days import DAYS, N_DAYS
from weekday_manifold.manifold.probes import build_mention_early
from weekday_manifold.manifold.steering import fit_steer_spline, template_demeaned_centroids
from weekday_manifold.plateau.model import load_plateau_model
from weekday_manifold.utils import data_dir, set_seed

MODEL = "meta-llama/Llama-3.1-8B"
PATCH_LAYER = 2        # layer where the ring is fitted and steered
READOUT_LAYER = 28     # layer where the tangential gain is read
N_U = 56               # points on the ring, eight per day
N_BACKGROUNDS = 7      # prompts steered, one per day
ABBR = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
OUT = os.path.join(data_dir("app_confound"), "steer_ring_output_logits.npz")


def weekday_token_ids(model) -> np.ndarray:
    """Ids of every single-token weekday spelling (full or short, either case, with or without a space)."""
    ids = set()
    for k, day in enumerate(DAYS):
        for w in (day, day.lower(), ABBR[k], ABBR[k].lower()):
            for form in (" " + w, w):
                t = model.to_tokens(form, prepend_bos=False)[0].tolist()
                if len(t) == 1:
                    ids.add(int(t[0]))
    return np.array(sorted(ids))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch", type=int, default=28, help="ring points per forward pass (memory only)")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    cfg = ManifoldConfig(model_name=MODEL, dtype="float16", layer_index=READOUT_LAYER,
                         formulation="mention_early", offset_style="word")
    set_seed(cfg.seed)
    model = load_plateau_model(cfg)
    model.requires_grad_(False)
    dev, dt = model.cfg.device, model.W_E.dtype
    pname, rname = f"blocks.{PATCH_LAYER}.hook_resid_post", f"blocks.{READOUT_LAYER}.hook_resid_post"

    specs = build_mention_early("stop")
    day_ids = np.array([int(sp.meta["z"]) for sp in specs])
    template_ids = np.array([int(sp.meta.get("content_id", 0)) for sp in specs])
    texts = [sp.text for sp in specs]
    toks_all = torch.cat([model.to_tokens(t, prepend_bos=cfg.prepend_bos) for t in texts], 0)
    seq_len = toks_all.shape[1]
    ro_pos = seq_len - 1                                            # read position: the final full stop

    store = {}

    def rec_all(act, hook):
        store[hook.name] = act.detach().float().cpu().numpy()
        return act
    with torch.no_grad():
        model.run_with_hooks(toks_all, return_type=None, fwd_hooks=[(pname, rec_all)])
    A_all = store[pname]

    # day-token position: the position with the largest day-centroid spread
    spreads = [float(np.linalg.norm(
        (Cp := template_demeaned_centroids(A_all[:, p, :], day_ids, template_ids, N_DAYS))
        - Cp.mean(0), axis=1).mean()) for p in range(seq_len)]
    day_pos = int(np.argmax(spreads))
    st = model.to_str_tokens(texts[0], prepend_bos=cfg.prepend_bos)
    print(f"patch L{PATCH_LAYER}@pos{day_pos} ({st[day_pos]!r}) -> read L{READOUT_LAYER}@pos{ro_pos} "
          f"({st[ro_pos]!r})", flush=True)

    C = template_demeaned_centroids(A_all[:, day_pos, :], day_ids, template_ids, N_DAYS)
    spline = fit_steer_spline(C)
    print(f"spline through 7 day centroids; ring radius {np.linalg.norm(C - C.mean(0), axis=1).mean():.2f}",
          flush=True)
    weekday_ids = weekday_token_ids(model)
    print(f"{len(weekday_ids)} single-token weekday spellings", flush=True)

    us = np.linspace(0.0, 1.0, N_U, endpoint=False)
    P = spline.forward(us)
    T = spline.derivative(us)
    VT = T / np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-30)   # unit tangent

    by_day = {k: [i for i in range(len(specs)) if day_ids[i] == k] for k in range(N_DAYS)}
    sel, rank = [], 0
    while len(sel) < N_BACKGROUNDS:
        grew = False
        for k in range(N_DAYS):
            if rank < len(by_day[k]) and len(sel) < N_BACKGROUNDS:
                sel.append(by_day[k][rank]); grew = True
        rank += 1
        if not grew:
            break

    GT = np.zeros((len(sel), N_U))
    OLOGIT = np.zeros((N_U, model.cfg.d_vocab), np.float32)
    for bi, i in enumerate(sel):
        toks = toks_all[i:i + 1]
        base_off = A_all[i, day_pos] - C[int(day_ids[i])]                   # this prompt's offset from its day centroid

        def patch_at(X):
            B = X.shape[0]

            def patch(resid, hook):
                return torch.cat([resid[:, :day_pos, :], X.view(B, 1, -1), resid[:, day_pos + 1:, :]], dim=1)
            return patch

        def F(X):
            s = {}

            def rec(act, hook):
                s["a"] = act[:, ro_pos, :]
                return act
            model.run_with_hooks(toks.expand(X.shape[0], -1), return_type=None,
                                 fwd_hooks=[(pname, patch_at(X)), (rname, rec)])
            return s["a"]

        for a0 in range(0, N_U, args.batch):
            sl = slice(a0, min(a0 + args.batch, N_U))
            x = torch.as_tensor(np.ascontiguousarray(base_off[None, :] + P[sl]), device=dev, dtype=dt)
            t = torch.as_tensor(np.ascontiguousarray(VT[sl]), device=dev, dtype=dt)
            DT = torch.func.jvp(F, (x,), (t,))[1].detach().float().cpu().numpy()
            GT[bi, sl] = np.linalg.norm(DT, axis=1)
            with torch.no_grad():
                lg = model.run_with_hooks(toks.expand(x.shape[0], -1), return_type="logits",
                                          fwd_hooks=[(pname, patch_at(x))])
                OLOGIT[sl] += lg[:, ro_pos, :].float().cpu().numpy()
        print(f"background {bi + 1}/{len(sel)} ({DAYS[day_ids[i]]})", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out,
             u=us,                                            # ring coordinate in [0, 1); day k at k/7
             gain=GT.mean(0),                                 # tangential gain, averaged over prompts
             out_logits=(OLOGIT / len(sel)).astype(np.float16),   # mean output logits at the final token
             day_ids=weekday_ids,
             meta=json.dumps({"model": MODEL, "patch_layer": PATCH_LAYER, "readout_layer": READOUT_LAYER,
                              "day_pos": day_pos, "ro_pos": ro_pos, "n_vocab": int(model.cfg.d_vocab),
                              "formulation": "mention_early"}))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
