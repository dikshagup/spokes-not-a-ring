"""Per-layer ring frames, the unsteered reference state, and the two steering hooks for figure 2C.

Each hook sets a target value rather than adding a step, so steering does not accumulate across layers.
"""
from __future__ import annotations

import numpy as np
import torch

from weekday_manifold.manifold.days import DAYS, N_DAYS
from weekday_manifold.timeofday import fit_ring_frame

MODS = {"neutral": "", "early": "early", "late": "late"}


def weekday_position(model, text, day, prepend_bos=True):
    """Index of the weekday's last sub-token in `text`."""
    toks = model.to_tokens(text, prepend_bos=prepend_bos)[0]
    did = model.to_tokens(" " + day, prepend_bos=False)[0]
    n = len(did)
    for i in range(len(toks) - n, -1, -1):
        if torch.equal(toks[i:i + n], did):
            return i + n - 1
    raise SystemExit(f"weekday {day!r} not found in {text!r}")


def _grab(model, text, layers):
    grab = {}
    with torch.no_grad():
        model.run_with_hooks(
            model.to_tokens(text, prepend_bos=True), return_type=None,
            fwd_hooks=[(f"blocks.{L}.hook_resid_post",
                        (lambda LL: lambda a, hook: grab.__setitem__(
                            LL, a[0].detach().float().cpu().numpy()))(L))
                       for L in layers])
    return grab


def fit_pos_frames(model, prompt_fn, carriers, layers, offsets, verbose=True):
    """Fit a ring frame per (layer, offset) from the neutral prompts. Returns (frames, acts, days)."""
    store = {(m, L, o): [] for m in MODS for L in layers for o in offsets}
    days = []
    for c in carriers:
        for d in range(N_DAYS):
            for m, word in MODS.items():
                text = prompt_fn(c, word, DAYS[d])
                P = weekday_position(model, text, DAYS[d])
                grab = _grab(model, text, layers)
                for L in layers:
                    for o in offsets:
                        if P + o >= grab[L].shape[0]:
                            raise SystemExit(f"offset {o} runs past the end of {text!r}")
                        store[(m, L, o)].append(grab[L][P + o])
            days.append(d)
    days = np.array(days)
    acts = {k: np.stack(v) for k, v in store.items()}
    frames = {(L, o): fit_ring_frame(acts[("neutral", L, o)], days)
              for L in layers for o in offsets}
    if verbose:
        print("[frames] plane variance share, ring radius, cyclic order per layer:")
        for L in layers:
            f = frames[(L, offsets[0])]
            hc = f.heptagon_check()
            print(f"    L{L:>2}  {f.evr_plane:.2f}  R={f.radius:6.3f}  "
                  f"{'cyclic' if hc['in_cyclic_order'] else 'NOT cyclic'}")
    return frames, acts, days


def clean_state(model, text, day, layers, offsets):
    """Unsteered activation at every (layer, offset), and the weekday position."""
    P = weekday_position(model, text, day)
    grab = _grab(model, text, layers)
    return {(L, o): grab[L][P + o] for L in layers for o in offsets}, P


def spline_set_hook(spline, u_target, positions, device, n_grid=2000):
    """Hook setting x_new = x + spline(u_target) - foot(x), where foot is the nearest of `n_grid` curve samples."""
    ug = np.linspace(0.0, 1.0, n_grid, endpoint=False)
    curve = torch.as_tensor(spline.forward(ug), device=device, dtype=torch.float32)  # [G, d]
    cn = (curve ** 2).sum(1)
    tgt_pt = torch.as_tensor(spline.forward(u_target), device=device, dtype=torch.float32)
    pos = torch.as_tensor(positions, device=device, dtype=torch.long)

    def fn(resid, hook):
        x = resid[:, pos, :].float()                                   # [B, n_pos, d]
        xf = x.reshape(-1, x.shape[-1])
        d2 = (xf ** 2).sum(1, keepdim=True) - 2.0 * xf @ curve.T + cn[None, :]
        foot = curve[d2.argmin(1)].reshape(x.shape)
        resid[:, pos, :] = (x + (tgt_pt - foot)).to(resid.dtype)
        return resid

    return fn


def clamp_dir_hook(u, target_coef, positions, device):
    """Hook that sets the component along unit direction `u` to `target_coef`."""
    uu = torch.as_tensor(np.asarray(u, dtype=float), device=device, dtype=torch.float32)
    uu = uu / uu.norm()
    tgt = torch.as_tensor(float(target_coef), device=device, dtype=torch.float32)
    pos = torch.as_tensor(positions, device=device, dtype=torch.long)

    def fn(resid, hook):
        x = resid[:, pos, :].float()                                   # [B, n_pos, d]
        delta = (tgt - x @ uu).unsqueeze(-1) * uu
        resid[:, pos, :] = (x + delta).to(resid.dtype)
        return resid

    return fn
