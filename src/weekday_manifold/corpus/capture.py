"""Capture of final-token residual-stream states at one layer."""
from __future__ import annotations

import time

import numpy as np
import torch


def capture_final_token(model, ids, layer, batch_size=32, prepend_bos=True, log_every=200):
    """Return [N, d_model] float16 ``blocks.<layer>.hook_resid_post`` states at the final token
    of each row of ``ids`` [N, n], with BOS prepended if asked."""
    dev = next(model.parameters()).device
    if not 0 <= layer < model.cfg.n_layers:
        raise ValueError(f"layer {layer} outside 0..{model.cfg.n_layers - 1}")
    acts = np.zeros((len(ids), model.cfg.d_model), dtype=np.float16)
    grab = {}

    def hook(act, hook):                                  # act [B, P, d]
        grab["h"] = act[:, -1, :].detach().float()

    fwd_hooks = [(f"blocks.{layer}.hook_resid_post", hook)]
    bos = torch.tensor([[model.tokenizer.bos_token_id]], device=dev) if prepend_bos else None

    def batch(a):
        chunk = torch.tensor(a.astype(np.int64), device=dev)
        return chunk if bos is None else torch.cat([bos.expand(chunk.shape[0], 1), chunk], 1)

    # Use stop_at_layer only if the hook still fires with it.
    stop_kw = {"stop_at_layer": layer + 1}
    try:
        with torch.no_grad():
            model.run_with_hooks(batch(ids[:1]), return_type=None, fwd_hooks=fwd_hooks, **stop_kw)
        if "h" not in grab:
            raise RuntimeError("hook did not fire under stop_at_layer")
    except (TypeError, RuntimeError):
        stop_kw = {}
    grab.clear()

    N, t0 = len(ids), time.time()
    for s in range(0, N, batch_size):
        with torch.no_grad():
            model.run_with_hooks(batch(ids[s:s + batch_size]), return_type=None,
                                 fwd_hooks=fwd_hooks, **stop_kw)
        h = grab.pop("h")
        acts[s:s + len(h)] = h.cpu().numpy().astype(np.float16)
        if s and (s // batch_size) % log_every == 0:
            el = time.time() - t0
            print(f"[capture] {s}/{N}  {el:.0f}s  eta {el / s * (N - s):.0f}s", flush=True)
    return acts


def load_model_for_capture(model_name, dtype="bfloat16"):
    """Load the model and return it with its resolved ``ManifoldConfig``."""
    from weekday_manifold.manifold.config import ManifoldConfig
    from weekday_manifold.plateau.model import load_plateau_model

    cfg = ManifoldConfig(model_name=model_name, dtype=dtype)
    model = load_plateau_model(cfg)
    return model, cfg.resolve(model.cfg.n_layers)
