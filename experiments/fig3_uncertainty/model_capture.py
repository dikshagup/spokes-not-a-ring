"""Load Llama-3.1-8B and capture the residual stream at the sentence-final period."""
from __future__ import annotations

import collections
import time
from typing import List

import numpy as np
import torch

from weekday_manifold.manifold.config import ManifoldConfig
from weekday_manifold.plateau.config import hook_name_for_layer
from weekday_manifold.plateau.model import load_plateau_model

MODEL = "meta-llama/Llama-3.1-8B"


def load_llama(layer: int):
    """Load Llama-3.1-8B in float16; returns the model and the resolved config."""
    cfg = ManifoldConfig(model_name=MODEL, dtype="float16", layer_index=layer,
                         formulation="mention_early")
    model = load_plateau_model(cfg)
    return model, cfg.resolve(model.cfg.n_layers)


def capture_final_token(model, texts: List[str], layer: int, prepend_bos: bool,
                        batch_size: int = 64) -> np.ndarray:
    """Residual at ``blocks.{layer}.hook_resid_post`` at the last token (the period), batched by length without padding."""
    toks = [model.to_tokens(t, prepend_bos=prepend_bos)[0] for t in texts]
    bad = [t for t, k in zip(texts, toks) if model.to_single_str_token(int(k[-1])).strip() != "."]
    if bad:
        raise SystemExit(f"{len(bad)} prompts do not end at the period, e.g. {bad[0]!r}")
    hook = hook_name_for_layer(layer)
    dev = next(model.parameters()).device
    acts = np.zeros((len(texts), model.cfg.d_model), dtype=np.float32)
    grab = {}

    def rec(act, hook):
        grab["x"] = act[:, -1, :].detach().float().cpu().numpy()

    by_len = collections.defaultdict(list)
    for i, t in enumerate(toks):
        by_len[int(t.shape[0])].append(i)
    t0 = time.time()
    for _, idx in sorted(by_len.items()):
        for s in range(0, len(idx), batch_size):
            sel = idx[s:s + batch_size]
            chunk = torch.stack([toks[i] for i in sel]).to(dev)
            with torch.no_grad():
                model.run_with_hooks(chunk, return_type=None, fwd_hooks=[(hook, rec)],
                                     stop_at_layer=layer + 1)
            acts[sel] = grab["x"]
    print(f"[capture] {len(texts)} prompts at layer {layer} in {time.time() - t0:.0f}s",
          flush=True)
    return acts
