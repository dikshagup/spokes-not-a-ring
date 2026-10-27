"""Load a HookedTransformer with the settings from a config."""

from __future__ import annotations

import torch
from transformer_lens import HookedTransformer

from weekday_manifold.model import get_device, load_model

_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def resolve_dtype(name: str) -> torch.dtype:
    if name not in _DTYPES:
        raise ValueError(f"Unsupported dtype {name!r}; choose from {sorted(_DTYPES)}.")
    return _DTYPES[name]


def load_plateau_model(config) -> HookedTransformer:
    """Load the model described by a ``PlateauConfig`` or ``ManifoldConfig``, with its dtype and processing flags."""
    return load_model(
        config.model_name,
        device=config.device or get_device(),
        dtype=resolve_dtype(config.dtype),
        fold_ln=config.fold_ln,
        center_writing_weights=config.center_writing_weights,
        center_unembed=config.center_unembed,
    )
