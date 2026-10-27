"""Model config and layer resolution for the capture scripts."""

from __future__ import annotations

from weekday_manifold.plateau.config import (
    LAYER_PRESETS,
    PlateauConfig,
    hook_name_for_layer,
    resolve_layer_index,
)

__all__ = [
    "PlateauConfig",
    "LAYER_PRESETS",
    "resolve_layer_index",
    "hook_name_for_layer",
]
