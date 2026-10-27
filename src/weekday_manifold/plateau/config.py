"""Model-loading configuration and layer-resolution helpers."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

# Fractions of depth; used only when layer_index is None.
LAYER_PRESETS: Dict[str, float] = {
    "paper_early": 1.0 / 12.0,  # ~0.083 -> block 4 of 48 (GPT-2 XL)
    "early_mid": 0.25,          # -> block 12 of 48 (GPT-2 XL)
    "mid": 0.5,                 # -> block 24 of 48 (GPT-2 XL)
}


def resolve_layer_index(
    n_layers: int,
    layer_frac: Optional[float] = None,
    layer_preset: Optional[str] = None,
    layer_index: Optional[int] = None,
) -> int:
    """Return the block index from ``layer_index``, ``layer_frac`` or ``layer_preset``, clamped to the model."""
    if layer_index is not None:
        idx = int(layer_index)
    else:
        if layer_frac is None:
            if layer_preset is None:
                raise ValueError(
                    "Provide one of layer_index, layer_frac, or layer_preset."
                )
            if layer_preset not in LAYER_PRESETS:
                raise ValueError(
                    f"Unknown layer_preset {layer_preset!r}; "
                    f"choose from {sorted(LAYER_PRESETS)}."
                )
            layer_frac = LAYER_PRESETS[layer_preset]
        idx = int(round(layer_frac * n_layers))
    return max(0, min(n_layers - 1, idx))


def hook_name_for_layer(layer_index: int) -> str:
    """Name of the TransformerLens hook at a block's output (``hook_resid_post``)."""
    return f"blocks.{layer_index}.hook_resid_post"


@dataclass
class PlateauConfig:
    """Model and patching settings for one run."""

    model_name: str = "meta-llama/Llama-3.1-8B"
    dtype: str = "float32"
    device: Optional[str] = None    # None -> auto (cuda > mps > cpu)

    # TransformerLens defaults; L2 distances depend on them, KL does not.
    fold_ln: bool = True
    center_writing_weights: bool = True
    center_unembed: bool = True

    # precedence: layer_index > layer_frac > layer_preset
    layer_preset: Optional[str] = "paper_early"  # used only when layer_index is None
    layer_frac: Optional[float] = None           # used only when layer_index is None
    layer_index: Optional[int] = 0               # default; resolve() clamps to range

    # negative indexes from the end; None -> final-token logits; must be >= layer_index
    record_layer: Optional[int] = -1

    # "last" or a 0-based index into the content tokens (BOS excluded)
    token_position: Union[str, int] = "last"
    prepend_bos: bool = True

    prompt_1: str = "The house was big"
    prompt_2: str = "The house was in"

    t_min: float = -0.
    t_max: float = 1.
    t_steps: int = 81

    # "linear" | "slerp"; excluded from cache_key()
    interp_mode: str = "linear"

    seed: int = 0
    cache_dir: str = "data/activations"
    notes: str = ""

    def resolve(self, n_layers: int) -> "PlateauConfig":
        """Resolve ``layer_index`` and ``record_layer`` against model depth."""
        idx = resolve_layer_index(
            n_layers=n_layers,
            layer_frac=self.layer_frac,
            layer_preset=self.layer_preset,
            layer_index=self.layer_index,
        )
        rec = self.record_layer
        if rec is not None and rec < 0:
            rec = max(0, min(n_layers - 1, n_layers + rec))
        return dataclasses.replace(self, layer_index=idx, record_layer=rec)

    @property
    def hook_name(self) -> str:
        if self.layer_index is None:
            raise ValueError(
                "layer_index is unresolved; call config.resolve(n_layers) first."
            )
        return hook_name_for_layer(self.layer_index)

    @property
    def record_hook_name(self):
        """Hook to measure distance at, or None for the output logits."""
        if self.record_layer is None:
            return None
        if self.record_layer < 0:
            raise ValueError(
                "record_layer is negative/unresolved; call resolve(n_layers) first."
            )
        return hook_name_for_layer(self.record_layer)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PlateauConfig":
        fields = {f.name for f in dataclasses.fields(cls)}
        unknown = set(d) - fields
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        return cls(**d)

    @classmethod
    def from_json(cls, path: str) -> "PlateauConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def cache_key(self) -> str:
        """Stable key for the on-disk activation cache."""
        if self.layer_index is None:
            raise ValueError("Resolve layer_index before computing a cache key.")
        import hashlib

        payload = {
            "model_name": self.model_name,
            "dtype": self.dtype,
            "fold_ln": self.fold_ln,
            "center_writing_weights": self.center_writing_weights,
            "center_unembed": self.center_unembed,
            "layer_index": self.layer_index,
            "token_position": self.token_position,
            "prepend_bos": self.prepend_bos,
            "prompt_1": self.prompt_1,
            "prompt_2": self.prompt_2,
        }
        blob = json.dumps(payload, sort_keys=True).encode()
        digest = hashlib.sha256(blob).hexdigest()[:16]
        return f"{self.model_name.replace('/', '_')}_L{self.layer_index}_{digest}"
