"""Configuration for the activation captures."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from weekday_manifold.plateau.config import hook_name_for_layer, resolve_layer_index

# Layer presets as fractions of depth. The paper's layer (block 28) is set with layer_index.
LATE_LAYER_PRESETS: Dict[str, float] = {
    "late_70": 0.70,   # -> block 22 of 32
    "late_78": 0.78,   # -> block 25 of 32
    "late_85": 0.85,   # -> block 27 of 32
}


@dataclass
class ManifoldConfig:
    """Settings for one activation-capture run."""

    model_name: str = "meta-llama/Llama-3.1-8B"
    dtype: str = "float32"
    device: Optional[str] = None        # None -> auto (cuda > mps > cpu)
    fold_ln: bool = True
    center_writing_weights: bool = True
    center_unembed: bool = True
    prepend_bos: bool = True

    # precedence: layer_index > layer_frac > layer_preset
    layer_preset: Optional[str] = "late_78"  # used only when layer_index is None
    layer_frac: Optional[float] = None       # used only when layer_index is None
    layer_index: Optional[int] = None        # set directly, or filled by resolve()

    # "trailing" is captured at the sentence-final "." token.
    formulation: str = "trailing"
    offset_style: str = "digit"          # arith only: digit ("3") | word ("three")
    seq_run_lengths: List[int] = field(default_factory=lambda: [2, 3, 4, 5, 6])
    seq_separators: List[str] = field(default_factory=lambda: ["plain", "comma"])
    # unused; kept so saved configs load
    group_by: str = "correct"            # correct | elicited

    # clamped to min(n_pca_dims, n_prompts-1, d_model)
    n_pca_dims: Optional[int] = 64
    day_scoring: str = "fullstring"      # fullstring (teacher-forced) | firsttoken
    plot_3d_dims: int = 3                # number of PCA dims to plot

    seed: int = 0
    cache_dir: str = "data/activations"
    out_dir: str = "experiments/results/manifold"
    notes: str = ""

    def resolve(self, n_layers: int) -> "ManifoldConfig":
        """Return a copy with ``layer_index`` set for a model with ``n_layers`` blocks."""
        frac = self.layer_frac
        if frac is None and self.layer_index is None and self.layer_preset is not None:
            if self.layer_preset not in LATE_LAYER_PRESETS:
                raise ValueError(
                    f"Unknown layer_preset {self.layer_preset!r}; "
                    f"choose from {sorted(LATE_LAYER_PRESETS)}."
                )
            frac = LATE_LAYER_PRESETS[self.layer_preset]
        idx = resolve_layer_index(
            n_layers=n_layers,
            layer_frac=frac,
            layer_preset=None,
            layer_index=self.layer_index,
        )
        return dataclasses.replace(self, layer_index=idx)

    @property
    def hook_name(self) -> str:
        if self.layer_index is None:
            raise ValueError("layer_index unresolved; call config.resolve(n_layers).")
        return hook_name_for_layer(self.layer_index)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ManifoldConfig":
        fields = {f.name for f in dataclasses.fields(cls)}
        unknown = set(d) - fields
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        return cls(**d)

    @classmethod
    def from_json(cls, path: str) -> "ManifoldConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))
