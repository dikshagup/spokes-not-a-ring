"""Prompt specification and concept definition."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np


@dataclass
class PromptSpec:
    """One prompt, its group label, and where to capture."""

    text: str
    answer_day: int
    capture_text: str
    formulation: str
    meta: Dict[str, object] = field(default_factory=dict)

    @property
    def label(self) -> int:
        """Class index of the prompt (same as answer_day)."""
        return self.answer_day

    @property
    def scored(self) -> bool:
        """Whether next-token accuracy is meaningful."""
        return self.formulation not in ("mention", "trailing")


@dataclass(frozen=True)
class Concept:
    """A concept: its labels, prompt builder and answer token ids."""

    name: str
    labels: Sequence[str]
    is_cyclic: bool
    build_prompts: Callable[..., List[PromptSpec]]
    token_ids: Callable[[Any], Dict[int, List[int]]]
    coordinate: Optional[np.ndarray] = None

    @property
    def n_labels(self) -> int:
        return len(self.labels)


def get_concept(name: str) -> Concept:
    """Return the concept with the given name."""
    if name == "days":
        from weekday_manifold.manifold.days import make_days_concept
        return make_days_concept()
    raise ValueError(f"unknown concept {name!r}; choose from ['days'].")
