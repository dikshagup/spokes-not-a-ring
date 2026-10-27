"""Next-token distributions over the seven days, embedded in Hellinger space and fitted with a closed curve."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from weekday_manifold.manifold.pca import group_centroids
from weekday_manifold.manifold.spline import PeriodicSpline


def restrict_to_concept(
    vocab_probs: np.ndarray,
    concept_ids: Sequence[int],
) -> np.ndarray:
    """Keep the probabilities of ``concept_ids`` and add one "other" entry for the rest."""
    vocab_probs = np.asarray(vocab_probs, dtype=float)
    ids = list(concept_ids)
    concept = vocab_probs[..., ids]                       # [..., K]
    other = 1.0 - concept.sum(axis=-1, keepdims=True)     # [..., 1]
    other = np.clip(other, 0.0, 1.0)
    return np.concatenate([concept, other], axis=-1)


def hellinger_embed(p: np.ndarray) -> np.ndarray:
    """Map a distribution to Hellinger space: ``p -> sqrt(p)``."""
    return np.sqrt(np.clip(np.asarray(p, dtype=float), 0.0, None))


def hellinger_decode(y: np.ndarray) -> np.ndarray:
    """Inverse of :func:`hellinger_embed`: square ``y`` and renormalize."""
    y = np.asarray(y, dtype=float)
    p = y ** 2
    total = p.sum(axis=-1, keepdims=True)
    return np.divide(p, total, out=np.zeros_like(p), where=total > 0)


def hellinger_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Hellinger distance between two distributions ``p``, ``q``."""
    return float(np.linalg.norm(hellinger_embed(p) - hellinger_embed(q)) / np.sqrt(2.0))


@dataclass
class BehaviorManifold:
    """A closed curve through the per-day mean output distributions, in Hellinger space."""

    centroids_dist: np.ndarray         # [K, K+1] distributions, in day order
    centroids_hell: np.ndarray         # [K, K+1] sqrt of centroids_dist
    spline: PeriodicSpline             # periodic curve through centroids_hell[order]
    day_order: List[int]               # day index of each spline knot
    labels: List[str]                  # name of each centroid row

    def forward(self, u) -> np.ndarray:
        """Point on the curve at ``u`` in [0, 1), in Hellinger space."""
        return self.spline.forward(u)

    def forward_dist(self, u) -> np.ndarray:
        """Output distribution at curve coordinate ``u``."""
        return hellinger_decode(self.forward(u))

    def inverse(self, y) -> np.ndarray:
        """Nearest curve coordinate ``u`` for Hellinger point(s) ``y``."""
        return self.spline.inverse(y)

    def tangent(self, u) -> np.ndarray:
        """Curve tangent d/du in Hellinger space."""
        return self.spline.derivative(u)

    def geodesic_matrix(self, n_samples: int = 2000) -> np.ndarray:
        """``[K, K]`` distances along the curve between days, indexed by day."""
        Dk = self.spline.geodesic_matrix(n_samples=n_samples)
        pos = [self.day_order.index(a) for a in range(len(self.day_order))]
        return Dk[np.ix_(pos, pos)]

    @property
    def knot_coords(self) -> np.ndarray:
        return self.spline.knots


def fit_behavior_manifold(
    distributions: np.ndarray,
    labels: Sequence[int],
    n_labels: Optional[int] = None,
    label_names: Optional[Sequence[str]] = None,
    day_order: Optional[Sequence[int]] = None,
) -> BehaviorManifold:
    """Average the distributions per class, take the square root, and fit a periodic spline."""
    distributions = np.asarray(distributions, dtype=float)
    labels = np.asarray(labels)
    if n_labels is None:
        n_labels = distributions.shape[-1] - 1
    if label_names is None:
        label_names = [str(i) for i in range(n_labels)]
    order = list(range(n_labels)) if day_order is None else list(day_order)

    centroids_dist = group_centroids(distributions, labels, n_labels)  # [K, K+1]
    centroids_hell = hellinger_embed(centroids_dist)
    spline = PeriodicSpline(centroids_hell[order])
    return BehaviorManifold(
        centroids_dist=centroids_dist,
        centroids_hell=centroids_hell,
        spline=spline,
        day_order=order,
        labels=list(label_names),
    )
