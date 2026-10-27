"""The weekday ring frame at one layer: centroid plane, orientation, cyclic-order check."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

N_DAYS = 7


@dataclass
class RingFrame:
    """The weekday reference frame at one layer, fitted from neutral prompts only."""

    mu: np.ndarray
    centroids: np.ndarray
    plane: np.ndarray
    span: np.ndarray
    evr: np.ndarray
    theta: np.ndarray
    radius: float
    orient: int

    @property
    def evr_plane(self) -> float:
        """Fraction of weekday-centroid variance captured by the 2-D plane."""
        return float(self.evr[:2].sum())

    @property
    def evr_beyond(self) -> float:
        """Fraction of weekday-centroid variance outside the plane."""
        return float(self.evr[2:].sum())

    def in_plane(self, X: np.ndarray) -> np.ndarray:
        """``[N, 2]`` coordinates of the rows of ``X`` in the plane, relative to ``mu``."""
        return (np.atleast_2d(X) - self.mu) @ self.plane

    def angle_of(self, X: np.ndarray) -> np.ndarray:
        """Ring angle (radians) of each row of ``X``, oriented so day index increases."""
        c = self.in_plane(X)
        return np.arctan2(self.orient * c[:, 1], c[:, 0])

    def heptagon_check(self) -> Dict[str, object]:
        """Check whether the seven centroids sit in weekday order around the ring."""
        ang = np.arctan2(self.orient * self.in_plane(self.centroids)[:, 1],
                         self.in_plane(self.centroids)[:, 0])
        order = list(np.argsort(ang))
        rot = [(order.index(d) - order.index(0)) % N_DAYS for d in range(N_DAYS)]
        in_order = rot == list(range(N_DAYS))
        gaps = np.diff(np.sort(ang))
        return {
            "in_cyclic_order": bool(in_order),
            "angles_deg": np.degrees(ang).tolist(),
            "order": [int(o) for o in order],
            "min_gap_deg": float(np.degrees(gaps.min())) if gaps.size else float("nan"),
            "mean_gap_deg": float(np.degrees(gaps.mean())) if gaps.size else float("nan"),
        }


def fit_ring_frame(acts: np.ndarray, days: np.ndarray) -> RingFrame:
    """Fit the weekday plane from the day centroids of `acts` (neutral prompts only)."""
    acts = np.asarray(acts, dtype=np.float64)
    days = np.asarray(days)
    cents = np.stack([acts[days == d].mean(axis=0) for d in range(N_DAYS)])
    mu = cents.mean(axis=0)
    C = cents - mu                                        # [7, d], rank <= 6
    U, S, Vt = np.linalg.svd(C, full_matrices=False)
    var = S ** 2
    evr = var / var.sum() if var.sum() > 0 else var
    plane = Vt[:2].T                                      # [d, 2]
    rank = int((S > 1e-8 * S.max()).sum()) if S.size else 0
    span = Vt[:rank].T                                    # [d, r<=6]
    coords = C @ plane                                    # [7, 2]
    raw_ang = np.arctan2(coords[:, 1], coords[:, 0])
    # orient so a positive angular shift points toward the next day.
    steps = np.angle(np.exp(1j * (np.roll(raw_ang, -1) - raw_ang)))
    orient = 1 if steps.mean() >= 0 else -1
    theta = np.arctan2(orient * coords[:, 1], coords[:, 0])
    radius = float(np.linalg.norm(C, axis=1).mean())
    return RingFrame(mu=mu, centroids=cents, plane=plane, span=span, evr=evr,
                     theta=theta, radius=radius, orient=orient)
