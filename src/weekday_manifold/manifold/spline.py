"""Periodic cubic spline through an ordered set of points."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline


def default_knots(n_points: int) -> np.ndarray:
    """Uniform intrinsic coordinates ``u_i = i / n_points`` in ``[0, 1)``."""
    return np.arange(n_points, dtype=float) / float(n_points)


class PeriodicSpline:
    """Closed cubic spline through ordered points in R^D, with coordinate u in [0, 1)."""

    def __init__(self, points: np.ndarray, knots: np.ndarray | None = None):
        self.points = np.asarray(points, dtype=float)
        if self.points.ndim != 2:
            raise ValueError(f"points must be [K, D], got shape {self.points.shape}.")
        K = self.points.shape[0]
        if K < 3:
            raise ValueError(f"need >=3 points for a periodic cubic spline, got {K}.")
        self.knots = default_knots(K) if knots is None else np.asarray(knots, float)
        self.period = 1.0
        # repeat the first point at u=1, as bc_type='periodic' requires.
        u_aug = np.concatenate([self.knots, [self.period]])
        y_aug = np.concatenate([self.points, self.points[:1]], axis=0)
        self._spline = CubicSpline(u_aug, y_aug, bc_type="periodic", axis=0)

    @property
    def n_points(self) -> int:
        return int(self.points.shape[0])

    @property
    def dim(self) -> int:
        return int(self.points.shape[1])

    def forward(self, u) -> np.ndarray:
        """Map intrinsic coordinate(s) ``u`` (wrapped into [0,1)) to point(s)."""
        u = np.mod(np.asarray(u, dtype=float), self.period)
        return self._spline(u)

    def derivative(self, u) -> np.ndarray:
        """Tangent d/du of the curve at ``u``."""
        u = np.mod(np.asarray(u, dtype=float), self.period)
        return self._spline(u, 1)

    def sample(self, n: int = 400) -> tuple[np.ndarray, np.ndarray]:
        """Dense closed polyline ``(u_grid, points)``; the last point repeats the first."""
        u = np.linspace(0.0, self.period, n + 1)
        return u, self._spline(u)

    def arc_length_table(self, n_samples: int = 2000) -> tuple[np.ndarray, np.ndarray]:
        """Cumulative arc length along the closed curve: ``(u_grid, s_grid)``."""
        u = np.linspace(0.0, self.period, n_samples + 1)
        pts = self._spline(u)                                  # [n+1, D]
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)     # [n]
        s = np.concatenate([[0.0], np.cumsum(seg)])
        return u, s

    def total_length(self, n_samples: int = 2000) -> float:
        """Perimeter of the closed curve (total arc length)."""
        return float(self.arc_length_table(n_samples)[1][-1])

    def geodesic_matrix(self, knots=None, n_samples: int = 2000) -> np.ndarray:
        """Pairwise distances along the curve between the knot points."""
        u_grid, s_grid = self.arc_length_table(n_samples)
        L = s_grid[-1]
        ks = self.knots if knots is None else np.asarray(knots, dtype=float)
        s_at = np.interp(np.mod(ks, self.period), u_grid, s_grid)  # arc length at each knot
        d = np.abs(s_at[:, None] - s_at[None, :])
        return np.minimum(d, L - d)

    def inverse(self, a, n_grid: int = 2000, refine: bool = True) -> np.ndarray:
        """Nearest intrinsic coordinate(s) ``u`` to point(s) ``a`` on the curve."""
        a = np.asarray(a, dtype=float)
        single = a.ndim == 1
        A = a[None, :] if single else a
        ug = np.linspace(0.0, self.period, n_grid, endpoint=False)
        curve = self._spline(ug)                       # [n_grid, D]
        d2 = ((A[:, None, :] - curve[None, :, :]) ** 2).sum(axis=-1)
        best = np.argmin(d2, axis=1)
        u_best = ug[best].astype(float)
        if refine:
            step = self.period / n_grid
            for i in range(A.shape[0]):
                u_best[i] = self._parabolic_refine(A[i], u_best[i], step)
        u_best = np.mod(u_best, self.period)
        return float(u_best[0]) if single else u_best

    def _dist2(self, point: np.ndarray, u: float) -> float:
        diff = self._spline(np.mod(u, self.period)) - point
        return float(np.dot(diff, diff))

    def _parabolic_refine(self, point: np.ndarray, u0: float, step: float) -> float:
        """One parabolic-interpolation step using the two neighboring grid nodes."""
        fm = self._dist2(point, u0 - step)
        f0 = self._dist2(point, u0)
        fp = self._dist2(point, u0 + step)
        denom = fm - 2.0 * f0 + fp
        if denom <= 1e-18:  # flat or concave -> keep the grid node
            return u0
        delta = 0.5 * (fm - fp) / denom
        delta = max(-1.0, min(1.0, delta))  # stay within the [u0-step, u0+step] bracket
        return u0 + delta * step


def fit_periodic_spline(points: np.ndarray, knots: np.ndarray | None = None) -> PeriodicSpline:
    return PeriodicSpline(points, knots=knots)
