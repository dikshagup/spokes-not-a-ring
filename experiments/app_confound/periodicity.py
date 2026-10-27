"""Periodicity measures for per-step logit traces around the weekday ring (confound appendix)."""
from __future__ import annotations

import numpy as np


def seven_fold(k, n_days: int = 7):
    """Share of each column's variance explained by a one-day-periodic template (k: [steps] or [steps, V])."""
    k = np.asarray(k, float)
    per = k.shape[0] // n_days
    tmpl = k.reshape(n_days, per, *k.shape[1:]).mean(0)
    tiled = np.concatenate([tmpl] * n_days, axis=0)
    return tiled.var(0) / np.maximum(k.var(0), 1e-30)
