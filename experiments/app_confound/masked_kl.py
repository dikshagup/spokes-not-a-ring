"""Masked KL between successive steered positions (confound appendix, panel B)."""
from __future__ import annotations

import numpy as np


class MaskedKL:
    """KL(P_i || P_{i-1}) per step around the closed ring, with masked columns set to -inf before the softmax."""

    def __init__(self, L):
        a = L - L.max(1, keepdims=True)
        self.E = np.exp(a)
        self.da = a - np.roll(a, 1, axis=0)
        self.S_all, self.T_all = self.E.sum(1), (self.E * self.da).sum(1)

    def __call__(self, idx):
        idx = np.asarray(idx, int)
        Z = self.S_all - self.E[:, idx].sum(1)
        T = self.T_all - (self.E[:, idx] * self.da[:, idx]).sum(1)
        return T / Z - (np.log(Z) - np.roll(np.log(Z), 1))


def kl_direct(L, idx):
    """Reference implementation of MaskedKL."""
    Lm = np.array(L, np.float64, copy=True)
    Lm[:, np.asarray(idx, int)] = -np.inf
    ls = Lm - Lm.max(1, keepdims=True)
    ls = ls - np.log(np.exp(ls).sum(1, keepdims=True))
    P = np.exp(ls)
    prev = np.roll(ls, 1, axis=0)
    with np.errstate(invalid="ignore"):
        term = np.where(P > 0, P * (ls - prev), 0.0)
    return term.sum(1)
