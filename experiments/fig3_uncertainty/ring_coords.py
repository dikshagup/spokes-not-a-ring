"""Reference ring, ring coordinates and capture files shared by the Figure 3 plates.

fit_ruler.py fits the ring on corpus templates, not on the mention prompts; the plates load it.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
from typing import Dict, List, Tuple

import numpy as np

from weekday_manifold.manifold.days import N_DAYS, named_days, swap_day
from weekday_manifold.manifold.spline import PeriodicSpline
from weekday_manifold.utils import data_dir

FIGURE = "fig3_uncertainty"
LAYER = 28                                   # 0.875 of Llama-3.1-8B's 32 blocks
RULER_K, RULER_Q = 6, 200                    # subspace rank; between-template directions removed


def ruler_path(layer: int = LAYER) -> str:
    return os.path.join(data_dir(FIGURE), f"ruler_L{layer}.npz")


def capture_path(family: str, layer: int = LAYER) -> str:
    return os.path.join(data_dir(FIGURE), f"{family}_L{layer}.npz")


def corpus_design(paths: List[str]) -> Tuple[List[str], List[str]]:
    """Every corpus sentence with one weekday, rewritten with each of the seven weekdays.

    Returns (bases, texts); texts are base-major, the seven days in order within each base.
    """
    bases, texts, seen = [], [], set()
    for path in paths:
        opener = gzip.open if path.endswith(".gz") else open
        for line in opener(path, "rt"):
            text = " ".join(json.loads(line)["text"].split())
            if text.lower() in seen:
                continue
            seen.add(text.lower())
            d = named_days(text)
            if len(d) != 1:
                continue
            variants = [text if w == d[0] else swap_day(text, d[0], w) for w in range(N_DAYS)]
            if all(v is not None for v in variants):
                bases.append(text)
                texts.extend(variants)
    return bases, texts


def fit_ruler(X: np.ndarray, k: int = RULER_K, q: int = RULER_Q) -> Dict[str, np.ndarray]:
    """Fit the weekday subspace on X [bases, 7, d]; returns mu, W (k x d), M (7 x k)."""
    n, _, d = X.shape
    flat = X.reshape(-1, d)
    mu = flat.mean(0)
    C = X.mean(0)
    W = np.linalg.svd(C - C.mean(0), full_matrices=False)[2][:k]
    A = X.mean(1) - mu
    Q = np.linalg.svd(A - A.mean(0), full_matrices=False)[2][:q]
    W = W - (W @ Q.T) @ Q
    if np.linalg.matrix_rank(W) < k:
        raise ValueError("centroid span collapsed under orthogonalisation; lower q")
    W = np.linalg.qr(W.T)[0].T
    Z = (flat - mu) @ W.T
    y = np.tile(np.arange(N_DAYS), n)
    M = np.stack([Z[y == day].mean(0) for day in range(N_DAYS)])
    acc = float(np.mean(((Z[:, None] - M[None]) ** 2).sum(-1).argmin(1) == y))
    return {"mu": mu, "W": W, "M": M, "in_sample_acc": acc}


def load_ruler(layer: int = LAYER, path: str | None = None):
    """Load the saved ruler; returns (mu, W, M, provenance)."""
    p = path or ruler_path(layer)
    if not os.path.exists(p):
        raise SystemExit(f"no ruler at {p}; run experiments/fig3_uncertainty/fit_ruler.py")
    z = np.load(p, allow_pickle=True)
    prov = {k: z[k].item() for k in ("corpus_sha1", "layer", "k", "q", "n_bases")}
    return z["mu"], z["W"], z["M"], prov


def corpus_sha1(texts: List[str]) -> str:
    """Short SHA-1 of the rendered fit prompts."""
    return hashlib.sha1("|".join(texts).encode()).hexdigest()[:16]


def ring_coords(Z: np.ndarray, M: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(u, r) of points Z in the ring's polar form p(u, r) = mu + r (sigma(u) - mu).

    u is in days, [0, 7), the best-aligned spoke; r is along that spoke, 1 on the ring, 0 at its centre.
    """
    ctr = M.mean(0)
    sp = PeriodicSpline(M - ctr)
    T = np.linspace(0, 1, 4000, endpoint=False)
    cv = sp.forward(T)
    cvr = np.linalg.norm(cv, axis=1)
    cvd = cv / np.maximum(cvr[:, None], 1e-12)
    v = Z - ctr
    n = np.linalg.norm(v, axis=1)
    cos = (v / np.maximum(n[:, None], 1e-12)) @ cvd.T
    j = cos.argmax(1)
    u = T[j] * N_DAYS
    r = n / cvr[j] * cos[np.arange(len(j)), j]
    return u, r


def signed_days(x: np.ndarray) -> np.ndarray:
    """Wrap a difference of day coordinates to (-3.5, 3.5]."""
    return ((np.asarray(x, float) + N_DAYS / 2) % N_DAYS) - N_DAYS / 2


def pair_position(u: np.ndarray, day_a: np.ndarray, day_b: np.ndarray) -> np.ndarray:
    """Angular position relative to the named pair: 0 on day A's spoke, 1 on day B's."""
    span = signed_days(day_b - day_a)
    return signed_days(u - day_a) / np.where(span == 0, 1, span)


META_KEYS = ("family_name", "level", "lean", "day_a", "day_b", "gap", "first", "order",
             "content_id")


def save_capture(path: str, acts: np.ndarray, specs, layer: int, model: str) -> None:
    np.savez(path, acts=acts.astype(np.float16), layer=layer, model=model,
             text=np.array([s.text for s in specs]),
             **{k: np.array([s.meta[k] for s in specs]) for k in META_KEYS})


def load_capture(path: str) -> Dict[str, np.ndarray]:
    """Load a capture as plain arrays, with acts as float64."""
    if not os.path.exists(path):
        raise SystemExit(f"no capture at {path}; run experiments/fig3_uncertainty/repro.sh")
    z = np.load(path, allow_pickle=True)
    out = {k: z[k] for k in META_KEYS + ("text",)}
    out["acts"] = np.asarray(z["acts"], dtype=np.float64)
    out["layer"] = int(z["layer"])
    return out


def family_geometry(cap: Dict[str, np.ndarray], mu, W, M) -> Dict[str, np.ndarray]:
    """Per-prompt lean, gap, pair position w and radius r."""
    u, r = ring_coords((cap["acts"] - mu) @ W.T, M)
    return {"lean": np.round(cap["lean"].astype(float), 3), "gap": cap["gap"].astype(int),
            "w": pair_position(u, cap["day_a"], cap["day_b"]), "r": r,
            "family_name": cap["family_name"], "level": cap["level"]}
