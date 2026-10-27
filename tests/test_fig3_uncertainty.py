"""Wiring tests for Figure 3 and the appendix families plate, on a synthetic capture (CPU only)."""
from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIG = os.path.join(ROOT, "experiments", "fig3_uncertainty")
PATHS = [os.path.join(ROOT, "src"), os.path.join(ROOT, "experiments", "common"), FIG]
for p in PATHS:
    sys.path.insert(0, p)

from weekday_manifold.manifold.days import N_DAYS                      # noqa: E402
from weekday_manifold.manifold.probes import (build_frequency_pairs,   # noqa: E402
                                              build_hedge_families)
from ring_coords import (fit_ruler, pair_position, ring_coords,        # noqa: E402
                         save_capture, signed_days)

D = 24


def synthetic_ruler(rng):
    """Seven centroids on a ring in a 24-d space, plus template noise."""
    ang = 2 * np.pi * np.arange(N_DAYS) / N_DAYS
    basis = rng.standard_normal((2, D))
    basis = np.linalg.qr(basis.T)[0].T
    centroids = 3 * (np.cos(ang)[:, None] * basis[0] + np.sin(ang)[:, None] * basis[1])
    n_bases = 40
    content = rng.standard_normal((n_bases, 1, D)) * 0.5
    X = centroids[None] + content + rng.standard_normal((n_bases, N_DAYS, D)) * 0.1
    return X, centroids, basis


def synthetic_capture(rng, specs, centroids, basis, path):
    """Each prompt lands on its favoured day's spoke, pulled inward with uncertainty."""
    acts = np.zeros((len(specs), D))
    for i, s in enumerate(specs):
        m = s.meta
        fav = m["day_a"] if m["lean"] >= 0 else m["day_b"]
        radius = 0.5 + 0.5 * abs(m["lean"])
        acts[i] = radius * centroids[fav] + rng.standard_normal(D) * 0.05
    save_capture(path, acts, specs, layer=28, model="synthetic")


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    rng = np.random.default_rng(0)
    root = tmp_path_factory.mktemp("fig3")
    X, centroids, basis = synthetic_ruler(rng)
    fit = fit_ruler(X, k=6, q=5)
    ruler = os.path.join(root, "ruler_L28.npz")
    np.savez(ruler, mu=fit["mu"], W=fit["W"], M=fit["M"], layer=28, model="synthetic",
             k=6, q=5, n_bases=X.shape[0], corpus_sha1="0" * 16,
             in_sample_acc=fit["in_sample_acc"])
    freq = build_frequency_pairs(bodies=["her flight leaves", "the parcel arrives"])
    hedges = build_hedge_families(bodies=["her flight leaves"])
    fp, hp = os.path.join(root, "frequency_L28.npz"), os.path.join(root, "hedges_L28.npz")
    synthetic_capture(rng, freq, centroids, basis, fp)
    synthetic_capture(rng, hedges, centroids, basis, hp)
    return {"root": str(root), "ruler": ruler, "frequency": fp, "hedges": hp,
            "fit": fit, "centroids": centroids}


def test_prompt_counts():
    freq = build_frequency_pairs()
    assert len(freq) == 5 * 21 * 2 * 32 == 6720
    hedges = build_hedge_families()
    assert len(hedges) == 4 * 6 * 21 * 2 * 12 + 5 * 21 * 2 * 12 == 14616
    for s in freq + hedges:
        assert s.text.endswith(".")
        assert (s.meta["day_b"] - s.meta["day_a"]) % N_DAYS == s.meta["gap"] in (1, 2, 3)


def test_ring_coords_recover_the_spokes(synthetic):
    fit, C = synthetic["fit"], synthetic["centroids"]
    Z = (C - fit["mu"]) @ fit["W"].T
    u, r = ring_coords(Z, fit["M"])
    assert np.allclose(np.round(u) % N_DAYS, np.arange(N_DAYS))
    assert np.allclose(r, 1, atol=0.1)       # centroids carry a little template noise
    # halfway in along Monday's spoke reads as Monday at half radius
    u2, r2 = ring_coords(0.5 * Z[:1], fit["M"])
    assert abs(r2[0] - 0.5) < 0.05 and abs(signed_days(u2[0])) < 0.1
    # pair position: day A's spoke is 0, day B's is 1, whichever way round the week
    assert pair_position(np.array([6.0, 0.0]), np.array([6, 6]), np.array([0, 0])).tolist() == [0.0, 1.0]


@pytest.mark.parametrize("script,capture,out", [
    ("plot_fig3.py", "frequency", "fig3_uncertainty"),
    ("plot_families.py", "hedges", "fig_uncertainty_families"),
])
def test_plates_render(synthetic, tmp_path, script, capture, out):
    stem = os.path.join(tmp_path, out)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(PATHS), MPLBACKEND="Agg")
    res = subprocess.run([sys.executable, os.path.join(FIG, script), "--capture",
                          synthetic[capture], "--ruler", synthetic["ruler"], "--out", stem],
                         capture_output=True, text=True, env=env)
    assert res.returncode == 0, res.stderr[-2000:]
    assert os.path.getsize(stem + ".png") > 10_000 and os.path.exists(stem + ".pdf")
    assert "apart" in res.stdout      # the printed summary ran
