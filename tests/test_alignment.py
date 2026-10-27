"""Tests for the Figure 4 translation, scale and rotation alignment, on synthetic
7-point configurations."""

import numpy as np
import pytest

from weekday_manifold.corpus.alignment import (ladder, noise_null, procrustes_proper,
                                               rotation_angle_deg, spoke_distances,
                                               stage_distances)
from weekday_manifold.corpus.split import fit_documents, fit_mask, held_out_mask


def heptagon(radius=5.0, d=2):
    ang = 2 * np.pi * np.arange(7) / 7
    k = np.zeros((7, d)); k[:, 0] = radius * np.cos(ang); k[:, 1] = radius * np.sin(ang)
    return k


def rot2(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])   # row-vector action


def test_recovers_similarity_in_2d():
    rng = np.random.default_rng(0)
    k = heptagon()
    F = 50
    t = rng.normal(size=(F, 2)) * 3
    s = rng.uniform(0.6, 1.5, F)
    th = rng.uniform(-40, 40, F)
    x = np.stack([s[i] * k @ rot2(th[i]) + t[i] + 0.01 * rng.normal(size=(7, 2))
                  for i in range(F)])
    L = ladder(x, k)
    assert np.allclose(L["t"], t, atol=0.02)
    assert np.allclose(L["s_rot"], s, atol=0.02)
    assert np.allclose(L["theta_deg"], th, atol=0.5)
    assert not L["reflect"].any()
    assert (L["sseTSR"] < 0.02).all()


def test_ladder_is_monotone_and_step_s_is_scale_only():
    rng = np.random.default_rng(1)
    k = heptagon()
    x = np.stack([1.3 * k + np.array([2.0, -1.0]) + 0.3 * rng.normal(size=(7, 2))
                  for _ in range(20)])
    L = ladder(x, k)
    assert (L["sse0"] >= L["sseT"]).all()
    assert (L["sseT"] >= L["sseTS"] - 1e-9).all()
    assert (L["sseTS"] >= L["sseTSR"] - 1e-9).all()
    assert np.allclose(L["s"], 1.3, atol=0.05)
    assert np.abs(L["theta_deg"]).max() < 5


def test_reflection_is_flagged_not_reported_as_rotation():
    k = heptagon()
    x = (k * np.array([1.0, -1.0]))[None]              # mirror image: day order reversed
    L = ladder(x, k)
    assert L["reflect"][0]
    assert np.isclose(np.linalg.det(L["R"][0]), 1.0)
    # a proper rotation fits a mirror image badly; an unconstrained fit with reflection fits it exactly
    assert L["sseTSR"][0] > 0.5 * L["sseTS"][0]


def test_angle_convention_matches_row_vector_action():
    R = rot2(30)[None]
    assert np.isclose(rotation_angle_deg(R)[0], 30)
    k = heptagon()
    y = k @ rot2(30)
    R2, s, refl = procrustes_proper(k, y[None])
    assert np.isclose(rotation_angle_deg(R2)[0], 30, atol=1e-6) and np.isclose(s[0], 1)
    assert not refl[0]


def test_6d_rotation_recovered():
    rng = np.random.default_rng(2)
    k = heptagon(d=6) + 0.5 * rng.normal(size=(7, 6)); k -= k.mean(0)
    Q, _ = np.linalg.qr(rng.normal(size=(6, 6)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    x = (0.8 * k @ Q + 1.5)[None]
    L = ladder(x, k)
    assert np.allclose(L["R"][0], Q, atol=1e-6)
    assert np.isclose(L["s_rot"][0], 0.8, atol=1e-6)
    assert L["sseTSR"][0] < 1e-9 and not L["reflect"][0]


def test_null_gain_brackets_a_translation_only_world():
    rng = np.random.default_rng(3)
    k = heptagon()
    F = 300
    x = np.stack([k + rng.normal(size=2) * 4 + 0.6 * rng.normal(size=(7, 2)) for _ in range(F)])
    L = ladder(x, k)
    obs = (L["sseTS"] - L["sseTSR"]).sum()
    gain, theta = noise_null(k, L["s"], L["sseTS"], 40, rng)
    lo, hi = np.percentile(gain.sum(1), [2.5, 97.5])
    assert lo <= obs <= hi
    assert (gain > -1e-9).all()
    assert 0 < np.median(np.abs(theta)) < 15


def test_stage_distances_shapes_and_hits():
    rng = np.random.default_rng(4)
    k = heptagon()
    x = np.stack([k + np.array([6.0, 0.0]) + 0.05 * rng.normal(size=(7, 2)) for _ in range(10)])
    L = ladder(x, k)
    dist, hit = stage_distances(L, k)
    assert dist["raw"].shape == (10, 7) and hit["T"].shape == (10, 7)
    assert hit["T"].mean() == 1.0 and hit["T+S+R"].mean() == 1.0
    assert np.median(dist["raw"]) > np.median(dist["T"])


def test_spoke_distance_is_scale_invariant_and_zero_on_the_spoke():
    rng = np.random.default_rng(5)
    k = heptagon()
    x = np.stack([1.4 * k + np.array([3.0, -2.0]) for _ in range(5)])      # on the spokes
    L = ladder(x, k)
    dist, hit = spoke_distances(L, k)
    assert np.allclose(dist["T"], 0, atol=1e-6) and hit["T"].all()
    assert np.allclose(dist["T"], dist["T+S"])                         # scale leaves spokes
    assert (dist["raw"] > 0).any()                                     # the shift does not


def test_split_is_document_disjoint_and_seeded():
    cls = np.array(["positive", "dirty_positive", "floor", "positive"] * 50)
    doc = np.array([f"d{i // 2}" for i in range(200)])
    f, h = fit_mask(cls, doc), held_out_mask(cls, doc)
    assert not (f & h).any()
    assert set(doc[f]).isdisjoint(set(doc[h]))
    assert not (f | h)[cls == "floor"].any()
    assert fit_documents(cls, doc, seed=0) == fit_documents(cls, doc, seed=0)
    assert fit_documents(cls, doc, seed=0) != fit_documents(cls, doc, seed=1)
    n_docs = len(set(doc[cls != "floor"]))
    assert len(fit_documents(cls, doc)) == round(0.34 * n_docs)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
