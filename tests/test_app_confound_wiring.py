"""Tests that figure_confound.py renders from a small synthetic capture on CPU, and tests
the two metric helpers."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "experiments", "app_confound"))

from masked_kl import MaskedKL, kl_direct          # noqa: E402
from periodicity import seven_fold                  # noqa: E402

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
G, V = 56, 400


class FakeTokenizer:
    """Ids 0-6 are ' Dayname', 7-13 'Dayname', 14-20 abbreviations, the rest filler (last is Hangul)."""

    def __init__(self):
        self.vocab = [" " + d for d in DAYS] + DAYS + [" " + d[:3] for d in DAYS]
        self.vocab += [f"w{i}" for i in range(len(self.vocab), V - 1)] + ["헌"]
        self.index = {s: i for i, s in enumerate(self.vocab)}

    def encode(self, s, add_special_tokens=False):
        return [self.index[s]] if s in self.index else [1, 2]

    def decode(self, ids):
        return "".join(self.vocab[i] for i in ids)


def synthetic_capture(path, seed=0):
    rng = np.random.default_rng(seed)
    u = np.linspace(0, 1, G, endpoint=False)
    phase = 2 * np.pi * 7 * u
    L = rng.normal(0, 0.3, (G, V))
    L[:, 30:60] += 2.0 * np.cos(phase)[:, None] * rng.uniform(0.5, 1.5, 30)   # periodic, non-weekday
    for k in range(7):                       # day names: a jump at their own day's crossing, then a decay
        x = (u * 7 - (k - 0.4)) % 7
        L[:, k] += 6.0 * np.exp(-x / 0.6)
        L[:, 7 + k] += 4.0 * np.exp(-x / 0.6)
    L[:, -1] += 1.5 * np.cos(phase)
    np.savez(path, u=u, gain=1 + np.cos(phase) ** 2, out_logits=L.astype(np.float16),
             day_ids=np.arange(21), meta='{"model": "synthetic"}')
    return path


def test_plate_renders_from_synthetic_capture(tmp_path):
    import figure_confound as fc
    from paper_style import helvetica

    z = np.load(synthetic_capture(str(tmp_path / "cap.npz")), allow_pickle=True)
    tok = FakeTokenizer()
    r = fc.analyse(z, tok, n_draws=20)
    assert len(r["cid"]) == 14 and len(r["per7"]) == 7
    assert not set(r["per7"]) & set(range(21)), "the most periodic non-day tokens must exclude every weekday spelling"
    assert all(r["own_hit"]), "a day name whose only bump is at its own day must peak at its own crossing"
    with helvetica.context(style=None):
        fig = fc.render(r, tok)
        out = tmp_path / "fig_confound.png"
        fig.savefig(out, dpi=50)
    assert out.stat().st_size > 0
    text = fc.caption(r, tok, 20)
    assert "(E)" in text and "U+D5CC" in fc.glyphs(tok.decode([V - 1]))
    assert "crossing" in fc.TITLES["D"] and "crossing" in fc.TITLES["E"]


def test_masked_kl_matches_direct():
    rng = np.random.default_rng(1)
    L = rng.normal(0, 2, (G, 50))
    idx = np.array([3, 7, 11])
    mk = MaskedKL(L)
    assert np.allclose(mk(idx), kl_direct(L, idx), atol=1e-10)
    assert np.allclose(mk(np.array([], int)), kl_direct(L, []), atol=1e-10)


def test_seven_fold_share():
    u = np.linspace(0, 1, G, endpoint=False)
    periodic = np.cos(2 * np.pi * 7 * u)
    assert seven_fold(periodic) == pytest.approx(1.0)
    assert seven_fold(np.cos(2 * np.pi * u)) < 0.05           # one cycle around the whole ring is not day-periodic
