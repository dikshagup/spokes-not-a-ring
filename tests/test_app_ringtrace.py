"""Tests for the ring-trace appendix: the sentence-end rule, the role proxy and the plate rendering."""

import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
for d in ("experiments/common", "experiments/app_ringtrace"):
    sys.path.insert(0, os.path.join(HERE, "..", d))

from select_period_windows import is_sentence_end, role_proxy, sentence_start  # noqa: E402

PERIOD, PERIOD_NL = 13, 627          # a lone "." and a merged ".\n", as in Llama's vocabulary
FAM = {PERIOD: ".", PERIOD_NL: ".\n"}


def tokenise(pieces):
    """A fake tokenisation: each piece is one token, ids are arbitrary except the periods."""
    text, offsets, ids = "", [], []
    for p in pieces:
        offsets.append((len(text), len(text) + len(p)))
        text += p
        ids.append(PERIOD_NL if p == ".\n" else PERIOD if p == "." else 1000 + len(ids))
    return ids, pieces, text, offsets


def ends(pieces, i):
    ids, pcs, text, off = tokenise(pieces)
    return is_sentence_end(i, ids, pcs, text, off, FAM)


def test_plain_sentence_end_before_a_capital():
    assert ends(["It", " was", " Monday", ".", " The", " end"], 3)


def test_end_of_document_counts():
    assert ends(["It", " was", " Monday", "."], 3)


def test_merged_period_newline_token_counts():
    assert ends(["It", " was", " Monday", ".\n", "Next", " line"], 3)


def test_decimal_point_is_not_a_sentence_end():
    assert not ends(["It", " cost", " 3", ".", "5", " dollars"], 3)


def test_abbreviation_and_initial_are_not_sentence_ends():
    assert not ends(["Call", " Dr", ".", " Smith"], 2)
    assert not ends(["in", " the", " U", ".", "S", ".", " today"], 3)
    assert not ends(["e", ".", "g", ".", " this"], 1)


def test_lowercase_continuation_is_not_a_sentence_end():
    assert not ends(["It", " was", " Monday", ".", " and", " then"], 3)


def test_opening_quote_after_period_counts():
    assert ends(["She", " left", " Monday", ".", ' "', "Fine", '"'], 3)


def test_sentence_start_walks_back_to_the_previous_closer():
    ids, pcs, text, off = tokenise(["Hi", ".", " It", " was", " Monday", "."])
    assert sentence_start(5, ids, pcs, FAM) == 1
    ids, pcs, text, off = tokenise(["It", " was", " Monday", "."])
    assert sentence_start(3, ids, pcs, FAM) == -1


def test_role_proxy():
    assert role_proxy("") == "initial"
    assert role_proxy("on") == "adverbial"
    assert role_proxy("On") == "adverbial"
    assert role_proxy("Market") == "incidental"


def synthetic_trace(path, n=140, layers=4, slots=16, k=6, seed=0):
    """A tiny trace_n16.npz at the real key schema, day structure in the first two coordinates."""
    rng = np.random.default_rng(seed)
    days = np.array(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    cls = np.array(["positive_clean"] * 84 + ["matched_negative_far"] * 28 + ["floor"] * 28)
    day = np.where(cls == "positive_clean", days[np.arange(n) % 7], "")
    doc = np.array([f"d{i // 2}" for i in range(n)])
    dix = np.arange(n) % 7
    ang = 2 * np.pi * dix / 7
    z = rng.normal(size=(n, layers, slots, k)).astype(np.float32)
    z[:, :, :, 0] += 5 * np.cos(ang)[:, None, None]
    z[:, :, :, 1] += 5 * np.sin(ang)[:, None, None]
    cents = np.zeros((layers, slots, 7, k), np.float32)
    cents[:, :, :, 0] = 5 * np.cos(2 * np.pi * np.arange(7) / 7)
    cents[:, :, :, 1] = 5 * np.sin(2 * np.pi * np.arange(7) / 7)
    is_fit = (cls == "positive_clean") & (np.arange(n) % 3 == 0)
    fit_doc = np.isin(doc, doc[is_fit])
    np.savez(path, z_own=z, z_tr=z, norm_own=np.full((n, layers, slots), 20, np.float32),
             norm_tr=np.full((n, layers, slots), 20, np.float32),
             valid=np.ones((n, slots), bool), slot_pos=np.zeros((n, slots), np.int8),
             cell_ok=np.ones(slots, bool), fit_counts=np.full((7, slots), 4),
             cents_own=cents, cents_tr=cents, is_fit=is_fit, fit_doc=fit_doc,
             layers=np.arange(layers), n=16, n_dims=k, cls=cls, day=day, doc=doc,
             dist=np.full(n, 3), anchor_idx16=np.full(n, 12))
    return str(path)


def test_plate_renders_from_synthetic_trace(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    import figure_ring_trace
    out = tmp_path / "fig_ring_trace.pdf"
    monkeypatch.setattr(sys, "argv", ["x", "--trace", synthetic_trace(tmp_path / "trace.npz"),
                                      "--out", str(out), "--layer", "3", "--max-offset", "4",
                                      "--strip", "0,1,2,15", "--also-png"])
    # write the numbers table to tmp instead of the repo's results directory
    monkeypatch.setattr(figure_ring_trace, "REPO", str(tmp_path))
    figure_ring_trace.main()
    assert out.exists() and (tmp_path / "fig_ring_trace.png").exists()
    assert (tmp_path / "experiments" / "results" / "fig_ring_trace_numbers.json").exists()
