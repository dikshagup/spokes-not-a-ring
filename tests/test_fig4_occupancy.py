"""Figure 4's plate and numbers scripts render from a tiny synthetic capture. CPU only."""
import json
import os
import sys

import numpy as np
import pytest

pytest.importorskip("matplotlib")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments", "fig4_occupancy"))

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
META = ("pos", "url", "pn_side", "title_case", "left_month", "left_coref", "left_year",
        "right_month", "right_year", "would_demote_on_year")


def synthetic_corpus(d, n_fam=60, n_ctrl=80, dim=32, seed=0):
    """raw, swap and capture-list npz files at the real key schema, days on a planar heptagon."""
    rng = np.random.default_rng(seed)
    ang = 2 * np.pi * np.arange(7) / 7
    cents = np.zeros((7, dim)); cents[:, 0] = 6 * np.cos(ang); cents[:, 1] = 6 * np.sin(ang)
    cls = np.array(["positive"] * (n_fam // 2) + ["dirty_positive"] * (n_fam - n_fam // 2)
                   + ["matched_negative"] * n_ctrl + ["near_miss"] * 10 + ["floor"] * 10)
    N = len(cls)
    doc = np.array([f"doc{i // 2}" for i in range(N)])
    day_ix = rng.integers(0, 7, N)
    day = np.array(DAYS)[day_ix]
    shift = rng.normal(size=(N, dim)) * 2
    acts = np.where(np.isin(cls, ["positive", "dirty_positive"])[:, None],
                    cents[day_ix], rng.normal(size=(N, dim)) * 3) + shift
    acts = acts + 0.3 * rng.normal(size=(N, dim))
    row = np.arange(N)
    np.savez(d / "raw_L28_n16.npz", acts=acts.astype(np.float16), cls=cls, day=day, doc=doc,
             is_fit=np.zeros(N, bool), row=row, layer=28, n=16, site="final_token",
             model="synthetic", dtype="bfloat16", prepend_bos=True, d_model=dim, split_seed=0)
    fam_rows = np.where(np.isin(cls, ["positive", "dirty_positive"]))[0]
    F = len(fam_rows)
    sw = np.tile(np.arange(7, dtype=np.int8), F)
    swap_acts = cents[sw] + np.repeat(shift[fam_rows], 7, axis=0) + 0.3 * rng.normal(size=(7 * F, dim))
    out = dict(acts=swap_acts.astype(np.float16), fam=np.repeat(np.arange(F, dtype=np.int32), 7),
               swap=sw, is_self=np.repeat(day_ix[fam_rows], 7) == sw, days=np.array(DAYS),
               fam_row=fam_rows.astype(np.int32), fam_cls=cls[fam_rows], fam_day=day[fam_rows],
               fam_form=np.array(["spaced"] * F), fam_doc=doc[fam_rows],
               fam_is_fit=np.zeros(F, bool), n=16, n_max=256, model="synthetic", split_seed=0,
               layer=28, site="final_token", dtype="bfloat16", prepend_bos=True, d_model=dim)
    for k in META:
        out["fam_" + k] = np.array([""] * F, dtype=object)
    np.savez(d / "swap_L28_n16.npz", **out)
    ids = rng.integers(1000, 2000, size=(N, 256)).astype(np.int32)
    np.savez(d / "capture_list_n256.npz", ids=ids, n_max=256, model="synthetic", seed=0,
             cls=cls, day=day, doc=doc, **{k: np.array([""] * N, dtype=object) for k in META})
    return N, F


def _run(module, argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["x"] + [str(a) for a in argv])
    assert module.main() == 0


def test_plate_and_numbers_render_from_synthetic_capture(tmp_path, monkeypatch):
    import figure_occupancy, alignment_numbers
    d = tmp_path / "data"; d.mkdir()
    synthetic_corpus(d)
    out_pdf = tmp_path / "fig4_occupancy.pdf"
    _run(figure_occupancy, ["--in", d, "--model", "no-such-tokenizer", "--max-points", 50,
                            "--max-contexts", 20, "--grid", 700, "--out", out_pdf,
                            "--results", tmp_path / "plate.json"], monkeypatch)
    assert out_pdf.exists() and (tmp_path / "fig4_occupancy.png").exists()
    plate = json.load(open(tmp_path / "plate.json"))
    assert plate["n_fit"] + plate["n_held"] == 60
    assert 0.5 < plate["own_centroid_rate"] <= 1.0          # the synthetic days are separable
    _run(alignment_numbers, ["--in", d, "--model", "no-such-tokenizer", "--n-null", 5,
                             "--results", tmp_path / "align.json"], monkeypatch)
    al = json.load(open(tmp_path / "align.json"))
    s = al["share"]["2D"]
    assert abs(sum(s.values()) - 1) < 1e-9 and s["translation"] > 0.5
    assert set(al["spoke"]) == {"raw", "T", "T+S", "T+S+R"}
