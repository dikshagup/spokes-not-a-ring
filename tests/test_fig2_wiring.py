"""Figure 2 renders from a synthetic capture and a synthetic ladder summary (CPU only)."""
import os
import sys

import numpy as np
import pytest

pytest.importorskip("matplotlib")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in ("src", "experiments/common", "experiments/fig2_timeofday"):
    sys.path.insert(0, os.path.join(ROOT, p))

from capture_prompts import build_prompts  # noqa: E402


def synthetic_capture(path, d=48, seed=0):
    rng = np.random.default_rng(seed)
    rows = build_prompts()
    day = np.array([r[1] for r in rows]); fam = np.array([r[2] for r in rows])
    lab = np.array([r[3] for r in rows]); hour = np.array([r[4] for r in rows], float)
    ang = 2 * np.pi * day / 7
    A = np.zeros((len(rows), d))
    A[:, 0], A[:, 1] = 8 * np.cos(ang), 8 * np.sin(ang)            # the weekday ring
    h = np.where(np.isnan(hour), 12.0, hour)
    A[:, 2] = 6 * (h - 12) / 12                                     # the hour axis
    A[:, 3] = 2 * np.sin(2 * np.pi * h / 24)
    A += 0.3 * rng.normal(size=A.shape)
    np.savez(path, A=A.astype(np.float32), day=day, fam=fam, lab=lab, hour=hour,
             layer=28, pos=6, n_tokens=8, texts=np.array([r[0] for r in rows]),
             model="synthetic")


def synthetic_ladder(ladder_dir, seed=0):
    import pandas as pd
    rng = np.random.default_rng(seed)
    os.makedirs(ladder_dir, exist_ok=True)
    rows = []
    carriers = [f"carrier {i}" for i in range(4)]
    for arm in ("manifold", "late_perp", "rand_perp"):
        for alpha in (-0.5, -0.3, -0.15, 0.0, 0.15, 0.3, 0.5):
            for c in carriers:
                for d in range(7):
                    eff = 2.0 * alpha if arm == "late_perp" else 0.0
                    rows.append(dict(carrier=c, day=d, alpha=alpha, arm=arm,
                                     d_cm=eff + 0.05 * rng.normal(), cm=15.0, cm0=15.0,
                                     conc=0.3, captured=0.5, ambiguous=0.1,
                                     displacement=28 * abs(alpha) + 0.1, target=28 * abs(alpha) + 0.1,
                                     cm_am=14.0, cm_pm=16.0))
    pd.DataFrame(rows).to_csv(os.path.join(ladder_dir, "rows.csv"), index=False)
    from steer_ladder import summarise
    summarise(os.path.join(ladder_dir, "rows.csv"), os.path.join(ladder_dir, "summary.csv"),
              n_boot=50)


def test_plate_and_numbers_render(tmp_path, monkeypatch):
    data = tmp_path / "fig2_timeofday"
    data.mkdir()
    synthetic_capture(data / "timeofday_L28.npz")
    synthetic_ladder(str(data / "ladder"))

    import analysis
    out_json = tmp_path / "numbers.json"
    monkeypatch.setattr(sys, "argv", ["x", "--data-dir", str(data), "--out", str(out_json)])
    assert analysis.main() == 0
    import json
    n = json.load(open(out_json))
    assert n["ladder"]["n_carriers"] == 4
    # the planted hour axis is orthogonal to the planted ring
    assert n["orthogonality"]["hour_variation_in_weekday_plane"] < 0.05
    assert {r["role"] for r in n["pca"]} >= {"weekday plane", "hour axis"}

    import plate
    out = tmp_path / "fig2_time_of_day.pdf"
    monkeypatch.setattr(sys, "argv", ["x", "--data-dir", str(data), "--out", str(out), "--also-png"])
    assert plate.main() == 0
    assert out.exists() and out.with_suffix(".png").exists()


def test_readout_forms_pin_the_hour():
    from hour_readout import unambiguous_forms, ambiguous_prefixes, circ_diff
    forms = unambiguous_forms()
    assert len(forms) == len({s for s, _ in forms})
    assert all(0 <= v < 24 for _, v in forms)
    assert (" 7pm", 19.0) in forms and (" noon", 12.0) in forms
    assert all(":" in s or "o'clock" in s or s.strip().isdigit() for s, _ in ambiguous_prefixes())
    assert circ_diff(1.0, 23.0) == 2.0 and circ_diff(23.0, 1.0) == -2.0
