"""Figures 1, 5, 6 and the methods schematic render end to end from synthetic captures. CPU only."""
import os
import sys

import pytest

import synthetic_captures as syn

HERE = os.path.dirname(os.path.abspath(__file__))
for d in ("fig1_sensitivity", "methods_schematic", "fig5_models", "fig6_arithmetic"):
    sys.path.insert(0, os.path.join(HERE, "..", "experiments", d))

pytest.importorskip("matplotlib")


def _rendered(out):
    """The PNG the plate was asked for and the PDF it writes beside it, both non-empty."""
    pdf = os.path.splitext(str(out))[0] + ".pdf"
    assert os.path.exists(out) and os.path.getsize(out) > 0, out
    assert os.path.exists(pdf) and os.path.getsize(pdf) > 0, pdf


def test_fig1_sensitivity_renders(tmp_path, monkeypatch):
    import figure_sensitivity
    out = tmp_path / "fig1_sensitivity.png"
    monkeypatch.setattr(sys, "argv", [
        "x", "--cascade", syn.ring_walk(tmp_path / "ring_walk.npz"),
        "--field", syn.polar_field(tmp_path / "jacobian_field.npz"),
        "--out", str(out), "--dpi", "60"])
    figure_sensitivity.main()
    _rendered(out)


def test_methods_schematic_renders(tmp_path, monkeypatch):
    import figure_schematic
    out = tmp_path / "fig4_schematic.png"
    monkeypatch.setattr(sys, "argv", [
        "x", "--field", syn.polar_field(tmp_path / "jacobian_field.npz"),
        "--out", str(out), "--dpi", "60"])
    figure_schematic.main()
    _rendered(out)


def test_fig5_models_renders(tmp_path, monkeypatch, capsys):
    import figure_models
    out = tmp_path / "fig5_models.png"
    monkeypatch.setattr(sys, "argv", [
        "x", "--llama", syn.polar_field(tmp_path / "llama.npz", with_off=True, seed=1),
        "--mistral", syn.polar_field(tmp_path / "mistral.npz", with_off=True, seed=2,
                                     model="mistralai/Mistral-7B-v0.1"),
        "--gpt2xl", syn.polar_field(tmp_path / "gpt2xl.npz", with_off=True, seed=3, d=48,
                                    model="gpt2-xl", readout_layer=42),
        "--out", str(out), "--dpi", "40"])
    figure_models.main()
    _rendered(out)
    printed = capsys.readouterr().out
    assert "GPT-2 XL (L2 -> L42)" in printed and "Mistral-7B (L2 -> L28)" in printed


def test_fig6_arithmetic_renders(tmp_path, monkeypatch, capsys):
    import figure_arithmetic
    out = tmp_path / "fig6_arithmetic.png"
    fam = "interrogative"
    monkeypatch.setattr(sys, "argv", [
        "x", "--rings", syn.rings(tmp_path / "rings.npz", formulation=fam),
        "--walk", syn.ring_walk(tmp_path / "ring_walk.npz", formulation=fam),
        "--field", syn.polar_field(tmp_path / "jacobian_field.npz", with_off=True,
                                   formulation=fam),
        "--out", str(out), "--dpi", "40"])
    figure_arithmetic.main()
    _rendered(out)
    printed = capsys.readouterr().out
    # the synthetic last steer layer is unstructured, so panel D must report dropping it
    assert "panel D excludes" in printed and f"L{syn.LAYERS[-1]}" in printed


def test_fig6_rejects_mismatched_halves(tmp_path, monkeypatch):
    import figure_arithmetic
    monkeypatch.setattr(sys, "argv", [
        "x", "--rings", syn.rings(tmp_path / "rings.npz"),
        "--walk", syn.ring_walk(tmp_path / "ring_walk.npz", formulation="interrogative"),
        "--field", syn.polar_field(tmp_path / "field.npz", with_off=True),   # mention family
        "--out", str(tmp_path / "x.png")])
    with pytest.raises(AssertionError, match="different prompt families"):
        figure_arithmetic.main()
