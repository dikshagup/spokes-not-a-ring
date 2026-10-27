"""Tests for the recording layer: None reads the output logits, R reads blocks.{R}.hook_resid_post."""

import pytest

from weekday_manifold.plateau.config import PlateauConfig


def test_record_layer_defaults_to_last_block():
    cfg = PlateauConfig()
    assert cfg.record_layer == -1
    resolved = cfg.resolve(n_layers=48)
    assert resolved.record_layer == 47
    assert resolved.record_hook_name == "blocks.47.hook_resid_post"


def test_negative_record_layer_unresolved_raises():
    with pytest.raises(ValueError):
        _ = PlateauConfig().record_hook_name  # -1 before resolve()


def test_record_layer_none_means_logits():
    cfg = PlateauConfig(record_layer=None)
    assert cfg.record_hook_name is None


def test_record_hook_name_for_residual_layer():
    cfg = PlateauConfig(record_layer=47)
    assert cfg.record_hook_name == "blocks.47.hook_resid_post"


def test_record_layer_survives_json_roundtrip(tmp_path):
    cfg = PlateauConfig(layer_index=4, record_layer=20)
    path = tmp_path / "cfg.json"
    cfg.to_json(str(path))
    loaded = PlateauConfig.from_json(str(path))
    assert loaded == cfg
    assert loaded.record_hook_name == "blocks.20.hook_resid_post"


def test_record_layer_not_in_cache_key():
    # the recording point changes the swept distance, not the captured activations
    base = PlateauConfig(layer_index=4, record_layer=None)
    recd = PlateauConfig(layer_index=4, record_layer=47)
    assert base.cache_key() == recd.cache_key()
