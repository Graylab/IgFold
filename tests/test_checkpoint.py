import json
import os

import pytest
import torch
from conftest import requires_weights
from safetensors.torch import safe_open

from igfold.model.IgFold import IgFold
from igfold.utils.checkpoint import (
    MODEL_CONFIG_KEYS,
    find_weights,
    load_model,
    model_config_from_dict,
    save_model,
)

SMALL_CONFIG = {
    "node_dim": 16,
    "depth": 1,
    "gt_depth": 1,
    "gt_heads": 2,
    "temp_ipa_depth": 1,
    "temp_ipa_heads": 2,
    "str_ipa_depth": 1,
    "str_ipa_heads": 2,
    "dev_ipa_depth": 1,
    "dev_ipa_heads": 2,
}


def test_model_config_from_dict_drops_extra_keys():
    cfg = model_config_from_dict({**SMALL_CONFIG, "initial_lr": 1e-3, "tokenizer": object()})
    assert set(cfg) == set(MODEL_CONFIG_KEYS)


def test_model_config_from_dict_missing_key():
    with pytest.raises(KeyError):
        model_config_from_dict({"node_dim": 16})


def test_save_and_load_roundtrip(tmp_path):
    model = IgFold(SMALL_CONFIG)
    path = save_model(model, str(tmp_path / "small.safetensors"))

    with safe_open(path, "pt") as f:
        assert json.loads(f.metadata()["config"]) == SMALL_CONFIG

    loaded = load_model(path, device="cpu")
    assert not loaded.training
    for k, v in model.state_dict().items():
        assert torch.equal(v, loaded.state_dict()[k])


def test_load_rejects_file_without_config(tmp_path):
    from safetensors.torch import save_file

    path = str(tmp_path / "bad.safetensors")
    save_file({"w": torch.zeros(2)}, path)
    with pytest.raises(ValueError):
        load_model(path)


def test_find_weights_missing_dir_has_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="pip install igfold"):
        find_weights(str(tmp_path))


def test_find_weights_env_override(tmp_path, monkeypatch):
    model = IgFold(SMALL_CONFIG)
    save_model(model, str(tmp_path / "a.safetensors"))
    save_model(model, str(tmp_path / "b.safetensors"))
    monkeypatch.setenv("IGFOLD_WEIGHTS_DIR", str(tmp_path))

    assert [os.path.basename(p) for p in find_weights()] == ["a.safetensors", "b.safetensors"]
    assert len(find_weights(num_models=1)) == 1


@requires_weights
def test_pretrained_weights_load():
    paths = find_weights()
    assert 1 <= len(paths) <= 4
    model = load_model(paths[0])
    assert sum(p.numel() for p in model.parameters()) == 1_557_238
