"""
Loading, saving and converting IgFold model weights.

IgFold weights are `safetensors` files with the model architecture config stored as JSON in the
file metadata under the key ``"config"``. Legacy PyTorch-Lightning ``.ckpt`` files (IgFold <= 0.4.0)
can be converted with :func:`convert_legacy_checkpoint`.
"""

import glob
import json
import os
import pickle
import types
import warnings
from typing import Dict, List, Optional, Tuple, Union

import torch
from safetensors.torch import load_file, safe_open, save_file

import igfold
from igfold.utils.general import exists

WEIGHTS_EXT = ".safetensors"
LEGACY_EXT = ".ckpt"
WEIGHTS_DIR_ENV = "IGFOLD_WEIGHTS_DIR"

# Keys that define the network architecture.
MODEL_CONFIG_KEYS = (
    "node_dim",
    "depth",
    "gt_depth",
    "gt_heads",
    "temp_ipa_depth",
    "temp_ipa_heads",
    "str_ipa_depth",
    "str_ipa_heads",
    "dev_ipa_depth",
    "dev_ipa_heads",
)


def get_default_weights_dir() -> str:
    """
    Directory searched for pre-trained weights: ``$IGFOLD_WEIGHTS_DIR`` if set,
    otherwise ``trained_models/IgFold`` inside the installed package.
    """
    env_dir = os.environ.get(WEIGHTS_DIR_ENV)
    if exists(env_dir) and len(env_dir) > 0:
        return env_dir

    package_dir = os.path.dirname(os.path.realpath(igfold.__file__))
    return os.path.join(package_dir, "trained_models", "IgFold")


def find_weights(
    weights_dir: Optional[str] = None,
    num_models: Optional[int] = None,
) -> List[str]:
    """
    Locate pre-trained weight files.

    :param weights_dir: Directory to search (defaults to :func:`get_default_weights_dir`).
    :param num_models: If given, return at most this many files (sorted by name).
    :raises FileNotFoundError: if no weight files are present.
    """
    weights_dir = weights_dir if exists(weights_dir) else get_default_weights_dir()

    paths = sorted(glob.glob(os.path.join(weights_dir, f"*{WEIGHTS_EXT}")))
    if len(paths) == 0:
        legacy = sorted(glob.glob(os.path.join(weights_dir, f"*{LEGACY_EXT}")))
        if len(legacy) > 0:
            warnings.warn(
                f"Found only legacy .ckpt weights in {weights_dir}. Converting them to "
                f"{WEIGHTS_EXT}; run `igfold convert-weights` once to silence this warning."
            )
            paths = [convert_legacy_checkpoint(p) for p in legacy]
        else:
            raise FileNotFoundError(
                f"No IgFold weights found in {weights_dir}. Pre-trained weights are distributed "
                f"with the PyPI package (`pip install igfold`), not the GitHub repository. Either "
                f"install from PyPI, copy the `trained_models/IgFold` directory into the package, "
                f"or point {WEIGHTS_DIR_ENV} at a directory containing the weights."
            )

    if exists(num_models):
        paths = paths[:num_models]

    return paths


def model_config_from_dict(config: Dict) -> Dict:
    """Extract the architecture-defining subset of a (possibly larger) config dict."""
    missing = [k for k in MODEL_CONFIG_KEYS if k not in config]
    if len(missing) > 0:
        raise KeyError(f"IgFold config is missing required keys: {missing}")

    return {k: int(config[k]) for k in MODEL_CONFIG_KEYS}


def save_model(model, path: str) -> str:
    """Save an IgFold model as a self-describing safetensors file."""
    state_dict = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    metadata = {"config": json.dumps(model_config_from_dict(model.config)), "format": "pt"}
    save_file(state_dict, path, metadata=metadata)

    return path


def load_model(
    path: str,
    device: Union[str, torch.device] = "cpu",
    strict: bool = True,
):
    """
    Load an IgFold model from a safetensors (or legacy .ckpt) file.

    :param path: Path to weights.
    :param device: Device to place the model on.
    :param strict: Passed to ``load_state_dict``.
    """
    from igfold.model.IgFold import IgFold  # local import to avoid a cycle at package import time

    if path.endswith(LEGACY_EXT):
        config, state_dict = read_legacy_checkpoint(path)
    else:
        with safe_open(path, framework="pt", device="cpu") as f:
            metadata = f.metadata() or {}
        if "config" not in metadata:
            raise ValueError(
                f"{path} has no 'config' entry in its safetensors metadata; is this an IgFold weights file?"
            )
        config = json.loads(metadata["config"])
        state_dict = load_file(path, device="cpu")

    model = IgFold(model_config_from_dict(config))
    model.load_state_dict(state_dict, strict=strict)
    model.eval()

    return model.to(torch.device(device))


class _Stub:
    """Placeholder for pickled objects that are not needed to rebuild the model."""

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        pass

    def __call__(self, *args, **kwargs):
        return _Stub()


class _LenientUnpickler(pickle.Unpickler):
    """Unpickler that replaces any class from ``transformers`` with a stub (legacy checkpoints
    contain pickled tokenizer/config objects that are not needed to rebuild the model)."""

    def find_class(self, module, name):
        if module.startswith("transformers"):
            return _Stub
        return super().find_class(module, name)


_lenient_pickle = types.SimpleNamespace(
    Unpickler=_LenientUnpickler,
    load=lambda f, **kw: _LenientUnpickler(f, **kw).load(),
    __name__="igfold_lenient_pickle",
)


def read_legacy_checkpoint(path: str) -> Tuple[Dict, Dict[str, torch.Tensor]]:
    """Read a legacy PyTorch-Lightning ``.ckpt`` and return ``(config, state_dict)``."""
    checkpoint = torch.load(
        path,
        map_location="cpu",
        pickle_module=_lenient_pickle,
        weights_only=False,
    )
    hparams = checkpoint.get("hyper_parameters", {})
    config = hparams.get("config", hparams)

    return model_config_from_dict(config), checkpoint["state_dict"]


def convert_legacy_checkpoint(ckpt_path: str, out_path: Optional[str] = None) -> str:
    """
    Convert a legacy ``.ckpt`` file to safetensors. Returns the output path
    (default: same name with a ``.safetensors`` extension).
    """
    if not exists(out_path):
        out_path = os.path.splitext(ckpt_path)[0] + WEIGHTS_EXT

    config, state_dict = read_legacy_checkpoint(ckpt_path)
    state_dict = {k: v.contiguous() for k, v in state_dict.items()}
    save_file(state_dict, out_path, metadata={"config": json.dumps(config), "format": "pt"})

    return out_path


if __name__ == "__main__":
    import sys

    from igfold.cli import main

    sys.exit(main(["convert-weights", *sys.argv[1:]]))
