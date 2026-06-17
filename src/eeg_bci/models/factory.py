from __future__ import annotations

import difflib
from collections.abc import Mapping
from typing import Any

import braindecode.models as bd_models
from omegaconf import DictConfig, OmegaConf

from eeg_bci.data.types import DatasetInfo


def build_model(model_cfg: DictConfig, dataset_info: DatasetInfo):
    params = _get_model_params(model_cfg)
    model_cls = _get_braindecode_model(str(model_cfg.name))
    return model_cls(
        n_chans=dataset_info.n_chans,
        n_outputs=dataset_info.n_outputs,
        n_times=dataset_info.n_times,
        **params,
    )


def _get_model_params(model_cfg: DictConfig) -> dict[str, Any]:
    params = OmegaConf.to_container(model_cfg.get("params", {}), resolve=True)
    if not isinstance(params, Mapping):
        raise ValueError("Model params must be a mapping/dictionary.")
    if not all(isinstance(key, str) for key in params):
        raise ValueError("Model params keys must be strings.")
    return dict(params)


def _get_braindecode_model(name: str) -> type[Any]:
    available_models = sorted(
        attr_name
        for attr_name in dir(bd_models)
        if attr_name[:1].isupper() and isinstance(getattr(bd_models, attr_name), type)
    )
    models_by_lower_name = {
        attr_name.lower(): getattr(bd_models, attr_name)
        for attr_name in available_models
    }

    exact_model = getattr(bd_models, name, None)
    if isinstance(exact_model, type):
        return exact_model

    model_cls = models_by_lower_name.get(name.lower())
    if model_cls is not None:
        return model_cls

    suggestions = difflib.get_close_matches(name, available_models, n=3)
    help_text = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
    raise ValueError(f"Unsupported Braindecode model '{name}'.{help_text}")
