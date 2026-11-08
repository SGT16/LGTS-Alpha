from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn


class NumericalError(FloatingPointError):
    pass


def require_finite(value, name: str) -> None:
    finite = torch.isfinite(value).all().item() if isinstance(value, torch.Tensor) else np.isfinite(value).all()
    if not finite:
        raise NumericalError(f"Non-finite values in {name}")


def validate_model_state(model: nn.Module) -> None:
    for name, value in model.state_dict().items():
        if value.is_floating_point() or value.is_complex():
            require_finite(value, f"model state: {name}")
        if name.endswith("running_var") and (value < 0).any().item():
            raise NumericalError(f"Negative BatchNorm running variance: {name}")


def save_model_snapshot(path: Path, model: nn.Module, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "format_version": 1,
        "purpose": "evaluation_recovery_not_epoch_resume",
        "metadata": json.loads(json.dumps(metadata, default=str)),
        "state_dict": {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
    }
    torch.save(payload, temporary)
    os.replace(temporary, path)
