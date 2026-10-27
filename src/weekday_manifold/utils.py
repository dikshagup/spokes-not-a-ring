"""Shared utilities: seeding, data paths, and results-table output."""

from __future__ import annotations

import datetime
import os
import random
import re

import numpy as np


def model_short_name(model_name: str) -> str:
    """Filesystem-safe short model id."""
    short = str(model_name).split("/")[-1]
    return re.sub(r"[^A-Za-z0-9._-]+", "-", short).strip("-") or "model"


def run_slug(run_name: str, model_name: str, with_date: bool = True) -> str:
    """Join a run name, the short model name and, optionally, today's ISO date."""
    parts = [run_name, model_short_name(model_name)]
    if with_date:
        parts.append(datetime.date.today().isoformat())
    return "_".join(parts)


def set_seed(seed: int = 0) -> None:
    """Seed the Python, NumPy and Torch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    # imported here so that importing this module does not import torch
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def data_dir(figure: str, create: bool = True) -> str:
    """Where a figure's captures live: ``$WM_DATA/<figure>`` (default ``~/.cache/weekday-manifold``)."""
    root = os.environ.get("WM_DATA") or os.path.join(os.path.expanduser("~/.cache"), "weekday-manifold")
    path = os.path.join(root, figure)
    if create:
        os.makedirs(path, exist_ok=True)
    return path


def dump_json(obj, path: str, ndigits: int = 6, **kw) -> None:
    """Write ``obj`` as JSON with every float rounded to ``ndigits``."""
    import json

    def rounded(x):
        if isinstance(x, float):
            return round(x, ndigits)
        if isinstance(x, dict):
            return {k: rounded(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [rounded(v) for v in x]
        return x

    with open(path, "w") as f:
        json.dump(rounded(obj), f, **kw)
