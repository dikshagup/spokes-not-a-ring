"""Shared code for the figures of "Spokes, not a Ring"."""

from weekday_manifold.utils import set_seed

__all__ = ["load_model", "set_seed"]


def __getattr__(name):
    if name == "load_model":
        from weekday_manifold.model import load_model as _f
        return _f
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
