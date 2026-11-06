"""MolToken public API. Heavy model dependencies are loaded lazily."""
__version__ = "0.1.0"


def __getattr__(name):
    if name == "MolTokenizer":
        from .tokenizer import MolTokenizer
        return MolTokenizer
    if name in {"load_pretrained", "load_sft"}:
        from . import checkpoints
        return getattr(checkpoints, name)
    raise AttributeError(name)


__all__ = ["MolTokenizer", "load_pretrained", "load_sft", "__version__"]
