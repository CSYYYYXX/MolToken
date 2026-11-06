"""Compression algorithms (BPE)."""

from .bpe_engine import BPEEngine

try:
    from .cpp_bpe_backend import CppBPEBackend  # noqa: F401
    _cpp_available = True
except Exception:
    _cpp_available = False

__all__ = ["BPEEngine"]
if _cpp_available:
    __all__.append("CppBPEBackend")
