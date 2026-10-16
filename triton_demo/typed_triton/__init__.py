"""Typed Triton JIT and checked host-to-kernel views."""

from .semantic_jit import constexpr
from .semantic_jit import semantic_jit as jit

__all__ = ["constexpr", "jit"]
