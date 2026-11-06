"""Typed failures for samples outside the frozen Axis codec domain.

These exceptions are intentionally lightweight so data materializers can catch
only known representation-limit failures without importing RDKit, NumPy, or the
compiled FDDFS backend at module import time.
"""

from __future__ import annotations


class FrozenAxisRepresentationLimitError(ValueError):
    """A valid source record cannot be represented by the frozen Axis codec."""


class RingIdOverflow(FrozenAxisRepresentationLimitError):
    """FDDFS needs more ring identifiers than the frozen vocabulary reserved."""

    def __init__(
        self,
        sample_id: str,
        max_ring_id: int,
        ring_capacity: int,
        first_colliding_token_id: int,
    ) -> None:
        self.sample_id = str(sample_id)
        self.max_ring_id = int(max_ring_id)
        self.ring_capacity = int(ring_capacity)
        self.first_colliding_token_id = int(first_colliding_token_id)
        self.message = (
            f"{self.sample_id}: max_ring_id={self.max_ring_id} "
            f"超过语料预留的 {self.ring_capacity} 个 ring 槽位；"
            "再往下转 int 会撞进 BPE merges 区"
            f"（id>={self.first_colliding_token_id}）"
        )
        # Structured args keep the exception picklable across ProcessPool workers.
        super().__init__(
            self.sample_id,
            self.max_ring_id,
            self.ring_capacity,
            self.first_colliding_token_id,
        )

    def __str__(self) -> str:
        return self.message


class UnknownFeatureError(FrozenAxisRepresentationLimitError):
    """A graph contains node or edge features absent from the frozen vocabulary."""

    def __init__(self, tokens: set[str] | tuple[str, ...] | list[str]) -> None:
        ordered = tuple(sorted(set(tokens)))
        self.tokens = set(ordered)
        super().__init__(ordered)

    def __str__(self) -> str:
        return f"未知 feature token: {sorted(self.tokens)[:5]}"


__all__ = [
    "FrozenAxisRepresentationLimitError",
    "RingIdOverflow",
    "UnknownFeatureError",
]
