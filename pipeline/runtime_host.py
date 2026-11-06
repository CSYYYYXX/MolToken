"""Optional cluster guard. Standalone users can use their own compute hosts.

Set MOLTOKEN_REQUIRE_WORKER=1 on cluster deployments that require worker-only
execution. No private hostname or scheduler dependency is imposed by default.
"""

from __future__ import annotations

import os
import socket


def is_compute_host(hostname: str | None = None) -> bool:
    """Return whether heavyweight work is allowed on *hostname*."""

    host = (hostname or socket.gethostname()).strip().lower()
    return os.environ.get("MOLTOKEN_REQUIRE_WORKER", "0") != "1" or "worker" in host


def require_compute_host(workload: str, hostname: str | None = None) -> str:
    """Apply the configured host policy and return the observed hostname."""

    host = (hostname or socket.gethostname()).strip().lower()
    if not is_compute_host(host):
        raise RuntimeError(f"{workload}: compute worker required, observed {host!r}")
    return host


__all__ = [
    "is_compute_host",
    "require_compute_host",
]
