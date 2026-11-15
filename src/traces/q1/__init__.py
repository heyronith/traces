"""TRACES Q1 infrastructure (protocol v1.3).

CPU-safe building blocks for single-BOS encoding, residual hybrids, activation
patching, donor caches, F1/F2/F4, identities 1–9, gates, and workload enumeration.

This package does **not** authorize model inference or scientific outcome collection.
Synthetic-tensor unit tests exercise logic without loading Gemma weights.
"""

from __future__ import annotations

__all__ = [
    "FROZEN_PROTOCOL_VERSION",
    "FROZEN_SCIENTIFIC_COMMIT",
]

FROZEN_PROTOCOL_VERSION = "1.3"
FROZEN_SCIENTIFIC_COMMIT = "b1e8874f45311678e6fa241f504e7e09d4a7808c"
