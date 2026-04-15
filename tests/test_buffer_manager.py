"""Tests for inference buffer manager."""

from __future__ import annotations

import numpy as np
from src.inference.buffer_manager import BufferManager


def test_buffer_update_and_ready() -> None:
    """Aircraft becomes ready after enough observations."""
    bm = BufferManager()

    features = np.zeros(8, dtype=np.float32)
    for t in range(10):
        bm.update("abc123", features, 1000 + t * 10)

    assert "abc123" in bm.get_ready_aircraft()
    assert bm.num_tracked == 1
    assert bm.num_ready == 1


def test_buffer_not_ready_until_min_observations() -> None:
    """Aircraft with too few observations is not ready."""
    bm = BufferManager()

    features = np.zeros(8, dtype=np.float32)
    for t in range(3):
        bm.update("abc123", features, 1000 + t * 10)

    assert "abc123" not in bm.get_ready_aircraft()
    assert bm.num_ready == 0


def test_buffer_garbage_collection() -> None:
    """Stale entries are removed by GC."""
    bm = BufferManager(stale_timeout=0.0)  # Immediate timeout

    features = np.zeros(8, dtype=np.float32)
    bm.update("abc123", features, 1000)

    import time

    time.sleep(0.01)
    removed = bm.garbage_collect()
    assert removed == 1
    assert bm.num_tracked == 0


def test_buffer_get_window() -> None:
    """Window extraction returns correct shape."""
    bm = BufferManager(context_length=32)

    features = np.arange(8, dtype=np.float32)
    for t in range(50):
        bm.update("abc123", features * t, 1000 + t * 10)

    buf = bm.get_buffer("abc123")
    assert buf is not None

    window_features, window_timestamps = buf.get_window(32)
    assert window_features.shape == (32, 8)
    assert window_timestamps.shape == (32,)


def test_multiple_aircraft() -> None:
    """Tracks multiple aircraft independently."""
    bm = BufferManager()

    features = np.zeros(8, dtype=np.float32)
    for t in range(10):
        bm.update("abc", features, 1000 + t * 10)
        bm.update("def", features, 1000 + t * 10)

    assert bm.num_tracked == 2
    assert bm.num_ready == 2
