import numpy as np
import pytest

from src.implementations.services.sequence_buffer import SequenceBuffer


def test_buffer_reports_not_ready_until_full():
    buf = SequenceBuffer(sequence_length=3, feature_size=4)
    assert not buf.is_ready()
    assert buf.snapshot() is None

    buf.append(np.ones(4, dtype=np.float32))
    buf.append(np.ones(4, dtype=np.float32) * 2)
    assert not buf.is_ready()

    buf.append(np.ones(4, dtype=np.float32) * 3)
    assert buf.is_ready()

    snap = buf.snapshot()
    assert snap is not None
    assert snap.shape == (3, 4)
    assert np.allclose(snap[0], 1.0)
    assert np.allclose(snap[2], 3.0)


def test_buffer_rolls_over_past_capacity():
    buf = SequenceBuffer(sequence_length=2, feature_size=2)
    buf.append(np.array([1.0, 1.0], dtype=np.float32))
    buf.append(np.array([2.0, 2.0], dtype=np.float32))
    buf.append(np.array([3.0, 3.0], dtype=np.float32))

    snap = buf.snapshot()
    assert snap is not None
    assert np.allclose(snap[0], 2.0)
    assert np.allclose(snap[1], 3.0)


def test_buffer_rejects_wrong_feature_shape():
    buf = SequenceBuffer(sequence_length=2, feature_size=4)
    with pytest.raises(ValueError):
        buf.append(np.zeros(3, dtype=np.float32))


def test_append_zero_advances_buffer():
    buf = SequenceBuffer(sequence_length=2, feature_size=3)
    buf.append_zero()
    buf.append_zero()
    snap = buf.snapshot()
    assert snap is not None
    assert np.all(snap == 0.0)


def test_clear_resets_buffer():
    buf = SequenceBuffer(sequence_length=2, feature_size=2)
    buf.append(np.ones(2, dtype=np.float32))
    buf.clear()
    assert len(buf) == 0
    assert not buf.is_ready()
