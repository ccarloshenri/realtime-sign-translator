import numpy as np
import pytest

from src.implementations.libras.libras_feature_extractor import (
    LibrasFeatureExtractor,
)


def test_feature_size_is_double_the_position_size():
    extractor = LibrasFeatureExtractor(include_both_hands=True)
    assert extractor.feature_size == 2 * extractor.position_size


def test_enrich_first_frame_velocity_is_zero():
    extractor = LibrasFeatureExtractor(include_both_hands=False)
    seq = np.random.default_rng(0).standard_normal(
        (5, extractor.position_size)
    ).astype(np.float32)
    enriched = extractor.enrich(seq)
    assert enriched.shape == (5, extractor.feature_size)
    # Velocity portion of the first frame must be zero.
    assert np.allclose(enriched[0, extractor.position_size:], 0.0)


def test_enrich_velocity_equals_frame_diff():
    extractor = LibrasFeatureExtractor(include_both_hands=False)
    seq = np.arange(
        3 * extractor.position_size, dtype=np.float32
    ).reshape(3, extractor.position_size)
    enriched = extractor.enrich(seq)
    assert np.allclose(
        enriched[1:, extractor.position_size:],
        np.diff(seq, axis=0),
    )


def test_enrich_rejects_wrong_shape():
    extractor = LibrasFeatureExtractor(include_both_hands=True)
    with pytest.raises(ValueError):
        extractor.enrich(np.zeros((4, 99), dtype=np.float32))
    with pytest.raises(ValueError):
        extractor.enrich(np.zeros(extractor.position_size, dtype=np.float32))
