import numpy as np

from src.implementations.services.landmark_normalizer import LandmarkNormalizer
from src.models.confidence import Confidence
from src.models.hand_landmarks import HandLandmarks, NUM_COORDS, NUM_LANDMARKS
from src.models.handedness import Handedness


def _synthetic_hand(offset: float, handedness: Handedness) -> HandLandmarks:
    points = np.zeros((NUM_LANDMARKS, NUM_COORDS), dtype=np.float32)
    points[0] = (offset, offset, 0.0)
    for i in range(1, NUM_LANDMARKS):
        points[i] = (offset + 0.01 * i, offset + 0.02 * i, 0.001 * i)
    return HandLandmarks(
        points=points,
        handedness=handedness,
        detection_confidence=Confidence(0.9),
    )


def test_normalizer_produces_translation_invariant_features():
    a = _synthetic_hand(offset=0.1, handedness=Handedness.RIGHT)
    b = _synthetic_hand(offset=0.4, handedness=Handedness.RIGHT)
    normalizer = LandmarkNormalizer(include_both_hands=False)

    features_a = normalizer.normalize((a,))
    features_b = normalizer.normalize((b,))

    assert np.allclose(features_a, features_b, atol=1e-6)


def test_normalizer_outputs_expected_size_for_both_hands():
    normalizer = LandmarkNormalizer(include_both_hands=True)
    left = _synthetic_hand(0.2, Handedness.LEFT)
    right = _synthetic_hand(0.5, Handedness.RIGHT)
    features = normalizer.normalize((left, right))
    assert features.shape == (2 * NUM_LANDMARKS * NUM_COORDS,)


def test_normalizer_returns_zeros_when_no_hands():
    normalizer = LandmarkNormalizer(include_both_hands=True)
    features = normalizer.normalize(())
    assert features.shape == (2 * NUM_LANDMARKS * NUM_COORDS,)
    assert np.all(features == 0.0)


def test_normalizer_slots_unknown_handedness_into_empty_slot():
    normalizer = LandmarkNormalizer(include_both_hands=True)
    unknown = _synthetic_hand(0.3, Handedness.UNKNOWN)
    features = normalizer.normalize((unknown,))
    per_hand = NUM_LANDMARKS * NUM_COORDS
    assert np.any(features[:per_hand] != 0.0)
    assert np.all(features[per_hand:] == 0.0)
