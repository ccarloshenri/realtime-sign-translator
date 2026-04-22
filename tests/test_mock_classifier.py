import numpy as np

from src.implementations.ml.mock_classifier import MockSequenceClassifier


def test_mock_classifier_returns_valid_distribution():
    classifier = MockSequenceClassifier(
        labels=("a", "b", "c"),
        sequence_length=4,
        feature_size=6,
    )
    rng = np.random.default_rng(0)
    sequence = rng.standard_normal((4, 6)).astype(np.float32)
    output = classifier.predict(sequence)

    assert output.label in {"a", "b", "c"}
    assert 0.0 <= float(output.confidence) <= 1.0
    assert output.probabilities is not None
    assert np.isclose(output.probabilities.sum(), 1.0, atol=1e-5)


def test_mock_classifier_on_zero_input_emits_uniform():
    classifier = MockSequenceClassifier(
        labels=("a", "b", "c", "d"),
        sequence_length=5,
        feature_size=8,
    )
    output = classifier.predict(np.zeros((5, 8), dtype=np.float32))
    assert output.probabilities is not None
    assert np.allclose(output.probabilities, 0.25, atol=1e-6)


def test_mock_classifier_rejects_wrong_shape():
    classifier = MockSequenceClassifier(
        labels=("a",),
        sequence_length=3,
        feature_size=4,
    )
    try:
        classifier.predict(np.zeros((2, 4), dtype=np.float32))
    except ValueError:
        return
    raise AssertionError("expected ValueError on wrong shape")
