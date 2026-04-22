import numpy as np

from src.implementations.services.prediction_smoother import PredictionSmoother
from src.interface.sequence_classifier import ClassifierOutput
from src.models.confidence import Confidence


LABELS = ("ola", "obrigado", "ajuda")


def _output(label: str, confidence: float) -> ClassifierOutput:
    probs = np.full(
        len(LABELS),
        (1.0 - confidence) / (len(LABELS) - 1),
        dtype=np.float32,
    )
    probs[LABELS.index(label)] = confidence
    return ClassifierOutput(
        label=label,
        confidence=Confidence(confidence),
        probabilities=probs,
    )


def test_smoother_requires_dwell_before_promoting_label():
    smoother = PredictionSmoother(
        labels=LABELS,
        min_confidence=0.6,
        smoothing_window=1,
        min_dwell_frames=3,
    )
    r1 = smoother.observe(_output("ola", 0.9))
    r2 = smoother.observe(_output("ola", 0.9))
    r3 = smoother.observe(_output("ola", 0.9))
    r4 = smoother.observe(_output("ola", 0.9))

    assert r1.should_publish is False
    assert r2.should_publish is False
    assert r3.should_publish is True
    assert r3.changed is True
    assert r4.should_publish is False


def test_smoother_drops_low_confidence_frames():
    smoother = PredictionSmoother(
        labels=LABELS,
        min_confidence=0.8,
        smoothing_window=1,
        min_dwell_frames=1,
    )
    result = smoother.observe(_output("ola", 0.5))
    assert result.should_publish is False


def test_smoother_averages_probabilities_over_window():
    smoother = PredictionSmoother(
        labels=LABELS,
        min_confidence=0.5,
        smoothing_window=3,
        min_dwell_frames=1,
    )
    smoother.observe(_output("ola", 0.9))
    smoother.observe(_output("obrigado", 0.9))
    result = smoother.observe(_output("obrigado", 0.9))
    assert result.label == "obrigado"
    assert result.should_publish is True


def test_reset_clears_state():
    smoother = PredictionSmoother(
        labels=LABELS,
        min_confidence=0.5,
        smoothing_window=1,
        min_dwell_frames=1,
    )
    smoother.observe(_output("ola", 0.9))
    smoother.reset()
    r = smoother.observe(_output("ola", 0.9))
    assert r.should_publish is True
    assert r.changed is True
