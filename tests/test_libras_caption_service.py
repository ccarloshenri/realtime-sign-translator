from datetime import datetime, timezone

from src.implementations.libras.libras_caption_service import (
    LIBRAS_SOURCE,
    LibrasCaptionService,
)
from src.models.confidence import Confidence
from src.models.sign_prediction import SignPrediction


class _StubPublisher:
    def __init__(self) -> None:
        self.published: list[SignPrediction] = []

    def publish(self, prediction: SignPrediction) -> None:
        self.published.append(prediction)


def _prediction(source: str) -> SignPrediction:
    return SignPrediction(
        label="olá",
        confidence=Confidence(0.9),
        sequence_size=30,
        source=source,
        timestamp=datetime(2026, 4, 22, 15, 30, tzinfo=timezone.utc),
    )


def test_retags_camera_source_to_libras():
    stub = _StubPublisher()
    service = LibrasCaptionService(downstream=stub)

    service.publish(_prediction(source="camera"))

    assert len(stub.published) == 1
    assert stub.published[0].source == LIBRAS_SOURCE
    assert stub.published[0].label == "olá"


def test_leaves_already_libras_source_unchanged():
    stub = _StubPublisher()
    service = LibrasCaptionService(downstream=stub)

    original = _prediction(source=LIBRAS_SOURCE)
    service.publish(original)

    assert stub.published[0] is original  # no replace() call needed
