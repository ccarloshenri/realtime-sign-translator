from datetime import datetime, timezone

from src.models.confidence import Confidence
from src.models.sign_prediction import SignPrediction


def test_sign_prediction_payload_matches_public_spec():
    prediction = SignPrediction(
        label="ajuda",
        confidence=Confidence(0.932),
        sequence_size=30,
        timestamp=datetime(2026, 4, 22, 15, 30, 0, tzinfo=timezone.utc),
    )
    payload = prediction.to_payload()
    assert payload == {
        "text": "ajuda",
        "confidence": 0.932,
        "timestamp": "2026-04-22T15:30:00Z",
        "source": "camera",
        "sequence_size": 30,
    }
