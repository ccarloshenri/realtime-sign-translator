"""Protocol interfaces — the contracts every implementation must honor."""

from .camera_provider import ICameraProvider
from .caption_publisher import ICaptionPublisher
from .configuration_provider import IConfigurationProvider
from .hand_landmark_extractor import IHandLandmarkExtractor
from .logger import ILogger
from .sequence_classifier import ClassifierOutput, ISequenceClassifier

__all__ = [
    "ClassifierOutput",
    "ICameraProvider",
    "ICaptionPublisher",
    "IConfigurationProvider",
    "IHandLandmarkExtractor",
    "ILogger",
    "ISequenceClassifier",
]
