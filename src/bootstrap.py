"""
Composition root.

Builds all concrete implementations (camera, extractor, classifier, API, etc.)
and wires them to the pipeline use case through the project's interfaces.
The rest of the codebase never constructs these classes directly — every
consumer reads them off the returned `AppServices` container.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.implementations.camera.opencv_camera import OpenCVCamera
from src.implementations.config.yaml_configuration import (
    AppConfig,
    YamlConfigurationProvider,
)
from src.implementations.logging.structured_logger import build_logger
from src.implementations.ml.mock_classifier import MockSequenceClassifier
from src.implementations.pipeline.run_translation_pipeline import (
    RunTranslationPipeline,
)
from src.implementations.services.landmark_normalizer import LandmarkNormalizer
from src.implementations.services.prediction_smoother import PredictionSmoother
from src.implementations.services.sequence_buffer import SequenceBuffer
from src.implementations.vision.mediapipe_extractor import (
    MediaPipeHandLandmarkExtractor,
)
from src.interface.logger import ILogger
from src.interface.sequence_classifier import ISequenceClassifier
from src.server.server import ApiServer, build_api
from src.server.state import ApiState
from src.server.websocket import WebSocketBroadcaster
from src.server.websocket_publisher import WebSocketCaptionPublisher
from src.ui.viewmodels.translation_viewmodel import TranslationViewModel


@dataclass(slots=True)
class AppServices:
    config: AppConfig
    logger: ILogger
    pipeline: RunTranslationPipeline
    view_model: TranslationViewModel
    api_server: ApiServer
    api_state: ApiState


def bootstrap(config_path: str | Path = "config.yaml") -> AppServices:
    config_provider = YamlConfigurationProvider(config_path)
    config = config_provider.get()

    logger = build_logger("signflow", config.logging)
    logger.info("bootstrap.start", backend=config.classifier.backend)

    normalizer = LandmarkNormalizer(include_both_hands=True)
    feature_size = normalizer.feature_size

    camera = OpenCVCamera(
        logger=logger,
        device_index=config.camera.device_index,
        width=config.camera.width,
        height=config.camera.height,
        target_fps=config.camera.target_fps,
        flip_horizontal=config.camera.flip_horizontal,
    )

    extractor = MediaPipeHandLandmarkExtractor(
        logger=logger,
        max_num_hands=config.vision.max_num_hands,
        min_detection_confidence=config.vision.min_detection_confidence,
        min_tracking_confidence=config.vision.min_tracking_confidence,
        model_complexity=config.vision.model_complexity,
    )

    classifier = _build_classifier(config, feature_size, logger)

    buffer = SequenceBuffer(
        sequence_length=classifier.sequence_length,
        feature_size=feature_size,
    )

    smoother = PredictionSmoother(
        labels=classifier.labels,
        min_confidence=config.pipeline.min_confidence,
        smoothing_window=config.pipeline.smoothing_window,
        min_dwell_frames=config.pipeline.min_dwell_frames,
        publish_unchanged=config.pipeline.publish_unchanged,
    )

    pipeline = RunTranslationPipeline(
        camera=camera,
        extractor=extractor,
        normalizer=normalizer,
        buffer=buffer,
        classifier=classifier,
        smoother=smoother,
        logger=logger,
        target_fps=config.camera.target_fps,
    )

    api_state = ApiState()
    broadcaster = WebSocketBroadcaster(logger)
    api_server = build_api(config, api_state, broadcaster, logger)

    ws_publisher = WebSocketCaptionPublisher(api_state, broadcaster)
    pipeline.callbacks.on_prediction.append(ws_publisher.publish)
    pipeline.callbacks.on_state.append(
        lambda st: api_state.set_pipeline_running(st.running)
    )

    view_model = TranslationViewModel(pipeline, language=config.ui.language)

    logger.info(
        "bootstrap.done",
        feature_size=feature_size,
        sequence_length=classifier.sequence_length,
        labels=len(classifier.labels),
    )

    return AppServices(
        config=config,
        logger=logger,
        pipeline=pipeline,
        view_model=view_model,
        api_server=api_server,
        api_state=api_state,
    )


def _build_classifier(
    config: AppConfig,
    feature_size: int,
    logger: ILogger,
) -> ISequenceClassifier:
    if config.classifier.backend == "keras":
        from src.implementations.ml.keras_classifier import (
            KerasSequenceClassifier,
        )

        try:
            return KerasSequenceClassifier(
                model_path=config.classifier.model_path,
                labels_path=config.classifier.labels_path,
                logger=logger,
            )
        except (FileNotFoundError, ImportError) as exc:
            logger.error("classifier.keras_fallback", error=str(exc))

    logger.info(
        "classifier.mock_selected",
        vocabulary=config.classifier.mock_vocabulary,
    )
    return MockSequenceClassifier(
        labels=tuple(config.classifier.mock_vocabulary),
        sequence_length=config.pipeline.sequence_length,
        feature_size=feature_size,
    )
