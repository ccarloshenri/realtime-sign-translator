"""
Fetch, extract and landmark-encode the MINDS-Libras dataset.

MINDS-Libras (Rezende et al., UFOP/NIAID) is a CC-BY 4.0 Libras dataset
hosted on Zenodo containing 20 isolated signs recorded 5 times by
12 signers in chroma-key conditions (~1200 videos, 64.8 GB total).

Signs: Acontecer, Aluno, Amarelo, América, Aproveitar, Bala, Banco,
Banheiro, Barulho, Cinco, Conhecer, Espelho, Esquina, Filho, Maçã,
Medo, Ruim, Sapo, Vacina, Vontade.

This script performs three phases (each skippable):

  1. Download selected signer zips from Zenodo.
  2. Unzip into a staging directory.
  3. Run MediaPipe on every video, extract a 30-frame centered window
     of normalized landmarks, and save each video as one
     `samples/<sign>/<uuid>.npz` file — exactly the format the existing
     training pipeline (`build_dataset` + `train_libras_model`) already
     consumes.

Usage (minimal — just one signer, ~2.5 GB):

    python -m training.libras.fetch_minds_libras --signers 1

Afterwards, continue with the existing pipeline:

    python -m training.preprocessing.build_dataset
    python -m training.libras.train_libras_model --allow-unknown-labels

Licensing: the dataset is CC-BY 4.0 (Rezende et al.); if you publish
anything trained on it you must cite the original authors. The script
does not redistribute the dataset — it downloads directly from Zenodo.
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
import unicodedata
import urllib.request
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.implementations.config.yaml_configuration import (  # noqa: E402
    YamlConfigurationProvider,
)
from src.implementations.logging.structured_logger import build_logger  # noqa: E402
from src.implementations.ml.feature_encoder import FeatureEncoder  # noqa: E402
from src.implementations.vision.mediapipe_extractor import (  # noqa: E402
    MediaPipeHandLandmarkExtractor,
)

ZENODO_RECORD = "2667329"
ZENODO_BASE = f"https://zenodo.org/records/{ZENODO_RECORD}/files"

# Canonical label list straight from the MINDS-Libras paper. Used to
# validate that whatever the zips look like, the labels we produce in
# `samples/` match the dataset vocabulary.
MINDS_LIBRAS_LABELS: tuple[str, ...] = (
    "Acontecer", "Aluno", "Amarelo", "América", "Aproveitar",
    "Bala", "Banco", "Banheiro", "Barulho", "Cinco",
    "Conhecer", "Espelho", "Esquina", "Filho", "Maçã",
    "Medo", "Ruim", "Sapo", "Vacina", "Vontade",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--signers",
        type=int,
        nargs="+",
        default=[1],
        choices=list(range(1, 13)),
        help="Which signer zips to pull from Zenodo (1..12). "
        "One signer ≈ 2.5-8.4 GB.",
    )
    parser.add_argument("--download-dir", default="data/minds-libras/raw")
    parser.add_argument("--extract-dir", default="data/minds-libras/videos")
    parser.add_argument("--samples-dir", default="samples")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-unzip", action="store_true")
    parser.add_argument("--skip-process", action="store_true")
    parser.add_argument("--cleanup-videos", action="store_true",
                        help="Delete the unzipped .mp4 files after processing")
    parser.add_argument("--config", default="config.yaml")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config = YamlConfigurationProvider(args.config).get()
    logger = build_logger("signflow.minds_libras", config.logging)

    download_dir = Path(args.download_dir)
    extract_dir = Path(args.extract_dir)
    samples_dir = Path(args.samples_dir)
    download_dir.mkdir(parents=True, exist_ok=True)
    extract_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------- 1. download
    zip_paths: list[Path] = []
    for signer in args.signers:
        fname = f"Sinalizador{signer:02d}.zip"
        dest = download_dir / fname
        url = f"{ZENODO_BASE}/{fname}"

        if args.skip_download:
            logger.info("minds.skip_download", path=str(dest))
        elif dest.exists() and dest.stat().st_size > 0:
            logger.info(
                "minds.already_downloaded",
                path=str(dest),
                mb=round(dest.stat().st_size / 1e6, 1),
            )
        else:
            logger.info("minds.downloading", url=url, dest=str(dest))
            _download(url, dest)

        if dest.exists():
            zip_paths.append(dest)

    # ----------------------------------------------------------- 2. unzip
    if args.skip_unzip:
        logger.info("minds.skip_unzip")
    else:
        for zip_path in zip_paths:
            _unzip(zip_path, extract_dir, logger)

    # -------------------------------------------------- 3. extract samples
    if args.skip_process:
        logger.info("minds.skip_process")
        return 0

    extractor = MediaPipeHandLandmarkExtractor(
        logger=logger,
        model_path=config.vision.model_path,
        max_num_hands=config.vision.max_num_hands,
        min_detection_confidence=config.vision.min_detection_confidence,
        min_tracking_confidence=config.vision.min_tracking_confidence,
        model_complexity=config.vision.model_complexity,
    )
    encoder = FeatureEncoder(include_both_hands=True)
    sequence_length = config.pipeline.sequence_length

    video_paths = sorted(extract_dir.rglob("*.mp4"))
    if not video_paths:
        logger.error("minds.no_videos_found", path=str(extract_dir))
        return 1

    logger.info("minds.processing_start", videos=len(video_paths))
    processed = 0
    skipped = 0
    for video_path in video_paths:
        label = _label_from_path(video_path)
        if label is None:
            logger.warning("minds.unknown_label", path=str(video_path))
            skipped += 1
            continue

        try:
            ok = _process_video(
                video_path=video_path,
                label=label,
                samples_dir=samples_dir,
                extractor=extractor,
                encoder=encoder,
                sequence_length=sequence_length,
            )
        except Exception:
            logger.exception("minds.video_failed", path=str(video_path))
            skipped += 1
            continue

        if ok:
            processed += 1
        else:
            skipped += 1

        if processed and processed % 25 == 0:
            logger.info(
                "minds.progress",
                processed=processed,
                skipped=skipped,
                total=len(video_paths),
            )

    extractor.close()
    logger.info(
        "minds.processing_done",
        processed=processed,
        skipped=skipped,
        samples_dir=str(samples_dir),
    )

    if args.cleanup_videos:
        _cleanup(extract_dir, logger)

    return 0


# --------------------------------------------------------------- helpers

def _download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")

    def _progress(block_num: int, block_size: int, total_size: int) -> None:
        downloaded = block_num * block_size
        pct = min(100.0, downloaded * 100 / max(total_size, 1))
        sys.stdout.write(
            f"\r    {pct:5.1f}%  {downloaded / 1e9:.2f} GB / {total_size / 1e9:.2f} GB"
        )
        sys.stdout.flush()

    urllib.request.urlretrieve(url, tmp, reporthook=_progress)
    sys.stdout.write("\n")
    tmp.rename(dest)


def _unzip(zip_path: Path, extract_dir: Path, logger) -> None:
    logger.info("minds.unzipping", src=str(zip_path), dest=str(extract_dir))
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)


_ACCENT_MAP = {
    "a": "á", "e": "é", "i": "í", "o": "ó", "u": "ú",
}


def _normalize_candidate(token: str) -> str:
    """Strip numeric suffixes and casefold+normalize accents for matching."""
    # Drop digits and common separators.
    token = re.sub(r"[0-9_\-]+", " ", token).strip()
    nfkd = unicodedata.normalize("NFKD", token)
    stripped = "".join(c for c in nfkd if not unicodedata.combining(c))
    return stripped.casefold()


_LABEL_INDEX: dict[str, str] = {
    _normalize_candidate(label): label for label in MINDS_LIBRAS_LABELS
}


def _label_from_path(video_path: Path) -> str | None:
    """
    Try to infer the sign label from the video path.

    MINDS-Libras layouts seen in the wild:
      Sinalizador01/Acontecer/Acontecer_1.mp4
      Sinalizador01/01Acontecer_Sin01_01.mp4
      Sinalizador01/Acontecer_Sin01_01.mp4
    We walk the path components (parent folders and the file stem) and
    look for any fragment that matches one of the 20 known labels.
    """
    candidates = list(video_path.parts[-4:-1]) + [video_path.stem]
    for candidate in candidates:
        # Try the whole candidate.
        key = _normalize_candidate(candidate)
        if key in _LABEL_INDEX:
            return _LABEL_INDEX[key]
        # Try individual whitespace-separated tokens.
        for token in key.split():
            if token in _LABEL_INDEX:
                return _LABEL_INDEX[token]
    return None


def _process_video(
    *,
    video_path: Path,
    label: str,
    samples_dir: Path,
    extractor: MediaPipeHandLandmarkExtractor,
    encoder: FeatureEncoder,
    sequence_length: int,
) -> bool:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < sequence_length:
        cap.release()
        return False

    # Take a centered window of sequence_length frames.
    start = max(0, (total - sequence_length) // 2)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    buffer: list[np.ndarray] = []
    for _ in range(sequence_length):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        hands = extractor.extract(frame)
        buffer.append(encoder.encode(hands))
    cap.release()

    if len(buffer) < sequence_length:
        return False

    features = np.stack(buffer, axis=0).astype(np.float32)
    out_dir = samples_dir / label
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{uuid.uuid4().hex}.npz"
    np.savez_compressed(
        out_path,
        features=features,
        meta=np.array(
            {
                "label": label,
                "source": "MINDS-Libras",
                "video": video_path.name,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "sequence_length": int(features.shape[0]),
            },
            dtype=object,
        ),
    )
    return True


def _cleanup(extract_dir: Path, logger) -> None:
    logger.info("minds.cleanup", path=str(extract_dir))
    shutil.rmtree(extract_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
