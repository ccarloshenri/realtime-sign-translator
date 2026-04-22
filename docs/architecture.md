# Architecture

SignFlow Realtime is organized around five top-level source folders, all
under `src/`:

```
src/
├── models/           # Data models shared across the project (entities + VOs)
├── interface/        # Protocol contracts that every implementation honors
├── implementations/  # Concrete adapters: camera, vision, ml, config, logging,
│                     # application services, and the realtime pipeline
├── server/           # Local FastAPI API + WebSocket broadcaster
├── ui/               # CustomTkinter desktop (main window, panels, viewmodel)
├── bootstrap.py      # Composition root — wires the whole object graph
└── main.py           # Entry point for the desktop app
```

The arrows of the dependency graph only ever point inward:
`ui` and `server` depend on `implementations`; `implementations` depends on
`interface` and `models`; `models` and `interface` depend on nothing.

## Why this split matters for a sign translator

Real-time gesture recognition evolves fast: the model, the feature set, and
the input modalities (RGB vs. RGB+depth, single hand vs. full body) will
keep changing. By keeping every integration point behind a Protocol in
`src/interface/`, we can swap:

- **Camera**: OpenCV today, RealSense / PyRealSense tomorrow.
- **Landmark extractor**: MediaPipe Hands today, MediaPipe Holistic or a
  custom YOLO-pose later.
- **Classifier**: Bi-LSTM MVP, Transformer encoder, seq2seq translator.
- **Caption publisher**: local WebSocket today, OBS browser source or
  Google Live Caption tomorrow.

…without touching the UI, the server or the composition root.

## Threading model

Three independent threads:

- **Main thread** — owns the Tk event loop and refreshes the UI at ~30 Hz.
- **Pipeline thread** — owns camera I/O, MediaPipe inference, and the
  temporal classifier. All its state is thread-safe.
- **API thread** — runs uvicorn with its own asyncio loop. Predictions are
  delivered to it via `asyncio.run_coroutine_threadsafe`.

## The sequence-first design

The critical non-obvious bit: we never ask the model to classify a single
frame. `SequenceBuffer` keeps the last `N` (default 30) normalized landmark
vectors; only when it's full does the pipeline invoke
`ISequenceClassifier.predict()`. The buffer slides by one frame per tick,
so the classifier sees a smooth, continuously-updating window of temporal
context. This is what makes the system suitable for motion-based signs
rather than isolated hand poses.

## Configuration

`config.yaml` → `YamlConfigurationProvider` → `AppConfig` (Pydantic v2).
The bootstrap layer reads it once and hands the typed object to every
constructor.

## Turning this MVP into a product

1. Replace the mock with a real trained network (`training/` pipeline).
2. Add velocity/acceleration channels to the feature encoder.
3. Add a "movement gate" to suppress predictions during still frames, and
   a rest-pose reset so the smoother clears its dwell counter cleanly.
4. Add segmentation (start/end of sign) so the pipeline can handle
   continuous signing, not isolated words.
5. Migrate to seq2seq translation with beam search for phrases/sentences.
