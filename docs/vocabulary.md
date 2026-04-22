# Growing the vocabulary

There is no ready-made Libras translator you can just install. Libras has
more than 10.000 signs, no public dataset covers all of them, and every
signer has personal variations. Every real-world sign-language recognizer —
academic or commercial — is built the same way: collect a dataset for the
signs you care about, train a temporal model, then grow from there.

SignFlow's architecture was designed for that loop. The vocabulary is not
hard-coded: it is whatever you write into `samples/` at collection time,
and the training pipeline emits a `labels.json` that the runtime reads at
startup. This document is the step-by-step.

## The loop

```
collect samples ──▶ build dataset ──▶ train model ──▶ run desktop app
        ▲                                                   │
        └───────────────── iterate ─────────────────────────┘
```

## 1. Collect samples

For each sign you want to recognize, run:

```
python -m training.data_collection.collect_samples --label <name> --samples 40
```

- Press SPACE to record one take. The sample is 30 frames (≈ 1 second at
  30 fps). Adjust `pipeline.sequence_length` in `config.yaml` if you want
  longer windows.
- Each take becomes one `.npz` file in `samples/<name>/`.
- Aim for **at least 30–50 takes per sign**, and ideally multiple signers,
  different lighting conditions, different angles.
- BACKSPACE undoes the last sample. ESC quits.

Libras-specific tips:
- Record both hands when the sign uses both. SignFlow already feeds a
  126-dim vector (2 hands × 21 points × 3 coords) per frame, so it sees
  both hands natively.
- Keep a consistent "neutral" rest pose between signs so the pipeline can
  later learn start/end boundaries.

## 2. Build the dataset

```
python -m training.preprocessing.build_dataset
```

This reads every folder under `samples/`, splits into train/val/test, and
writes:

- `training/datasets/signflow.npz`
- `artifacts/labels.json` (ordered list — the index of each label equals
  the model's output axis)

## 3. Train the model

```
pip install tensorflow
python -m training.model_training.train_model
```

The baseline is a mask-aware Bi-LSTM → Dense → Softmax. Good enough to
validate the pipeline; not SOTA. When you outgrow it, just replace the
model file inside `train_model.py` with a Temporal Convolutional Network,
a small Transformer encoder, or a seq2seq translator. The runtime
classifier (`KerasSequenceClassifier`) only cares that the input is
`(batch, T, F)` and the output is `(batch, num_labels)`.

## 4. Run the trained model

Open `config.yaml` and switch the backend:

```yaml
classifier:
  backend: keras
```

`python -m src.main` — done. The app now loads `artifacts/signflow_lstm.keras`
and `artifacts/labels.json`, and the UI will show whatever label the model
predicts.

## Roadmap to a useful Libras translator

Growing from a 10-sign demo to something practical is engineering, not
magic:

1. **Dataset size.** The single biggest lever. Libras benchmarks like
   MINDS-Libras or V-LIBRASIL can bootstrap; your own recordings close
   the domain gap.
2. **Multiple signers.** A model trained on one person won't generalize.
3. **Feature engineering.** Add velocity/acceleration channels, world
   coordinates, facial landmarks (many Libras signs are
   non-manual-marker-dependent).
4. **Sign segmentation.** Continuous signing needs start/end detection,
   not just windowed classification.
5. **Sequence-to-sequence.** For sentences, move from isolated-sign
   classification to a seq2seq model with CTC or attention decoding.
6. **Multi-modal.** Face + body + hands together massively improve
   disambiguation on visually similar signs.

None of these require re-architecting the project — every piece plugs
into an existing interface in `src/interface/`.
