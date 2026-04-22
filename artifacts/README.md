# Trained artifacts

Trained model files and the label vocabulary live here:

- `signflow_lstm.keras` — Keras model produced by `training/model_training/train_model.py`.
- `labels.json` — ordered label vocabulary produced by
  `training/preprocessing/build_dataset.py`. The index of each label must
  match the model's output axis.

Both files are gitignored; only check in `labels.json` if your team wants to
share a frozen vocabulary.

Point `config.yaml` at these paths and set `classifier.backend: keras` to
switch the live pipeline from the mock to the trained model.
