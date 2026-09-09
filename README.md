# HAR Platform V2 — Neural Networks & Glasses Streaming


1. A **neural-network model** (PyTorch MLP) as an alternative to the forest.
2. **One-tap glasses recording** via the Meta DAT SDK (currently blocked).

## Status
- Everything V1 does still works here; only the *model* and *video path* change.

## Files (`pipeline/`)
- `HAR_NN_Training.ipynb` — **new:** trains a PyTorch MLP, compares it against the forest (same data, same 39 features, same leave-one-session-out split), exports to ExecuTorch.
- `HAR_Training_v3.ipynb`, `HAR_Batch_v3.ipynb`, `HAR_Pipeline_v3.ipynb`, `har_pipeline_v3.py` — reused from V1 (forest training, labeling, pipeline). **Keep all.**

## Run the neural network (no hardware)
Open `HAR_NN_Training.ipynb` in Colab → set `SESSIONS_DIR` to your Drive sessions folder → **Run all**. It prints a forest-vs-NN comparison table.
Key knobs (cell #1): `LABEL_COLUMN`, `EPOCHS`, `HIDDEN` (MLP layer sizes).

> Expect the forest to win or tie — neural nets need far more, ideally **multi-subject**, data (tens of hours, 10+ people). "Forest ≥ NN" is a valid finding; the NN path is built for when that data exists.

## Notes
- On-device NN needs an ExecuTorch runtime in Flutter (unimplemented); the saved model includes scaler stats that must be applied before inference.