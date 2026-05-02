# Pre-trained Model Weights

This directory contains pre-trained EMA (Exponential Moving Average) checkpoints for the PI-DSN model.

## Files

| File | Filament | Focal Length | Seed | Description |
| ---- | -------- | ------------ | ---- | ----------- |
| `FIL001_75mm_seed124_ema.pth` | FIL001 (d=100.2 um) | 75 mm | 124 | Primary result |
| `FIL001_75mm_seed212_ema.pth` | FIL001 (d=100.2 um) | 75 mm | 212 | Seed stability |
| `FIL001_120mm_seed124_ema.pth` | FIL001 (d=100.2 um) | 120 mm | 124 | Primary result |
| `FIL001_120mm_seed212_ema.pth` | FIL001 (d=100.2 um) | 120 mm | 212 | Seed stability |
| `FIL002_75mm_seed42_ema.pth` | FIL002 (d=100.0 um) | 75 mm | 42 | Cross-filament |
| `FIL002_120mm_seed42_ema.pth` | FIL002 (d=100.0 um) | 120 mm | 42 | Cross-filament |

## EMA vs. Standard Checkpoints

During training, the pipeline saves two types of checkpoints:

- **`best_ema_model.pth` (EMA)**: Exponentially averaged model parameters. These provide smoother, more stable predictions and are used for all results reported in the paper.
- **`best_model.pth` (Standard)**: Raw model parameters from the training step with the best validation metric.

The EMA checkpoint is selected via a label-free criterion (no ground-truth labels required), making it suitable for real-world deployment where true values are unknown.

## How to Use

These weights are automatically loaded when running evaluation with `ALLOW_TRAIN=0`:

```bash
export FILAMENT_PROJECT_ROOT="/path/to/publish"
export ALLOW_TRAIN=0
cd src/Code_75/experiments
python run_all_experiments.py
```

To use a specific weight file manually in Python:

```python
import torch

checkpoint = torch.load("weights/FIL001_75mm_seed124_ema.pth", map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])
```

## File Size

Each checkpoint is approximately 50 MB.
