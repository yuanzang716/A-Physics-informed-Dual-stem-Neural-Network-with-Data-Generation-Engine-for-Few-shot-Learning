# Diffraction-Based Filament Metrology: A Few-shot Inverse Problem Workflow

Code and data for the paper:

> **A Physics-informed Dual-Stream Neural Network with Data Generation Engine for Label-scarce and Sparse-data Inverse Problems**
>
> Yuan Zhang, Jiao Zhao, Lin Chen, MingYang Li,  JiaHao Han, Qiang Lin, Bin Wu, ZhengHui Hu
>
> 2026

---

## Overview

This repository provides a complete, reproducible implementation of the Physics-Informed Dual-Stream Network (PI-DSN) with Measurement-Guided Data Augmentation (MDGA) for few-shot inverse problems. The method estimates filament diameters from Fraunhofer diffraction patterns using only 8--10 real measurements per focal length.

**Key components:**
- Four-stage training pipeline (coarse fitting, simulation bank generation, PI-DSN training, label-free checkpoint selection)
- Physics-informed loss with adaptive weighting
- EMA-based checkpoint selection without ground-truth labels
- Cross-filament generalization (FIL001 -> FIL002)

---

## Directory Structure

```
publish/
├── README.md                              # This file
├── PATH_CONFIGURATION_GUIDE.md            # Detailed path configuration guide
├── HARDCODED_PATHS_INVENTORY.md           # Complete hardcoded path inventory
├── requirements.txt                       # pip dependencies
├── environment.yml                        # conda environment
├── verify_paths.py                        # Path verification tool
├── fix_hardcoded_paths.py                 # Automatic path fixer for plotting scripts
├── setup_env.sh                           # Linux/Mac environment setup
├── setup_env.bat                          # Windows environment setup
│
├── data/filaments/                        # Raw diffraction pattern images
│   ├── FIL001/
│   │   ├── focal_75mm/raw/               # 10 BMP images
│   │   └── focal_120mm/raw/              # 8 BMP images
│   ├── FIL002/
│   │   ├── focal_75mm/raw/               # 5 BMP images
│   │   └── focal_120mm/raw/              # 9 BMP images
│   ├── registry.csv                       # Filament metadata registry
│   └── migration_map.csv                  # Data migration mapping
│
├── weights/                               # Pre-trained EMA model weights
│   ├── FIL001_75mm_seed124_ema.pth
│   ├── FIL001_75mm_seed212_ema.pth
│   ├── FIL001_120mm_seed124_ema.pth
│   ├── FIL001_120mm_seed212_ema.pth
│   ├── FIL002_75mm_seed42_ema.pth
│   ├── FIL002_120mm_seed42_ema.pth
│   └── README.md                          # Weight file documentation
│
├── src/                                   # Core source code
│   ├── filament_layout.py                 # Centralized path configuration
│   ├── bootstrap_filament_layout.py       # Bootstrap path resolver
│   ├── Code_75/                           # 75mm focal length pipeline
│   │   ├── core/main_75.py               # Core training module
│   │   ├── experiments/
│   │   │   ├── run_all_experiments.py     # Main entry point (4-stage pipeline)
│   │   │   ├── ablation.py               # Ablation experiments
│   │   │   └── sensitivity.py            # Sensitivity analysis
│   │   └── analysis/
│   │       └── analyze_probe_stats.py     # Probe statistics analysis
│   └── Code_120/                          # 120mm focal length pipeline
│       └── (same structure as Code_75)
│
├── scripts/                               # Plotting and analysis scripts
│   ├── fft_analysis.py                    # Table 2: FFT baseline
│   ├── synthetic_multi_basin_analysis.py  # Fig.1: parameter coupling
│   ├── unified_mismatch_analysis.py       # Fig.4: sim-to-real mismatch
│   ├── plot_network_full_analysis_nm.py   # Fig.5: training dynamics + ablation
│   └── plot_network_probe_results_nm.py   # Sec.3.6: seed stability
│
└── examples/                              # Example scripts
    ├── quick_start.sh                     # Quick evaluation demo
    ├── train_fil001_75mm.sh               # Full training example
    └── reproduce_all_figures.sh           # Reproduce all paper figures
```

---

## Quick Start (3 Steps)

### Step 1: Set Up Environment

```bash
# Option A: conda (recommended)
conda env create -f environment.yml
conda activate diffraction-metrology

# Option B: pip
pip install -r requirements.txt
```

### Step 2: Configure Paths

The code uses an environment variable `FILAMENT_PROJECT_ROOT` to locate data and outputs.

```bash
# Linux / macOS
export FILAMENT_PROJECT_ROOT="/path/to/this/publish"

# Windows (Command Prompt)
set FILAMENT_PROJECT_ROOT=D:\path\to\this\publish

# Windows (PowerShell)
$env:FILAMENT_PROJECT_ROOT = "D:\path\to\this\publish"
```

Or use the provided setup scripts:
```bash
# Linux / macOS
source setup_env.sh

# Windows
setup_env.bat
```

> **Important:** Many plotting scripts contain hardcoded paths from the original development environment. Run the automatic path fixer before using them:
> ```bash
> python fix_hardcoded_paths.py /path/to/this/publish
> ```
> See [PATH_CONFIGURATION_GUIDE.md](PATH_CONFIGURATION_GUIDE.md) and [HARDCODED_PATHS_INVENTORY.md](HARDCODED_PATHS_INVENTORY.md) for full details.

### Step 3: Evaluate Pre-trained Models

```bash
# Evaluate FIL001 at 75mm focal length (seed 124)
export FILAMENT_PROJECT_ROOT="/path/to/this/publish"
export FILAMENT_ID=FIL001
export FOCAL_MM=75
export GT_DIAMETER_UM=100.2
export SEED=124
export ALLOW_TRAIN=0

cd src/Code_75/experiments
python run_all_experiments.py
```

This loads the pre-trained EMA weights and runs evaluation without training.

---

## Complete Training Workflow (From Scratch)

The entry point is `run_all_experiments.py`, which orchestrates four stages automatically:

### Stage 1: Coarse Fitting (FFT + Differential Evolution)
- Extracts initial diameter estimate from diffraction pattern FFT
- Refines via Differential Evolution (DE) optimization
- Output: coarse parameter estimates

### Stage 2: Simulation Bank Generation (Latin Hypercube Sampling)
- Generates synthetic diffraction patterns using LHS over the parameter space
- Creates training pairs (simulated pattern, known parameters)
- Output: simulation bank in `sim_bank/`

### Stage 3: PI-DSN Training
- Trains the Physics-Informed Dual-Stream Network
- Uses MDGA for data augmentation
- Applies adaptive loss weighting and gradual loss introduction
- Output: model checkpoints in `runs/`

### Stage 4: Label-free Checkpoint Selection
- Selects the best checkpoint using EMA-based evaluation
- No ground-truth labels required
- Output: `best_ema_model.pth`

**To run full training:**

```bash
export FILAMENT_PROJECT_ROOT="/path/to/this/publish"
export FILAMENT_ID=FIL001
export FOCAL_MM=75
export GT_DIAMETER_UM=100.2
export SEED=124
export NUM_EPOCHS=60
export ALLOW_TRAIN=1

cd src/Code_75/experiments
python run_all_experiments.py
```

For 120mm focal length, use `src/Code_120/experiments/run_all_experiments.py` with `FOCAL_MM=120`.

---

## Reproducing Paper Results

### Table 2: FFT Baseline Estimates
```bash
cd scripts
python fft_analysis.py
```
Computes coarse FFT-based diameter estimates for all filaments.

### Figure 1: Parameter Coupling Analysis
```bash
cd scripts
python synthetic_multi_basin_analysis.py
```
Generates the loss landscape showing parameter coupling in the inverse problem.

### Figure 4: Simulation-to-Measurement Mismatch
```bash
cd scripts
python unified_mismatch_analysis.py
```
Analyzes and visualizes the structured mismatch between simulated and real diffraction patterns.

### Figure 5: Training Dynamics and Ablation Study
```bash
cd scripts
python plot_network_full_analysis_nm.py
```
Plots training curves, ablation results, and convergence analysis.

> Requires completed training runs. Pre-trained results are included in `weights/`.

### Section 3.6: Seed Stability (Probe Experiment)
```bash
cd scripts
python plot_network_probe_results_nm.py
```
Analyzes prediction stability across different random seeds.

---

## Cross-Filament Generalization (FIL002)

To evaluate on a different filament (FIL002) without retraining from scratch:

```bash
export FILAMENT_PROJECT_ROOT="/path/to/this/publish"
export FILAMENT_ID=FIL002
export EVAL_FILAMENT_ID=FIL002
export FOCAL_MM=75
export GT_DIAMETER_UM=100.0
export SEED=42
export ALLOW_TRAIN=1

cd src/Code_75/experiments
python run_all_experiments.py
```

Pre-trained FIL002 weights are provided in `weights/FIL002_75mm_seed42_ema.pth` and `weights/FIL002_120mm_seed42_ema.pth`.

---

## Pre-trained Weights

| File | Filament | Focal Length | Seed | Size |
|------|----------|-------------|------|------|
| `FIL001_75mm_seed124_ema.pth` | FIL001 | 75 mm | 124 | ~50 MB |
| `FIL001_75mm_seed212_ema.pth` | FIL001 | 75 mm | 212 | ~50 MB |
| `FIL001_120mm_seed124_ema.pth` | FIL001 | 120 mm | 124 | ~50 MB |
| `FIL001_120mm_seed212_ema.pth` | FIL001 | 120 mm | 212 | ~50 MB |
| `FIL002_75mm_seed42_ema.pth` | FIL002 | 75 mm | 42 | ~50 MB |
| `FIL002_120mm_seed42_ema.pth` | FIL002 | 120 mm | 42 | ~50 MB |

All weights are EMA (Exponential Moving Average) checkpoints, which are the primary weights used for all results reported in the paper.

See [weights/README.md](weights/README.md) for details on EMA vs. standard checkpoints.

---

## Raw Data

| Dataset | Focal Length | Images | Format |
|---------|-------------|--------|--------|
| FIL001 | 75 mm | 10 | BMP |
| FIL001 | 120 mm | 8 | BMP |
| FIL002 | 75 mm | 5 | BMP |
| FIL002 | 120 mm | 9 | BMP |

Located in `data/filaments/`. These are the original, unprocessed diffraction pattern images captured from real filament samples.

---

## Path Configuration

This is the most common source of errors when reproducing results. The codebase uses a centralized path system based on the `FILAMENT_PROJECT_ROOT` environment variable.

**Quick fix:** Set the environment variable and run the path fixer:
```bash
export FILAMENT_PROJECT_ROOT="/path/to/this/publish"
python fix_hardcoded_paths.py "$FILAMENT_PROJECT_ROOT"
python verify_paths.py
```

For detailed instructions, see:
- [PATH_CONFIGURATION_GUIDE.md](PATH_CONFIGURATION_GUIDE.md) — Step-by-step configuration guide
- [HARDCODED_PATHS_INVENTORY.md](HARDCODED_PATHS_INVENTORY.md) — Complete list of hardcoded paths

---

## FAQ

**Q: Why do I get "FileNotFoundError" when running plotting scripts?**
A: The plotting scripts contain hardcoded paths from the development environment. Run `python fix_hardcoded_paths.py /path/to/publish` to fix them automatically, or see [HARDCODED_PATHS_INVENTORY.md](HARDCODED_PATHS_INVENTORY.md) for manual fixes.

**Q: What is the difference between EMA and standard weights?**
A: EMA weights are exponentially averaged model parameters that provide smoother, more stable predictions. All paper results use EMA checkpoints. Standard checkpoints (`best_model.pth`) are the raw training weights without averaging.

**Q: Can I train on a CPU?**
A: Training is designed for GPU (CUDA). CPU training is technically possible but extremely slow and not recommended. Evaluation of pre-trained models can run on CPU.

**Q: How long does training take?**
A: It depends on the amount of data and the number of simulated data generated for each image.

**Q: What does `ALLOW_TRAIN=0` do?**
A: It skips the training stage and only runs evaluation using existing checkpoints. Set `ALLOW_TRAIN=1` to enable training from scratch.

**Q: How do I reproduce results for a specific seed?**
A: Set the `SEED` environment variable (e.g., `export SEED=124`) before running `run_all_experiments.py`. The paper reports results for seeds 124 and 212 (FIL001) and seed 42 (FIL002).

---

## Requirements

- Python 3.10+
- PyTorch 2.0+ with CUDA support
- See `requirements.txt` or `environment.yml` for full dependency list

---

## License

This code is provided for academic research and reproducibility purposes.

---

## Citation

If you use this code or data, please cite:

```bibtex
@article{zhang2026workflow,
  title={A Physics-informed Dual-stem Neural Network with Data Generation Engine for Label-scarce and
Sparse-data Inverse Problems},
  author={Zhang, Yuan and Chen, Lin and Li, MingYang and Zhao, Jiao 
          and Han, JiaHao and Lin, Qiang and Wu, Bin and Hu, ZhengHui},
  journal={},
  year={2026}
}
```
