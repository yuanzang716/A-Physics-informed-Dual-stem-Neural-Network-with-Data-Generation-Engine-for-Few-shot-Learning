# Physics-informed Dual-stem Neural Network for Few-shot Inverse Problems

This repository contains the official implementation of the paper: **"A Physics-informed Dual-stem Neural Network with Data Generation Engine for Few-shot Learning"**.

## Overview
We propose a physics-informed learning framework to solve high-precision inverse problems in the few-shot regime. The framework integrates:
1. **MDGA (Measurement-guided local Data Generation and Alignment)**: Transforms a few real measurements into dense, measurement-specific simulation banks.
2. **DSN (Dual-Stem Network)**: A network architecture designed to fuse physical structure with residual evidence while mitigating sim-to-real mismatch.

The method is demonstrated on **Fraunhofer diffraction-based filament diameter metrology**, achieving significant precision improvements over classical fitting-based baselines, and we define a general pinn to compere with out framwork, the result shows that our framwork is better than the general pinn.

## System Requirements
- **Hardware**: NVIDIA GPU (RTX 4090 24GB recommended for speed), 16+ vCPUs.
- **OS**: Ubuntu 22.04 LTS.
- **Training Time**: ~300 seconds per epoch (on RTX 4090).

## Installation
We recommend using Conda to manage the environment:

```bash
# Create environment
conda create -n fewshot-metrology python=3.12
conda activate fewshot-metrology

# Install core dependencies
pip install torch==2.8.0 torchvision --index-url https://download.pytorch.org/whl/cu121
pip install numpy scipy pandas opencv-python matplotlib tqdm joblib pillow scikit-learn openpyxl
```

## Directory Structure
```text
.
├── src/
│   ├── code_75/         # 75mm focal length training & evaluation
│   ├── code_120/        # 120mm focal length training & evaluation
│   ├── de_1d_fit/       # Differential Evolution 1D profile fitting
│   ├── fft_analysis.py  # FFT-based diameter initialization
│   └── ...
├── scripts/             # Analysis and plotting scripts for paper figures
├── data/
│   ├── Diameter_100.2_75mm/
│   │   ├── train_real/  # All real BMP images (Training)
│   │   ├── val_real/    # All real BMP images (Validation)
│   │   └── simulation_dataset/ (Demo samples only)
│   └── Diameter_100.2_120mm/
└── README.md
```

## Data Preparation
Following the user's requirement:
- **Real Images**: All real `.bmp` measurements for 75mm and 120mm setups are included in the `data/` folder.
- **Simulated Data**: Only representative demo samples are provided in `simulation_dataset/` to maintain a compact repository size. The full simulation bank can be regenerated using the `MDGA` engine provided in the source code.

## Quick Start
To reproduce the results in the paper, follow the pipeline in the order described below.

### 1. FFT Baseline Analysis
Calculate the coarse diameter estimates using the FFT-based method.
```bash
# From the project root
python src/fft_analysis.py --ground_truth 100.2
```
- **Output**: `fft_analysis_results.csv` and summary statistics in the terminal.

### 2. Numerical Simulation (Ill-posedness Analysis)
Generate the synthetic error landscapes to visualize the multi-basin structure and parameter coupling.
```bash
python src/synthetic_multi_basin_analysis.py
```
- **Output**: `src/synthetic_outputs/` containing PDF/PNG landscapes.

### 3. Main Training (PI-DSN)
The core training scripts for different focal lengths (75mm or 120mm). 
**Path Setting**: These scripts read paths and security flags from environment variables.

```bash
# Example for 75mm Focal Length
export ALLOW_TRAIN=1
export DATA_ROOT=$(pwd)/data/Diameter_100.2_75mm
cd src/code_75/Code_75/core
python main_75.py
```
- **Environment Variables**:
  - `ALLOW_TRAIN=1`: Security flag required to enable GPU training.
  - `DATA_ROOT`: Absolute path to the dataset folder (containing `train_real/` and `val_real/`).
  - `NUM_EPOCHS` (Optional): Override default training length (e.g., `export NUM_EPOCHS=60`).
- **Output**: `results/` directory containing `best_ema_model.pth`, `metrics.csv`, and `ablation_env_effective.json`.

### 4. Complete Experiment Workflow (Multi-seed)
Run the full suite of experiments (Probe, Baseline, Ablation, Sensitivity) across multiple random seeds.
```bash
cd src/code_75/Code_75/experiments
python run_all_experiments.py
```
- **Configuration**: Edit the `CONFIG` dictionary at the top of `run_all_experiments.py` to specify the `main_path`.
- **Output**: A structured hierarchy in `results/seed_*/` for each experiment stage.

### 5. Custom Ablation Sweep
Run a systematic ablation study by overriding specific components of the baseline strategy.
```bash
export ALLOW_TRAIN=1
export BASELINE_DIR=/path/to/your/seed_124/baseline
export SWEEP_ROOT=/path/to/output/ablation_sweep
cd src/code_75/Code_75/experiments
python ablation.py
```
- **Required Env Vars**:
  - `BASELINE_DIR`: Path to the successful baseline run folder.
  - `SWEEP_ROOT`: Destination directory for the ablation results.
- **Output**: `ablation_env_effective.json` in each subfolder to audit the exact configuration used.

## Execution Modes
The core scripts (`main_75.py` and `main_120.py`) support two primary execution modes controlled via environment variables.

### 1. Training Mode (Default)
Used to run the full data-to-training pipeline. For safety, a hardware/cost protection flag is required.
```bash
export ALLOW_TRAIN=1
export DATA_ROOT=$(pwd)/data/Diameter_100.2_75mm
export BASE_MODEL_SAVE_DIR=$(pwd)/results/my_experiment
python src/code_75/Code_75/core/main_75.py
```
- **Key Variables**:
  - `ALLOW_TRAIN=1`: **Mandatory** to enable training.
  - `DATA_ROOT`: Path to the raw measurement data.
  - `BASE_MODEL_SAVE_DIR`: Where to save weights, logs, and snapshots.
  - `SEED`: Random seed for reproducibility.

### 2. Export Mode
Used to load a trained model and export diameter predictions for all images (Train/Val) into CSV files.
```bash
export MODE=export_preds
export EXPORT_CKPT=ema
export BASE_MODEL_SAVE_DIR=$(pwd)/results/my_experiment
export EXPORT_OUT_DIR=$(pwd)/exports
python src/code_75/Code_75/core/main_75.py
```
- **Key Variables**:
  - `MODE=export_preds`: Triggers the prediction export logic.
  - `EXPORT_CKPT`: Choose which checkpoint to use: `best` (default) or `ema` (recommended).
  - `BASE_MODEL_SAVE_DIR`: Must point to the folder containing your `.pth` files.
  - `EXPORT_OUT_DIR`: (Optional) Destination for the generated `.csv` files. Defaults to `BASE_MODEL_SAVE_DIR/pred_exports`.

## Analysis and Plotting
After training, use the scripts in the `scripts/` folder to generate paper-ready figures and source data tables.

### 1. Network Training and Ablation Analysis
Generates the comprehensive training dynamics and ablation study summary (corresponding to Figure 4 in the paper).
```bash
python scripts/plot_network_full_analysis_nm.py
```
- **Function**: Plots loss/score curves for both focal lengths and compiles the ablation summary bar chart (Panel E).
- **Outputs**: 
  - `focal_length_comparison_outputs/figures/Fig_results_network_training_ablation_sensitivity.pdf`
  - `focal_length_comparison_outputs/tables/source_data_seed124_ablation_summary.csv` (and other source data).

### 2. Sim-to-Real Mismatch Analysis
Characterizes the structured discrepancy between idealized physics simulations and real measurements (corresponding to Figure 2/3).
```bash
python scripts/unified_mismatch_analysis.py
```
- **Function**: Computes Correlation, NRMSE, SSIM, and PSD ratios across paired real/sim images. Performs statistical morphology analysis on mismatch distributions.
- **Outputs**:
  - `figures/Fig1_example_tiles.pdf`: Representative real vs. simulated diffraction patterns.
  - `figures/Fig2_mismatch_analysis.pdf`: Violin plots of metrics and radial PSD ratio curves.
  - `tables/table_mismatch_stats_75_vs_120.csv`: Statistical comparison between focal lengths.
  - `tables/mismatch_per_pair_***mm.csv`: Detailed per-image mismatch metrics.

### 3. Seed Robustness (Probe Stage) Results
Visualizes the stability of the training protocol across multiple random seeds (Probe stage analysis).
```bash
python scripts/plot_network_probe_results_nm.py
```
- **Function**: Displays the relative error of EMA-selected checkpoints for multiple probe runs (e.g., 10 seeds), highlighting the success rate.
- **Outputs**:
  - `focal_length_comparison_outputs/figures/Fig_results_network_seed_robustness.pdf`
  - `focal_length_comparison_outputs/tables/source_data_probe_stats_***mm.csv`.
*Note: Ensure the `seed_roots` in the plotting scripts point to your `results/` directories.*

## Citation
If you find this work useful, please cite:
```text
 Zhang, Y., et al. "A Physics-informed Dual-stem Neural Network with Data Generation Engine for Few-shot Learning." (2026).
```

## License
This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

