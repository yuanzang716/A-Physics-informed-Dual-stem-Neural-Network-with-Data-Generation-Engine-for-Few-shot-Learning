#!/bin/bash
# setup_env.sh
# Environment Variable Configuration Script
#
# Usage:
#   source setup_env.sh
#
# Note:
#   Must use 'source' command, not './setup_env.sh'

# ============================================================
# Core Path Configuration
# ============================================================

# Please modify to your actual path
# Examples:
#   Linux:   /home/username/research/diffraction-metrology
#   Mac:     /Users/username/research/diffraction-metrology
#   Windows: /d/Projects/diffraction-metrology (Git Bash)

export FILAMENT_PROJECT_ROOT="/root/autodl-tmp"

# If you are already in the publish directory, you can use relative path:
# export FILAMENT_PROJECT_ROOT="$(pwd)"

# ============================================================
# Experiment Configuration
# ============================================================

# Filament ID
export FILAMENT_ID="FIL001"

# Evaluation Filament ID (for cross-filament generalization)
export EVAL_FILAMENT_ID="FIL001"

# Focal Length (mm)
export FOCAL_MM="75"

# Ground Truth Diameter (um)
export GT_DIAMETER_UM="100.2"

# ============================================================
# Training Parameters
# ============================================================

# Random Seed
export SEED="124"

# Number of Epochs
export NUM_EPOCHS="60"

# Allow Training
export ALLOW_TRAIN="1"

# ============================================================
# Advanced Configuration
# ============================================================

# Early pair-only epochs
export EARLY_PAIR_ONLY_EPOCHS="10"

# Loss gradual introduction
export LOSS_GRADUAL_INTRO_ENABLE="1"
export LOSS_GRADUAL_INTRO_EPOCHS="10"
export LOSS_GRADUAL_INTRO_MODE="cosine"
export LOSS_GRADUAL_INTRO_TARGETS="phys,weak,vic"

# Conditional loss
export LOSS_CONDITIONAL_ENABLE="1"
export LOSS_CONDITIONAL_PAIR_THRESHOLD="0.025"
export LOSS_CONDITIONAL_STABLE_EPOCHS="2"

# Physics loss weight
export W_PHYS_REDUCED="0.5"

# Adaptive pair loss boost
export LOSS_ADAPTIVE_PAIR_BOOST="1"
export LOSS_ADAPTIVE_PAIR_THRESHOLD="0.030"
export LOSS_ADAPTIVE_PAIR_MAX_BOOST="2.5"

# ============================================================
# Output Paths
# ============================================================

# Run output root
export RUN_ROOT="${FILAMENT_PROJECT_ROOT}/runs/train_${FILAMENT_ID}/focal_${FOCAL_MM}mm"

# Report output root
export REPORT_ROOT="${FILAMENT_PROJECT_ROOT}/reports/${FILAMENT_ID}"

# ============================================================
# Verify Configuration
# ============================================================

echo "============================================================"
echo "Environment Variables Set"
echo "============================================================"
echo "Core Paths:"
echo "  FILAMENT_PROJECT_ROOT: ${FILAMENT_PROJECT_ROOT}"
echo ""
echo "Experiment Configuration:"
echo "  FILAMENT_ID:           ${FILAMENT_ID}"
echo "  EVAL_FILAMENT_ID:      ${EVAL_FILAMENT_ID}"
echo "  FOCAL_MM:              ${FOCAL_MM}"
echo "  GT_DIAMETER_UM:        ${GT_DIAMETER_UM}"
echo ""
echo "Training Parameters:"
echo "  SEED:                  ${SEED}"
echo "  NUM_EPOCHS:            ${NUM_EPOCHS}"
echo "  ALLOW_TRAIN:           ${ALLOW_TRAIN}"
echo ""
echo "Output Paths:"
echo "  RUN_ROOT:              ${RUN_ROOT}"
echo "  REPORT_ROOT:           ${REPORT_ROOT}"
echo "============================================================"
echo ""
echo "Next Steps:"
echo "  1. Verify paths: python verify_paths.py"
echo "  2. Start training: cd src/Code_75/experiments && python run_all_experiments.py"
echo "============================================================"
