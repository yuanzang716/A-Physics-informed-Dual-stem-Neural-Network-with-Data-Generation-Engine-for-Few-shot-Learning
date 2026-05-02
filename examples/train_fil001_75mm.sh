#!/bin/bash
# train_fil001_75mm.sh — Full training pipeline for FIL001 at 75mm
#
# Usage:
#   chmod +x train_fil001_75mm.sh
#   ./train_fil001_75mm.sh /path/to/publish
#
# This runs the complete 4-stage pipeline:
#   1. Coarse fitting (FFT + DE)
#   2. Simulation bank generation (LHS)
#   3. PI-DSN training
#   4. Label-free checkpoint selection (EMA)

set -e

if [ -z "$1" ]; then
    echo "Usage: ./train_fil001_75mm.sh <path_to_publish_directory>"
    echo "Example: ./train_fil001_75mm.sh /home/user/publish"
    exit 1
fi

PUBLISH_DIR="$(cd "$1" && pwd)"

echo "============================================================"
echo "Full Training: FIL001 at 75mm focal length"
echo "============================================================"
echo "PUBLISH_DIR: $PUBLISH_DIR"
echo ""

# Set environment variables
export FILAMENT_PROJECT_ROOT="$PUBLISH_DIR"
export FILAMENT_ID=FIL001
export EVAL_FILAMENT_ID=FIL001
export FOCAL_MM=75
export GT_DIAMETER_UM=100.2
export SEED=124
export NUM_EPOCHS=60
export ALLOW_TRAIN=1

# Advanced training parameters
export EARLY_PAIR_ONLY_EPOCHS=10
export LOSS_GRADUAL_INTRO_ENABLE=1
export LOSS_GRADUAL_INTRO_EPOCHS=10
export LOSS_GRADUAL_INTRO_MODE=cosine
export LOSS_GRADUAL_INTRO_TARGETS=phys,weak,vic
export LOSS_CONDITIONAL_ENABLE=1
export LOSS_CONDITIONAL_PAIR_THRESHOLD=0.025
export LOSS_CONDITIONAL_STABLE_EPOCHS=2
export W_PHYS_REDUCED=0.5
export LOSS_ADAPTIVE_PAIR_BOOST=1
export LOSS_ADAPTIVE_PAIR_THRESHOLD=0.030
export LOSS_ADAPTIVE_PAIR_MAX_BOOST=2.5

echo "Configuration:"
echo "  Filament:     $FILAMENT_ID"
echo "  Focal length: ${FOCAL_MM}mm"
echo "  Seed:         $SEED"
echo "  Epochs:       $NUM_EPOCHS"
echo "  Training:     ENABLED"
echo ""

# Run full pipeline
echo "Starting 4-stage training pipeline..."
cd "$PUBLISH_DIR/src/Code_75/experiments"
python run_all_experiments.py

echo ""
echo "============================================================"
echo "Training complete!"
echo "Results saved to: $PUBLISH_DIR/runs/train_FIL001/focal_75mm/"
echo "============================================================"
