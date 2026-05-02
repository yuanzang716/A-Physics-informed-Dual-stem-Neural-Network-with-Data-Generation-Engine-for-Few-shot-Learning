#!/bin/bash
# quick_start.sh — Evaluate pre-trained models without training
#
# Usage:
#   chmod +x quick_start.sh
#   ./quick_start.sh /path/to/publish
#
# This script evaluates the pre-trained EMA weights on FIL001 (75mm, seed 124).

set -e

if [ -z "$1" ]; then
    echo "Usage: ./quick_start.sh <path_to_publish_directory>"
    echo "Example: ./quick_start.sh /home/user/publish"
    exit 1
fi

PUBLISH_DIR="$(cd "$1" && pwd)"

echo "============================================================"
echo "Quick Start: Evaluate Pre-trained Model"
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
export ALLOW_TRAIN=0

echo "Configuration:"
echo "  Filament:     $FILAMENT_ID"
echo "  Focal length: ${FOCAL_MM}mm"
echo "  Seed:         $SEED"
echo "  Training:     DISABLED (evaluation only)"
echo ""

# Verify paths
echo "Verifying paths..."
python "$PUBLISH_DIR/verify_paths.py" || true
echo ""

# Run evaluation
echo "Running evaluation..."
cd "$PUBLISH_DIR/src/Code_75/experiments"
python run_all_experiments.py

echo ""
echo "============================================================"
echo "Evaluation complete!"
echo "============================================================"
