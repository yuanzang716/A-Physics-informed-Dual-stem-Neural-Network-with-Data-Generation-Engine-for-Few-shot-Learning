#!/bin/bash
# reproduce_all_figures.sh — Reproduce all figures and tables from the paper
#
# Usage:
#   chmod +x reproduce_all_figures.sh
#   ./reproduce_all_figures.sh /path/to/publish
#
# Prerequisites:
#   - Completed training runs (or use pre-trained weights)
#   - Run fix_hardcoded_paths.py first to fix plotting script paths

set -e

if [ -z "$1" ]; then
    echo "Usage: ./reproduce_all_figures.sh <path_to_publish_directory>"
    echo "Example: ./reproduce_all_figures.sh /home/user/publish"
    exit 1
fi

PUBLISH_DIR="$(cd "$1" && pwd)"

echo "============================================================"
echo "Reproducing All Paper Figures and Tables"
echo "============================================================"
echo "PUBLISH_DIR: $PUBLISH_DIR"
echo ""

export FILAMENT_PROJECT_ROOT="$PUBLISH_DIR"

# Fix hardcoded paths in plotting scripts
echo "[0/5] Fixing hardcoded paths in plotting scripts..."
python "$PUBLISH_DIR/fix_hardcoded_paths.py" "$PUBLISH_DIR"
echo ""

# Create output directories
mkdir -p "$PUBLISH_DIR/figures"
mkdir -p "$PUBLISH_DIR/tables"

cd "$PUBLISH_DIR/scripts"

# Table 2: FFT Baseline
echo "[1/5] Table 2: FFT baseline estimates..."
python fft_analysis.py
echo ""

# Figure 1: Parameter coupling
echo "[2/5] Figure 1: Parameter coupling analysis..."
python synthetic_multi_basin_analysis.py
echo ""

# Figure 4: Sim-to-real mismatch
echo "[3/5] Figure 4: Simulation-to-measurement mismatch..."
python unified_mismatch_analysis.py
echo ""

# Figure 5: Training dynamics + ablation
echo "[4/5] Figure 5: Training dynamics and ablation study..."
python plot_network_full_analysis_nm.py
echo ""

# Section 3.6: Seed stability
echo "[5/5] Section 3.6: Seed stability (probe experiment)..."
python plot_network_probe_results_nm.py
echo ""

echo "============================================================"
echo "All figures and tables generated!"
echo "  Figures: $PUBLISH_DIR/figures/"
echo "  Tables:  $PUBLISH_DIR/tables/"
echo "============================================================"
