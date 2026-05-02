#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Path Configuration Verification Script

This script verifies that all paths are correctly configured before running
experiments or plotting scripts.

Usage:
    python verify_paths.py
"""

import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
SRC_DIR = SCRIPT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

try:
    from filament_layout import (
        PROJECT_ROOT,
        DATA_ROOT_BASE,
        RUNS_ROOT_BASE,
        REPORTS_ROOT_BASE,
        canonical_data_root,
        canonical_run_root,
    )
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)


def print_header(title):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def check_path(path, description):
    """Check if a path exists and print status."""
    exists = os.path.exists(path)
    status = "OK" if exists else "FAIL"
    print(f"{status} {description}")
    print(f"  Path: {path}")
    if not exists:
        print(f"  WARN  Directory does not exist!")
    return exists


def count_files(directory, pattern="*.BMP"):
    """Count files matching a pattern in a directory."""
    if not os.path.exists(directory):
        return 0
    from glob import glob
    return len(glob(os.path.join(directory, pattern)))


def main():
    print_header("Path Configuration Verification")

    # Check if filament_layout can be imported
    if not IMPORT_SUCCESS:
        print("\nERROR: Cannot import filament_layout")
        print(f"   {IMPORT_ERROR}")
        print("\nPlease ensure:")
        print("  1. src/filament_layout.py exists")
        print("  2. Current directory is the publish/ root")
        print("  3. Python path is correctly configured")
        return 1

    print("\nOK Successfully imported filament_layout")

    # Print environment variable
    print_header("Environment Variables")
    env_var = os.environ.get("FILAMENT_PROJECT_ROOT")
    if env_var:
        print(f"OK FILAMENT_PROJECT_ROOT is set: {env_var}")
    else:
        print(f"WARN  FILAMENT_PROJECT_ROOT is not set (using default)")

    # Print core paths
    print_header("Core Paths")
    print(f"PROJECT_ROOT:       {PROJECT_ROOT}")
    print(f"DATA_ROOT_BASE:     {DATA_ROOT_BASE}")
    print(f"RUNS_ROOT_BASE:     {RUNS_ROOT_BASE}")
    print(f"REPORTS_ROOT_BASE:  {REPORTS_ROOT_BASE}")

    # Check if core directories exist
    print_header("Directory Check")
    all_exist = True
    all_exist &= check_path(PROJECT_ROOT, "Project Root")
    all_exist &= check_path(DATA_ROOT_BASE, "Data Root Base")
    all_exist &= check_path(RUNS_ROOT_BASE, "Runs Root Base")

    # Check FIL001 data
    print_header("FIL001 Data Check")

    for focal_mm in [75, 120]:
        print(f"\n--- Focal Length: {focal_mm}mm ---")
        data_root = canonical_data_root('FIL001', focal_mm)
        print(f"Data root: {data_root}")

        raw_dir = os.path.join(data_root, 'raw')
        if os.path.exists(raw_dir):
            bmp_count = count_files(raw_dir, "*.BMP")
            expected = 10 if focal_mm == 75 else 8
            if bmp_count == expected:
                print(f"  OK Found {bmp_count}/{expected} raw images")
            elif bmp_count > 0:
                print(f"  WARN  Found {bmp_count}/{expected} raw images (expected {expected})")
            else:
                print(f"  FAIL No BMP files found in {raw_dir}")
        else:
            print(f"  FAIL Raw directory does not exist: {raw_dir}")

    # Check FIL002 data
    print_header("FIL002 Data Check")

    fil002_exists = False
    for focal_mm in [75, 120]:
        data_root = canonical_data_root('FIL002', focal_mm)
        raw_dir = os.path.join(data_root, 'raw')
        if os.path.exists(raw_dir):
            bmp_count = count_files(raw_dir, "*.BMP")
            if bmp_count > 0:
                print(f"  OK FIL002 @ {focal_mm}mm: {bmp_count} images")
                fil002_exists = True

    if not fil002_exists:
        print("  WARN  FIL002 data not found (optional for cross-filament evaluation)")

    # Check for trained weights
    print_header("Trained Weights Check")

    weights_found = False
    for filament_id in ['FIL001', 'FIL002']:
        for focal_mm in [75, 120]:
            for seed in [124, 212, 42]:
                run_root = canonical_run_root(filament_id, focal_mm)
                weight_path = os.path.join(run_root, 'baseline', f'seed_{seed}', 'best_ema_model.pth')
                if os.path.exists(weight_path):
                    print(f"  OK Found: {filament_id} @ {focal_mm}mm, seed {seed}")
                    weights_found = True

    if not weights_found:
        print("  WARN  No trained weights found")
        print("     You can either:")
        print("     1. Train from scratch using run_all_experiments.py")
        print("     2. Download pre-trained weights (see README.md)")

    # Check plotting script paths
    print_header("Plotting Script Path Check")

    plot_script = SCRIPT_DIR / "scripts" / "plot_network_full_analysis_nm.py"
    if plot_script.exists():
        print(f"OK Found plotting script: {plot_script}")

        # Read and check for hardcoded paths
        with open(plot_script, 'r', encoding='utf-8') as f:
            content = f.read()
            if '/root/autodl-tmp' in content:
                print("  WARN  WARNING: Script contains hardcoded paths '/root/autodl-tmp'")
                print("     Please modify the following in plot_network_full_analysis_nm.py:")
                print("     - BASELINE_METRICS dictionary (around line 563)")
                print("     - ABLATION_ROOTS dictionary (around line 571)")
                print("     - out_dir and tab_dir (around line 650)")
                print("     See PATH_CONFIGURATION_GUIDE.md for details")
            else:
                print("  OK No hardcoded '/root/autodl-tmp' paths found")
    else:
        print(f"FAIL Plotting script not found: {plot_script}")

    # Summary
    print_header("Summary")

    if all_exist:
        print("OK All core directories exist")
    else:
        print("FAIL Some core directories are missing")
        print("  Please check the paths above and ensure data is in the correct location")

    print("\nNext Steps:")
    if not weights_found:
        print("  1. Train a model: cd src/Code_75/experiments && python run_all_experiments.py")
    else:
        print("  1. Evaluate existing model: cd src/Code_75/experiments && python run_all_experiments.py --eval_only")
    print("  2. Generate figures: python scripts/plot_network_full_analysis_nm.py")
    print("  3. See README.md for detailed instructions")

    print("\n" + "=" * 70)

    return 0 if all_exist else 1


if __name__ == "__main__":
    sys.exit(main())
