#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sensitivity.py - Sensitivity analysis script

Run sensitivity analysis on top of the baseline to test model robustness under different perturbations.

Usage:
1) Configure via environment variables:
   - SEED: random seed
   - MAIN_PATH: path to main_120.py
   - SWEEP_ROOT: sensitivity analysis output directory
   - BASELINE_DIR: baseline results directory
2) Or run this file directly (uses default settings)
"""

import os
import sys
import subprocess
import json
import time
from typing import Dict, List, Any


# =============================================================================
#  Project-level Path Configuration
# =============================================================================

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


# =============================================================================
#  Baseline Strategy Env (sync with run_all_experiments.py baseline)
# =============================================================================
BASELINE_ENV = {
    "EARLY_PAIR_ONLY_EPOCHS": "10",
    "LOSS_GRADUAL_INTRO_ENABLE": "1",
    "LOSS_GRADUAL_INTRO_EPOCHS": "10",
    "LOSS_GRADUAL_INTRO_MODE": "cosine",
    "LOSS_GRADUAL_INTRO_TARGETS": "phys,weak,vic",
    "LOSS_CONDITIONAL_ENABLE": "1",
    "LOSS_CONDITIONAL_PAIR_THRESHOLD": "0.025",
    "LOSS_CONDITIONAL_STABLE_EPOCHS": "2",
    "W_PHYS_REDUCED": "0.5",
    "LOSS_ADAPTIVE_PAIR_BOOST": "1",
    "LOSS_ADAPTIVE_PAIR_THRESHOLD": "0.030",
    "LOSS_ADAPTIVE_PAIR_MAX_BOOST": "2.5",
}

# =============================================================================
#  Sensitivity Analysis Configuration
# =============================================================================
SENSITIVITY_CONFIGS = [
    {
        "name": "coarse_perturb_plus3pct",
        "desc": "Coarse diameter perturbation +3%", 
        "env": {
            "SENSITIVITY_COARSE_PERTURB": "0.03",
        }
    },
    {
        "name": "coarse_perturb_minus3pct",
        "desc": "Coarse diameter perturbation -3%", 
        "env": {
            "SENSITIVITY_COARSE_PERTURB": "-0.03",
        }
    },
    {
        "name": "image_noise_std001",
        "desc": "Image noise std=0.01",
        "env": {
            "SENSITIVITY_IMAGE_NOISE": "0.01",
        }
    },
    {
        "name": "params_zero",
        "desc": "Zero out parameters (test parameter dependence)",
        "env": {
            "SENSITIVITY_PARAMS_ZERO": "1",
        }
    },
]


# =============================================================================
#  User Settings (can be overridden by environment variables)
# =============================================================================
USER_SETTINGS = {
    "SEED": int(os.environ.get("SEED", "42")),
    "MAIN_PATH": os.environ.get("MAIN_PATH", os.path.join(PROJECT_ROOT, "core", "main_120.py")),
    "SWEEP_ROOT": os.environ.get("SWEEP_ROOT", ""),
    "BASELINE_DIR": os.environ.get("BASELINE_DIR", ""),
    "NUM_EPOCHS": int(os.environ.get("NUM_EPOCHS", "60")),
    "SKIP_IF_EXISTS": True,
    "MAX_BATCHES": int(os.environ.get("MAX_BATCHES", "50")),  # Sensitivity analysis only runs a small number of batches
}


def run_sensitivity_config(config: Dict[str, Any], exp_dir: str) -> bool:
    """Run single sensitivity analysis configuration"""
    print(f"\n  [SENSITIVITY] {config['name']}: {config['desc']}")
    print(f"     Output: {exp_dir}")
    
    # Check if already completed
    done_file = os.path.join(exp_dir, "DONE.txt")
    if USER_SETTINGS["SKIP_IF_EXISTS"] and os.path.exists(done_file):
        print(f"     [SKIP] Already exists, skipping")
        return True
    
    os.makedirs(exp_dir, exist_ok=True)
    
    # Prepare environment variables
    env = os.environ.copy()
    env.update(BASELINE_ENV)
    env.update({
        "SEED": str(USER_SETTINGS["SEED"]),
        "BASE_MODEL_SAVE_DIR": exp_dir,
        "BASELINE_DIR": USER_SETTINGS["BASELINE_DIR"],  # Pass baseline directory
        "NUM_EPOCHS": str(USER_SETTINGS["NUM_EPOCHS"]),
        "SENSITIVITY_MODE": "1",  # Enable sensitivity analysis mode
        "MAX_BATCHES": str(USER_SETTINGS["MAX_BATCHES"]),
    })
    env.update(config.get("env", {}))
    
    try:
        result = subprocess.run(
            [sys.executable, USER_SETTINGS["MAIN_PATH"]],
            env=env,
            cwd=os.path.dirname(USER_SETTINGS["MAIN_PATH"]),
            check=True,
            capture_output=False,
        )
        
        # Mark as completed
        with open(done_file, "w") as f:
            f.write(f"Completed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        print(f"     [OK] Completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"     [ERROR] Failed: {e}")
        return False


def main():
    """Main function"""
    print("="*70)
    print("Sensitivity Analysis")
    print("="*70)
    
    # Check baseline directory
    baseline_dir = USER_SETTINGS["BASELINE_DIR"]
    if not baseline_dir or not os.path.exists(baseline_dir):
        print(f"[ERROR] Baseline directory does not exist: {baseline_dir}")
        print("   Please set environment variable BASELINE_DIR")
        return
    
    # Set output directory
    sweep_root = USER_SETTINGS["SWEEP_ROOT"]
    if not sweep_root:
        # Default: create sensitivity_analysis in same directory as baseline
        sweep_root = os.path.join(os.path.dirname(baseline_dir), "sensitivity_analysis")
    os.makedirs(sweep_root, exist_ok=True)
    
    print(f"\nConfiguration:")
    print(f"   Seed: {USER_SETTINGS['SEED']}")
    print(f"   Baseline: {baseline_dir}")
    print(f"   Output: {sweep_root}")
    print(f"   Max Batches: {USER_SETTINGS['MAX_BATCHES']}")
    print(f"\nNumber of sensitivity configurations: {len(SENSITIVITY_CONFIGS)}")
    
    # Run all sensitivity configurations
    results = {}
    for config in SENSITIVITY_CONFIGS:
        exp_dir = os.path.join(sweep_root, config["name"])
        success = run_sensitivity_config(config, exp_dir)
        results[config["name"]] = success
    
    # Save summary results
    summary_file = os.path.join(sweep_root, "sensitivity_summary.json")
    with open(summary_file, "w") as f:
        json.dump({
            "seed": USER_SETTINGS["SEED"],
            "baseline_dir": baseline_dir,
            "sweep_root": sweep_root,
            "configs": SENSITIVITY_CONFIGS,
            "results": results,
            "completed_at": time.strftime('%Y-%m-%d %H:%M:%S'),
        }, f, indent=2)
    
    print("\n" + "="*70)
    print("[OK] Sensitivity analysis completed!")
    print("="*70)
    print(f"Summary results: {summary_file}")


if __name__ == "__main__":
    main()
