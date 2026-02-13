#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Orchestrates a sensitivity analysis by running the training script with various
perturbations applied to inputs or model parameters.

This script is designed to assess model robustness by systematically introducing
disturbances and measuring their impact on performance.
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
#  Sensitivity Analysis Configurations
# =============================================================================

SENSITIVITY_CONFIGS = [
    {
        "name": "coarse_perturb_plus3pct",
        "desc": "Coarse diameter perturbation: +3%",
        "env": {"SENSITIVITY_COARSE_PERTURB": "0.03"}
    },
    {
        "name": "coarse_perturb_minus3pct",
        "desc": "Coarse diameter perturbation: -3%",
        "env": {"SENSITIVITY_COARSE_PERTURB": "-0.03"}
    },
    {
        "name": "image_noise_std001",
        "desc": "Image noise injection: std=0.01",
        "env": {"SENSITIVITY_IMAGE_NOISE": "0.01"}
    },
    {
        "name": "params_zero",
        "desc": "Parameter zeroing test (assesses parameter dependency)",
        "env": {"SENSITIVITY_PARAMS_ZERO": "1"}
    },
]

# =============================================================================
#  User Settings
# =============================================================================

USER_SETTINGS = {
    "seed": int(os.environ.get("SEED", "42")),
    "main_path": os.environ.get("MAIN_PATH", os.path.join(PROJECT_ROOT, "core", "main_75.py")),
    "sweep_root": os.environ.get("SWEEP_ROOT", ""),
    "baseline_dir": os.environ.get("BASELINE_DIR", ""),
    "num_epochs": int(os.environ.get("NUM_EPOCHS", "60")),
    "max_batches": int(os.environ.get("MAX_BATCHES", "50")),  # Limit batches for speed.
    "skip_if_exists": os.environ.get("SKIP_IF_EXISTS", "true").lower() in ["true", "1"],
}

# =============================================================================
#  Core Logic
# =============================================================================

def _run_subprocess(script_path: str, env: Dict[str, str]) -> bool:
    """Helper to run a script in a subprocess."""
    try:
        subprocess.run(
            [sys.executable, script_path],
            env=env,
            cwd=os.path.dirname(script_path),
            check=True,
            capture_output=False,
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"  Error running script {os.path.basename(script_path)}: {e}", file=sys.stderr)
        return False

def run_sensitivity_config(config: Dict[str, Any]) -> bool:
    """Runs a single sensitivity analysis configuration."""
    exp_name = config["name"]
    exp_dir = os.path.join(USER_SETTINGS["sweep_root"], exp_name)
    print(f"\n--- Processing Sensitivity Config: {exp_name} ({config['desc']})")
    print(f"  Output Directory: {exp_dir}")

    done_file = os.path.join(exp_dir, "DONE.txt")
    if USER_SETTINGS["skip_if_exists"] and os.path.exists(done_file):
        print(f"  Skipping completed experiment: {exp_name}")
        return True

    os.makedirs(exp_dir, exist_ok=True)

    env = os.environ.copy()
    env.update({
        "SEED": str(USER_SETTINGS["seed"]),
        "BASE_MODEL_SAVE_DIR": exp_dir,
        "BASELINE_DIR": os.path.abspath(USER_SETTINGS["baseline_dir"]),
        "NUM_EPOCHS": str(USER_SETTINGS["num_epochs"]),
        "SENSITIVITY_MODE": "1",
        "MAX_BATCHES": str(USER_SETTINGS["max_batches"]),
    })
    env.update(config.get("env", {}))

    if _run_subprocess(USER_SETTINGS["main_path"], env):
        with open(done_file, "w") as f:
            f.write(f"Completed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        print(f"  Successfully completed: {exp_name}")
        return True
    else:
        print(f"  Failed: {exp_name}", file=sys.stderr)
        return False

def main():
    """Main entry point for the sensitivity analysis sweep."""
    print("="*70)
    print("Sensitivity Analysis Runner")
    print("="*70)

    if not USER_SETTINGS["baseline_dir"] or not os.path.exists(USER_SETTINGS["baseline_dir"]):
        print(f"Error: Baseline directory not found: {USER_SETTINGS['baseline_dir']}", file=sys.stderr)
        print("Please set the BASELINE_DIR environment variable.", file=sys.stderr)
        sys.exit(1)

    if not USER_SETTINGS["sweep_root"]:
        USER_SETTINGS["sweep_root"] = os.path.join(
            os.path.dirname(USER_SETTINGS["baseline_dir"]),
            "sensitivity_analysis"
        )
    os.makedirs(USER_SETTINGS["sweep_root"], exist_ok=True)

    print(f"\nConfiguration:")
    print(f"  Seed: {USER_SETTINGS['seed']}")
    print(f"  Baseline Directory: {USER_SETTINGS['baseline_dir']}")
    print(f"  Sweep Root: {USER_SETTINGS['sweep_root']}")
    print(f"  Max Batches: {USER_SETTINGS['max_batches']}")
    print(f"  Number of Sensitivity Configs: {len(SENSITIVITY_CONFIGS)}")

    results = {}
    for config in SENSITIVITY_CONFIGS:
        success = run_sensitivity_config(config)
        results[config["name"]] = success

    summary_path = os.path.join(USER_SETTINGS["sweep_root"], "sensitivity_summary.json")
    summary_data = {
        "run_at": time.strftime('%Y-%m-%d %H:%M:%S'),
        "seed": USER_SETTINGS["seed"],
        "baseline_dir": USER_SETTINGS["baseline_dir"],
        "sweep_root": USER_SETTINGS["sweep_root"],
        "configs_run": [c["name"] for c in SENSITIVITY_CONFIGS],
        "results": results,
    }
    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2)

    print("\n" + "="*70)
    print("Sensitivity analysis sweep finished.")
    print(f"Summary written to: {summary_path}")
    print("="*70)

if __name__ == "__main__":
    main()
