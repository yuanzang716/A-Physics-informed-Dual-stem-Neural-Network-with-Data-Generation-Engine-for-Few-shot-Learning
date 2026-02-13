#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Orchestrates an ablation study by systematically running different training
configurations based on a baseline model.

This script is designed to be called with environment variables to specify
the necessary paths and settings, allowing for flexible integration into
larger experiment workflows.
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

# Dynamically define the project root. This script is in experiments/.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# =============================================================================
#  Ablation Configurations
# =============================================================================

# Each dictionary defines a specific ablation experiment by overriding
# environment variables of the main training script.
ABLATION_CONFIGS = [
    # --- Baseline and Strategy Ablations (E0x) ---
    {
        "name": "E00_baseline",
        "desc": "Baseline: The full, recommended strategy.",
        "env": {
            "EARLY_PAIR_ONLY_EPOCHS": "10",
            "LOSS_GRADUAL_INTRO_ENABLE": "1",
            "LOSS_GRADUAL_INTRO_EPOCHS": "10",
            "LOSS_CONDITIONAL_ENABLE": "1",
            "LOSS_CONDITIONAL_PAIR_THRESHOLD": "0.025",
            "W_PHYS_REDUCED": "0.5",
            "LOSS_ADAPTIVE_PAIR_BOOST": "1",
        }
    },
    {
        "name": "E01_no_early_pair",
        "desc": "Removes the initial pair-only training phase.",
        "env": {"EARLY_PAIR_ONLY_EPOCHS": "0"}
    },
    {
        "name": "E02_no_gradual_intro",
        "desc": "Disables gradual introduction of losses.",
        "env": {"LOSS_GRADUAL_INTRO_ENABLE": "0"}
    },
    {
        "name": "E03_no_conditional",
        "desc": "Disables conditional loss enablement.",
        "env": {"LOSS_CONDITIONAL_ENABLE": "0"}
    },
    {
        "name": "E04_no_adaptive_boost",
        "desc": "Disables adaptive pair loss boosting.",
        "env": {"LOSS_ADAPTIVE_PAIR_BOOST": "0"}
    },
    {
        "name": "E05_full_phys_weight",
        "desc": "Uses the full physics loss weight (W_PHYS_REDUCED=1.0).",
        "env": {"W_PHYS_REDUCED": "1.0"}
    },
    {
        "name": "E06_no_physics",
        "desc": "Disables the physics loss entirely (W_PHYS_REDUCED=0.0).",
        "env": {"W_PHYS_REDUCED": "0.0"}
    },
    # --- Objectives / physics (E07-E09) ---
    {
        "name": "E07_phys_pair_only",
        "desc": "Only physics and pair losses enabled",
        "env": {
            "W_WEAK": "0.0",
            "W_VICREG": "0.0",
            "W_TTA": "0.0",
            "W_GROUP": "0.0"
        }
    },
    {
        "name": "E08_phys_only",
        "desc": "Only physics loss enabled from the beginning",
        "env": {
            "EARLY_PAIR_ONLY_EPOCHS": "0",
            "W_PAIR": "0.0",
            "W_WEAK": "0.0",
            "W_VICREG": "0.0",
            "W_TTA": "0.0",
            "W_GROUP": "0.0"
        }
    },
    {
        "name": "E09_pure_pinn_style",
        "desc": "Pure PINN Baseline: Single-stem + Pair + Phys only",
        "env": {
            "FORCE_SINGLE_BRANCH": "1",
            "W_PAIR": "1.5",
            "W_PHYS": "3.0",
            "W_WEAK": "0.0",
            "W_VICREG": "0.0",
            "W_TTA": "0.0",
            "W_GROUP": "0.0"
        }
    },
    # --- Structural and Input Ablations (E10-E12) ---
    {
        "name": "E10_no_fft",
        "desc": "Disables FFT features as input.",
        "env": {"USE_FFT_INPUT": "0"}
    },
    {
        "name": "E11_use_diff_input",
        "desc": "Enables the explicit difference map as an input channel.",
        "env": {"USE_DIFF_INPUT": "1"}
    },
    {
        "name": "E12_single_branch",
        "desc": "Forces the sim branch to use the real stem (single-branch architecture).",
        "env": {"FORCE_SINGLE_BRANCH": "1"}
    },
    # --- Data Quantity Ablations (E13-E14) ---
    {
        "name": "E13_sim_samples_300",
        "desc": "Ablates data quantity by using only 300 sim samples per real image.",
        "env": {"SIM_SAMPLE_COUNT": "300"}
    },
    {
        "name": "E14_sim_samples_600",
        "desc": "Ablates data quantity by using 600 sim samples per real image.",
        "env": {"SIM_SAMPLE_COUNT": "600"}
    }
]

# =============================================================================
#  User Settings
# =============================================================================

USER_SETTINGS = {
    "seed": int(os.environ.get("SEED", "42")),
    "baseline_dir": os.environ.get("BASELINE_DIR", ""),
    "sweep_root": os.environ.get("SWEEP_ROOT", ""),
    "main_path": os.path.join(PROJECT_ROOT, "core", "main_75.py"),
    "num_epochs": int(os.environ.get("NUM_EPOCHS", "60")),
    "skip_if_exists": os.environ.get("SKIP_IF_EXISTS", "true").lower() in ["true", "1"],
}

# =============================================================================
#  Core Logic
# =============================================================================

def _run_subprocess(script_path: str, env: Dict[str, str]) -> bool:
    """Helper to run a script in a subprocess, ensuring it runs from its own directory."""
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

def run_ablation_config(config: Dict[str, Any], baseline_strategy: Dict[str, Any]):
    """Runs a single ablation experiment configuration."""
    exp_name = config["name"]
    exp_dir = os.path.join(USER_SETTINGS["sweep_root"], exp_name)
    print(f"\n--- Processing Ablation: {exp_name} ({config['desc']})")
    print(f"  Output Directory: {exp_dir}")

    # For the baseline run, simply copy results to avoid re-training.
    if exp_name == "E00_baseline":
        baseline_done_file = os.path.join(USER_SETTINGS["baseline_dir"], "DONE.txt")
        if os.path.exists(baseline_done_file):
            print("  INFO: Replicating baseline results without re-training.")
            os.makedirs(exp_dir, exist_ok=True)
            import shutil
            for fname in ["best_model.pth", "best_ema_model.pth", "metrics.csv", "DONE.txt"]:
                src = os.path.join(USER_SETTINGS["baseline_dir"], fname)
                if os.path.exists(src):
                    shutil.copy2(src, os.path.join(exp_dir, fname))
            return
        else:
            print("  WARNING: Baseline results not found. Training from scratch.")

    # Standard check to skip completed experiments.
    done_file = os.path.join(exp_dir, "DONE.txt")
    if USER_SETTINGS["skip_if_exists"] and os.path.exists(done_file):
        print(f"  Skipping completed experiment: {exp_name}")
        return

    os.makedirs(exp_dir, exist_ok=True)

    # Prepare environment: start with baseline, then apply ablation-specific overrides.
    env = os.environ.copy()
    env.update(baseline_strategy)
    env.update(config.get("env", {}))
    env.update({
        "SEED": str(USER_SETTINGS["seed"]),
        "BASE_MODEL_SAVE_DIR": exp_dir,
        "NUM_EPOCHS": str(USER_SETTINGS["num_epochs"]),
    })

    # Persist the *effective* environment to the experiment folder for auditability.
    # This avoids confusion when main_75.py snapshots are incomplete or selectively filtered.
    # NOTE: This is written even if the experiment later fails.
    effective_env_path = os.path.join(exp_dir, "ablation_env_effective.json")
    try:
        env_snapshot = {
            "created_at": time.strftime('%Y-%m-%d %H:%M:%S'),
            "exp_name": exp_name,
            "desc": config.get("desc", ""),
            "env": {k: str(v) for k, v in sorted(env.items())},
        }
        with open(effective_env_path, "w") as f:
            json.dump(env_snapshot, f, indent=2)
    except Exception as e:
        print(f"  Warning: failed to write {os.path.basename(effective_env_path)}: {e}", file=sys.stderr)

    effective_cfg_path = os.path.join(exp_dir, "ablation_config_effective.json")
    try:
        cfg_snapshot = {
            "created_at": time.strftime('%Y-%m-%d %H:%M:%S'),
            "exp_name": exp_name,
            "desc": config.get("desc", ""),
            "baseline_strategy": baseline_strategy,
            "overrides": config.get("env", {}),
            "resolved": {
                "SEED": str(USER_SETTINGS["seed"]),
                "BASE_MODEL_SAVE_DIR": exp_dir,
                "NUM_EPOCHS": str(USER_SETTINGS["num_epochs"]),
            },
        }
        with open(effective_cfg_path, "w") as f:
            json.dump(cfg_snapshot, f, indent=2)
    except Exception as e:
        print(f"  Warning: failed to write {os.path.basename(effective_cfg_path)}: {e}", file=sys.stderr)

    if _run_subprocess(USER_SETTINGS["main_path"], env):
        with open(done_file, "w") as f:
            f.write(f"Completed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        print(f"  Successfully completed: {exp_name}")
    else:
        print(f"  Failed: {exp_name}", file=sys.stderr)

def main():
    """Main entry point for the ablation sweep."""
    print("="*70)
    print("Ablation Study Runner")
    print("="*70)

    # Validate required environment variables.
    if not USER_SETTINGS["baseline_dir"] or not USER_SETTINGS["sweep_root"]:
        print("Error: BASELINE_DIR and SWEEP_ROOT environment variables are required.", file=sys.stderr)
        sys.exit(1)

    print(f"Configuration:")
    print(f"  Seed: {USER_SETTINGS['seed']}")
    print(f"  Baseline Directory: {USER_SETTINGS['baseline_dir']}")
    print(f"  Ablation Sweep Root: {USER_SETTINGS['sweep_root']}")
    print(f"  Number of Epochs: {USER_SETTINGS['num_epochs']}")
    print(f"  Number of Ablation Configs: {len(ABLATION_CONFIGS)}")

    # The baseline strategy is E00_baseline's environment.
    baseline_strategy = next((c["env"] for c in ABLATION_CONFIGS if c["name"] == "E00_baseline"), {})
    if not baseline_strategy:
        print("Error: E00_baseline configuration not found.", file=sys.stderr)
        sys.exit(1)

    # Run all ablation configurations.
    for config in ABLATION_CONFIGS:
        run_ablation_config(config, baseline_strategy)

    # Save a summary of the run.
    summary_path = os.path.join(USER_SETTINGS["sweep_root"], "ablation_summary.json")
    summary_data = {
        "run_at": time.strftime('%Y-%m-%d %H:%M:%S'),
        "seed": USER_SETTINGS["seed"],
        "baseline_dir": USER_SETTINGS["baseline_dir"],
        "configs_run": [c["name"] for c in ABLATION_CONFIGS],
    }
    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2)

    print("\n" + "="*70)
    print("Ablation sweep finished.")
    print(f"Summary written to: {summary_path}")
    print("="*70)

if __name__ == "__main__":
    main()
