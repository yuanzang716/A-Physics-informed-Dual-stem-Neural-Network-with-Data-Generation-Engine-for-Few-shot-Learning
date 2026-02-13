#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ablation.py - Ablation study script

Run ablation studies on top of the baseline by systematically removing or modifying strategy components.

Usage:
1) Configure via environment variables:
   - SEED: random seed
   - BASELINE_DIR: baseline results directory
   - SWEEP_ROOT: ablation sweep output directory
2) Or run this file directly (uses default settings)
"""

import os
import sys
import subprocess
import json
import time
from typing import Dict, Any


# =============================================================================
#  Project-level Path Configuration
# =============================================================================

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


# =============================================================================
#  Ablation Experiment Configuration
# =============================================================================
ABLATION_CONFIGS = [
    {
        "name": "E00_baseline",
        "desc": "Baseline (full strategy)",
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
        "desc": "Remove Early Pair-Only",
        "env": {
            "EARLY_PAIR_ONLY_EPOCHS": "0",
            "LOSS_GRADUAL_INTRO_ENABLE": "1",
            "LOSS_GRADUAL_INTRO_EPOCHS": "10",
            "LOSS_CONDITIONAL_ENABLE": "1",
            "W_PHYS_REDUCED": "0.5",
            "LOSS_ADAPTIVE_PAIR_BOOST": "1",
        }
    },
    {
        "name": "E02_no_gradual_intro",
        "desc": "Remove Gradual Introduction",
        "env": {
            "EARLY_PAIR_ONLY_EPOCHS": "10",
            "LOSS_GRADUAL_INTRO_ENABLE": "0",
            "LOSS_CONDITIONAL_ENABLE": "1",
            "W_PHYS_REDUCED": "0.5",
            "LOSS_ADAPTIVE_PAIR_BOOST": "1",
        }
    },
    {
        "name": "E03_no_conditional",
        "desc": "Remove Conditional Enable",
        "env": {
            "EARLY_PAIR_ONLY_EPOCHS": "10",
            "LOSS_GRADUAL_INTRO_ENABLE": "1",
            "LOSS_GRADUAL_INTRO_EPOCHS": "10",
            "LOSS_CONDITIONAL_ENABLE": "0",
            "W_PHYS_REDUCED": "0.5",
            "LOSS_ADAPTIVE_PAIR_BOOST": "1",
        }
    },
    {
        "name": "E04_no_adaptive_boost",
        "desc": "Remove Adaptive Pair Boost",
        "env": {
            "EARLY_PAIR_ONLY_EPOCHS": "10",
            "LOSS_GRADUAL_INTRO_ENABLE": "1",
            "LOSS_GRADUAL_INTRO_EPOCHS": "10",
            "LOSS_CONDITIONAL_ENABLE": "1",
            "W_PHYS_REDUCED": "0.5",
            "LOSS_ADAPTIVE_PAIR_BOOST": "0",
        }
    },
    {
        "name": "E05_full_phys_weight",
        "desc": "Use full physics weight (no reduction)",
        "env": {
            "EARLY_PAIR_ONLY_EPOCHS": "10",
            "LOSS_GRADUAL_INTRO_ENABLE": "1",
            "LOSS_GRADUAL_INTRO_EPOCHS": "10",
            "LOSS_CONDITIONAL_ENABLE": "1",
            "W_PHYS_REDUCED": "1.0",
            "LOSS_ADAPTIVE_PAIR_BOOST": "1",
        }
    },
    {
        "name": "E06_no_physics",
        "desc": "Remove physics loss entirely (verify the necessity of physics constraints)",
        "env": {
            "EARLY_PAIR_ONLY_EPOCHS": "10",
            "LOSS_GRADUAL_INTRO_ENABLE": "1",
            "LOSS_GRADUAL_INTRO_EPOCHS": "10",
            "LOSS_CONDITIONAL_ENABLE": "1",
            "W_PHYS_REDUCED": "0.0",  # Set the physics weight to 0, equivalent to disabling it
            "LOSS_ADAPTIVE_PAIR_BOOST": "1",
            # Note: if the code supports it, you can also set ENABLE_PHYS=0, but W_PHYS_REDUCED=0.0 is simpler
        }
    },
    {
        "name": "E07_no_fft",
        "desc": "Disable FFT input (verify the contribution of frequency-domain input)",
        "env": {
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
            "USE_FFT_INPUT": "0",
        }
    },
    {
        "name": "E08_use_diff_input",
        "desc": "Enable Diff input channel (verify the contribution of explicit residual input)",
        "env": {
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
            "USE_DIFF_INPUT": "1",
        }
    },
    {
        "name": "E09_single_branch",
        "desc": "Single-branch: route sim branch through stem_real (pad to 4ch) (verify the necessity of dual-stem)",
        "env": {
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
            "FORCE_SINGLE_BRANCH": "1",
        }
    },
    {
        "name": "E10_sim_samples_300",
        "desc": "Data-size ablation: train with only 300 sim samples per real",
        "env": {
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
            "SIM_SAMPLE_COUNT": "300",
        }
    },
    {
        "name": "E11_sim_samples_600",
        "desc": "Data-size ablation: train with only 600 sim samples per real",
        "env": {
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
            "SIM_SAMPLE_COUNT": "600",
        }
    },
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
]


# =============================================================================
#  User Settings (can be overridden by environment variables)
# =============================================================================
USER_SETTINGS = {
    "SEED": int(os.environ.get("SEED", "42")),
    "BASELINE_DIR": os.environ.get("BASELINE_DIR", ""),
    "SWEEP_ROOT": os.environ.get("SWEEP_ROOT", ""),
    "MAIN_PATH": os.path.join(PROJECT_ROOT, "core", "main_120.py"),
    "NUM_EPOCHS": int(os.environ.get("NUM_EPOCHS", "60")),
    "SKIP_IF_EXISTS": True,
}


def run_ablation_config(config: Dict[str, Any], exp_dir: str, baseline_dir: str) -> bool:
    """Run single ablation experiment configuration"""
    print(f"\n  [ABLATION] {config['name']}: {config['desc']}")
    print(f"     Output: {exp_dir}")
    
    # Special handling: E00_baseline directly uses existing baseline results to avoid duplicate training
    if config["name"] == "E00_baseline":
        baseline_done = os.path.join(baseline_dir, "DONE.txt")
        if os.path.exists(baseline_dir) and os.path.exists(baseline_done):
            print(f"     [INFO] Using existing baseline results (no duplicate training)")
            os.makedirs(exp_dir, exist_ok=True)
            
            # Create symlink or copy key files to E00_baseline directory
            # For compatibility, we copy key files instead of creating symlinks
            import shutil
            key_files = ["best_model.pth", "metrics.csv", "config_snapshot.json", "DONE.txt"]
            for fname in key_files:
                src = os.path.join(baseline_dir, fname)
                dst = os.path.join(exp_dir, fname)
                if os.path.exists(src):
                    shutil.copy2(src, dst)
            
            # Create reference marker
            with open(os.path.join(exp_dir, "BASELINE_REFERENCE.txt"), "w") as f:
                f.write(f"This result references: {baseline_dir}\n")
                f.write(f"Created at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            print(f"     [OK] Completed (referenced baseline result)")
            return True
        else:
            print(f"     [WARN] Baseline does not exist, will run training")
            # If baseline doesn't exist, continue with training (this shouldn't happen)
    
    # Check if already completed
    done_file = os.path.join(exp_dir, "DONE.txt")
    if USER_SETTINGS["SKIP_IF_EXISTS"] and os.path.exists(done_file):
        print(f"     [SKIP] Already exists, skipping")
        return True
    
    os.makedirs(exp_dir, exist_ok=True)
    
    # Prepare environment variables
    env = os.environ.copy()
    env.update({
        "SEED": str(USER_SETTINGS["SEED"]),
        "BASE_MODEL_SAVE_DIR": exp_dir,
        "NUM_EPOCHS": str(USER_SETTINGS["NUM_EPOCHS"]),
    })
    env.update(config.get("env", {}))

    # Persist the *effective* environment to the experiment folder for auditability.
    # This avoids confusion when main_120.py snapshots are incomplete or selectively filtered.
    # NOTE: This is written even if the experiment later fails.
    effective_env_path = os.path.join(exp_dir, "ablation_env_effective.json")
    try:
        env_snapshot = {
            "created_at": time.strftime('%Y-%m-%d %H:%M:%S'),
            "exp_name": config["name"],
            "desc": config.get("desc", ""),
            "env": {k: str(v) for k, v in sorted(env.items())},
        }
        with open(effective_env_path, "w") as f:
            json.dump(env_snapshot, f, indent=2)
    except Exception as e:
        print(f"     [WARN] Failed to write {os.path.basename(effective_env_path)}: {e}")

    effective_cfg_path = os.path.join(exp_dir, "ablation_config_effective.json")
    try:
        cfg_snapshot = {
            "created_at": time.strftime('%Y-%m-%d %H:%M:%S'),
            "exp_name": config["name"],
            "desc": config.get("desc", ""),
            "overrides": config.get("env", {}),
            "resolved": {
                "SEED": str(USER_SETTINGS["SEED"]),
                "BASE_MODEL_SAVE_DIR": exp_dir,
                "NUM_EPOCHS": str(USER_SETTINGS["NUM_EPOCHS"]),
            },
        }
        with open(effective_cfg_path, "w") as f:
            json.dump(cfg_snapshot, f, indent=2)
    except Exception as e:
        print(f"     [WARN] Failed to write {os.path.basename(effective_cfg_path)}: {e}")

    try:
        subprocess.run(
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
    print("Ablation Experiments")
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
        # Default: create ablation_sweep in same directory as baseline
        sweep_root = os.path.join(os.path.dirname(baseline_dir), "ablation_sweep")
    os.makedirs(sweep_root, exist_ok=True)
    
    print(f"\nConfiguration:")
    print(f"   Seed: {USER_SETTINGS['SEED']}")
    print(f"   Baseline: {baseline_dir}")
    print(f"   Output: {sweep_root}")
    print(f"   Epochs: {USER_SETTINGS['NUM_EPOCHS']}")
    print(f"\nNumber of ablation configurations: {len(ABLATION_CONFIGS)}")
    
    # Run all ablation configurations
    results = {}
    for config in ABLATION_CONFIGS:
        exp_dir = os.path.join(sweep_root, config["name"])
        success = run_ablation_config(config, exp_dir, baseline_dir)
        results[config["name"]] = success
    
    # Save summary results
    summary_file = os.path.join(sweep_root, "ablation_summary.json")
    with open(summary_file, "w") as f:
        json.dump({
            "seed": USER_SETTINGS["SEED"],
            "baseline_dir": baseline_dir,
            "sweep_root": sweep_root,
            "configs": ABLATION_CONFIGS,
            "results": results,
            "completed_at": time.strftime('%Y-%m-%d %H:%M:%S'),
        }, f, indent=2)
    
    print("\n" + "="*70)
    print("[OK] Ablation experiments completed!")
    print("="*70)
    print(f"Summary results: {summary_file}")


if __name__ == "__main__":
    main()
