#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_all_experiments.py (successful-strategy version)

Run the complete experiment pipeline with one command:
1. Probe stage: run short trainings under multiple random seeds and compute success/failure rates
2. Baseline: run the successful strategy (err=0.60% at epoch 38, seed 212)
3. Ablation: run ablation studies on top of the baseline
4. Sensitivity: run sensitivity analysis on top of the baseline

Successful strategy configuration (recommended_base):
- EARLY_PAIR_ONLY_EPOCHS=10
- LOSS_GRADUAL_INTRO_ENABLE=1, LOSS_GRADUAL_INTRO_EPOCHS=10
- LOSS_CONDITIONAL_ENABLE=1, LOSS_CONDITIONAL_PAIR_THRESHOLD=0.025
- W_PHYS_REDUCED=0.5
- LOSS_ADAPTIVE_PAIR_BOOST=1

Usage:
1) Modify the CONFIG below
2) Run this file directly
"""

import os
import sys
import subprocess
import time
import json
import random
from typing import List, Dict, Any


# =============================================================================
#  Project-level Path Configuration
# =============================================================================

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


# =============================================================================
#  Experiment Configuration
# =============================================================================

CONFIG = {
    # --- Path Configurations ---
    "main_path": os.path.join(PROJECT_ROOT, "core", "main_120.py"),
    "ablation_path": os.path.join(PROJECT_ROOT, "experiments", "ablation.py"),
    "sensitivity_path": os.path.join(PROJECT_ROOT, "experiments", "sensitivity.py"),
    "stats_path": os.path.join(PROJECT_ROOT, "analysis", "analyze_probe_stats.py"),
    "output_root": os.path.join(PROJECT_ROOT, "results"),
    
    # --- Random Seed Configuration ---
    "SEEDS": [212],  # Legacy behavior: used when USE_SEED_SAMPLING=False
    
    # --- Paper Protocol: Seed Sampling (Recommended) ---
    "USE_SEED_SAMPLING": True,
    "SEED_SAMPLING_FRAME": [1, 300],        # Sampling frame: {1..300}
    "SEED_SAMPLING_RNG_SEED": 20260108,     # Sampling RNG seed (not training seed)
    "FULL_RUN_N_SEEDS": 1,                  # Number of seeds for FULL stage (optimized to 1)
    
    # --- PROBE Stage: Multi-seed Statistics ---
    "PROBE_RUN_ENABLE": True,               # Enable probe stage
    "PROBE_N_SEEDS": 10,                    # Number of probe seeds (10 is sufficient for statistics)
                                           # Time: 10 seeds × 20 epochs × 3 min = 600 min = 10 hours
                                           # Can reduce to 8 if time is tight (about 8 hours), but statistical significance will decrease
    "PROBE_EPOCHS": 20,                     # Probe training epochs (must be at least 20, as strategy needs 20 epochs to fully take effect)
                                           # Strategy config: EARLY_PAIR_ONLY_EPOCHS=10 (first 10 epochs only pair loss)
                                           #                  + LOSS_GRADUAL_INTRO_EPOCHS=10 (next 10 epochs gradual introduction)
                                           #                  = at least 20 epochs needed to see full effect
    "PROBE_SUCCESS_THRESHOLD": 0.01,        # Success threshold: err < 1% considered success
    "PROBE_PARALLEL": False,                # Whether to run probe in parallel (requires multi-GPU or sequential multiple processes)
    "PROBE_MAX_PARALLEL": 4,                # Maximum parallel count (if PROBE_PARALLEL=True)
    
    # --- FULL Stage: Complete Training ---
    "STAGE2_RUN_FULL": True,                # Whether to run full training after probe
    "FULL_RUN_N_SEEDS": 2,                  # Number of seeds for FULL stage (reduced from 2 to 1 to save time)
                                           # Time: 1 seed × 60 epochs × 3 min = 180 min = 3 hours
                                           # Can increase to 2 if multiple seeds need verification (6 hours)
    "FULL_EPOCHS": 60,                      # Full training epochs (60 epochs × 3 min = 180 min/seed ≈ 3 hours/seed)
    
    # --- Convenience Override: Directly Specify Seeds to Run ---
    "FORCE_FULL_SEEDS": [],                 # Force specify FULL seeds (highest priority)
    "FORCE_PROBE_SEEDS": [],                # Force specify PROBE seeds (highest priority)
    
    "BASELINE_STRATEGY": {
        "name": "baseline",
        "exp_subdir": "baseline",
        "desc": "Successful strategy baseline: err=0.60% at epoch 38 (recommended_base config)",
        "env": {
            "EARLY_PAIR_ONLY_EPOCHS": "10",  # First 10 epochs only train pair loss
            "LOSS_GRADUAL_INTRO_ENABLE": "1",  # Enable gradual introduction
            "LOSS_GRADUAL_INTRO_EPOCHS": "10",  # 10 epochs cosine ramp
            "LOSS_GRADUAL_INTRO_MODE": "cosine",
            "LOSS_GRADUAL_INTRO_TARGETS": "phys,weak,vic",
            "LOSS_CONDITIONAL_ENABLE": "1",  # Conditional enable
            "LOSS_CONDITIONAL_PAIR_THRESHOLD": "0.025",  # Enable other losses when pair < 0.025
            "LOSS_CONDITIONAL_STABLE_EPOCHS": "2",  # Need stable 2 epochs
            "W_PHYS_REDUCED": "0.5",  # Reduce phys weight to 0.5x
            "LOSS_ADAPTIVE_PAIR_BOOST": "1",  # Adaptive pair loss boost
            "LOSS_ADAPTIVE_PAIR_THRESHOLD": "0.030",
            "LOSS_ADAPTIVE_PAIR_MAX_BOOST": "2.5",
        }
    },
    
    # --- Experiment Configuration ---
    "ABLATION_SWEEP_NAME": "ablation_sweep",
    "SENSITIVITY_SWEEP_NAME": "sensitivity_analysis",  # Sensitivity output directory name
    
    # --- Runtime Control ---
    "SKIP_COMPLETED": True,
    "CONTINUE_ON_ERROR": True,
    "CUDA_VISIBLE_DEVICES": "0",
    
    # --- Result Cleanup ---
    "CLEAR_RESULTS_ON_START": False,  # Clear results directory before running (to avoid confusion)
    # Set to False: Keep completed results, only run incomplete experiments (avoid re-running time-consuming tasks)
    # Set to True: Clear all results before each run, re-run all experiments
}


def _sample_seeds(frame: List[int], n: int, rng_seed: int) -> List[int]:
    """Randomly sample n seeds from frame"""
    rng = random.Random(rng_seed)
    available = list(range(frame[0], frame[1] + 1))
    return sorted(rng.sample(available, min(n, len(available))))


def get_seed_dir(seed: int) -> str:
    """Get output directory for seed"""
    return os.path.join(CONFIG["output_root"], f"seed_{seed}")


def run_probe(seed: int, epochs: int) -> bool:
    """Run probe experiment (short-term training for statistics)"""
    strategy = CONFIG["BASELINE_STRATEGY"]
    seed_dir = get_seed_dir(seed)
    exp_dir = os.path.join(seed_dir, "probe")
    os.makedirs(exp_dir, exist_ok=True)
    
    # Check if already completed
    done_file = os.path.join(exp_dir, "DONE.txt")
    if CONFIG["SKIP_COMPLETED"] and os.path.exists(done_file):
        print(f"  [SKIP] Probe already exists, skipping (seed={seed})")
        return True
    
    env = os.environ.copy()
    env.update({
        "SEED": str(seed),
        "BASE_MODEL_SAVE_DIR": exp_dir,
        "NUM_EPOCHS": str(epochs),
        "ALLOW_TRAIN": "1",
    })
    env.update(strategy.get("env", {}))
    
    print(f"  [PROBE] (seed={seed}, epochs={epochs})")
    print(f"     Strategy: {strategy['name']}")
    print(f"     Output: {exp_dir}")
    
    try:
        result = subprocess.run(
            [sys.executable, CONFIG["main_path"]],
            env=env,
            cwd=os.path.dirname(CONFIG["main_path"]),
            check=True,
            capture_output=False,
        )
        
        # Mark as completed
        with open(done_file, "w") as f:
            f.write(f"Completed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        print(f"  [OK] Probe completed (seed={seed})")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  [ERROR] Probe failed (seed={seed}): {e}")
        if not CONFIG["CONTINUE_ON_ERROR"]:
            raise
        return False


def export_predictions_for_seed(seed: int) -> bool:
    """After baseline training, export per-image predictions for train/val using best and best_ema weights."""
    strategy = CONFIG["BASELINE_STRATEGY"]
    seed_dir = get_seed_dir(seed)
    exp_dir = os.path.join(seed_dir, strategy["exp_subdir"])

    # Only export if the baseline exists
    if not os.path.exists(exp_dir):
        print(f"  [WARN] Baseline directory not found, skipping export (seed={seed})")
        return False

    out_dir = os.path.join(CONFIG["output_root"], "pred_exports")
    os.makedirs(out_dir, exist_ok=True)

    print(f"  [EXPORT] Exporting predictions (seed={seed}) -> {out_dir}")

    for ckpt in ["best", "ema"]:
        env = os.environ.copy()
        env.update({
            "SEED": str(seed),
            "BASE_MODEL_SAVE_DIR": exp_dir,
            "MODE": "export_preds",
            "EXPORT_CKPT": ckpt,
            "EXPORT_OUT_DIR": out_dir,
        })
        try:
            subprocess.run(
                [sys.executable, CONFIG["main_path"]],
                env=env,
                cwd=os.path.dirname(CONFIG["main_path"]),
                check=True,
                capture_output=False,
            )
        except subprocess.CalledProcessError as e:
            print(f"  [ERROR] Export failed (seed={seed}, ckpt={ckpt}): {e}")
            if not CONFIG["CONTINUE_ON_ERROR"]:
                raise
            return False

    print(f"  [OK] Export completed (seed={seed})")
    return True


def run_baseline(seed: int, epochs: int) -> bool:
    """Run baseline experiment (complete training)"""
    strategy = CONFIG["BASELINE_STRATEGY"]
    seed_dir = get_seed_dir(seed)
    exp_dir = os.path.join(seed_dir, "baseline")
    os.makedirs(exp_dir, exist_ok=True)
    
    # Check if already completed
    done_file = os.path.join(exp_dir, "DONE.txt")
    if CONFIG["SKIP_COMPLETED"] and os.path.exists(done_file):
        print(f"  [SKIP] Baseline already exists, skipping (seed={seed})")
        return True
    
    env = os.environ.copy()
    env.update({
        "SEED": str(seed),
        "BASE_MODEL_SAVE_DIR": exp_dir,
        "NUM_EPOCHS": str(epochs),
        "ALLOW_TRAIN": "1",
    })
    env.update(strategy.get("env", {}))
    
    print(f"  [BASELINE] (seed={seed}, epochs={epochs})")
    print(f"     Strategy: {strategy['name']}")
    print(f"     Output: {exp_dir}")
    
    try:
        result = subprocess.run(
            [sys.executable, CONFIG["main_path"]],
            env=env,
            cwd=os.path.dirname(CONFIG["main_path"]),
            check=True,
            capture_output=False,
        )
        
        # Mark as completed
        with open(done_file, "w") as f:
            f.write(f"Completed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        print(f"  [OK] Baseline completed (seed={seed})")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  [ERROR] Baseline failed (seed={seed}): {e}")
        if not CONFIG["CONTINUE_ON_ERROR"]:
            raise
        return False


def run_ablation(seed: int) -> bool:
    """Run ablation experiment"""
    seed_dir = get_seed_dir(seed)
    baseline_dir = os.path.join(seed_dir, "baseline")
    sweep_dir = os.path.join(seed_dir, CONFIG["ABLATION_SWEEP_NAME"])
    
    # Check if baseline exists
    if not os.path.exists(baseline_dir):
        print(f"  [WARN] Baseline does not exist, skipping Ablation (seed={seed})")
        return False
    
    # Check if already completed
    done_file = os.path.join(sweep_dir, "DONE.txt")
    if CONFIG["SKIP_COMPLETED"] and os.path.exists(done_file):
        print(f"  [SKIP] Ablation already exists, skipping (seed={seed})")
        return True
    
    env = os.environ.copy()
    env.update({
        "SEED": str(seed),
        "BASELINE_DIR": baseline_dir,
        "SWEEP_ROOT": sweep_dir,
    })
    
    print(f"  [ABLATION] (seed={seed})")
    print(f"     Baseline: {baseline_dir}")
    print(f"     Output: {sweep_dir}")
    
    try:
        result = subprocess.run(
            [sys.executable, CONFIG["ablation_path"]],
            env=env,
            cwd=os.path.dirname(CONFIG["ablation_path"]),
            check=True,
            capture_output=False,
        )
        
        # Mark as completed
        with open(done_file, "w") as f:
            f.write(f"Completed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        print(f"  [OK] Ablation completed (seed={seed})")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  [ERROR] Ablation failed (seed={seed}): {e}")
        if not CONFIG["CONTINUE_ON_ERROR"]:
            raise
        return False


def run_sensitivity(seed: int) -> bool:
    """Run sensitivity experiment"""
    strategy = CONFIG["BASELINE_STRATEGY"]
    seed_dir = get_seed_dir(seed)
    baseline_dir = os.path.join(seed_dir, "baseline")
    sweep_dir = os.path.join(seed_dir, CONFIG["SENSITIVITY_SWEEP_NAME"])
    
    # Check if baseline exists
    if not os.path.exists(baseline_dir):
        print(f"  [WARN] Baseline does not exist, skipping Sensitivity (seed={seed})")
        return False
    
    # Check if already completed
    done_file = os.path.join(sweep_dir, "sensitivity_summary.json")
    if CONFIG["SKIP_COMPLETED"] and os.path.exists(done_file):
        print(f"  [SKIP] Sensitivity already exists, skipping (seed={seed})")
        return True
    
    env = os.environ.copy()
    env.update({
        "SEED": str(seed),
        "MAIN_PATH": CONFIG["main_path"],
        "SWEEP_ROOT": sweep_dir,
        "BASELINE_DIR": baseline_dir,
        "NUM_EPOCHS": str(CONFIG["FULL_EPOCHS"]),  # Use same number of epochs as baseline
    })
    # Pass baseline strategy config to ensure sensitivity uses the same strategy
    env.update(strategy.get("env", {}))
    
    print(f"  [SENSITIVITY] (seed={seed})")
    print(f"     Strategy: {strategy['name']}")
    print(f"     Baseline: {baseline_dir}")
    print(f"     Output: {sweep_dir}")
    
    try:
        result = subprocess.run(
            [sys.executable, CONFIG["sensitivity_path"]],
            env=env,
            cwd=os.path.dirname(CONFIG["sensitivity_path"]),
            check=True,
            capture_output=False,
        )
        
        # Mark as completed (sensitivity.py will create summary file, no need to create here)
        
        print(f"  [OK] Sensitivity completed (seed={seed})")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  [ERROR] Sensitivity failed (seed={seed}): {e}")
        if not CONFIG["CONTINUE_ON_ERROR"]:
            raise
        return False


def analyze_probe_stats() -> bool:
    """Analyze probe stage statistics"""
    stats_dir = os.path.join(CONFIG["output_root"], "probe_statistics")
    os.makedirs(stats_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("Analyzing Probe Statistics")
    print("="*60)
    
    try:
        result = subprocess.run(
            [sys.executable, CONFIG["stats_path"]],
            cwd=os.path.dirname(CONFIG["stats_path"]),
            check=True,
            capture_output=False,
        )
        print("[OK] Statistical analysis completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Statistical analysis failed: {e}")
        return False


def main():
    """Main function"""
    print("="*70)
    print("Starting Complete Experiment Pipeline")
    print("="*70)
    
    # Clear results directory (if configured)
    if CONFIG["CLEAR_RESULTS_ON_START"]:
        if os.path.exists(CONFIG["output_root"]):
            print(f"\n[WARN] Clearing results directory: {CONFIG['output_root']}")
            import shutil
            shutil.rmtree(CONFIG["output_root"])
        os.makedirs(CONFIG["output_root"], exist_ok=True)
    
    # Get seed list
    if CONFIG["USE_SEED_SAMPLING"]:
        if CONFIG["FORCE_FULL_SEEDS"]:
            full_seeds = CONFIG["FORCE_FULL_SEEDS"]
        else:
            full_seeds = _sample_seeds(CONFIG["SEED_SAMPLING_FRAME"], CONFIG["FULL_RUN_N_SEEDS"], CONFIG["SEED_SAMPLING_RNG_SEED"])
        
        if CONFIG["FORCE_PROBE_SEEDS"]:
            probe_seeds = CONFIG["FORCE_PROBE_SEEDS"]
        else:
            probe_seeds = _sample_seeds(CONFIG["SEED_SAMPLING_FRAME"], CONFIG["PROBE_N_SEEDS"], CONFIG["SEED_SAMPLING_RNG_SEED"] + 1)
            # Ensure probe seeds don't overlap with full seeds
            probe_seeds = [s for s in probe_seeds if s not in full_seeds]
    else:
        full_seeds = CONFIG["SEEDS"]
        probe_seeds = []
    
    print(f"\nSeed Configuration:")
    print(f"   Full seeds: {full_seeds}")
    print(f"   Probe seeds: {probe_seeds}")
    
    # Stage 1: Probe
    if CONFIG["PROBE_RUN_ENABLE"] and probe_seeds:
        print("\n" + "="*70)
        print("Stage 1: Probe Stage (Multi-seed Statistics)")
        print("="*70)
        for seed in probe_seeds:
            run_probe(seed, CONFIG["PROBE_EPOCHS"])
        
        # Analyze probe statistics
        analyze_probe_stats()
    
    # Stage 2: Baseline
    if CONFIG["STAGE2_RUN_FULL"] and full_seeds:
        print("\n" + "="*70)
        print("Stage 2: Baseline Stage (Complete Training)")
        print("="*70)
        for seed in full_seeds:
            if run_baseline(seed, CONFIG["FULL_EPOCHS"]):
                export_predictions_for_seed(seed)
    
    # Stage 3: Ablation
    if CONFIG["STAGE2_RUN_FULL"] and full_seeds:
        print("\n" + "="*70)
        print("Stage 3: Ablation Stage (Ablation Experiments)")
        print("="*70)
        for seed in full_seeds:
            run_ablation(seed)
    
    # Stage 4: Sensitivity
    if CONFIG["STAGE2_RUN_FULL"] and full_seeds:
        print("\n" + "="*70)
        print("Stage 4: Sensitivity Stage (Sensitivity Analysis)")
        print("="*70)
        for seed in full_seeds:
            run_sensitivity(seed)
    
    print("\n" + "="*70)
    print("[OK] All experiments completed!")
    print("="*70)
    print(f"Results saved to: {CONFIG['output_root']}")


if __name__ == "__main__":
    main()
