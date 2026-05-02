#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main runner for orchestrating the complete experiment workflow.

This script provides a centralized entry point to run:
1.  Probe Stage: Short training runs across multiple seeds to assess strategy stability.
2.  Baseline Stage: Full training run of the recommended best-performing strategy.
3.  Ablation Stage: Systematic removal of components from the baseline strategy.
4.  Sensitivity Stage: Analysis of model robustness to various perturbations.
"""

import os
import sys
import subprocess
import time
import random
from typing import List, Dict, Any

# =============================================================================
#  Project-level Path Configuration
# =============================================================================

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
LAYOUT_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, '..'))
if LAYOUT_ROOT not in sys.path:
    sys.path.insert(0, LAYOUT_ROOT)

from filament_layout import canonical_report_root, canonical_run_root, default_export_out_dir

# =============================================================================
#  Experiment Configuration
# =============================================================================

CONFIG = {
    # --- Path Configurations ---
    "main_path": os.path.join(PROJECT_ROOT, "core", "main_75.py"),
    "ablation_path": os.path.join(PROJECT_ROOT, "experiments", "ablation.py"),
    "sensitivity_path": os.path.join(PROJECT_ROOT, "experiments", "sensitivity.py"),
    "stats_path": os.path.join(PROJECT_ROOT, "analysis", "analyze_probe_stats.py"),
    "filament_id": os.environ.get("FILAMENT_ID", "FIL001"),
    "eval_filament_id": os.environ.get("EVAL_FILAMENT_ID", os.environ.get("FILAMENT_ID", "FIL001")),
    "focal_mm": int(os.environ.get("FOCAL_MM", "75")),
    "gt_diameter_um": float(os.environ.get("GT_DIAMETER_UM", "100.2")),
    "output_root": canonical_run_root(os.environ.get("FILAMENT_ID", "FIL001"), int(os.environ.get("FOCAL_MM", "75"))),
    "report_root": canonical_report_root(os.environ.get("FILAMENT_ID", "FIL001")),

    # --- Seed Configuration ---
    "seeds": [212],  # Fallback for non-sampling mode.

    # --- Seed Sampling Protocol (Recommended) ---
    "use_seed_sampling": True,
    "seed_sampling_frame": [1, 300],
    "seed_sampling_rng_seed": 20260108,
    "full_run_n_seeds": 1,

    # --- Probe Stage: Multi-seed Stability Assessment ---
    "probe_run_enable": True,
    "probe_n_seeds": 10,
    "probe_epochs": 20,  # Must be >= 20 for strategies to fully manifest.
    "probe_success_threshold": 0.01,  # err < 1% is considered a success.
    "probe_parallel": False,
    "probe_max_parallel": 4,

    # --- Full Stage: Full Training Runs ---
    "stage2_run_full": True,
    "full_epochs": 60,

    # --- Overrides for Specific Seeds ---
    "force_full_seeds": [],
    "force_probe_seeds": [],

    # --- Baseline Strategy (Recommended Best Configuration) ---
    # This configuration is used as the baseline for ablation and sensitivity analyses.
    "baseline_strategy": {
        "name": "baseline",
        "exp_subdir": "baseline",
        "desc": "The recommended baseline strategy.",
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
        }
    },

    # --- Experiment Naming ---
    "ablation_sweep_name": "ablation_sweep",
    "sensitivity_sweep_name": "sensitivity_analysis",

    # --- Execution Control ---
    "skip_completed": True,
    "continue_on_error": True,
    "cuda_visible_devices": "0",

    # --- Result Directory Management ---
    "clear_results_on_start": False,
}


def _sample_seeds(frame: List[int], n: int, rng_seed: int) -> List[int]:
    """Randomly sample n unique seeds from a given frame."""
    rng = random.Random(rng_seed)
    available = list(range(frame[0], frame[1] + 1))
    return sorted(rng.sample(available, min(n, len(available))))


def stage_seed_dir(stage: str, seed: int) -> str:
    """Construct canonical stage/seed directory."""
    return os.path.join(CONFIG["output_root"], stage, f"seed_{seed}")


def _run_subprocess(script_path: str, env: Dict[str, str]) -> bool:
    """Helper function to run a script in a subprocess."""
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

def run_probe(seed: int, epochs: int) -> bool:
    """Run a short-term probe experiment for a single seed."""
    strategy = CONFIG["baseline_strategy"]
    exp_dir = stage_seed_dir("probe", seed)
    os.makedirs(exp_dir, exist_ok=True)

    done_file = os.path.join(exp_dir, "DONE.txt")
    if CONFIG["skip_completed"] and os.path.exists(done_file):
        print(f"  Skipping completed probe for seed={seed}")
        return True

    print(f"  Running probe for seed={seed}...")
    env = os.environ.copy()
    env.update({
        "SEED": str(seed),
        "BASE_MODEL_SAVE_DIR": exp_dir,
        "NUM_EPOCHS": str(epochs),
        "ALLOW_TRAIN": "1",
        "FILAMENT_ID": CONFIG["filament_id"],
        "EVAL_FILAMENT_ID": CONFIG["eval_filament_id"],
        "FOCAL_MM": str(CONFIG["focal_mm"]),
        "GT_DIAMETER_UM": str(CONFIG["gt_diameter_um"]),
        "RUN_ROOT": CONFIG["output_root"],
        "REPORT_ROOT": CONFIG["report_root"],
    })
    env.update(strategy.get("env", {}))

    if _run_subprocess(CONFIG["main_path"], env):
        with open(done_file, "w") as f:
            f.write(f"Completed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        print(f"  Probe for seed={seed} completed.")
        return True
    else:
        if not CONFIG["continue_on_error"]:
            raise RuntimeError(f"Probe failed for seed={seed} and continue_on_error is False.")
        return False

def export_predictions_for_seed(seed: int) -> bool:
    """After training, export per-image predictions for train/val using best and best_ema weights."""
    exp_dir = stage_seed_dir("baseline", seed)
    split_role = "external_eval" if CONFIG["eval_filament_id"] != CONFIG["filament_id"] else "internal_val"

    # Only export if the baseline exists
    if not os.path.exists(exp_dir):
        print(f"  Warning: Baseline directory not found for seed={seed}. Skipping prediction export.")
        return False

    out_dir = default_export_out_dir(exp_dir, CONFIG["eval_filament_id"], split_role)
    os.makedirs(out_dir, exist_ok=True)

    print(f"  Exporting predictions for seed={seed}...")

    for ckpt in ["best", "ema"]:
        env = os.environ.copy()
        env.update({
            "SEED": str(seed),
            "BASE_MODEL_SAVE_DIR": exp_dir,
            "MODE": "export_preds",
            "EXPORT_CKPT": ckpt,
            "EXPORT_OUT_DIR": out_dir,
            "FILAMENT_ID": CONFIG["filament_id"],
            "EVAL_FILAMENT_ID": CONFIG["eval_filament_id"],
            "FOCAL_MM": str(CONFIG["focal_mm"]),
            "GT_DIAMETER_UM": str(CONFIG["gt_diameter_um"]),
            "RUN_ROOT": CONFIG["output_root"],
            "REPORT_ROOT": CONFIG["report_root"],
            "SPLIT_ROLE": split_role,
        })
        ok = _run_subprocess(CONFIG["main_path"], env)
        if not ok:
            if not CONFIG["continue_on_error"]:
                raise RuntimeError(f"Prediction export failed for seed={seed}, ckpt={ckpt}.")
            return False

    print(f"  Prediction export for seed={seed} completed.")
    return True


def run_baseline(seed: int, epochs: int) -> bool:
    """Run the full baseline experiment for a single seed."""
    strategy = CONFIG["baseline_strategy"]
    exp_dir = stage_seed_dir("baseline", seed)
    os.makedirs(exp_dir, exist_ok=True)

    done_file = os.path.join(exp_dir, "DONE.txt")
    if CONFIG["skip_completed"] and os.path.exists(done_file):
        print(f"  Skipping completed baseline for seed={seed}")
        return True

    print(f"  Running baseline for seed={seed}...")
    env = os.environ.copy()
    env.update({
        "SEED": str(seed),
        "BASE_MODEL_SAVE_DIR": exp_dir,
        "NUM_EPOCHS": str(epochs),
        "ALLOW_TRAIN": "1",
        "FILAMENT_ID": CONFIG["filament_id"],
        "EVAL_FILAMENT_ID": CONFIG["eval_filament_id"],
        "FOCAL_MM": str(CONFIG["focal_mm"]),
        "GT_DIAMETER_UM": str(CONFIG["gt_diameter_um"]),
        "RUN_ROOT": CONFIG["output_root"],
        "REPORT_ROOT": CONFIG["report_root"],
    })
    env.update(strategy.get("env", {}))

    if _run_subprocess(CONFIG["main_path"], env):
        with open(done_file, "w") as f:
            f.write(f"Completed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        print(f"  Baseline for seed={seed} completed.")
        return True
    else:
        if not CONFIG["continue_on_error"]:
            raise RuntimeError(f"Baseline failed for seed={seed} and continue_on_error is False.")
        return False

def run_ablation(seed: int) -> bool:
    """Run the ablation sweep for a single seed."""
    baseline_dir = stage_seed_dir("baseline", seed)
    sweep_dir = stage_seed_dir(CONFIG["ablation_sweep_name"].replace("_sweep", ""), seed)
    if CONFIG["ablation_sweep_name"] == "ablation_sweep":
        sweep_dir = stage_seed_dir("ablation", seed)

    if not os.path.exists(baseline_dir):
        print(f"  Warning: Baseline directory not found for seed={seed}. Skipping ablation.")
        return False

    print(f"  Running ablation sweep for seed={seed}...")
    env = os.environ.copy()
    env.update({
        "SEED": str(seed),
        "BASELINE_DIR": os.path.abspath(baseline_dir),
        "SWEEP_ROOT": os.path.abspath(sweep_dir),
        "FILAMENT_ID": CONFIG["filament_id"],
        "EVAL_FILAMENT_ID": CONFIG["eval_filament_id"],
        "FOCAL_MM": str(CONFIG["focal_mm"]),
        "GT_DIAMETER_UM": str(CONFIG["gt_diameter_um"]),
        "RUN_ROOT": CONFIG["output_root"],
        "REPORT_ROOT": CONFIG["report_root"],
    })

    if _run_subprocess(CONFIG["ablation_path"], env):
        print(f"  Ablation sweep for seed={seed} completed.")
        return True
    else:
        if not CONFIG["continue_on_error"]:
            raise RuntimeError(f"Ablation failed for seed={seed} and continue_on_error is False.")
        return False

def run_sensitivity(seed: int) -> bool:
    """Run the sensitivity analysis for a single seed."""
    strategy = CONFIG["baseline_strategy"]
    baseline_dir = stage_seed_dir("baseline", seed)
    sweep_dir = stage_seed_dir("sensitivity", seed)

    if not os.path.exists(baseline_dir):
        print(f"  Warning: Baseline directory not found for seed={seed}. Skipping sensitivity analysis.")
        return False

    print(f"  Running sensitivity analysis for seed={seed}...")
    env = os.environ.copy()
    env.update({
        "SEED": str(seed),
        "MAIN_PATH": os.path.abspath(CONFIG["main_path"]),
        "SWEEP_ROOT": os.path.abspath(sweep_dir),
        "BASELINE_DIR": os.path.abspath(baseline_dir),
        "NUM_EPOCHS": str(CONFIG["full_epochs"]),
        "FILAMENT_ID": CONFIG["filament_id"],
        "EVAL_FILAMENT_ID": CONFIG["eval_filament_id"],
        "FOCAL_MM": str(CONFIG["focal_mm"]),
        "GT_DIAMETER_UM": str(CONFIG["gt_diameter_um"]),
        "RUN_ROOT": CONFIG["output_root"],
        "REPORT_ROOT": CONFIG["report_root"],
    })
    env.update(strategy.get("env", {}))

    if _run_subprocess(CONFIG["sensitivity_path"], env):
        print(f"  Sensitivity analysis for seed={seed} completed.")
        return True
    else:
        if not CONFIG["continue_on_error"]:
            raise RuntimeError(f"Sensitivity analysis failed for seed={seed} and continue_on_error is False.")
        return False

def analyze_probe_stats() -> bool:
    """Analyze statistics from the probe stage."""
    print("\nAnalyzing probe statistics...")
    if _run_subprocess(
        CONFIG["stats_path"],
        {
            **os.environ.copy(),
            "RESULTS_ROOT": CONFIG["output_root"],
            "REPORT_ROOT": CONFIG["report_root"],
            "FILAMENT_ID": CONFIG["filament_id"],
            "EVAL_FILAMENT_ID": CONFIG["eval_filament_id"],
            "FOCAL_MM": str(CONFIG["focal_mm"]),
            "GT_DIAMETER_UM": str(CONFIG["gt_diameter_um"]),
            "SPLIT_ROLE": "probe",
        },
    ):
        print("Probe statistics analysis completed.")
        return True
    else:
        print("Probe statistics analysis failed.", file=sys.stderr)
        return False


def main():
    """Main entry point for the experiment runner."""
    print("="*70)
    print("Starting Full Experiment Workflow")
    print("="*70)

    if CONFIG["clear_results_on_start"]:
        output_root = CONFIG["output_root"]
        if os.path.exists(output_root):
            print(f"WARNING: `clear_results_on_start` is True. Deleting {output_root}")
            response = input("Type 'YES' to confirm deletion: ")
            if response == 'YES':
                import shutil
                shutil.rmtree(output_root)
                print(f"Directory {output_root} deleted.")
            else:
                print("Deletion cancelled. Exiting.")
                return
        os.makedirs(output_root, exist_ok=True)

    if CONFIG["use_seed_sampling"]:
        full_seeds = CONFIG["force_full_seeds"] or _sample_seeds(CONFIG["seed_sampling_frame"], CONFIG["full_run_n_seeds"], CONFIG["seed_sampling_rng_seed"])
        probe_seeds = CONFIG["force_probe_seeds"] or _sample_seeds(CONFIG["seed_sampling_frame"], CONFIG["probe_n_seeds"], CONFIG["seed_sampling_rng_seed"] + 1)
        probe_seeds = [s for s in probe_seeds if s not in full_seeds]
    else:
        full_seeds = CONFIG["seeds"]
        probe_seeds = []

    print(f"\nSeed Configuration:")
    print(f"  Full training seeds: {full_seeds}")
    print(f"  Probe seeds: {probe_seeds}")

    # Stage 1: Probe
    if CONFIG["probe_run_enable"] and probe_seeds:
        print("\n--- Stage 1: Probe ---")
        for seed in probe_seeds:
            run_probe(seed, CONFIG["probe_epochs"])
        analyze_probe_stats()

    # Stages 2, 3, 4: Full runs
    if CONFIG["stage2_run_full"] and full_seeds:
        for seed in full_seeds:
            print(f"\n--- Processing Full Run for Seed: {seed} ---")
            # Stage 2: Baseline
            print("\n--- Stage 2: Baseline ---")
            if not run_baseline(seed, CONFIG["full_epochs"]):
                continue  # Skip to next seed if baseline fails

            # Export predictions (best + best_ema) for train/val
            export_predictions_for_seed(seed)

            # Stage 3: Ablation
            print("\n--- Stage 3: Ablation Sweep ---")
            run_ablation(seed)

            # Stage 4: Sensitivity
            print("\n--- Stage 4: Sensitivity Analysis ---")
            run_sensitivity(seed)

    print("\n" + "="*70)
    print("All experiment stages completed.")
    print(f"Results are saved in: {CONFIG['output_root']}")
    print("="*70)

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = CONFIG["cuda_visible_devices"]
    main()
