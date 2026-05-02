#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Probe-stage statistics aggregator with filament-aware metadata.

Supports two directory layouts:
1. Canonical layout:   <results_root>/probe/seed_<seed>/metrics.csv
2. Flat layout:        <results_root>/seed_<seed>/probe/metrics.csv
"""

from __future__ import annotations

import glob
import json
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LAYOUT_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, ".."))
if LAYOUT_ROOT not in sys.path:
    sys.path.insert(0, LAYOUT_ROOT)

from filament_layout import add_identity_columns, canonical_report_root, report_path


def _default_results_root() -> str:
    return os.path.join(PROJECT_ROOT, "results")


CONFIG = {
    "results_root": os.path.abspath(os.environ.get("RESULTS_ROOT", _default_results_root())),
    "report_root": os.path.abspath(
        os.environ.get("REPORT_ROOT", canonical_report_root(os.environ.get("FILAMENT_ID", "FIL001")))
    ),
    "train_filament_id": os.environ.get("FILAMENT_ID", "FIL001"),
    "eval_filament_id": os.environ.get("EVAL_FILAMENT_ID", os.environ.get("FILAMENT_ID", "FIL001")),
    "focal_mm": int(os.environ.get("FOCAL_MM", "75")),
    "gt_diameter_um": float(os.environ.get("GT_DIAMETER_UM", "100.2")),
    "success_threshold": float(os.environ.get("SUCCESS_THRESHOLD", "0.01")),
    "split_role": os.environ.get("SPLIT_ROLE", "probe"),
}


def to_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_metrics_csv(csv_path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(csv_path):
        return None

    try:
        df = pd.read_csv(csv_path)
        if df.empty or "val_ema_score" not in df.columns:
            return None

        scores = pd.to_numeric(df["val_ema_score"], errors="coerce")
        valid = df.loc[scores.notna()].copy()
        if valid.empty:
            return None
        valid["val_ema_score"] = scores.loc[valid.index]
        best_row = valid.loc[valid["val_ema_score"].idxmin()]

        pred_mean = to_float(best_row.get("val_ema_pred_mean", best_row.get("val_pred_mean")))
        result = {
            "epoch": int(to_float(best_row.get("epoch"), -1)),
            "val_ema_err_obs": to_float(best_row.get("val_ema_err_obs")),
            "val_ema_score": to_float(best_row.get("val_ema_score")),
            "val_ema_pred_mean": pred_mean,
            "val_err_obs": to_float(best_row.get("val_err_obs")),
            "val_score": to_float(best_row.get("val_score")),
            "val_pred_mean": to_float(best_row.get("val_pred_mean")),
            "val_pred_std": to_float(best_row.get("val_pred_std")),
            "train_loss": to_float(best_row.get("train_loss", best_row.get("train_loss_total"))),
        }

        if not np.isnan(pred_mean):
            result["bias_um"] = abs(pred_mean - CONFIG["gt_diameter_um"])
        else:
            result["bias_um"] = float("nan")

        result["is_success"] = (
            not np.isnan(result["val_ema_err_obs"])
            and result["val_ema_err_obs"] < CONFIG["success_threshold"]
        )
        return result
    except Exception as exc:
        print(f"Warning: failed to read {csv_path}: {exc}")
        return None


def _iter_probe_dirs(results_root: str) -> List[str]:
    canonical = sorted(glob.glob(os.path.join(results_root, "probe", "seed_*")))
    if canonical:
        return canonical

    flat_seed_roots = sorted(glob.glob(os.path.join(results_root, "seed_*")))
    flat = []
    for seed_root in flat_seed_roots:
        probe_dir = os.path.join(seed_root, "probe")
        if os.path.isdir(probe_dir):
            flat.append(probe_dir)
    return flat


def _seed_from_path(path: str) -> Optional[int]:
    for part in reversed(os.path.normpath(path).split(os.sep)):
        if part.startswith("seed_"):
            try:
                return int(part.split("_", 1)[1])
            except ValueError:
                return None
    return None


def collect_all_probe_results() -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for probe_dir in _iter_probe_dirs(CONFIG["results_root"]):
        metrics_csv = os.path.join(probe_dir, "metrics.csv")
        metrics = load_metrics_csv(metrics_csv)
        if metrics is None:
            continue

        seed = _seed_from_path(probe_dir)
        metrics["seed"] = seed
        metrics["probe_dir"] = probe_dir
        results.append(metrics)

    return sorted(results, key=lambda row: (row.get("seed") is None, row.get("seed", 10**9)))


def calculate_statistics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {}

    ema_errors = [row["val_ema_err_obs"] for row in results if not np.isnan(row["val_ema_err_obs"])]
    biases = [row["bias_um"] for row in results if not np.isnan(row["bias_um"])]
    ema_scores = [row["val_ema_score"] for row in results if not np.isnan(row["val_ema_score"])]
    successes = [bool(row["is_success"]) for row in results]

    stats: Dict[str, Any] = {}
    if ema_errors:
        stats["ema_error"] = {
            "mean": float(np.mean(ema_errors)),
            "std": float(np.std(ema_errors)),
            "median": float(np.median(ema_errors)),
            "min": float(np.min(ema_errors)),
            "max": float(np.max(ema_errors)),
            "q25": float(np.percentile(ema_errors, 25)),
            "q75": float(np.percentile(ema_errors, 75)),
            "n": len(ema_errors),
        }
    if biases:
        stats["bias"] = {
            "mean": float(np.mean(biases)),
            "std": float(np.std(biases)),
            "median": float(np.median(biases)),
            "min": float(np.min(biases)),
            "max": float(np.max(biases)),
            "q25": float(np.percentile(biases, 25)),
            "q75": float(np.percentile(biases, 75)),
            "n": len(biases),
        }
    if ema_scores:
        stats["ema_score"] = {
            "mean": float(np.mean(ema_scores)),
            "std": float(np.std(ema_scores)),
            "median": float(np.median(ema_scores)),
            "min": float(np.min(ema_scores)),
            "max": float(np.max(ema_scores)),
            "n": len(ema_scores),
        }
    if successes:
        n_success = sum(successes)
        n_total = len(successes)
        stats["success_rate"] = {
            "n_success": int(n_success),
            "n_total": int(n_total),
            "rate": float(n_success / n_total) if n_total else 0.0,
            "failure_rate": float((n_total - n_success) / n_total) if n_total else 0.0,
        }
    return stats


def _identity_kwargs() -> Dict[str, Any]:
    return {
        "train_filament_id": CONFIG["train_filament_id"],
        "eval_filament_id": CONFIG["eval_filament_id"],
        "focal_mm": CONFIG["focal_mm"],
        "split_role": CONFIG["split_role"],
    }


def _build_summary_df(stats: Dict[str, Any]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if "ema_error" in stats:
        rows.append({"metric": "EMA Relative Error", **stats["ema_error"]})
    if "bias" in stats:
        rows.append({"metric": "Absolute Bias (um)", **stats["bias"]})
    if "ema_score" in stats:
        row = {"metric": "EMA Score", **stats["ema_score"]}
        row.setdefault("q25", np.nan)
        row.setdefault("q75", np.nan)
        rows.append(row)
    if "success_rate" in stats:
        rows.append(
            {
                "metric": "Success Rate",
                "mean": stats["success_rate"]["rate"],
                "std": np.nan,
                "median": np.nan,
                "min": np.nan,
                "max": np.nan,
                "q25": np.nan,
                "q75": np.nan,
                "n": stats["success_rate"]["n_total"],
                "n_success": stats["success_rate"]["n_success"],
                "n_total": stats["success_rate"]["n_total"],
                "failure_rate": stats["success_rate"]["failure_rate"],
            }
        )
    summary_df = pd.DataFrame(rows)
    if summary_df.empty:
        return pd.DataFrame(columns=["metric", "mean", "std", "median", "min", "max", "q25", "q75", "n"])
    return add_identity_columns(summary_df, **_identity_kwargs())


def _write_compatibility_outputs(results_df: pd.DataFrame, summary_df: pd.DataFrame, payload: Dict[str, Any]) -> None:
    """Write additional copies of results to the probe_statistics subdirectory."""
    compat_dir = os.path.join(CONFIG["results_root"], "probe_statistics")
    os.makedirs(compat_dir, exist_ok=True)

    results_df.to_csv(os.path.join(compat_dir, "metrics_per_seed.csv"), index=False)
    summary_df.to_csv(os.path.join(compat_dir, "summary_stats.csv"), index=False)
    with open(os.path.join(compat_dir, "detailed_results.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def save_results(results: List[Dict[str, Any]], stats: Dict[str, Any]) -> None:
    os.makedirs(CONFIG["report_root"], exist_ok=True)

    results_df = pd.DataFrame(results)
    if results_df.empty:
        results_df = pd.DataFrame(
            columns=[
                "seed",
                "epoch",
                "val_ema_err_obs",
                "val_ema_score",
                "val_ema_pred_mean",
                "val_err_obs",
                "val_score",
                "val_pred_mean",
                "val_pred_std",
                "train_loss",
                "bias_um",
                "is_success",
                "probe_dir",
            ]
        )
    results_df = add_identity_columns(results_df, **_identity_kwargs())
    summary_df = _build_summary_df(stats)

    payload = {
        "config": CONFIG,
        "statistics": stats,
        "results": results_df.to_dict(orient="records"),
    }

    detailed_json = report_path(
        CONFIG["train_filament_id"],
        "probe_detailed",
        CONFIG["focal_mm"],
        CONFIG["eval_filament_id"],
        CONFIG["split_role"],
        "json",
        report_root=CONFIG["report_root"],
    )
    metrics_csv = report_path(
        CONFIG["train_filament_id"],
        "probe_metrics",
        CONFIG["focal_mm"],
        CONFIG["eval_filament_id"],
        CONFIG["split_role"],
        "csv",
        report_root=CONFIG["report_root"],
    )
    summary_csv = report_path(
        CONFIG["train_filament_id"],
        "probe_summary",
        CONFIG["focal_mm"],
        CONFIG["eval_filament_id"],
        CONFIG["split_role"],
        "csv",
        report_root=CONFIG["report_root"],
    )

    with open(detailed_json, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    results_df.to_csv(metrics_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    _write_compatibility_outputs(results_df, summary_df, payload)

    print(f"Saved detailed results to: {detailed_json}")
    print(f"Saved per-seed metrics to: {metrics_csv}")
    print(f"Saved summary statistics to: {summary_csv}")


def main() -> None:
    results = collect_all_probe_results()
    stats = calculate_statistics(results)
    save_results(results, stats)

    print("\nProbe-stage summary")
    print(f"  results_root: {CONFIG['results_root']}")
    print(f"  seeds_found: {len(results)}")
    if "success_rate" in stats:
        print(
            "  success_rate: "
            f"{100.0 * stats['success_rate']['rate']:.2f}% "
            f"({stats['success_rate']['n_success']}/{stats['success_rate']['n_total']})"
        )


if __name__ == "__main__":
    main()
