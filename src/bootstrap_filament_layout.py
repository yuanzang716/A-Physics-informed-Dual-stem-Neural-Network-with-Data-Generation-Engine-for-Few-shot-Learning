#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import shutil
from pathlib import Path

from filament_layout import (
    DATA_ROOT_BASE,
    LEGACY_DATASETS,
    MIGRATION_FIELDS,
    MIGRATION_MAP_PATH,
    REGISTRY_FIELDS,
    REGISTRY_PATH,
    RUNS_ROOT_BASE,
    canonical_data_root,
    canonical_run_root,
    iter_legacy_layout_rows,
)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _safe_unlink(path: str) -> None:
    p = Path(path)
    if p.is_symlink() or p.is_file():
        p.unlink()
    elif p.is_dir():
        shutil.rmtree(p)


def _symlink(src: str, dst: str) -> None:
    if os.path.lexists(dst):
        return
    os.symlink(src, dst)


def _symlink_dir_contents(src_dir: str, dst_dir: str, pattern: str = "*.BMP") -> None:
    _ensure_dir(dst_dir)
    if not os.path.isdir(src_dir):
        return
    for src in sorted(Path(src_dir).glob(pattern)):
        dst = Path(dst_dir) / src.name
        if dst.exists() or dst.is_symlink():
            continue
        dst.symlink_to(src)


def _symlink_file(src: str, dst: str) -> None:
    if not os.path.exists(src):
        return
    parent = os.path.dirname(dst)
    _ensure_dir(parent)
    _symlink(src, dst)


def _ensure_dir_or_symlink(src_dir: str, dst_dir: str) -> None:
    """Reuse a legacy directory when it exists; otherwise materialize a canonical directory."""
    if os.path.isdir(src_dir):
        _symlink(src_dir, dst_dir)
        return

    if os.path.islink(dst_dir) and not os.path.exists(dst_dir):
        _safe_unlink(dst_dir)
    _ensure_dir(dst_dir)


def _write_csv(path: str, fieldnames, rows) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _bootstrap_registry() -> None:
    rows = []
    seen = set()
    for filament_id, by_focal in LEGACY_DATASETS.items():
        info = by_focal[min(by_focal.keys())]
        key = (filament_id, info["legacy_alias"])
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "filament_id": filament_id,
                "legacy_alias": info["legacy_alias"],
                "nominal_diameter_um": info["nominal_diameter_um"],
                "reference_diameter_um": info["reference_diameter_um"],
                "notes": info["notes"],
                "status": info["status"],
            }
        )
    _write_csv(REGISTRY_PATH, REGISTRY_FIELDS, rows)


def _bootstrap_migration_map() -> None:
    _write_csv(MIGRATION_MAP_PATH, MIGRATION_FIELDS, list(iter_legacy_layout_rows()))


def _canonical_artifacts_for_legacy_root(legacy_root: str):
    return {
        "initial_diameter.txt": os.path.join(legacy_root, "train_real", "initial_diameter.txt"),
        "optimized_params.csv": os.path.join(legacy_root, "train_real", "optimized_params.csv"),
        "train_real_rp.csv": os.path.join(legacy_root, "train_real", "train_real_rp.csv"),
        "val_real_rp.csv": os.path.join(legacy_root, "val_real", "val_real_rp.csv"),
    }


def _bootstrap_data_roots() -> None:
    for filament_id, by_focal in LEGACY_DATASETS.items():
        for focal_mm, info in by_focal.items():
            legacy_root = str(info["legacy_root"])
            canonical_root = canonical_data_root(filament_id, focal_mm)
            raw_dir = os.path.join(canonical_root, "raw")
            split_train = os.path.join(canonical_root, "split", "train_real")
            split_val = os.path.join(canonical_root, "split", "val_real")
            artifacts_dir = os.path.join(canonical_root, "artifacts")
            sim_dir = os.path.join(canonical_root, "simulation_dataset")
            cache_dir = os.path.join(canonical_root, "cache")
            fft_dir = os.path.join(cache_dir, "fft_tensors")
            manifests_dir = os.path.join(canonical_root, "manifests")

            for p in [canonical_root, raw_dir, os.path.join(canonical_root, "split"), artifacts_dir, cache_dir, manifests_dir]:
                _ensure_dir(p)

            _symlink_dir_contents(legacy_root, raw_dir)
            _ensure_dir_or_symlink(os.path.join(legacy_root, "train_real"), split_train)
            _ensure_dir_or_symlink(os.path.join(legacy_root, "val_real"), split_val)

            artifacts = _canonical_artifacts_for_legacy_root(legacy_root)
            for name, src in artifacts.items():
                _symlink_file(src, os.path.join(artifacts_dir, name))

            if os.path.exists(os.path.join(legacy_root, "simulation_dataset")) and not os.path.lexists(sim_dir):
                _symlink(os.path.join(legacy_root, "simulation_dataset"), sim_dir)

            legacy_fft = os.path.join(legacy_root, "cache", "fft_tensors")
            if os.path.exists(legacy_fft):
                if not os.path.lexists(fft_dir):
                    _symlink(legacy_fft, fft_dir)
            else:
                _ensure_dir(fft_dir)

            manifest = {
                "filament_id": filament_id,
                "focal_mm": focal_mm,
                "legacy_root": legacy_root,
                "canonical_root": canonical_root,
                "reference_diameter_um": info["reference_diameter_um"],
                "nominal_diameter_um": info["nominal_diameter_um"],
                "status": info["status"],
            }
            with open(os.path.join(manifests_dir, "dataset_manifest.json"), "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)


def _symlink_stage_dir(src: str, dst: str) -> None:
    if not os.path.exists(src):
        return
    parent = os.path.dirname(dst)
    _ensure_dir(parent)
    _symlink(src, dst)


def _bootstrap_existing_run_symlinks() -> None:
    legacy_results = {
        ("FIL001", 75): os.path.join("/root/autodl-tmp", "Code_75", "results"),
        ("FIL001", 120): os.path.join("/root/autodl-tmp", "Code_120", "results"),
    }

    for (filament_id, focal_mm), legacy_root in legacy_results.items():
        if not os.path.isdir(legacy_root):
            continue
        run_root = canonical_run_root(filament_id, focal_mm)
        for stage in ["probe", "baseline", "ablation", "sensitivity"]:
            _ensure_dir(os.path.join(run_root, stage))

        for seed_dir in sorted(Path(legacy_root).glob("seed_*")):
            seed_name = seed_dir.name
            probe_src = seed_dir / "probe"
            baseline_src = seed_dir / "baseline"
            ablation_src = seed_dir / "ablation_sweep"
            sensitivity_src = seed_dir / "sensitivity_analysis"
            _symlink_stage_dir(str(probe_src), os.path.join(run_root, "probe", seed_name))
            _symlink_stage_dir(str(baseline_src), os.path.join(run_root, "baseline", seed_name))
            _symlink_stage_dir(str(ablation_src), os.path.join(run_root, "ablation", seed_name))
            _symlink_stage_dir(str(sensitivity_src), os.path.join(run_root, "sensitivity", seed_name))


def main() -> None:
    _ensure_dir(DATA_ROOT_BASE)
    _ensure_dir(RUNS_ROOT_BASE)
    _bootstrap_registry()
    _bootstrap_migration_map()
    _bootstrap_data_roots()
    _bootstrap_existing_run_symlinks()
    print("Bootstrapped filament-aware layout:")
    print(f"  registry: {REGISTRY_PATH}")
    print(f"  migration_map: {MIGRATION_MAP_PATH}")


if __name__ == "__main__":
    main()
