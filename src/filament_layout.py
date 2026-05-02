from __future__ import annotations

import csv
import os
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional


PROJECT_ROOT = os.environ.get("FILAMENT_PROJECT_ROOT", "/root/autodl-tmp")
DATA_ROOT_BASE = os.path.join(PROJECT_ROOT, "data", "filaments")
RUNS_ROOT_BASE = os.path.join(PROJECT_ROOT, "runs")
REPORTS_ROOT_BASE = os.path.join(PROJECT_ROOT, "reports")
REGISTRY_PATH = os.path.join(DATA_ROOT_BASE, "registry.csv")
MIGRATION_MAP_PATH = os.path.join(DATA_ROOT_BASE, "migration_map.csv")


REGISTRY_FIELDS = [
    "filament_id",
    "legacy_alias",
    "nominal_diameter_um",
    "reference_diameter_um",
    "notes",
    "status",
]


MIGRATION_FIELDS = [
    "filament_id",
    "focal_mm",
    "legacy_root",
    "canonical_root",
    "raw_source",
    "split_train_source",
    "split_val_source",
    "simulation_dataset_source",
    "cache_fft_source",
]


LEGACY_DATASETS: Dict[str, Dict[int, Dict[str, object]]] = {
    "FIL001": {
        75: {
            "legacy_alias": "Diameter_100.2_75mm",
            "legacy_root": os.path.join(PROJECT_ROOT, "Diameter_100.2_75mm"),
            "nominal_diameter_um": 100.2,
            "reference_diameter_um": 100.2,
            "r_bounds_mm": (-5 * 1.05, -5 * 0.95),  # (-5.25, -4.75)
            "notes": "Development filament (legacy 100.2um, 75mm focal)",
            "status": "active",
        },
        120: {
            "legacy_alias": "Diameter_100.2_120mm",
            "legacy_root": os.path.join(PROJECT_ROOT, "Diameter_100.2_120mm"),
            "nominal_diameter_um": 100.2,
            "reference_diameter_um": 100.2,
            "r_bounds_mm": (-15 * 1.05, -15 * 0.95),  # (-15.75, -14.25)
            "notes": "Development filament (legacy 100.2um, 120mm focal)",
            "status": "active",
        },
    },
    "FIL002": {
        75: {
            "legacy_alias": "Diameter_100.0_75mm",
            "legacy_root": os.path.join(PROJECT_ROOT, "Diameter_100.0_75mm"),
            "nominal_diameter_um": 100.0,
            "reference_diameter_um": 100.0,
            "r_bounds_mm": (-5.5 * 1.05, -4.5 * 0.95),  # (-5.775, -4.275)
            "notes": "External-validation filament (legacy 100.0um, 75mm focal)",
            "status": "active",
        },
        120: {
            "legacy_alias": "Diameter_100.0_120mm",
            "legacy_root": os.path.join(PROJECT_ROOT, "Diameter_100.0_120mm"),
            "nominal_diameter_um": 100.0,
            "reference_diameter_um": 100.0,
            "r_bounds_mm": (-6.0 * 1.05, -4.0 * 0.95),  # (-6.3, -3.8)
            "notes": "External-validation filament (legacy 100.0um, 120mm focal)",
            "status": "active",
        },
    },
}


FALLBACK_FILAMENT_BY_GT = {
    "100.2": "FIL001",
    "100.0": "FIL002",
}


@dataclass(frozen=True)
class FilamentContext:
    filament_id: str
    eval_filament_id: str
    focal_mm: int
    gt_diameter_um: float
    data_root: str
    raw_dir: str
    split_dir: str
    train_real_dir: str
    val_real_dir: str
    artifacts_dir: str
    simulation_dataset_dir: str
    cache_dir: str
    fft_cache_dir: str
    manifests_dir: str
    run_root: str
    report_root: str
    is_canonical_data_root: bool
    r_bounds_mm: Optional[tuple] = None  # (low, high) for DE parameter 'r'

    def to_metadata(self) -> Dict[str, object]:
        return asdict(self)


def ensure_parent_on_path(path: str) -> None:
    import sys

    if path not in sys.path:
        sys.path.insert(0, path)


def canonical_data_root(filament_id: str, focal_mm: int) -> str:
    return os.path.join(DATA_ROOT_BASE, filament_id, f"focal_{int(focal_mm)}mm")


def canonical_run_root(train_filament_id: str, focal_mm: int) -> str:
    return os.path.join(RUNS_ROOT_BASE, f"train_{train_filament_id}", f"focal_{int(focal_mm)}mm")


def canonical_report_root(train_filament_id: str) -> str:
    return os.path.join(REPORTS_ROOT_BASE, f"train_{train_filament_id}")


def canonical_stage_dir(run_root: str, stage: str, seed: Optional[int] = None, leaf: Optional[str] = None) -> str:
    out = os.path.join(run_root, stage)
    if seed is not None:
        out = os.path.join(out, f"seed_{int(seed)}")
    if leaf:
        out = os.path.join(out, leaf)
    return out


def default_export_out_dir(base_model_save_dir: str, eval_filament_id: str, split_role: str) -> str:
    if split_role == "external_eval":
        return os.path.join(base_model_save_dir, "external_eval", eval_filament_id)
    return os.path.join(base_model_save_dir, "internal_eval")


def report_tag(focal_mm: int, eval_filament_id: str, split_role: str) -> str:
    return f"focal_{int(focal_mm)}mm_eval_{eval_filament_id}_{split_role}"


def report_path(
    train_filament_id: str,
    stem: str,
    focal_mm: int,
    eval_filament_id: str,
    split_role: str,
    ext: str,
    report_root: Optional[str] = None,
) -> str:
    return os.path.join(
        os.path.abspath(report_root or canonical_report_root(train_filament_id)),
        f"{stem}_{report_tag(focal_mm, eval_filament_id, split_role)}.{ext.lstrip('.')}",
    )


def default_probe_stats_path(
    train_filament_id: str,
    focal_mm: int,
    eval_filament_id: Optional[str] = None,
    split_role: str = "probe",
) -> str:
    resolved_eval = eval_filament_id or train_filament_id
    return report_path(
        train_filament_id,
        "probe_detailed",
        focal_mm,
        resolved_eval,
        split_role,
        "json",
    )


def default_probe_summary_csv(
    train_filament_id: str,
    focal_mm: int,
    eval_filament_id: Optional[str] = None,
    split_role: str = "probe",
) -> str:
    resolved_eval = eval_filament_id or train_filament_id
    return report_path(
        train_filament_id,
        "probe_summary",
        focal_mm,
        resolved_eval,
        split_role,
        "csv",
    )


def read_registry(path: str = REGISTRY_PATH) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def registry_row_for_filament(filament_id: str, path: str = REGISTRY_PATH) -> Optional[Dict[str, str]]:
    for row in read_registry(path):
        if row.get("filament_id") == filament_id:
            return row
    return None


def reference_diameter_for_filament(filament_id: str, path: str = REGISTRY_PATH) -> Optional[float]:
    row = registry_row_for_filament(filament_id, path=path)
    if row is not None:
        try:
            return float(row["reference_diameter_um"])
        except (KeyError, TypeError, ValueError):
            pass

    legacy_info = LEGACY_DATASETS.get(filament_id, {})
    for info in legacy_info.values():
        try:
            return float(info["reference_diameter_um"])
        except (KeyError, TypeError, ValueError):
            continue
    return None


def infer_filament_id(default_data_root: str, default_gt: float, fallback: Optional[str] = None) -> str:
    if fallback:
        return fallback

    norm_gt = f"{float(default_gt):.1f}"
    if norm_gt in FALLBACK_FILAMENT_BY_GT:
        return FALLBACK_FILAMENT_BY_GT[norm_gt]

    lower_root = os.path.abspath(default_data_root).lower()
    if "100.2" in lower_root:
        return "FIL001"
    if "100.0" in lower_root:
        return "FIL002"
    return "FIL001"


def _path_exists(path: str) -> bool:
    return bool(path) and os.path.exists(path)


def _is_canonical_root(path: str) -> bool:
    ap = os.path.abspath(path)
    data_base = os.path.abspath(DATA_ROOT_BASE)
    return ap == data_base or ap.startswith(data_base + os.sep)


def _resolve_data_root(
    filament_id: str,
    focal_mm: int,
    default_data_root: str,
) -> str:
    explicit = os.environ.get("DATA_ROOT")
    if explicit:
        return os.path.abspath(explicit)

    canonical = canonical_data_root(filament_id, focal_mm)
    if os.path.exists(canonical):
        return canonical
    return os.path.abspath(default_data_root)


def _legacy_subpaths(data_root: str) -> Dict[str, str]:
    train_real = os.path.join(data_root, "train_real")
    val_real = os.path.join(data_root, "val_real")
    return {
        "raw_dir": data_root,
        "split_dir": data_root,
        "train_real_dir": train_real,
        "val_real_dir": val_real,
        "artifacts_dir": data_root,
        "simulation_dataset_dir": os.path.join(data_root, "simulation_dataset"),
        "cache_dir": os.path.join(data_root, "cache"),
        "fft_cache_dir": os.path.join(data_root, "cache", "fft_tensors"),
        "manifests_dir": os.path.join(data_root, "manifests"),
    }


def _canonical_subpaths(data_root: str) -> Dict[str, str]:
    return {
        "raw_dir": os.path.join(data_root, "raw"),
        "split_dir": os.path.join(data_root, "split"),
        "train_real_dir": os.path.join(data_root, "split", "train_real"),
        "val_real_dir": os.path.join(data_root, "split", "val_real"),
        "artifacts_dir": os.path.join(data_root, "artifacts"),
        "simulation_dataset_dir": os.path.join(data_root, "simulation_dataset"),
        "cache_dir": os.path.join(data_root, "cache"),
        "fft_cache_dir": os.path.join(data_root, "cache", "fft_tensors"),
        "manifests_dir": os.path.join(data_root, "manifests"),
    }


def resolve_dataset_subpaths(data_root: str) -> Dict[str, str]:
    if _is_canonical_root(data_root):
        return _canonical_subpaths(data_root)
    return _legacy_subpaths(data_root)


def resolve_filament_context(
    *,
    default_data_root: str,
    default_gt: float,
    default_focal_mm: int,
    default_filament_id: Optional[str] = None,
) -> FilamentContext:
    focal_mm = int(os.environ.get("FOCAL_MM", str(default_focal_mm)))
    gt_diameter_um = float(os.environ.get("GT_DIAMETER_UM", str(default_gt)))
    filament_id = os.environ.get(
        "FILAMENT_ID",
        infer_filament_id(default_data_root, default_gt, fallback=default_filament_id),
    )
    eval_filament_id = os.environ.get("EVAL_FILAMENT_ID", filament_id)

    data_root = _resolve_data_root(filament_id, focal_mm, default_data_root)
    is_canonical = _is_canonical_root(data_root)
    subpaths = resolve_dataset_subpaths(data_root)

    run_root = os.path.abspath(
        os.environ.get("RUN_ROOT", canonical_run_root(filament_id, focal_mm))
    )
    report_root = os.path.abspath(
        os.environ.get("REPORT_ROOT", canonical_report_root(filament_id))
    )

    # Resolve per-filament r bounds from LEGACY_DATASETS or env vars
    r_bounds_mm = None
    env_r_lo = os.environ.get("R_BOUND_LOW")
    env_r_hi = os.environ.get("R_BOUND_HIGH")
    if env_r_lo is not None and env_r_hi is not None:
        r_bounds_mm = (float(env_r_lo), float(env_r_hi))
    else:
        fil_info = LEGACY_DATASETS.get(filament_id, {}).get(focal_mm, {})
        r_bounds_mm = fil_info.get("r_bounds_mm")

    return FilamentContext(
        filament_id=filament_id,
        eval_filament_id=eval_filament_id,
        focal_mm=focal_mm,
        gt_diameter_um=gt_diameter_um,
        data_root=data_root,
        raw_dir=subpaths["raw_dir"],
        split_dir=subpaths["split_dir"],
        train_real_dir=subpaths["train_real_dir"],
        val_real_dir=subpaths["val_real_dir"],
        artifacts_dir=subpaths["artifacts_dir"],
        simulation_dataset_dir=subpaths["simulation_dataset_dir"],
        cache_dir=subpaths["cache_dir"],
        fft_cache_dir=subpaths["fft_cache_dir"],
        manifests_dir=subpaths["manifests_dir"],
        run_root=run_root,
        report_root=report_root,
        is_canonical_data_root=is_canonical,
        r_bounds_mm=r_bounds_mm,
    )


def add_identity_columns(
    df,
    *,
    train_filament_id: str,
    eval_filament_id: str,
    focal_mm: int,
    split_role: str,
):
    df = df.copy()
    df["train_filament_id"] = train_filament_id
    df["eval_filament_id"] = eval_filament_id
    df["focal_mm"] = int(focal_mm)
    df["split_role"] = split_role
    return df


def build_snapshot_payload(
    ctx: FilamentContext,
    *,
    split_role: Optional[str] = None,
    extra: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    payload: Dict[str, object] = {
        "filament_id": ctx.filament_id,
        "eval_filament_id": ctx.eval_filament_id,
        "focal_mm": ctx.focal_mm,
        "gt_diameter_um": ctx.gt_diameter_um,
        "data_root": ctx.data_root,
        "run_root": ctx.run_root,
        "report_root": ctx.report_root,
    }
    if split_role is not None:
        payload["split_role"] = split_role
    if extra:
        payload.update(extra)
    return payload


def known_filament_ids() -> Iterable[str]:
    return LEGACY_DATASETS.keys()


def iter_legacy_layout_rows() -> Iterable[Dict[str, object]]:
    for filament_id, by_focal in LEGACY_DATASETS.items():
        for focal_mm, info in by_focal.items():
            legacy_root = str(info["legacy_root"])
            canonical_root = canonical_data_root(filament_id, focal_mm)
            yield {
                "filament_id": filament_id,
                "focal_mm": focal_mm,
                "legacy_root": legacy_root,
                "canonical_root": canonical_root,
                "raw_source": legacy_root,
                "split_train_source": os.path.join(legacy_root, "train_real"),
                "split_val_source": os.path.join(legacy_root, "val_real"),
                "simulation_dataset_source": os.path.join(legacy_root, "simulation_dataset"),
                "cache_fft_source": os.path.join(legacy_root, "cache", "fft_tensors"),
            }
