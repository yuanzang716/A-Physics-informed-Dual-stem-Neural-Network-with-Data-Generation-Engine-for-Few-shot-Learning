# Complete Hardcoded Path Inventory

This document lists **ALL** hardcoded paths in the codebase for users to reference when reproducing results.

---

## Summary Table

| Priority | File | Line(s) | Hardcoded Content | Must Modify? | How to Fix |
|----------|------|---------|-------------------|--------------|------------|
| HIGH | `filament_layout.py` | 9 | `PROJECT_ROOT = "/root/autodl-tmp"` | No (use env var) | Set `FILAMENT_PROJECT_ROOT` environment variable |
| MEDIUM | `Code_75/core/main_75.py` | 46 | `default_data_root="/root/autodl-tmp/Diameter_100.2_75mm"` | No (overridden by env var) | Usually no change needed |
| MEDIUM | `Code_120/core/main_120.py` | 46 | `default_data_root="/root/autodl-tmp/Diameter_100.2_120mm"` | No (overridden by env var) | Usually no change needed |
| HIGH | `scripts/plot_network_full_analysis_nm.py` | 563-574 | `BASELINE_METRICS` and `ABLATION_ROOTS` dicts | **Yes** | Manually update to your paths |
| HIGH | `scripts/plot_network_full_analysis_nm.py` | 650, 657 | `out_dir` and `tab_dir` | **Yes** | Manually update to your paths |
| HIGH | `scripts/plot_network_probe_results_nm.py` | 121-122 | `p120` and `p75` paths | **Yes** | Manually update to your paths |
| HIGH | `scripts/plot_network_probe_results_nm.py` | 202, 209 | `out_dir` and `tab_dir` | **Yes** | Manually update to your paths |

---

## Detailed Explanation

### 1. `filament_layout.py` (line 9)

**Original code:**
```python
PROJECT_ROOT = os.environ.get("FILAMENT_PROJECT_ROOT", "/root/autodl-tmp")
```

**Notes:**
- This is the **core path configuration**; all other paths are derived from it
- It reads the `FILAMENT_PROJECT_ROOT` environment variable first
- Falls back to `/root/autodl-tmp` if the variable is not set

**How to fix (pick one):**

**Option 1: Set the environment variable (recommended)**
```bash
export FILAMENT_PROJECT_ROOT="/your/path/to/publish"
```

**Option 2: Change the default value**
```python
PROJECT_ROOT = os.environ.get("FILAMENT_PROJECT_ROOT", "/your/path/to/publish")
```

**Option 3: Hardcode directly (not recommended)**
```python
PROJECT_ROOT = "/your/path/to/publish"
```

---

### 2. `Code_75/core/main_75.py` and `Code_120/core/main_120.py` (line 46)

**Original code:**
```python
LAYOUT = resolve_filament_context(
    default_data_root="/root/autodl-tmp/Diameter_100.2_75mm",  # or 120mm
    default_gt=100.2,
    default_focal_mm=75,  # or 120
    default_filament_id="FIL001",
)
```

**Notes:**
- These are **fallback defaults**, overridden by environment variables
- `run_all_experiments.py` sets the correct environment variables automatically
- **Usually no change needed**

**If you do need to change them:**
```python
LAYOUT = resolve_filament_context(
    default_data_root=os.path.join(PROJECT_ROOT, "data/filaments/FIL001/focal_75mm"),
    default_gt=100.2,
    default_focal_mm=75,
    default_filament_id="FIL001",
)
```

---

### 3. `scripts/plot_network_full_analysis_nm.py` (lines 563-574)

**Original code:**
```python
BASELINE_METRICS = {
    (75, 124): '/root/autodl-tmp/Code_75/results/seed_124/baseline/metrics.csv',
    (120, 124): '/root/autodl-tmp/Code_120/results/seed_124/baseline/metrics.csv',
    (75, 212): '/root/autodl-tmp/Code_75/results/seed_212/baseline/metrics.csv',
    (120, 212): '/root/autodl-tmp/Code_120/results/seed_212/baseline/metrics.csv',
}

ABLATION_ROOTS = {
    (75, 124): '/root/autodl-tmp/Code_75/results/seed_124',
    (120, 124): '/root/autodl-tmp/Code_120/results/seed_124',
    (75, 212): '/root/autodl-tmp/Code_75/results/seed_212',
    (120, 212): '/root/autodl-tmp/Code_120/results/seed_212',
}
```

**This must be changed manually.**

**Updated version (assuming PROJECT_ROOT = /your/path/publish):**
```python
# Option 1: Absolute paths
BASELINE_METRICS = {
    (75, 124): '/your/path/publish/runs/train_FIL001/focal_75mm/baseline/seed_124/metrics.csv',
    (120, 124): '/your/path/publish/runs/train_FIL001/focal_120mm/baseline/seed_124/metrics.csv',
    (75, 212): '/your/path/publish/runs/train_FIL001/focal_75mm/baseline/seed_212/metrics.csv',
    (120, 212): '/your/path/publish/runs/train_FIL001/focal_120mm/baseline/seed_212/metrics.csv',
}

ABLATION_ROOTS = {
    (75, 124): '/your/path/publish/runs/train_FIL001/focal_75mm/ablation/seed_124',
    (120, 124): '/your/path/publish/runs/train_FIL001/focal_120mm/ablation/seed_124',
    (75, 212): '/your/path/publish/runs/train_FIL001/focal_75mm/ablation/seed_212',
    (120, 212): '/your/path/publish/runs/train_FIL001/focal_120mm/ablation/seed_212',
}

# Option 2: Environment variable (recommended)
import os
PROJECT_ROOT = os.environ.get("FILAMENT_PROJECT_ROOT", "/root/autodl-tmp")

BASELINE_METRICS = {
    (75, 124): f'{PROJECT_ROOT}/runs/train_FIL001/focal_75mm/baseline/seed_124/metrics.csv',
    (120, 124): f'{PROJECT_ROOT}/runs/train_FIL001/focal_120mm/baseline/seed_124/metrics.csv',
    (75, 212): f'{PROJECT_ROOT}/runs/train_FIL001/focal_75mm/baseline/seed_212/metrics.csv',
    (120, 212): f'{PROJECT_ROOT}/runs/train_FIL001/focal_120mm/baseline/seed_212/metrics.csv',
}

ABLATION_ROOTS = {
    (75, 124): f'{PROJECT_ROOT}/runs/train_FIL001/focal_75mm/ablation/seed_124',
    (120, 124): f'{PROJECT_ROOT}/runs/train_FIL001/focal_120mm/ablation/seed_124',
    (75, 212): f'{PROJECT_ROOT}/runs/train_FIL001/focal_75mm/ablation/seed_212',
    (120, 212): f'{PROJECT_ROOT}/runs/train_FIL001/focal_120mm/ablation/seed_212',
}
```

---

### 4. `scripts/plot_network_full_analysis_nm.py` (lines 650, 657)

**Original code:**
```python
out_dir = '/root/autodl-tmp/focal_length_comparison_outputs/figures'
# ...
tab_dir = '/root/autodl-tmp/focal_length_comparison_outputs/tables'
```

**Updated version:**
```python
# Option 1: Absolute paths
out_dir = '/your/path/publish/figures'
tab_dir = '/your/path/publish/tables'

# Option 2: Environment variable (recommended)
import os
PROJECT_ROOT = os.environ.get("FILAMENT_PROJECT_ROOT", "/root/autodl-tmp")
out_dir = f'{PROJECT_ROOT}/figures'
tab_dir = f'{PROJECT_ROOT}/tables'

# Option 3: Relative to script location
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # assumes script is in scripts/
out_dir = os.path.join(PROJECT_ROOT, 'figures')
tab_dir = os.path.join(PROJECT_ROOT, 'tables')
```

---

### 5. `scripts/plot_network_probe_results_nm.py` (lines 121-122)

**Original code:**
```python
p120 = '/root/autodl-tmp/Code_120/results/probe_statistics/detailed_results.json'
p75 = '/root/autodl-tmp/Code_75/results/probe_statistics/detailed_results.json'
```

**Updated version:**
```python
import os
PROJECT_ROOT = os.environ.get("FILAMENT_PROJECT_ROOT", "/root/autodl-tmp")

p120 = f'{PROJECT_ROOT}/runs/train_FIL001/focal_120mm/probe_statistics/detailed_results.json'
p75 = f'{PROJECT_ROOT}/runs/train_FIL001/focal_75mm/probe_statistics/detailed_results.json'
```

---

### 6. `scripts/plot_network_probe_results_nm.py` (lines 202, 209)

**Original code:**
```python
out_dir = '/root/autodl-tmp/focal_length_comparison_outputs/figures'
# ...
tab_dir = '/root/autodl-tmp/focal_length_comparison_outputs/tables'
```

**Updated version:**
```python
import os
PROJECT_ROOT = os.environ.get("FILAMENT_PROJECT_ROOT", "/root/autodl-tmp")
out_dir = f'{PROJECT_ROOT}/figures'
tab_dir = f'{PROJECT_ROOT}/tables'
```

---

## Batch Modification Script

A convenience script is provided to apply all path fixes automatically:

```python
#!/usr/bin/env python3
# fix_hardcoded_paths.py
"""Automatically fix hardcoded paths in plotting scripts."""

import os
import sys
import re

def fix_plot_network_full_analysis(file_path, project_root):
    """Fix paths in plot_network_full_analysis_nm.py."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Add PROJECT_ROOT definition
    if 'PROJECT_ROOT = os.environ.get' not in content:
        import_section = 'import os\n'
        if 'import os' in content:
            content = content.replace(
                'import os',
                f'import os\n\nPROJECT_ROOT = os.environ.get("FILAMENT_PROJECT_ROOT", "{project_root}")'
            )
    
    # Replace BASELINE_METRICS
    old_pattern = r"BASELINE_METRICS = \{[^}]+\}"
    new_metrics = """BASELINE_METRICS = {
    (75, 124): f'{PROJECT_ROOT}/runs/train_FIL001/focal_75mm/baseline/seed_124/metrics.csv',
    (120, 124): f'{PROJECT_ROOT}/runs/train_FIL001/focal_120mm/baseline/seed_124/metrics.csv',
    (75, 212): f'{PROJECT_ROOT}/runs/train_FIL001/focal_75mm/baseline/seed_212/metrics.csv',
    (120, 212): f'{PROJECT_ROOT}/runs/train_FIL001/focal_120mm/baseline/seed_212/metrics.csv',
}"""
    content = re.sub(old_pattern, new_metrics, content, flags=re.DOTALL)
    
    # Replace ABLATION_ROOTS
    old_pattern = r"ABLATION_ROOTS = \{[^}]+\}"
    new_roots = """ABLATION_ROOTS = {
    (75, 124): f'{PROJECT_ROOT}/runs/train_FIL001/focal_75mm/ablation/seed_124',
    (120, 124): f'{PROJECT_ROOT}/runs/train_FIL001/focal_120mm/ablation/seed_124',
    (75, 212): f'{PROJECT_ROOT}/runs/train_FIL001/focal_75mm/ablation/seed_212',
    (120, 212): f'{PROJECT_ROOT}/runs/train_FIL001/focal_120mm/ablation/seed_212',
}"""
    content = re.sub(old_pattern, new_roots, content, flags=re.DOTALL)
    
    # Replace output directories
    content = re.sub(
        r"out_dir = '[^']*'",
        "out_dir = f'{PROJECT_ROOT}/figures'",
        content
    )
    content = re.sub(
        r"tab_dir = '[^']*'",
        "tab_dir = f'{PROJECT_ROOT}/tables'",
        content
    )
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Fixed: {file_path}")

def fix_plot_network_probe_results(file_path, project_root):
    """Fix paths in plot_network_probe_results_nm.py."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Add PROJECT_ROOT definition
    if 'PROJECT_ROOT = os.environ.get' not in content:
        if 'import os' in content:
            content = content.replace(
                'import os',
                f'import os\n\nPROJECT_ROOT = os.environ.get("FILAMENT_PROJECT_ROOT", "{project_root}")'
            )
    
    # Replace probe data paths
    content = re.sub(
        r"p120 = '[^']*'",
        "p120 = f'{PROJECT_ROOT}/runs/train_FIL001/focal_120mm/probe_statistics/detailed_results.json'",
        content
    )
    content = re.sub(
        r"p75 = '[^']*'",
        "p75 = f'{PROJECT_ROOT}/runs/train_FIL001/focal_75mm/probe_statistics/detailed_results.json'",
        content
    )
    
    # Replace output directories
    content = re.sub(
        r"out_dir = '[^']*'",
        "out_dir = f'{PROJECT_ROOT}/figures'",
        content
    )
    content = re.sub(
        r"tab_dir = '[^']*'",
        "tab_dir = f'{PROJECT_ROOT}/tables'",
        content
    )
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Fixed: {file_path}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python fix_hardcoded_paths.py <project_root>")
        print("Example: python fix_hardcoded_paths.py /home/user/publish")
        sys.exit(1)
    
    project_root = sys.argv[1]
    script_dir = os.path.join(project_root, 'scripts')
    
    print(f"Fixing hardcoded paths with PROJECT_ROOT = {project_root}")
    print("=" * 60)
    
    # Fix both plotting scripts
    fix_plot_network_full_analysis(
        os.path.join(script_dir, 'plot_network_full_analysis_nm.py'),
        project_root
    )
    fix_plot_network_probe_results(
        os.path.join(script_dir, 'plot_network_probe_results_nm.py'),
        project_root
    )
    
    print("=" * 60)
    print("All hardcoded paths have been fixed!")
    print("\nNext steps:")
    print("  1. Verify: python verify_paths.py")
    print("  2. Run plotting scripts")

if __name__ == "__main__":
    main()
```

**Usage:**
```bash
python fix_hardcoded_paths.py /your/path/to/publish
```

---

## Verification Checklist

After making changes, confirm the following:

- [ ] Set the `FILAMENT_PROJECT_ROOT` environment variable, or edited `filament_layout.py`
- [ ] `python verify_paths.py` runs without errors
- [ ] Updated all hardcoded paths in `plot_network_full_analysis_nm.py`
- [ ] Updated all hardcoded paths in `plot_network_probe_results_nm.py`
- [ ] Test-ran a plotting script and confirmed the output directory is correct

---

## Need Help?

If path issues persist after making changes:
1. Run `python verify_paths.py` and review the output
2. Check that the environment variable is set correctly: `echo $FILAMENT_PROJECT_ROOT`
3. Confirm file permissions: `ls -la /your/path/publish`
4. Include the full error message and stack trace when reporting issues

---

**Last Updated:** 2026-05-01
