# Path Configuration Guide

## IMPORTANT

The codebase contains **hardcoded paths**, which is the most common obstacle when reproducing results. Please read this guide carefully and modify the configurations to match your environment.

---

## Hardcoded Path Inventory

### 1. Core Path Configuration

**File:** `filament_layout.py` (line 9)

```python
PROJECT_ROOT = os.environ.get("FILAMENT_PROJECT_ROOT", "/root/autodl-tmp")
```

**Default:** `/root/autodl-tmp`
**Purpose:** Root directory for all data, weights, and outputs.

**Derived paths:**

- `DATA_ROOT_BASE = {PROJECT_ROOT}/data/filaments`
- `RUNS_ROOT_BASE = {PROJECT_ROOT}/runs`
- `REPORTS_ROOT_BASE = {PROJECT_ROOT}/reports`

---

### 2. Training Script Defaults

**Files:**

- `src/Code_75/core/main_75.py` (line 46)
- `src/Code_120/core/main_120.py` (line 46)

```python
LAYOUT = resolve_filament_context(
    default_data_root="/root/autodl-tmp/Diameter_100.2_75mm",
    default_gt=100.2,
    default_focal_mm=75,
    default_filament_id="FIL001",
)
```

**Note:** These defaults are overridden by environment variables and usually do not need to be changed.

---

### 3. Plotting Script Hardcoded Paths

**File:** `scripts/plot_network_full_analysis_nm.py` (lines 563-574)

```python
BASELINE_METRICS = {
    (75, 124): '/root/autodl-tmp/Code_75/results/seed_124/baseline/metrics.csv',
    (120, 124): '/root/autodl-tmp/Code_120/results/seed_124/baseline/metrics.csv',
    ...
}

ABLATION_ROOTS = {
    (75, 124): '/root/autodl-tmp/Code_75/results/seed_124',
    ...
}

out_dir = '/root/autodl-tmp/focal_length_comparison_outputs/figures'
tab_dir = '/root/autodl-tmp/focal_length_comparison_outputs/tables'
```

**These paths must be modified manually.**

---

## Configuration Methods

### Method 1: Set Environment Variable (Recommended)

This is the **simplest** approach and requires no code changes.

**Linux/Mac:**

```bash
# Temporary (current terminal session)
export FILAMENT_PROJECT_ROOT="/path/to/your/publish"

# Permanent (add to ~/.bashrc or ~/.zshrc)
echo 'export FILAMENT_PROJECT_ROOT="/path/to/your/publish"' >> ~/.bashrc
source ~/.bashrc
```

**Windows (PowerShell):**

```powershell
# Temporary
$env:FILAMENT_PROJECT_ROOT="D:\path\to\your\publish"

# Permanent (user-level environment variable)
[System.Environment]::SetEnvironmentVariable('FILAMENT_PROJECT_ROOT', 'D:\path\to\your\publish', 'User')
```

**Windows (CMD):**

```cmd
set FILAMENT_PROJECT_ROOT=D:\path\to\your\publish
```

---

### Method 2: Modify filament_layout.py (Once and For All)

**Steps:**

1. Open `filament_layout.py`
2. Find line 9:

   ```python
   PROJECT_ROOT = os.environ.get("FILAMENT_PROJECT_ROOT", "/root/autodl-tmp")
   ```

3. Change the default value to your actual path:

   ```python
   PROJECT_ROOT = os.environ.get("FILAMENT_PROJECT_ROOT", "/your/actual/path/publish")
   ```

**Examples:**

- Linux: `"/home/username/projects/diffraction-metrology"`
- Windows: `"D:/Projects/diffraction-metrology"` (use forward slashes)
- Mac: `"/Users/username/research/diffraction-metrology"`

---

### Method 3: Modify Plotting Scripts (Required)

The plotting scripts contain paths that **must be changed manually** because they do not use environment variables.

**Files to modify:**

#### `scripts/plot_network_full_analysis_nm.py`

Find lines 563-574 and update to your actual paths:

```python
# Original
BASELINE_METRICS = {
    (75, 124): '/root/autodl-tmp/Code_75/results/seed_124/baseline/metrics.csv',
    ...
}

# Change to
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

out_dir = '/your/path/publish/figures'
tab_dir = '/your/path/publish/tables'
```

#### `scripts/unified_mismatch_analysis.py`

If this script contains hardcoded paths, update them in the same way.

---

## Standard Directory Structure

After configuration, your directory layout should look like this:

```
/your/path/publish/                    # <- PROJECT_ROOT
├── data/
│   └── filaments/                     # <- DATA_ROOT_BASE
│       ├── FIL001/
│       │   ├── focal_75mm/
│       │   │   └── raw/*.BMP
│       │   └── focal_120mm/
│       │       └── raw/*.BMP
│       └── FIL002/
│           └── ...
├── runs/                              # <- RUNS_ROOT_BASE
│   ├── train_FIL001/
│   │   ├── focal_75mm/
│   │   │   ├── baseline/
│   │   │   │   ├── seed_124/
│   │   │   │   │   ├── best_ema_model.pth
│   │   │   │   │   └── metrics.csv
│   │   │   │   └── seed_212/
│   │   │   ├── ablation/
│   │   │   └── probe/
│   │   └── focal_120mm/
│   │       └── ...
│   └── train_FIL002/
│       └── ...
├── reports/                           # <- REPORTS_ROOT_BASE
├── figures/                           # Plot output
├── tables/                            # Table output
├── weights/                           # Pre-trained weights (optional)
└── src/
    ├── Code_75/
    ├── Code_120/
    └── filament_layout.py
```

---

## Verify Configuration

Run the following script to check that paths are configured correctly:

```python
# verify_paths.py
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from filament_layout import (
    PROJECT_ROOT,
    DATA_ROOT_BASE,
    RUNS_ROOT_BASE,
    canonical_data_root,
    canonical_run_root,
)

print("=" * 60)
print("Path Configuration Verification")
print("=" * 60)
print(f"PROJECT_ROOT:     {PROJECT_ROOT}")
print(f"DATA_ROOT_BASE:   {DATA_ROOT_BASE}")
print(f"RUNS_ROOT_BASE:   {RUNS_ROOT_BASE}")
print()
print("FIL001 @ 75mm:")
print(f"  Data root:  {canonical_data_root('FIL001', 75)}")
print(f"  Runs root:  {canonical_run_root('FIL001', 75)}")
print()
print("Checking whether directories exist:")
print(f"  DATA_ROOT_BASE exists: {os.path.exists(DATA_ROOT_BASE)}")
print(f"  RUNS_ROOT_BASE exists: {os.path.exists(RUNS_ROOT_BASE)}")
print()

# Check raw data
raw_dir = os.path.join(canonical_data_root('FIL001', 75), 'raw')
if os.path.exists(raw_dir):
    bmp_files = [f for f in os.listdir(raw_dir) if f.endswith('.BMP')]
    print(f"Found {len(bmp_files)} raw images")
else:
    print(f"Raw data directory not found: {raw_dir}")

print("=" * 60)
```

**Run:**

```bash
cd /your/path/publish
python verify_paths.py
```

**Expected output:**

```
============================================================
Path Configuration Verification
============================================================
PROJECT_ROOT:     /your/path/publish
DATA_ROOT_BASE:   /your/path/publish/data/filaments
RUNS_ROOT_BASE:   /your/path/publish/runs

FIL001 @ 75mm:
  Data root:  /your/path/publish/data/filaments/FIL001/focal_75mm
  Runs root:  /your/path/publish/runs/train_FIL001/focal_75mm

Checking whether directories exist:
  DATA_ROOT_BASE exists: True
  RUNS_ROOT_BASE exists: True

Found 10 raw images
============================================================
```

---

## Common Issues

### Issue 1: FileNotFoundError: No such file or directory

**Cause:** Incorrect path configuration or data not placed in the expected location.

**Solution:**

1. Run `verify_paths.py` to inspect paths
2. Confirm the `FILAMENT_PROJECT_ROOT` environment variable is set
3. Check that data files are in `{PROJECT_ROOT}/data/filaments/FIL001/focal_75mm/raw/`

---

### Issue 2: Training script cannot find data

**Cause:** The default path in `main_75.py` was not overridden by environment variables.

**Solution:**

```bash
# Explicitly set all relevant environment variables
export FILAMENT_PROJECT_ROOT="/your/path/publish"
export FILAMENT_ID="FIL001"
export FOCAL_MM="75"
export GT_DIAMETER_UM="100.2"

cd src/Code_75/experiments
python run_all_experiments.py
```

---

### Issue 3: Plotting script cannot find metrics.csv

**Cause:** Hardcoded paths in the plotting script have not been updated.

**Solution:**

1. Open `scripts/plot_network_full_analysis_nm.py`
2. Update the paths on lines 563-574 to match your environment
3. Alternatively, use command-line arguments if the script supports them

---

### Issue 4: Windows path separator problems

**Cause:** Windows uses backslashes `\`, but Python may interpret them as escape characters.

**Solution:**

```python
# Wrong
PROJECT_ROOT = "D:\Projects\publish"  # \P is treated as an escape sequence

# Correct (option 1: forward slashes)
PROJECT_ROOT = "D:/Projects/publish"

# Correct (option 2: raw string)
PROJECT_ROOT = r"D:\Projects\publish"

# Correct (option 3: escaped backslashes)
PROJECT_ROOT = "D:\\Projects\\publish"
```

---

## Quick Configuration Checklist

Before starting training or plotting, confirm:

- [ ] Set the `FILAMENT_PROJECT_ROOT` environment variable, or edited `filament_layout.py`
- [ ] Ran `verify_paths.py` and confirmed paths are correct
- [ ] Raw data is in `{PROJECT_ROOT}/data/filaments/FIL001/focal_75mm/raw/`
- [ ] If plotting, updated hardcoded paths in `scripts/plot_network_full_analysis_nm.py`
- [ ] If using pre-trained weights, placed `.pth` files in the correct location

---

## Best Practice Recommendations

1. **Use environment variables**: set `FILAMENT_PROJECT_ROOT` permanently in `~/.bashrc` or `~/.zshrc`
2. **Create a setup script**: put all environment variables in `setup_env.sh` and `source` it before each session
3. **Use absolute paths**: avoid confusion caused by relative paths
4. **Version control**: do not commit modified paths to Git (add them to `.gitignore`)

**Example setup script:**

```bash
# setup_env.sh
#!/bin/bash

# Set project root
export FILAMENT_PROJECT_ROOT="/home/username/research/diffraction-metrology"

# Set experiment parameters
export FILAMENT_ID="FIL001"
export FOCAL_MM="75"
export GT_DIAMETER_UM="100.2"

# Set training parameters
export SEED="124"
export NUM_EPOCHS="60"
export ALLOW_TRAIN="1"

echo "Environment variables set:"
echo "  PROJECT_ROOT: $FILAMENT_PROJECT_ROOT"
echo "  FILAMENT_ID:  $FILAMENT_ID"
echo "  FOCAL_MM:     $FOCAL_MM"
```

**Usage:**

```bash
source setup_env.sh
cd src/Code_75/experiments
python run_all_experiments.py
```

---

## Need Help?

If path issues persist:

1. Run `verify_paths.py` and share the output
2. Check the `PROJECT_ROOT` value on line 9 of `filament_layout.py`
3. Confirm your operating system and Python version
4. Provide the full error message and stack trace

---

**Last Updated:** 2026-05-01
