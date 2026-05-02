#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fix_hardcoded_paths.py

Automatically fix hardcoded paths in plotting scripts

Usage:
    python fix_hardcoded_paths.py /your/path/to/publish
"""

import os
import sys
import re
from pathlib import Path


def fix_plot_network_full_analysis(file_path, project_root):
    """Fix plot_network_full_analysis_nm.py"""
    print(f"\nFixing: {file_path}")

    if not os.path.exists(file_path):
        print(f"  FAIL File not found: {file_path}")
        return False

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content

    # 1. Add PROJECT_ROOT definition (after imports at the top of the file)
    if 'PROJECT_ROOT = os.environ.get' not in content:
        # Find the position of the last import statement
        import_lines = []
        lines = content.split('\n')
        last_import_idx = 0
        for i, line in enumerate(lines):
            if line.strip().startswith('import ') or line.strip().startswith('from '):
                last_import_idx = i

        # Insert PROJECT_ROOT definition after the last import
        lines.insert(last_import_idx + 1, '')
        lines.insert(last_import_idx + 2, f'# Project root for path resolution')
        lines.insert(last_import_idx + 3, f'PROJECT_ROOT = os.environ.get("FILAMENT_PROJECT_ROOT", "{project_root}")')
        lines.insert(last_import_idx + 4, '')
        content = '\n'.join(lines)

    # 2. Replace BASELINE_METRICS
    baseline_pattern = r'BASELINE_METRICS\s*=\s*\{[^}]+\}'
    new_baseline = """BASELINE_METRICS = {
        (75, 124): f'{PROJECT_ROOT}/runs/train_FIL001/focal_75mm/baseline/seed_124/metrics.csv',
        (120, 124): f'{PROJECT_ROOT}/runs/train_FIL001/focal_120mm/baseline/seed_124/metrics.csv',
        (75, 212): f'{PROJECT_ROOT}/runs/train_FIL001/focal_75mm/baseline/seed_212/metrics.csv',
        (120, 212): f'{PROJECT_ROOT}/runs/train_FIL001/focal_120mm/baseline/seed_212/metrics.csv',
    }"""
    content = re.sub(baseline_pattern, new_baseline, content, flags=re.DOTALL)

    # 3. Replace ABLATION_ROOTS
    ablation_pattern = r'ABLATION_ROOTS\s*=\s*\{[^}]+\}'
    new_ablation = """ABLATION_ROOTS = {
        (75, 124): f'{PROJECT_ROOT}/runs/train_FIL001/focal_75mm/ablation/seed_124',
        (120, 124): f'{PROJECT_ROOT}/runs/train_FIL001/focal_120mm/ablation/seed_124',
        (75, 212): f'{PROJECT_ROOT}/runs/train_FIL001/focal_75mm/ablation/seed_212',
        (120, 212): f'{PROJECT_ROOT}/runs/train_FIL001/focal_120mm/ablation/seed_212',
    }"""
    content = re.sub(ablation_pattern, new_ablation, content, flags=re.DOTALL)

    # 4. Replace out_dir
    content = re.sub(
        r"out_dir\s*=\s*['\"][^'\"]+['\"]",
        "out_dir = f'{PROJECT_ROOT}/figures'",
        content
    )

    # 5. Replace tab_dir
    content = re.sub(
        r"tab_dir\s*=\s*['\"][^'\"]+['\"]",
        "tab_dir = f'{PROJECT_ROOT}/tables'",
        content
    )

    if content != original_content:
        # Backup original file
        backup_path = file_path + '.backup'
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(original_content)
        print(f"  OK Backup created: {backup_path}")

        # Write modified content
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  OK Successfully fixed")
        return True
    else:
        print(f"  WARN  No changes needed")
        return True


def fix_plot_network_probe_results(file_path, project_root):
    """Fix plot_network_probe_results_nm.py"""
    print(f"\nFixing: {file_path}")

    if not os.path.exists(file_path):
        print(f"  FAIL File not found: {file_path}")
        return False

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content

    # 1. Add PROJECT_ROOT definition
    if 'PROJECT_ROOT = os.environ.get' not in content:
        import_lines = []
        lines = content.split('\n')
        last_import_idx = 0
        for i, line in enumerate(lines):
            if line.strip().startswith('import ') or line.strip().startswith('from '):
                last_import_idx = i

        lines.insert(last_import_idx + 1, '')
        lines.insert(last_import_idx + 2, f'# Project root for path resolution')
        lines.insert(last_import_idx + 3, f'PROJECT_ROOT = os.environ.get("FILAMENT_PROJECT_ROOT", "{project_root}")')
        lines.insert(last_import_idx + 4, '')
        content = '\n'.join(lines)

    # 2. Replace p120 and p75
    content = re.sub(
        r"p120\s*=\s*['\"][^'\"]+['\"]",
        "p120 = f'{PROJECT_ROOT}/runs/train_FIL001/focal_120mm/probe_statistics/detailed_results.json'",
        content
    )
    content = re.sub(
        r"p75\s*=\s*['\"][^'\"]+['\"]",
        "p75 = f'{PROJECT_ROOT}/runs/train_FIL001/focal_75mm/probe_statistics/detailed_results.json'",
        content
    )

    # 3. Replace out_dir
    content = re.sub(
        r"out_dir\s*=\s*['\"][^'\"]+['\"]",
        "out_dir = f'{PROJECT_ROOT}/figures'",
        content
    )

    # 4. Replace tab_dir
    content = re.sub(
        r"tab_dir\s*=\s*['\"][^'\"]+['\"]",
        "tab_dir = f'{PROJECT_ROOT}/tables'",
        content
    )

    if content != original_content:
        # Backup original file
        backup_path = file_path + '.backup'
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(original_content)
        print(f"  OK Backup created: {backup_path}")

        # Write modified content
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  OK Successfully fixed")
        return True
    else:
        print(f"  WARN  No changes needed")
        return True


def main():
    print("=" * 70)
    print("Hardcoded Path Auto-Fix Tool")
    print("=" * 70)

    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  python fix_hardcoded_paths.py <project_root>")
        print("\nExample:")
        print("  python fix_hardcoded_paths.py /home/user/publish")
        print("  python fix_hardcoded_paths.py D:/Projects/publish")
        print("\nOr use current directory:")
        print("  python fix_hardcoded_paths.py .")
        sys.exit(1)

    project_root = sys.argv[1]

    # Handle relative paths
    if project_root == '.':
        project_root = os.getcwd()
    else:
        project_root = os.path.abspath(project_root)

    # Convert to forward slashes (cross-platform compatibility)
    project_root = project_root.replace('\\', '/')

    print(f"\nPROJECT_ROOT will be set to:")
    print(f"  {project_root}")

    # Check if directory exists
    if not os.path.exists(project_root):
        print(f"\nFAIL ERROR: Directory does not exist")
        sys.exit(1)

    script_dir = os.path.join(project_root, 'scripts')
    if not os.path.exists(script_dir):
        print(f"\nFAIL ERROR: scripts directory not found")
        print(f"  Expected: {script_dir}")
        sys.exit(1)

    print("\nStarting fixes...")
    print("-" * 70)

    # Fix both plotting scripts
    success = True
    success &= fix_plot_network_full_analysis(
        os.path.join(script_dir, 'plot_network_full_analysis_nm.py'),
        project_root
    )
    success &= fix_plot_network_probe_results(
        os.path.join(script_dir, 'plot_network_probe_results_nm.py'),
        project_root
    )

    print("\n" + "=" * 70)
    if success:
        print("OK All hardcoded paths have been fixed!")
        print("\nNext steps:")
        print("  1. Verify paths:")
        print("     python verify_paths.py")
        print("  2. Set environment variable:")
        print(f"     export FILAMENT_PROJECT_ROOT=\"{project_root}\"")
        print("  3. Run plotting scripts:")
        print("     python scripts/plot_network_full_analysis_nm.py")
    else:
        print("FAIL Some files failed to fix")
        print("Please check error messages and fix manually")
        sys.exit(1)

    print("=" * 70)


if __name__ == "__main__":
    main()
