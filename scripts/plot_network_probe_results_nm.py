#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import sys
import warnings
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore', category=RuntimeWarning)


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from filament_layout import canonical_report_root, report_path


def set_pub_style():
    plt.rcParams.update({
        'font.size': 8,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
        'axes.linewidth': 1.0,
        'axes.edgecolor': '#333333',
        'axes.labelcolor': '#333333',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.labelsize': 8,
        'axes.titlesize': 9,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.size': 3.0,
        'ytick.major.size': 3.0,
        'xtick.major.width': 1.0,
        'ytick.major.width': 1.0,
        'xtick.color': '#333333',
        'ytick.color': '#333333',
        'legend.fontsize': 7,
        'legend.frameon': False,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'figure.dpi': 300,
        'savefig.dpi': 600,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.03,
        'lines.linewidth': 1.0,
        'patch.linewidth': 1.0,
        'axes.unicode_minus': False,
    })
    plt.rcParams['figure.facecolor'] = 'white'


PALETTE = {
    'blue': '#2E86AB',
    'orange': '#D55E00',
    'gray': '#4D4D4D',
    'lightgray': '#9CA3AF',
}


def read_json(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def json_to_df(d: Dict[str, Any]) -> pd.DataFrame:
    rows = d.get('results', [])
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Normalize types
    for c in ['seed', 'epoch']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Relative error (%) from obs error fraction
    if 'val_ema_err_obs' in df.columns:
        df['rel_err_ema_pct'] = 100.0 * pd.to_numeric(df['val_ema_err_obs'], errors='coerce')
    if 'val_err_obs' in df.columns:
        df['rel_err_raw_pct'] = 100.0 * pd.to_numeric(df['val_err_obs'], errors='coerce')

    # Bias already in um
    if 'bias_um' in df.columns:
        df['bias_um'] = pd.to_numeric(df['bias_um'], errors='coerce')

    if 'val_pred_std' in df.columns:
        df['val_pred_std'] = pd.to_numeric(df['val_pred_std'], errors='coerce')

    if 'val_ema_score' in df.columns:
        df['val_ema_score'] = pd.to_numeric(df['val_ema_score'], errors='coerce')

    if 'is_success' in df.columns:
        df['is_success'] = df['is_success'].astype(bool)

    return df


def summarize(d: Dict[str, Any]) -> Tuple[str, float]:
    cfg = d.get('config', {})
    gt = float(cfg.get('gt_diameter_um', cfg.get('GT_DIAMETER', np.nan)))
    succ_th = float(cfg.get('success_threshold', cfg.get('SUCCESS_THRESHOLD', np.nan)))
    title = f"GT={gt:.1f} µm, success ≤ {100*succ_th:.1f}%"
    return title, gt


def add_panel_label(ax, label: str):
    ax.text(-0.16, 1.10, label, transform=ax.transAxes, fontsize=11,
            fontweight='bold', va='top', ha='left')


def _strip_x(ax):
    ax.tick_params(axis='x', which='both', bottom=True, top=False)


def resolve_probe_json(train_filament_id: str, eval_filament_id: str, focal_mm: int) -> str:
    candidates = [
        report_path(train_filament_id, 'probe_detailed', focal_mm, eval_filament_id, 'probe', 'json'),
        os.path.join(canonical_report_root(train_filament_id), f'probe_statistics_focal_{focal_mm}mm_internal_eval_{train_filament_id}.json'),
        os.path.join(PROJECT_ROOT, f'Code_{focal_mm}', 'results', 'probe_statistics', 'detailed_results.json'),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f'No probe statistics JSON found for focal={focal_mm}mm, train={train_filament_id}, eval={eval_filament_id}')


def main():
    set_pub_style()

    train_filament_id = os.environ.get('TRAIN_FILAMENT_ID', os.environ.get('FILAMENT_ID', 'FIL001'))
    eval_filament_id = os.environ.get('EVAL_FILAMENT_ID', train_filament_id)

    p120 = resolve_probe_json(train_filament_id, eval_filament_id, 120)
    p75 = resolve_probe_json(train_filament_id, eval_filament_id, 75)

    d120 = read_json(p120)
    d75 = read_json(p75)

    df120 = json_to_df(d120)
    df75 = json_to_df(d75)

    title120, _ = summarize(d120)
    title75, _ = summarize(d75)

    # NM double-column max width ~183mm ≈ 7.2in
    fig = plt.figure(figsize=(7.2, 4.2))
    gs = fig.add_gridspec(1, 2, wspace=0.35, left=0.10, right=0.98, top=0.92, bottom=0.18)

    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])

    def plot_seed_summary(ax, df: pd.DataFrame, payload: Dict[str, Any], color: str, title: str):
        ax.set_title(title)

        if df is None or df.empty:
            ax.text(0.5, 0.5, 'No probe statistics found', ha='center', va='center')
            ax.set_xticks([])
            ax.set_yticks([])
            return

        # Sort by seed for stable x
        df = df.sort_values('seed')
        x = np.arange(len(df))

        y = df['rel_err_ema_pct'].to_numpy(dtype=float)
        gt = float(payload.get('config', {}).get('gt_diameter_um', payload.get('config', {}).get('GT_DIAMETER', np.nan)))
        if 'val_pred_std' in df.columns and np.isfinite(gt) and gt > 0:
            yerr = 100.0 * df['val_pred_std'].to_numpy(dtype=float) / gt
        else:
            yerr = None

        # Success threshold line
        cfg = payload.get('config', {})
        succ_th = float(cfg.get('success_threshold', cfg.get('SUCCESS_THRESHOLD', 0.01)))
        ax.axhline(100.0 * succ_th, color='#111111', linestyle='--', linewidth=1.0, alpha=0.9)

        # Scatter by success
        succ = df['is_success'].to_numpy(dtype=bool) if 'is_success' in df.columns else np.ones_like(y, dtype=bool)

        ax.scatter(x[succ], y[succ], s=26, color=color, alpha=0.95, edgecolors='white', linewidths=0.3, zorder=3, label='success')
        ax.scatter(x[~succ], y[~succ], s=26, color=color, alpha=0.25, edgecolors='white', linewidths=0.3, zorder=3, label='fail')

        # Median + IQR (robust summary)
        med = float(np.nanmedian(y))
        q25 = float(np.nanquantile(y, 0.25))
        q75 = float(np.nanquantile(y, 0.75))
        ax.axhline(med, color=color, linewidth=1.4, alpha=0.95)
        ax.fill_between([-0.5, len(x) - 0.5], [q25, q25], [q75, q75], color=color, alpha=0.15, linewidth=0)

        ax.set_xlim(-0.6, len(x) - 0.4)
        ax.set_ylim(bottom=0)
        ax.set_xlabel('Random seed')
        ax.set_ylabel('Relative error (%)')

        # x tick labels: seed values
        ax.set_xticks(x)
        ax.set_xticklabels([str(int(s)) for s in df['seed'].to_numpy()], rotation=45, ha='right')

        _strip_x(ax)
        ax.grid(True, axis='y', alpha=0.18, linewidth=0.6)

        # Compact legend explaining elements
        handles = [
            plt.Line2D([0], [0], marker='o', color='none', markerfacecolor=color, markeredgecolor='white', markeredgewidth=0.3, markersize=6, label='seed (success)'),
            plt.Line2D([0], [0], marker='o', color='none', markerfacecolor=color, alpha=0.25, markeredgecolor='white', markeredgewidth=0.3, markersize=6, label='seed (fail)'),
            plt.Line2D([0, 1], [0, 0], color=color, linewidth=1.4, label='median'),
            plt.Rectangle((0, 0), 1, 1, facecolor=color, alpha=0.15, edgecolor='none', label='IQR'),
            plt.Line2D([0, 1], [0, 0], color='#111111', linestyle='--', linewidth=1.0, label='success threshold'),
        ]
        ax.legend(handles=handles, loc='upper left')

    plot_seed_summary(axA, df75, d75, PALETTE['blue'], f"75 mm\n{title75}")
    plot_seed_summary(axB, df120, d120, PALETTE['orange'], f"120 mm\n{title120}")

    add_panel_label(axA, 'A')
    add_panel_label(axB, 'B')

    out_dir = '/root/autodl-tmp/focal_length_comparison_outputs/figures'
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f'Fig_results_network_seed_robustness_train_{train_filament_id}_eval_{eval_filament_id}')
    fig.savefig(out + '.pdf')
    fig.savefig(out + '.png')

    # Source data export
    tab_dir = '/root/autodl-tmp/focal_length_comparison_outputs/tables'
    os.makedirs(tab_dir, exist_ok=True)
    df75.to_csv(os.path.join(tab_dir, f'source_data_probe_stats_75mm_train_{train_filament_id}_eval_{eval_filament_id}.csv'), index=False)
    df120.to_csv(os.path.join(tab_dir, f'source_data_probe_stats_120mm_train_{train_filament_id}_eval_{eval_filament_id}.csv'), index=False)

    print('Saved:')
    print(' ', out + '.pdf')
    print(' ', out + '.png')
    print('Source data:')
    print(' ', os.path.join(tab_dir, f'source_data_probe_stats_75mm_train_{train_filament_id}_eval_{eval_filament_id}.csv'))
    print(' ', os.path.join(tab_dir, f'source_data_probe_stats_120mm_train_{train_filament_id}_eval_{eval_filament_id}.csv'))


if __name__ == '__main__':
    main()
