#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import warnings
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore', category=RuntimeWarning)


def set_pub_style():
    plt.rcParams.update({
        'font.size': 7,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial'],
        'axes.linewidth': 1.0,
        'axes.edgecolor': '#333333',
        'axes.labelcolor': '#333333',
        'axes.spines.top': True,
        'axes.spines.right': True,
        'axes.labelsize': 7,
        'axes.titlesize': 8,
        'axes.titlepad': 4,
        'xtick.labelsize': 6,
        'ytick.labelsize': 6,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.size': 3.0,
        'ytick.major.size': 3.0,
        'xtick.major.width': 1.0,
        'ytick.major.width': 1.0,
        'legend.fontsize': 6,
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
    # Okabe–Ito colorblind-safe palette (Nature Methods-friendly)
    # https://jfly.uni-koeln.de/color/
    'blue': '#0072B2',        # blue
    'blue_light': '#56B4E9',  # sky blue
    'orange': '#D55E00',      # vermillion (use for 120mm)
    'orange_light': '#E69F00',
    'gray': '#4D4D4D',
    'lightgray': '#B0B0B0',
    'black': '#111111',
}


LABEL_MAP_ABLATION = {
    'E00_baseline': 'Full protocol (baseline)',

    # Training schedule
    'E01_no_early_pair': 'No early pair-only phase',
    'E02_no_gradual_intro': 'No gradual introduction',
    'E03_no_conditional': 'No conditional activation',
    'E04_no_adaptive_boost': 'No adaptive pair boosting',

    # Objectives / physics
    'E05_full_phys_weight': 'Full physics weight',
    'E06_no_physics': 'No physics consistency',
    'E07_phys_pair_only': 'Physics + pair only',
    'E08_phys_only': 'Physics only',
    'E09_pure_pinn_style': 'GPINN (single-stem)',

    # Inputs / architecture
    'E10_no_fft': 'No Fourier input',
    'E11_use_diff_input': 'With residual input',
    'E12_single_branch': 'Single-stem (shared)',

    # Data efficiency
    'E13_sim_samples_300': '300 sims per real',
    'E14_sim_samples_600': '600 sims per real',
}


def add_panel_label(ax, label: str):
    ax.text(-0.16, 1.10, label.lower(), transform=ax.transAxes, fontsize=8,
            fontweight='bold', va='top', ha='left')


def _style_ax(ax, grid_axis: str = 'y'):
    ax.set_axisbelow(True)
    if grid_axis:
        ax.grid(True, axis=grid_axis, alpha=0.18, linewidth=0.6)
    ax.tick_params(length=3.0, width=1.0)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color('#333333')


def read_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if 'epoch' in df.columns:
        df['epoch'] = pd.to_numeric(df['epoch'], errors='coerce')
    return df


def read_json(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _infer_best_epoch(df: pd.DataFrame, prefer_ema: bool = True, which: str = 'last') -> Optional[int]:
    """Infer best epoch from flags.

    which:
      - 'last': use the last occurrence (matches "last best" checkpoint semantics)
      - 'first': use the first occurrence
    """
    if df is None or df.empty:
        return None

    col = 'is_best_ema' if prefer_ema and 'is_best_ema' in df.columns else 'is_best_model'
    if col not in df.columns:
        return None

    hits = df.loc[pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int) == 1]
    if hits.empty:
        return None

    if which == 'first':
        return int(hits['epoch'].iloc[0])
    return int(hits['epoch'].iloc[-1])


def _plot_loss_and_score(ax_loss, ax_score, df: pd.DataFrame, title: str, color: str):
    """Two y-axes per subplot.

    Left y-axis: original curve (train loss / selection score)
    Right y-axis: relative error to GT (%)
    """
    if df is None or df.empty:
        ax_loss.text(0.5, 0.5, 'No metrics.csv', ha='center', va='center')
        ax_score.text(0.5, 0.5, 'No metrics.csv', ha='center', va='center')
        return

    x = df['epoch'].to_numpy(dtype=float)

    # Right-axis: relative error to GT (%)
    # We plot both the instantaneous model error (non-EMA) and EMA error when available.
    err_ema = None
    if 'val_ema_err_obs' in df.columns:
        err_ema = 100.0 * pd.to_numeric(df['val_ema_err_obs'], errors='coerce').to_numpy(dtype=float)

    err_raw = None
    if 'val_err_obs' in df.columns:
        err_raw = 100.0 * pd.to_numeric(df['val_err_obs'], errors='coerce').to_numpy(dtype=float)

    best_ema_ep = _infer_best_epoch(df, prefer_ema=True, which='last')
    best_runtime_ep = _infer_best_epoch(df, prefer_ema=False, which='last')

    # ---- Loss subplot (left: train loss, right: rel err) ----
    loss = None
    if 'train_loss_total' in df.columns:
        loss = pd.to_numeric(df['train_loss_total'], errors='coerce').to_numpy(dtype=float)
        ax_loss.plot(x, loss, color=color, linewidth=1.6)

    ax_loss.set_title(title)
    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('Train loss')
    _style_ax(ax_loss, grid_axis='y')

    ax_loss_r = ax_loss.twinx()
    err_color = PALETTE['blue_light'] if color == PALETTE['blue'] else (PALETTE['orange_light'] if color == PALETTE['orange'] else PALETTE['lightgray'])
    # Right-axis styling: keep the dataset hue but differentiate EMA vs non-EMA by linestyle.
    if err_raw is not None:
        ax_loss_r.plot(x, err_raw, color=color, linestyle='--', linewidth=1.2, alpha=0.55, label='non-EMA error')
    if err_ema is not None:
        ax_loss_r.plot(x, err_ema, color=color, linestyle='-.', linewidth=1.6, alpha=0.90, label='EMA error')
    ax_loss_r.set_ylabel('Rel. err to GT (%)')
    ax_loss_r.tick_params(length=3.0, width=1.0)
    ax_loss_r.spines['right'].set_linewidth(1.0)
    ax_loss_r.spines['right'].set_color('#333333')

    # Mark epochs on both axes with vertical lines
    def _mark_vline(ep: Optional[int], marker_color: str, ls: str):
        if ep is None:
            return
        ax_loss.axvline(ep, color=marker_color, linestyle=ls, linewidth=1.0, alpha=0.90)
        ax_loss_r.axvline(ep, color=marker_color, linestyle=ls, linewidth=1.0, alpha=0.90)

    _mark_vline(best_ema_ep, PALETTE['black'], '--')
    _mark_vline(best_runtime_ep, PALETTE['gray'], ':')

    # ---- Score subplot (left: selection score, right: rel err) ----
    score = None
    if 'val_ema_score' in df.columns:
        score = pd.to_numeric(df['val_ema_score'], errors='coerce').to_numpy(dtype=float)
        ax_score.plot(x, score, color=color, linewidth=1.6)

    ax_score.set_xlabel('Epoch')
    ax_score.set_ylabel('Selection score')
    _style_ax(ax_score, grid_axis='y')

    ax_score_r = ax_score.twinx()
    err_color = PALETTE['blue_light'] if color == PALETTE['blue'] else (PALETTE['orange_light'] if color == PALETTE['orange'] else PALETTE['lightgray'])
    # Right-axis styling: keep the dataset hue but differentiate EMA vs non-EMA by linestyle.
    if err_raw is not None:
        ax_score_r.plot(x, err_raw, color=color, linestyle='--', linewidth=1.2, alpha=0.55, label='non-EMA error')
    if err_ema is not None:
        ax_score_r.plot(x, err_ema, color=color, linestyle='-.', linewidth=1.6, alpha=0.90, label='EMA error')
    ax_score_r.set_ylabel('Rel. err to GT (%)')
    ax_score_r.tick_params(length=3.0, width=1.0)
    ax_score_r.spines['right'].set_linewidth(1.0)
    ax_score_r.spines['right'].set_color('#333333')

    def _mark_vline2(ep: Optional[int], marker_color: str, ls: str):
        if ep is None:
            return
        ax_score.axvline(ep, color=marker_color, linestyle=ls, linewidth=1.0, alpha=0.90)
        ax_score_r.axvline(ep, color=marker_color, linestyle=ls, linewidth=1.0, alpha=0.90)

    _mark_vline2(best_ema_ep, PALETTE['black'], '--')
    _mark_vline2(best_runtime_ep, PALETTE['gray'], ':')

    # Legend (only once on lower axis)
    handles = [
        plt.Line2D([0, 1], [0, 0], color=color, linewidth=1.6, label='Left axis (loss/score)'),
        plt.Line2D([0, 1], [0, 0], color=color, linestyle='-.', linewidth=1.6, alpha=0.90, label='Right axis: EMA error'),
        plt.Line2D([0, 1], [0, 0], color=color, linestyle='--', linewidth=1.2, alpha=0.55, label='Right axis: non-EMA error'),
        plt.Line2D([0, 1], [0, 0], color=PALETTE['black'], linestyle='--', linewidth=1.0, label='Best EMA checkpoint'),
        plt.Line2D([0, 1], [0, 0], color=PALETTE['gray'], linestyle=':', linewidth=1.0, label='Best non-EMA checkpoint'),
    ]
    ax_score.legend(handles=handles, loc='best')


def _load_ablation_summary(seed_root_75: str, seed_root_120: str) -> pd.DataFrame:
    """Load ablation results for plotting.

    NOTE:
      - Do NOT rely on ablation_summary.json being up-to-date.
      - Instead, scan the ablation_sweep directory for E*/metrics.csv.
      - If ablation_summary.json exists, use it only to enrich human-readable descriptions.
    """
    rows: List[Dict[str, Any]] = []

    for focal, root in [(75, seed_root_75), (120, seed_root_120)]:
        sweep_root = os.path.join(root, 'ablation_sweep')
        if not os.path.isdir(sweep_root):
            continue

        # Optional desc map from ablation_summary.json
        desc_map: Dict[str, str] = {}
        p = os.path.join(sweep_root, 'ablation_summary.json')
        if os.path.exists(p):
            try:
                d = read_json(p)
                for cfg in d.get('configs', []) or []:
                    name = cfg.get('name')
                    if not name:
                        continue
                    desc_map[str(name)] = str(cfg.get('desc', name))
            except Exception:
                desc_map = {}

        # Scan all experiments under ablation_sweep
        for name in sorted(os.listdir(sweep_root)):
            if not (isinstance(name, str) and name.startswith('E')):
                continue
            mpath = os.path.join(sweep_root, name, 'metrics.csv')
            if not os.path.exists(mpath):
                continue

            df = read_csv(mpath)
            best_ep = _infer_best_epoch(df, prefer_ema=True)
            if best_ep is None:
                continue
            row = df.loc[df['epoch'] == best_ep]
            if row.empty:
                continue

            r = {
                'focal_mm': focal,
                'config': name,
                'desc': desc_map.get(name, name),
                'best_epoch': int(best_ep),
                'val_ema_err_obs': float(pd.to_numeric(row.get('val_ema_err_obs', np.nan), errors='coerce').iloc[0]) if 'val_ema_err_obs' in row.columns else np.nan,
                'val_ema_score': float(pd.to_numeric(row.get('val_ema_score', np.nan), errors='coerce').iloc[0]) if 'val_ema_score' in row.columns else np.nan,
            }
            rows.append(r)

    out = pd.DataFrame(rows)
    if not out.empty and 'val_ema_err_obs' in out.columns:
        # rel_err_ema_pct = 100 * |pred - GT| / GT (computed during evaluation; stored as val_ema_err_obs)
        out['rel_err_ema_pct'] = 100.0 * out['val_ema_err_obs']

    return out


# --- Modified to add baseline vertical lines

def _plot_ablation(ax, df_ab: pd.DataFrame, title: str):
    ax.set_title(title)

    if df_ab is None or df_ab.empty:
        ax.text(0.5, 0.5, 'No ablation summary found', ha='center', va='center')
        ax.set_xticks([])
        ax.set_yticks([])
        return

    # Identify baseline errors for vertical reference lines
    baseline75 = df_ab[(df_ab['focal_mm'] == 75) & (df_ab['config'].str.contains('baseline', case=False, regex=False))]
    baseline120 = df_ab[(df_ab['focal_mm'] == 120) & (df_ab['config'].str.contains('baseline', case=False, regex=False))]
    base_val_75 = float(baseline75['rel_err_ema_pct'].iloc[0]) if not baseline75.empty else None
    base_val_120 = float(baseline120['rel_err_ema_pct'].iloc[0]) if not baseline120.empty else None

    # --- Grouped label design (Nature Methods-ish): group headers + spacing ---
    GROUPS: List[Tuple[str, List[str]]] = [
        ('Training schedule', [
            'E00_baseline',
            'E01_no_early_pair',
            'E02_no_gradual_intro',
            'E03_no_conditional',
            'E04_no_adaptive_boost',
        ]),
        ('Physics / objectives', [
            'E05_full_phys_weight',
            'E06_no_physics',
            'E07_phys_pair_only',
            'E08_phys_only',
            'E09_pure_pinn_style',
        ]),
        ('Inputs / architecture', [
            'E10_no_fft',
            'E11_use_diff_input',
            'E12_single_branch',
        ]),
        ('Local simulation bank size', [
            'E13_sim_samples_300',
            'E14_sim_samples_600',
        ]),
    ]

    # Build the final y-order: header row + items + spacer
    present = set(df_ab['config'].astype(str).unique().tolist())

    ordered: List[Tuple[str, str]] = []  # (kind, key)
    for header, items in GROUPS:
        kept = [c for c in items if c in present]
        if not kept:
            continue
        ordered.append(('header', header))
        for c in kept:
            ordered.append(('item', c))
        ordered.append(('spacer', ''))

    # Add any leftover configs not covered by GROUPS
    covered = set([k for kind, k in ordered if kind == 'item'])
    leftovers = sorted([c for c in present if c not in covered])
    if leftovers:
        ordered.append(('header', 'Other'))
        for c in leftovers:
            ordered.append(('item', c))

    # Remove trailing spacer if exists
    while ordered and ordered[-1][0] == 'spacer':
        ordered.pop()

    bar_height = 0.48
    y_pos = np.arange(len(ordered))

    # Data maps
    data75 = df_ab[df_ab['focal_mm'] == 75].set_index('config')
    data120 = df_ab[df_ab['focal_mm'] == 120].set_index('config')

    err75 = []
    err120 = []
    y_tick_labels = []
    y_tick_pos = []

    for i, (kind, key) in enumerate(ordered):
        if kind == 'item':
            err75.append(float(data75.loc[key, 'rel_err_ema_pct']) if key in data75.index else np.nan)
            err120.append(float(data120.loc[key, 'rel_err_ema_pct']) if key in data120.index else np.nan)
            y_tick_pos.append(i)
            y_tick_labels.append(LABEL_MAP_ABLATION.get(key, key.replace('_', ' ')))
        else:
            err75.append(np.nan)
            err120.append(np.nan)

    # Draw bars only for items
    for i, (kind, key) in enumerate(ordered):
        if kind != 'item':
            continue
        v75 = err75[i]
        v120 = err120[i]
        if np.isfinite(v75):
            ax.barh(i + bar_height / 2, v75, height=bar_height, color=PALETTE['blue'], edgecolor=PALETTE['black'], linewidth=0.6)
        if np.isfinite(v120):
            ax.barh(i - bar_height / 2, v120, height=bar_height, color=PALETTE['orange'], edgecolor=PALETTE['black'], linewidth=0.6)

    # Baseline reference lines
    if base_val_75 is not None:
        ax.axvline(base_val_75, color=PALETTE['blue_light'], linestyle='--', linewidth=1.0, label='baseline 75mm')
    if base_val_120 is not None:
        ax.axvline(base_val_120, color=PALETTE['orange_light'], linestyle='--', linewidth=1.0, label='baseline 120mm')

    # Group header text + separator lines
    for i, (kind, key) in enumerate(ordered):
        if kind == 'header':
            ax.text(-0.02, i, key, transform=ax.get_yaxis_transform(),
                    ha='right', va='center', fontsize=7, fontweight='bold', color=PALETTE['gray'])
            ax.axhline(i - 0.5, color=PALETTE['lightgray'], linewidth=0.6, alpha=0.6)
        elif kind == 'spacer':
            ax.axhline(i - 0.5, color=PALETTE['lightgray'], linewidth=0.6, alpha=0.25)

    ax.set_yticks(y_tick_pos)
    ax.set_yticklabels(y_tick_labels)
    ax.invert_yaxis()
    ax.set_xlabel('Relative error to GT (%)')
    _style_ax(ax, grid_axis='x')

    # Legend: focal length colors + baseline refs
    handles = [
        plt.Line2D([0, 1], [0, 0], color=PALETTE['blue'], linewidth=6, label='75mm'),
        plt.Line2D([0, 1], [0, 0], color=PALETTE['orange'], linewidth=6, label='120mm'),
    ]
    if base_val_75 is not None:
        handles.append(plt.Line2D([0, 1], [0, 0], color=PALETTE['blue_light'], linestyle='--', linewidth=1.0, label='baseline 75mm'))
    if base_val_120 is not None:
        handles.append(plt.Line2D([0, 1], [0, 0], color=PALETTE['orange_light'], linestyle='--', linewidth=1.0, label='baseline 120mm'))
    ax.legend(handles=handles, loc='lower right')


def _load_sensitivity(seed_root_75: str, seed_root_120: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for focal, root in [(75, seed_root_75), (120, seed_root_120)]:
        sens_root = os.path.join(root, 'sensitivity_analysis')
        if not os.path.isdir(sens_root):
            continue
        for cond in sorted(os.listdir(sens_root)):
            mpath = os.path.join(sens_root, cond, 'metrics.csv')
            if not os.path.exists(mpath):
                continue
            df = read_csv(mpath)
            best_ep = _infer_best_epoch(df, prefer_ema=True)
            if best_ep is None:
                continue
            row = df.loc[df['epoch'] == best_ep]
            if row.empty:
                continue
            rows.append({
                'focal_mm': focal,
                'condition': cond,
                'best_epoch': int(best_ep),
                'val_ema_err_obs': float(pd.to_numeric(row.get('val_ema_err_obs', np.nan), errors='coerce').iloc[0]) if 'val_ema_err_obs' in row.columns else np.nan,
            })

    out = pd.DataFrame(rows)
    if not out.empty:
        out['rel_err_ema_pct'] = 100.0 * out['val_ema_err_obs']
    return out


# --- Modified to plot delta vs baseline and use box/bar

def _plot_sensitivity(ax, df_sens: pd.DataFrame, title: str):
    ax.set_title(title)

    if df_sens is None or df_sens.empty:
        ax.text(0.5, 0.5, 'No sensitivity results found', ha='center', va='center')
        ax.set_xticks([])
        ax.set_yticks([])
        return

    # Compute baseline values for delta calculation
    base75 = df_sens[(df_sens['focal_mm'] == 75) & (df_sens['condition'] == 'baseline')]
    base120 = df_sens[(df_sens['focal_mm'] == 120) & (df_sens['condition'] == 'baseline')]
    base_val_75 = float(base75['rel_err_ema_pct'].iloc[0]) if not base75.empty else np.nan
    base_val_120 = float(base120['rel_err_ema_pct'].iloc[0]) if not base120.empty else np.nan

    # Remove baseline row from plotting (we only keep deltas)
    df_plot = df_sens[df_sens['condition'] != 'baseline'].copy()
    if df_plot.empty:
        ax.text(0.5, 0.5, 'No non-baseline sensitivity', ha='center', va='center')
        return

    # Calculate delta vs baseline (percentage points)
    def _delta(row):
        base = base_val_75 if row['focal_mm'] == 75 else base_val_120
        return row['rel_err_ema_pct'] - base

    df_plot['delta_pp'] = df_plot.apply(_delta, axis=1)

    # Sort conditions
    conds_sorted = sorted(df_plot['condition'].unique())
    x = np.arange(len(conds_sorted))
    bar_width = 0.36

    data75 = df_plot[df_plot['focal_mm'] == 75].set_index('condition')
    data120 = df_plot[df_plot['focal_mm'] == 120].set_index('condition')

    y75 = [float(data75.loc[c, 'delta_pp']) if c in data75.index else np.nan for c in conds_sorted]
    y120 = [float(data120.loc[c, 'delta_pp']) if c in data120.index else np.nan for c in conds_sorted]

    ax.bar(x - bar_width/2, y75, width=bar_width, color=PALETTE['blue'], label='75mm')
    ax.bar(x + bar_width/2, y120, width=bar_width, color=PALETTE['orange'], label='120mm')

    ax.axhline(0, color=PALETTE['lightgray'], linewidth=1.0)

    ax.set_xticks(x)
    ax.set_xticklabels([c.replace('_', '\n') for c in conds_sorted], rotation=0, ha='center')
    ax.set_ylabel('Δ error vs baseline (p.p.)')
    _style_ax(ax, grid_axis='y')
    ax.legend(loc='best')
    ax.set_title(title)

    if df_sens is None or df_sens.empty:
        ax.text(0.5, 0.5, 'No sensitivity results found', ha='center', va='center')
        ax.set_xticks([])
        ax.set_yticks([])
        return

    order = ['baseline', 'params_zero', 'coarse_perturb_minus3pct', 'coarse_perturb_plus3pct', 'image_noise_std001']
    conds = df_sens['condition'].unique().tolist()
    conds_sorted = [c for c in order if c in conds] + [c for c in sorted(conds) if c not in order]

    x = np.arange(len(conds_sorted))
    bar_width = 0.36

    data75 = df_sens[df_sens['focal_mm'] == 75].set_index('condition')
    data120 = df_sens[df_sens['focal_mm'] == 120].set_index('condition')

    y75 = [float(data75.loc[c, 'rel_err_ema_pct']) if c in data75.index else np.nan for c in conds_sorted]
    y120 = [float(data120.loc[c, 'rel_err_ema_pct']) if c in data120.index else np.nan for c in conds_sorted]

    ax.bar(x - bar_width/2, y75, width=bar_width, color=PALETTE['blue'], label='75mm')
    ax.bar(x + bar_width/2, y120, width=bar_width, color=PALETTE['orange'], label='120mm')

    ax.set_xticks(x)
    ax.set_xticklabels([c.replace('_', '\n') for c in conds_sorted], rotation=0, ha='center')
    ax.set_ylabel('Relative error to GT (%)')
    _style_ax(ax, grid_axis='y')
    ax.legend(loc='best')


def main():
    set_pub_style()

    seeds = [124, 212]

    # Baseline metrics for loss/score curves
    baselines = {
        (75, 124): '/root/autodl-tmp/Code_75/results/seed_124/baseline/metrics.csv',
        (120, 124): '/root/autodl-tmp/Code_120/results/seed_124/baseline/metrics.csv',
        (75, 212): '/root/autodl-tmp/Code_75/results/seed_212/baseline/metrics.csv',
        (120, 212): '/root/autodl-tmp/Code_120/results/seed_212/baseline/metrics.csv',
    }

    # Seed roots for ablation/sensitivity
    seed_roots = {
        (75, 124): '/root/autodl-tmp/Code_75/results/seed_124',
        (120, 124): '/root/autodl-tmp/Code_120/results/seed_124',
        (75, 212): '/root/autodl-tmp/Code_75/results/seed_212',
        (120, 212): '/root/autodl-tmp/Code_120/results/seed_212',
    }

    # Figure: two-column width, multi-panel but compact.
    # Layout: 3 rows x 2 cols
    # Row1: loss (75/120) for seed_124
    # Row2: loss (75/120) for seed_212
    # Row3: ablation summary (seed_124) spanning both columns
    fig = plt.figure(figsize=(7.09, 7.6))
    gs = fig.add_gridspec(3, 2, hspace=0.42, wspace=0.28, left=0.10, right=0.98, top=0.95, bottom=0.07)

    # Two rows: each row has two columns (75mm left, 120mm right), each column has two stacked axes (loss+score)
    def make_nested(cell):
        return gs[cell].subgridspec(2, 1, hspace=0.15)

    # Seed 124
    g00 = make_nested((0, 0))
    axA1 = fig.add_subplot(g00[0, 0])
    axA2 = fig.add_subplot(g00[1, 0], sharex=axA1)

    g01 = make_nested((0, 1))
    axB1 = fig.add_subplot(g01[0, 0])
    axB2 = fig.add_subplot(g01[1, 0], sharex=axB1)

    # Seed 212
    g10 = make_nested((1, 0))
    axC1 = fig.add_subplot(g10[0, 0])
    axC2 = fig.add_subplot(g10[1, 0], sharex=axC1)

    g11 = make_nested((1, 1))
    axD1 = fig.add_subplot(g11[0, 0])
    axD2 = fig.add_subplot(g11[1, 0], sharex=axD1)

    # Ablation for seed 124 (summary across conditions)
    axE = fig.add_subplot(gs[2, :])

    # Plot curves
    df_75_124 = read_csv(baselines[(75, 124)])
    df_120_124 = read_csv(baselines[(120, 124)])
    df_75_212 = read_csv(baselines[(75, 212)])
    df_120_212 = read_csv(baselines[(120, 212)])

    _plot_loss_and_score(axA1, axA2, df_75_124, 'seed 124 (75 mm)', PALETTE['blue'])
    _plot_loss_and_score(axB1, axB2, df_120_124, 'seed 124 (120 mm)', PALETTE['orange'])
    _plot_loss_and_score(axC1, axC2, df_75_212, 'seed 212 (75 mm)', PALETTE['blue'])
    _plot_loss_and_score(axD1, axD2, df_120_212, 'seed 212 (120 mm)', PALETTE['orange'])

    add_panel_label(axA1, 'A')
    add_panel_label(axB1, 'B')
    add_panel_label(axC1, 'C')
    add_panel_label(axD1, 'D')

    # Ablation / sensitivity from seed 124 (you can extend to seed 212 if needed)
    df_ab = _load_ablation_summary(seed_roots[(75, 124)], seed_roots[(120, 124)])
    _plot_ablation(axE, df_ab, 'Ablation (best checkpoint error)')
    add_panel_label(axE, 'E')

    # NOTE: keep a deterministic y-order so newly added experiments (e.g., E12/E13)
    # appear in a predictable position.
    if df_ab is not None and not df_ab.empty:
        def _cfg_key(c: str) -> Tuple[int, str]:
            # Sort by numeric experiment id if possible: E00, E01, ...
            try:
                if isinstance(c, str) and c.startswith('E'):
                    n = int(c.split('_')[0][1:])
                    return (n, c)
            except Exception:
                pass
            return (10**9, str(c))

        df_ab = df_ab.copy()
        df_ab['__cfg_key'] = df_ab['config'].map(lambda x: _cfg_key(str(x)))
        df_ab = df_ab.sort_values(['__cfg_key', 'focal_mm']).drop(columns=['__cfg_key'])



    out_dir = '/root/autodl-tmp/focal_length_comparison_outputs/figures'
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, 'Fig_results_network_training_ablation_sensitivity')
    fig.savefig(out + '.pdf')
    fig.savefig(out + '.png')

    # Source data export
    tab_dir = '/root/autodl-tmp/focal_length_comparison_outputs/tables'
    os.makedirs(tab_dir, exist_ok=True)
    df_75_124.to_csv(os.path.join(tab_dir, 'source_data_seed124_75mm_baseline_metrics.csv'), index=False)
    df_120_124.to_csv(os.path.join(tab_dir, 'source_data_seed124_120mm_baseline_metrics.csv'), index=False)
    df_75_212.to_csv(os.path.join(tab_dir, 'source_data_seed212_75mm_baseline_metrics.csv'), index=False)
    df_120_212.to_csv(os.path.join(tab_dir, 'source_data_seed212_120mm_baseline_metrics.csv'), index=False)
    df_ab.to_csv(os.path.join(tab_dir, 'source_data_seed124_ablation_summary.csv'), index=False)


    print('Saved:')
    print(' ', out + '.pdf')
    print(' ', out + '.png')
    print('Source data:')
    print(' ', os.path.join(tab_dir, 'source_data_seed124_75mm_baseline_metrics.csv'))
    print(' ', os.path.join(tab_dir, 'source_data_seed124_120mm_baseline_metrics.csv'))
    print(' ', os.path.join(tab_dir, 'source_data_seed212_75mm_baseline_metrics.csv'))
    print(' ', os.path.join(tab_dir, 'source_data_seed212_120mm_baseline_metrics.csv'))
    print(' ', os.path.join(tab_dir, 'source_data_seed124_ablation_summary.csv'))
    print(' ', os.path.join(tab_dir, 'source_data_seed124_sensitivity_summary.csv'))


if __name__ == '__main__':
    main()

