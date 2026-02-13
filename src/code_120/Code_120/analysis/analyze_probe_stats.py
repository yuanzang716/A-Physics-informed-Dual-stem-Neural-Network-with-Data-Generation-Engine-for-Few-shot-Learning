#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_probe_stats.py

统计分析脚本：从probe实验结果中提取统计指标

功能：
1. 提取所有probe实验的指标（EMA误差、预测均值、偏差等）
2. 计算成功率、失败率
3. 计算统计量（mean, std, median, min, max, quartiles）
4. 计算最大偏差和方差分布
5. 生成统计摘要表格（CSV和JSON）

输出：
- probe_statistics/summary_stats.csv：统计摘要表格
- probe_statistics/detailed_results.json：详细结果
- probe_statistics/metrics_per_seed.csv：每个种子的详细指标
"""

import os
import sys
import json
import csv
import glob
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from pathlib import Path


# ==========================
# 配置
# ==========================
CONFIG = {
    "OUTPUT_ROOT": os.path.join(os.path.dirname(os.path.dirname(__file__)), "results"),  # 结果保存到new_strategy/results（与run_all_experiments.py一致）
    "GT_DIAMETER": 100.2,  # Ground truth diameter (um)
    "SUCCESS_THRESHOLD": 0.01,  # 成功率阈值：err < 1%
    "OUTPUT_DIR": None,  # 自动设置为 OUTPUT_ROOT/probe_statistics
}


def to_float(x: Any, default: float = float('nan')) -> float:
    """安全转换为float"""
    try:
        return float(x)
    except (ValueError, TypeError):
        return default


def load_metrics_csv(csv_path: str) -> Optional[Dict[str, Any]]:
    """从metrics.csv中提取最佳EMA模型指标"""
    if not os.path.exists(csv_path):
        return None
    
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            return None
        
        # 找到val_ema_score最小的行（最佳EMA模型）
        if 'val_ema_score' not in df.columns:
            return None
        
        df['val_ema_score'] = pd.to_numeric(df['val_ema_score'], errors='coerce')
        best_row = df.loc[df['val_ema_score'].idxmin()]
        
        # 提取关键指标
        result = {
            'epoch': int(best_row.get('epoch', -1)),
            'val_ema_err_obs': to_float(best_row.get('val_ema_err_obs')),
            'val_ema_score': to_float(best_row.get('val_ema_score')),
            'val_ema_pred_mean': to_float(best_row.get('val_pred_mean')),  # EMA预测均值
            'val_err_obs': to_float(best_row.get('val_err_obs')),
            'val_score': to_float(best_row.get('val_score')),
            'val_pred_mean': to_float(best_row.get('val_pred_mean')),
            'val_pred_std': to_float(best_row.get('val_pred_std')),
            'train_loss': to_float(best_row.get('train_loss')),
        }
        
        # 计算偏差
        if not np.isnan(result['val_ema_pred_mean']):
            result['bias_um'] = abs(result['val_ema_pred_mean'] - CONFIG['GT_DIAMETER'])
        else:
            result['bias_um'] = float('nan')
        
        # 判断是否成功
        result['is_success'] = (
            not np.isnan(result['val_ema_err_obs']) and 
            result['val_ema_err_obs'] < CONFIG['SUCCESS_THRESHOLD']
        )
        
        return result
    except Exception as e:
        print(f"警告：无法读取 {csv_path}: {e}")
        return None


def collect_all_probe_results() -> List[Dict[str, Any]]:
    """收集所有probe实验的结果"""
    output_root = CONFIG["OUTPUT_ROOT"]
    results = []
    
    # 查找所有seed_XX/probe目录
    seed_dirs = glob.glob(os.path.join(output_root, "seed_*"))
    
    for seed_dir in seed_dirs:
        seed_name = os.path.basename(seed_dir)
        try:
            seed = int(seed_name.split('_')[1])
        except (ValueError, IndexError):
            continue
        
        probe_dir = os.path.join(seed_dir, "probe")
        metrics_csv = os.path.join(probe_dir, "metrics.csv")
        
        if not os.path.exists(metrics_csv):
            continue
        
        metrics = load_metrics_csv(metrics_csv)
        if metrics is None:
            continue
        
        metrics['seed'] = seed
        metrics['probe_dir'] = probe_dir
        results.append(metrics)
    
    return sorted(results, key=lambda x: x['seed'])


def calculate_statistics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """计算统计量"""
    if not results:
        return {}
    
    # 提取关键指标
    ema_errors = [r['val_ema_err_obs'] for r in results if not np.isnan(r['val_ema_err_obs'])]
    biases = [r['bias_um'] for r in results if not np.isnan(r['bias_um'])]
    ema_scores = [r['val_ema_score'] for r in results if not np.isnan(r['val_ema_score'])]
    successes = [r['is_success'] for r in results]
    
    stats = {}
    
    # EMA误差统计
    if ema_errors:
        stats['ema_error'] = {
            'mean': float(np.mean(ema_errors)),
            'std': float(np.std(ema_errors)),
            'median': float(np.median(ema_errors)),
            'min': float(np.min(ema_errors)),
            'max': float(np.max(ema_errors)),
            'q25': float(np.percentile(ema_errors, 25)),
            'q75': float(np.percentile(ema_errors, 75)),
            'n': len(ema_errors),
        }
    
    # 偏差统计
    if biases:
        stats['bias'] = {
            'mean': float(np.mean(biases)),
            'std': float(np.std(biases)),
            'median': float(np.median(biases)),
            'min': float(np.min(biases)),
            'max': float(np.max(biases)),
            'q25': float(np.percentile(biases, 25)),
            'q75': float(np.percentile(biases, 75)),
            'n': len(biases),
        }
    
    # EMA score统计
    if ema_scores:
        stats['ema_score'] = {
            'mean': float(np.mean(ema_scores)),
            'std': float(np.std(ema_scores)),
            'median': float(np.median(ema_scores)),
            'min': float(np.min(ema_scores)),
            'max': float(np.max(ema_scores)),
            'n': len(ema_scores),
        }
    
    # 成功率统计
    if successes:
        n_success = sum(successes)
        n_total = len(successes)
        stats['success_rate'] = {
            'n_success': n_success,
            'n_total': n_total,
            'rate': float(n_success / n_total) if n_total > 0 else 0.0,
            'failure_rate': float((n_total - n_success) / n_total) if n_total > 0 else 0.0,
        }
    
    return stats


def save_results(results: List[Dict[str, Any]], stats: Dict[str, Any], output_dir: str):
    """保存结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 保存详细结果（JSON）
    detailed_path = os.path.join(output_dir, "detailed_results.json")
    with open(detailed_path, 'w', encoding='utf-8') as f:
        json.dump({
            'config': CONFIG,
            'statistics': stats,
            'results': results,
        }, f, indent=2, ensure_ascii=False)
    
    # 2. 保存每个种子的指标（CSV）
    metrics_path = os.path.join(output_dir, "metrics_per_seed.csv")
    if results:
        df = pd.DataFrame(results)
        df.to_csv(metrics_path, index=False)
    
    # 3. 保存统计摘要（CSV）
    summary_path = os.path.join(output_dir, "summary_stats.csv")
    summary_rows = []
    
    # EMA误差统计
    if 'ema_error' in stats:
        summary_rows.append({
            'metric': 'EMA Relative Error',
            'mean': stats['ema_error']['mean'],
            'std': stats['ema_error']['std'],
            'median': stats['ema_error']['median'],
            'min': stats['ema_error']['min'],
            'max': stats['ema_error']['max'],
            'q25': stats['ema_error']['q25'],
            'q75': stats['ema_error']['q75'],
            'n': stats['ema_error']['n'],
        })
    
    # 偏差统计
    if 'bias' in stats:
        summary_rows.append({
            'metric': 'Absolute Bias (um)',
            'mean': stats['bias']['mean'],
            'std': stats['bias']['std'],
            'median': stats['bias']['median'],
            'min': stats['bias']['min'],
            'max': stats['bias']['max'],
            'q25': stats['bias']['q25'],
            'q75': stats['bias']['q75'],
            'n': stats['bias']['n'],
        })
    
    # EMA score统计
    if 'ema_score' in stats:
        summary_rows.append({
            'metric': 'EMA Score',
            'mean': stats['ema_score']['mean'],
            'std': stats['ema_score']['std'],
            'median': stats['ema_score']['median'],
            'min': stats['ema_score']['min'],
            'max': stats['ema_score']['max'],
            'q25': None,
            'q75': None,
            'n': stats['ema_score']['n'],
        })
    
    # 成功率统计
    if 'success_rate' in stats:
        summary_rows.append({
            'metric': 'Success Rate',
            'mean': stats['success_rate']['rate'],
            'std': None,
            'median': None,
            'min': None,
            'max': None,
            'q25': None,
            'q75': None,
            'n': stats['success_rate']['n_total'],
        })
    
    if summary_rows:
        df_summary = pd.DataFrame(summary_rows)
        df_summary.to_csv(summary_path, index=False)
    
    print(f"\n✅ 结果已保存到:")
    print(f"   - {detailed_path}")
    print(f"   - {metrics_path}")
    print(f"   - {summary_path}")


def print_summary(stats: Dict[str, Any], n_results: int):
    """打印统计摘要"""
    print("\n" + "=" * 80)
    print("Probe 统计分析摘要")
    print("=" * 80)
    print(f"\n总实验数: {n_results}")
    
    if 'success_rate' in stats:
        sr = stats['success_rate']
        print(f"\n成功率统计:")
        print(f"  成功数: {sr['n_success']}/{sr['n_total']}")
        print(f"  成功率: {sr['rate']*100:.2f}%")
        print(f"  失败率: {sr['failure_rate']*100:.2f}%")
    
    if 'ema_error' in stats:
        ee = stats['ema_error']
        print(f"\nEMA相对误差统计:")
        print(f"  均值: {ee['mean']*100:.4f}%")
        print(f"  标准差: {ee['std']*100:.4f}%")
        print(f"  中位数: {ee['median']*100:.4f}%")
        print(f"  范围: [{ee['min']*100:.4f}%, {ee['max']*100:.4f}%]")
        print(f"  四分位数: Q25={ee['q25']*100:.4f}%, Q75={ee['q75']*100:.4f}%")
    
    if 'bias' in stats:
        bias = stats['bias']
        print(f"\n绝对偏差统计 (um):")
        print(f"  均值: {bias['mean']:.4f} um")
        print(f"  标准差: {bias['std']:.4f} um")
        print(f"  中位数: {bias['median']:.4f} um")
        print(f"  范围: [{bias['min']:.4f}, {bias['max']:.4f}] um")
        print(f"  最大偏差: {bias['max']:.4f} um")
    
    print("\n" + "=" * 80)


def main():
    """主函数"""
    # 设置输出目录
    if CONFIG["OUTPUT_DIR"] is None:
        CONFIG["OUTPUT_DIR"] = os.path.join(CONFIG["OUTPUT_ROOT"], "probe_statistics")
    
    print("=" * 80)
    print("Probe 统计分析")
    print("=" * 80)
    print(f"输出根目录: {CONFIG['OUTPUT_ROOT']}")
    print(f"GT直径: {CONFIG['GT_DIAMETER']} um")
    print(f"成功率阈值: err < {CONFIG['SUCCESS_THRESHOLD']*100}%")
    
    # 收集所有probe结果
    print("\n正在收集probe实验结果...")
    results = collect_all_probe_results()
    
    if not results:
        print("❌ 未找到任何probe实验结果！")
        print("   请确保已运行probe阶段实验。")
        return
    
    print(f"✅ 找到 {len(results)} 个probe实验结果")
    
    # 计算统计量
    print("\n正在计算统计量...")
    stats = calculate_statistics(results)
    
    # 打印摘要
    print_summary(stats, len(results))
    
    # 保存结果
    print("\n正在保存结果...")
    save_results(results, stats, CONFIG["OUTPUT_DIR"])
    
    print("\n✅ 统计分析完成！")
    print(f"\n下一步：运行可视化脚本生成图表")
    print(f"  python3 visualize_probe_stats.py")


if __name__ == "__main__":
    main()
