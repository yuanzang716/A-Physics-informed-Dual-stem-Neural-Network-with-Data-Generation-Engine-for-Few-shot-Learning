# 衍射式细丝测量：少样本逆问题计算工作流

本仓库提供论文的完整代码与数据：

> **面向标签稀缺与稀疏数据逆问题的具备数据生成引擎的物理信息双分支神经网络**
>
> Yuan Zhang, Lin Chen, MingYang Li, Jiao Zhao, JiaHao Han, Qiang Lin, Bin Wu, ZhengHui Hu
>
> 2026

---

## 概述

本仓库实现了物理信息双流网络（PI-DSN）配合实测引导数据增强（MDGA）的完整可复现方案，用于少样本逆问题。该方法仅需每个焦距 8～10 张真实测量即可从弗朗霍夫衍射图样估计细丝直径。

**核心组件：**
- 四阶段训练流程（粗估计、仿真库生成、PI-DSN 训练、无标签检查点筛选）
- 自适应加权的物理信息损失函数
- 无需真值标签的 EMA 检查点筛选机制
- 跨细丝泛化能力（FIL001 → FIL002）

---

## 目录结构

```
publish/
├── README.md                              # 英文版说明文档
├── README_CH.md                           # 本文件：中文说明文档
├── PATH_CONFIGURATION_GUIDE.md             # 路径配置详细指南
├── HARDCODED_PATHS_INVENTORY.md           # 所有硬编码路径清单
├── requirements.txt                       # pip 依赖列表
├── environment.yml                        # conda 环境配置
├── verify_paths.py                        # 路径验证工具
├── fix_hardcoded_paths.py                 # 绘图脚本路径自动修复工具
├── setup_env.sh                           # Linux/macOS 环境变量配置
├── setup_env.bat                          # Windows 环境变量配置
│
├── data/filaments/                        # 原始衍射图样图像
│   ├── FIL001/
│   │   ├── focal_75mm/raw/               # 10 张 BMP 图像
│   │   └── focal_120mm/raw/              # 8 张 BMP 图像
│   ├── FIL002/
│   │   ├── focal_75mm/raw/               # 5 张 BMP 图像
│   │   └── focal_120mm/raw/              # 9 张 BMP 图像
│   ├── registry.csv                       # 细丝元数据注册表
│   └── migration_map.csv                  # 数据迁移映射表
│
├── weights/                               # 预训练 EMA 模型权重
│   ├── FIL001_75mm_seed124_ema.pth
│   ├── FIL001_75mm_seed212_ema.pth
│   ├── FIL001_120mm_seed124_ema.pth
│   ├── FIL001_120mm_seed212_ema.pth
│   ├── FIL002_75mm_seed42_ema.pth
│   ├── FIL002_120mm_seed42_ema.pth
│   └── README.md                          # 权重文件说明
│
├── src/                                   # 核心源代码
│   ├── filament_layout.py                 # 集中式路径配置
│   ├── bootstrap_filament_layout.py       # 路径解析引导脚本
│   ├── Code_75/                           # 75mm 焦距流程
│   │   ├── core/main_75.py              # 核心训练模块
│   │   ├── experiments/
│   │   │   ├── run_all_experiments.py     # 主入口（四阶段流程）
│   │   │   ├── ablation.py              # 消融实验
│   │   │   └── sensitivity.py            # 敏感度分析
│   │   └── analysis/
│   │       └── analyze_probe_stats.py     # 探测统计分析
│   └── Code_120/                          # 120mm 焦距流程
│       └── （结构与 Code_75/ 相同）
│
├── scripts/                               # 绘图与分析脚本
│   ├── fft_analysis.py                   # 表 2：FFT 基线估计
│   ├── synthetic_multi_basin_analysis.py  # 图 1：参数耦合分析
│   ├── unified_mismatch_analysis.py       # 图 4：仿真-实测失配
│   ├── plot_network_full_analysis_nm.py   # 图 5：训练动力学与消融
│   └── plot_network_probe_results_nm.py   # 3.6 节：随机种子稳定性
│
└── examples/                              # 示例脚本
    ├── quick_start.sh                    # 快速评估演示
    ├── train_fil001_75mm.sh             # 完整训练示例
    └── reproduce_all_figures.sh            # 复现论文所有图表
```

---

## 快速开始（3 步）

### 第一步：配置环境

```bash
# 方式 A：conda（推荐）
conda env create -f environment.yml
conda activate diffraction-metrology

# 方式 B：pip
pip install -r requirements.txt
```

### 第二步：配置路径

代码通过环境变量 `FILAMENT_PROJECT_ROOT` 定位数据和输出路径。

```bash
# Linux / macOS
export FILAMENT_PROJECT_ROOT="/path/to/this/publish"

# Windows（命令提示符）
set FILAMENT_PROJECT_ROOT=D:\path\to\this\publish

# Windows（PowerShell）
$env:FILAMENT_PROJECT_ROOT = "D:\path\to\this\publish"
```

或直接运行提供的环境配置脚本：

```bash
# Linux / macOS
source setup_env.sh

# Windows
setup_env.bat
```

> **重要提示：** 部分绘图脚本中包含原开发环境的硬编码路径。使用前请先运行自动路径修复工具：
> ```bash
> python fix_hardcoded_paths.py /path/to/this/publish
> ```
> 详见 [PATH_CONFIGURATION_GUIDE.md](PATH_CONFIGURATION_GUIDE.md) 和 [HARDCODED_PATHS_INVENTORY.md](HARDCODED_PATHS_INVENTORY.md)。

### 第三步：使用预训练模型评估

```bash
export FILAMENT_PROJECT_ROOT="/path/to/this/publish"
export FILAMENT_ID=FIL001
export FOCAL_MM=75
export GT_DIAMETER_UM=100.2
export SEED=124
export ALLOW_TRAIN=0

cd src/Code_75/experiments
python run_all_experiments.py
```

这将加载预训练的 EMA 权重并仅运行评估（不训练）。

---

## 完整训练流程（从零开始）

入口脚本为 `run_all_experiments.py`，自动编排四个阶段：

### 阶段一：粗估计（FFT + 差分进化）
- 从衍射图样的 FFT 中提取初始直径估计
- 通过差分进化（DE）优化进一步精化
- 输出：粗估计参数

### 阶段二：仿真库生成（拉丁超立方采样）
- 使用 LHS 在参数空间内生成合成衍射图样
- 创建训练对（仿真图样，已知参数）
- 输出：`sim_bank/` 中的仿真库

### 阶段三：PI-DSN 训练
- 训练物理信息双流网络
- 使用 MDGA 进行数据增强
- 应用自适应损失加权与渐进式损失引入
- 输出：`runs/` 中的模型检查点

### 阶段四：无标签检查点筛选
- 基于 EMA 评估结果选择最佳检查点
- 无需任何真值标签
- 输出：`best_ema_model.pth`

**完整训练命令：**

```bash
export FILAMENT_PROJECT_ROOT="/path/to/this/publish"
export FILAMENT_ID=FIL001
export FOCAL_MM=75
export GT_DIAMETER_UM=100.2
export SEED=124
export NUM_EPOCHS=60
export ALLOW_TRAIN=1

cd src/Code_75/experiments
python run_all_experiments.py
```

120mm 焦距请使用 `src/Code_120/experiments/run_all_experiments.py`，并设置 `FOCAL_MM=120`。

---

## 复现论文结果

### 表 2：FFT 基线估计
```bash
cd scripts
python fft_analysis.py
```
计算所有细丝的 FFT 粗直径估计值。

### 图 1：参数耦合分析
```bash
cd scripts
python synthetic_multi_basin_analysis.py
```
生成展示逆问题参数耦合特性的损失地形图。

### 图 4：仿真-实测失配
```bash
cd scripts
python unified_mismatch_analysis.py
```
分析与可视化仿真与真实衍射图样之间的结构化失配。

### 图 5：训练动力学与消融研究
```bash
cd scripts
python plot_network_full_analysis_nm.py
```
绘制训练曲线、消融实验结果与收敛分析。

> 需要已完成的训练运行记录。预训练结果已包含在 `weights/` 中。

### 3.6 节：随机种子稳定性（探测实验）
```bash
cd scripts
python plot_network_probe_results_nm.py
```
分析不同随机种子下预测结果的稳定性。

---

## 跨细丝泛化（FIL002）

无需从零训练即可在新细丝（FIL002）上评估：

```bash
export FILAMENT_PROJECT_ROOT="/path/to/this/publish"
export FIL_ID=FIL002
export EVAL_FILAMENT_ID=FIL002
export FOCAL_MM=75
export GT_DIAMETER_UM=100.0
export SEED=42
export ALLOW_TRAIN=1

cd src/Code_75/experiments
python run_all_experiments.py
```

预训练的 FIL002 权重已提供：`weights/FIL002_75mm_seed42_ema.pth` 和 `weights/FIL002_120mm_seed42_ema.pth`。

---

## 预训练权重

| 文件名 | 细丝 | 焦距 | 随机种子 | 大小 |
|--------|------|------|----------|------|
| `FIL001_75mm_seed124_ema.pth` | FIL001 | 75 mm | 124 | ~50 MB |
| `FIL001_75mm_seed212_ema.pth` | FIL001 | 75 mm | 212 | ~50 MB |
| `FIL001_120mm_seed124_ema.pth` | FIL001 | 120 mm | 124 | ~50 MB |
| `FIL001_120mm_seed212_ema.pth` | FIL001 | 120 mm | 212 | ~50 MB |
| `FIL002_75mm_seed42_ema.pth` | FIL002 | 75 mm | 42 | ~50 MB |
| `FIL002_120mm_seed42_ema.pth` | FIL002 | 120 mm | 42 | ~50 MB |

所有权重均为 EMA（指数移动平均）检查点，是论文所有结果所使用的权重。

EMA 与标准权重的区别详见 [weights/README.md](weights/README.md)。

---

## 原始数据

| 数据集 | 焦距 | 图像数量 | 格式 |
|--------|------|----------|------|
| FIL001 | 75 mm | 10 张 | BMP |
| FIL001 | 120 mm | 8 张 | BMP |
| FIL002 | 75 mm | 5 张 | BMP |
| FIL002 | 120 mm | 9 张 | BMP |

存放于 `data/filaments/`。这些是来自真实细丝样本的原始未处理衍射图样图像。

---

## 路径配置

这是复现结果时最常见的错误来源。代码库采用基于 `FILAMENT_PROJECT_ROOT` 环境变量的集中式路径系统。

**快速修复：** 设置环境变量后运行路径修复工具：

```bash
export FILAMENT_PROJECT_ROOT="/path/to/this/publish"
python fix_hardcoded_paths.py "$FILAMENT_PROJECT_ROOT"
python verify_paths.py
```

详细说明请参阅：
- [PATH_CONFIGURATION_GUIDE.md](PATH_CONFIGURATION_GUIDE.md) — 分步配置指南
- [HARDCODED_PATHS_INVENTORY.md](HARDCODED_PATHS_INVENTORY.md) — 硬编码路径完整清单

---

## 常见问题

**问：运行绘图脚本时出现 "FileNotFoundError" 怎么办？**
答：绘图脚本中包含原开发环境的硬编码路径。先运行 `python fix_hardcoded_paths.py /path/to/publish` 自动修复，或参阅 [HARDCODED_PATHS_INVENTORY.md](HARDCODED_PATHS_INVENTORY.md) 手动修复。

**问：EMA 权重和标准权重有什么区别？**
答：EMA 权重是对模型参数进行指数移动平均后得到的，预测更平滑稳定。论文所有结果均使用 EMA 检查点。标准权重（`best_model.pth`）是未经平均的原始训练权重。

**问：可以在 CPU 上训练吗？**
答：训练专为 GPU（CUDA）设计，CPU 训练在技术上可行但极慢，不推荐使用。预训练模型的评估可在 CPU 上运行。

**问：训练需要多长时间？**
答：依赖数据量和每个图像对应生成的模拟数据数量决定。

**问：`ALLOW_TRAIN=0` 的作用是什么？**
答：跳过训练阶段，仅使用已有检查点进行评估。设为 `ALLOW_TRAIN=1` 可从零开始训练。

**问：如何复现特定随机种子的结果？**
答：运行 `run_all_experiments.py` 前设置 `SEED` 环境变量（如 `export SEED=124`）。论文报告了 FIL001 的种子 124 和 212，以及 FIL002 的种子 42 的结果。

---

## 环境要求

- Python 3.10+
- PyTorch 2.0+（支持 CUDA）
- 详见 `requirements.txt` 或 `environment.yml`

---

## 许可

本代码仅供学术研究与结果复现目的使用。

---

## 引用

如使用本代码或数据，请引用：

```bibtex
@article{zhang2026workflow,
  title={A Physics-informed Dual-stem Neural Network with Data Generation Engine for Label-scarce and
Sparse-data Inverse Problems},
  author={Zhang, Yuan and Chen, Lin and Li, MingYang and Zhao, Jiao
          and Han, JiaHao and Lin, Qiang and Wu, Bin and Hu, ZhengHui},
  journal={},
  year={2026}
}
```
