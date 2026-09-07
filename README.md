# MultiGraph Ponzi Scheme Detector

## 项目概述 / Project Overview

本项目实现了一个基于双层图神经网络的以太坊庞氏骗局检测系统。该系统通过构建交易图和应用多头注意力机制来识别可疑的庞氏骗局合约。

This project implements a Ponzi scheme detection system for Ethereum smart contracts using a two-layer graph neural network. It identifies suspicious Ponzi contracts by constructing transaction graphs and applying multi-head attention mechanisms.

---

## 环境要求 / Environment Requirements

- Python 3.8+
- PyTorch 1.10+
- PyTorch Geometric 2.0+
- scikit-learn
- pandas
- numpy
- tqdm

---

## 文件结构 / File Structure

```
Research2_Submit/
├── Main.py                 # 主训练脚本（5折交叉验证）
├── Graph_cache_builder.py  # 图数据缓存构建脚本
├── Graph_construction.py   # 双层图构建核心模块
├── Dataset.py              # 数据预处理工具
├── Utils.py            # Etherscan API 工具类
└── Dataset/
    ├── PonziCombine/
    │   ├── Dataset1.csv    # 数据集1（地址,标签）
    │   └── Dataset2.csv    # 数据集2（地址,标签）
    └── graph_cache_multigraph_*/  # 生成的图缓存目录
```

---

## 运行步骤 / Running Steps

### 步骤0：准备数据集文件 / Step 0: Prepare Dataset File

将包含地址和标签的 CSV 文件放入 `./Dataset/Ponzi/` 目录，文件格式如下：

```csv
Address,VerifiedLabel
0x1234567890abcdef1234567890abcdef12345678,1
0xabcdef1234567890abcdef1234567890abcdef12,0
```

### 步骤1：获取交易数据/ Step 1: Fetch Transaction Data 

如果需要从头获取区块链交易数据，使用 `Dataset.py`。该脚本会**自动读取 Ponzi 文件夹下的所有 CSV 文件**并一起处理：

```bash
# 获取中心地址的交易数据（自动处理所有数据集）
python Dataset.py --get_transactions

# 获取相关地址标签
python Dataset.py --get_labels

# 获取相关地址的交易数据
python Dataset.py --get_address_transactions

# 提取邻居节点关系
python Dataset.py --extract_neighbors

# 提取中心节点特征
python Dataset.py --extract_center_features

# 提取邻居节点特征
python Dataset.py --extract_neighbor_features
```

### 步骤2：构建图数据缓存 / Step 2: Build Graph Cache

使用 `Graph_cache_builder.py` 为数据集构建图数据缓存：

```bash
# 为 Dataset1 构建图缓存
python Graph_cache_builder.py --dataset Dataset1 --max_neighbors 200 --max_txs 200

# 为 Dataset2 构建图缓存
python Graph_cache_builder.py --dataset Dataset2 --max_neighbors 200 --max_txs 200
```

### 步骤3：训练模型 / Step 3: Train Model

使用 `Main.py` 运行5折交叉验证训练：

```bash
# 使用 Dataset1 训练
python Main.py --dataset Dataset1 --max_neighbors 200 --max_txs 200

# 使用 Dataset2 训练
python Main.py --dataset Dataset2 --max_neighbors 200 --max_txs 200
```

---

## 命令行参数说明 / Command Line Arguments

### Dataset.py 参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `--get_transactions` | flag | 获取中心地址的交易数据（**自动处理 Ponzi 文件夹下所有 CSV**） |
| `--get_labels` | flag | 获取相关地址标签 |
| `--get_address_transactions` | flag | 获取相关地址的交易数据 |
| `--extract_neighbors` | flag | 提取邻居节点关系 |
| `--extract_center_features` | flag | 提取中心节点特征 |
| `--extract_neighbor_features` | flag | 提取邻居节点特征 |
| `--workers` | int | 工作线程数（默认自动） |
| `--batch_size` | int | 每批处理地址数（默认200） |
| `--reverse` | flag | 反向处理地址列表 |

### Graph_cache_builder.py 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--dataset` | str | Dataset1 | 数据集名称（Dataset1 或 Dataset2） |
| `--max_neighbors` | int | 200 | 每个中心节点保留的最大邻居数 |
| `--max_txs` | int | 200 | 每个邻居节点保留的最大交易数 |
| `--cache_dir` | str | None | 自定义缓存目录路径 |

### Main.py 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--dataset` | str | Dataset1 | 数据集名称（Dataset1 或 Dataset2） |
| `--output_dir` | str | None | 自定义输出目录路径 |
| `--cache_dir` | str | None | 自定义图缓存目录路径 |
| `--max_neighbors` | int | 200 | 构图时的最大邻居数，用于定位默认缓存目录 |
| `--max_txs` | int | 200 | 每个邻居的最大交易数，用于定位默认缓存目录 |

---

## 输出说明 / Output Description

### 图缓存目录 / Graph Cache Directory

图缓存文件保存在 `./Dataset/graph_cache_multigraph_{DATASET}_maxneighbors{K}_maxtxs{M}/` 目录下，每个地址对应一个 `.pt` 文件。论文主实验配置为 `K=200, M=200`。

### 训练结果 / Training Results

训练结果保存在 `./5fold_results_{DATASET}/` 目录下：
- `5fold_results.csv` - 包含5折交叉验证的 Precision、Recall、F1、AUC 等指标

---

## 模型架构 / Model Architecture

本项目使用的模型包含以下核心组件：

1. **SingleGraphEncoder**: 基于 GAT 的单层图编码器，用于提取图级特征
2. **MultiGraphClassifier**: 多头注意力机制的多图分类器
3. **Class-Balanced Focal Loss**: 类别平衡的焦点损失函数
4. **动态阈值优化**: 基于验证集 F1 分数自动寻找最佳分类阈值

---

## 数据集格式 / Dataset Format

CSV 文件需包含两列：
- `Address`: 以太坊合约地址（以 0x 开头，42位十六进制）
- `VerifiedLabel`: 标签（0=正常合约，1=庞氏骗局合约）

示例：
```csv
Address,VerifiedLabel
0x1234567890abcdef1234567890abcdef12345678,1
0xabcdef1234567890abcdef1234567890abcdef12,0
```

---

## 注意事项 / Notes

1. 确保数据集文件 `Dataset1.csv` 和 `Dataset2.csv` 存在于 `./Dataset/Ponzi/` 目录下
2. 首次运行构图脚本时会自动生成图缓存，后续运行会跳过已存在的文件
3. 训练过程会自动创建输出目录并保存结果
4. 建议在 GPU 环境下运行以获得更好的性能

---

## 联系方式 / Contact

如有问题，请联系jwxu@m.fudan.edu.cn。

---

*Last updated: June 2026*
