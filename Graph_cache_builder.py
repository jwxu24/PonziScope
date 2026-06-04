# File: Graph_multi_batch.py
import os
import torch
import time
import pandas as pd
import argparse
from tqdm import tqdm
from Graph_construction import build_two_layer_multigraph

# ==================== 命令行参数解析 ====================
parser = argparse.ArgumentParser(description="Build and cache multigraph dataset")
parser.add_argument("--dataset", type=str, default="Dataset1", 
                    choices=["Dataset1", "Dataset2"], help="选择数据集")
parser.add_argument("--max_neighbors", type=int, default=50, 
                    help="最大邻居数量")
parser.add_argument("--max_txs", type=int, default=20, 
                    help="每个邻居的最大交易数量")
parser.add_argument("--cache_dir", type=str, default=None, 
                    help="缓存目录（默认自动生成）")
args = parser.parse_args()

# 根据参数设置配置
DATASET_NAME = args.dataset
MAX_NEIGHBORS = args.max_neighbors
MAX_TXS_PER_NEIGHBOR = args.max_txs


# 设置缓存目录
if args.cache_dir:
    CACHE_DIR = args.cache_dir
else:
    CACHE_DIR = f"./Dataset/graph_cache_multigraph_{DATASET_NAME}_maxneighbors{MAX_NEIGHBORS}_maxtxs{MAX_TXS_PER_NEIGHBOR}"

os.makedirs(CACHE_DIR, exist_ok=True)

print(f"数据集: {DATASET_NAME}")
print(f"最大邻居数: {MAX_NEIGHBORS}")
print(f"每个邻居最大交易数: {MAX_TXS_PER_NEIGHBOR}")
print(f"缓存目录: {CACHE_DIR}")


def save_multigraph_data(address: str, label: int, max_neighbors=200, max_txs=200):
    """
    为单个地址构建并保存主图 + 邻居图组
    """
    save_path = os.path.join(CACHE_DIR, f"{address}.pt")

    if os.path.exists(save_path):
        # print(f"[SKIP] {address} 已存在，跳过")
        return True

    try:
        main_graph, main_addr_to_index, neighbor_graphs = build_two_layer_multigraph(
            address=address,
            max_neighbors=max_neighbors,
            max_txs=max_txs
        )

        if main_graph is None:
            print(f"[FAIL] {address} 主图构建失败")
            return False

        # 提取每个邻居图的 Data 对象
        neighbor_datas = [data for _, data, _ in neighbor_graphs]

        save_dict = {
            "address": address,
            "label": torch.tensor(label, dtype=torch.long),
            "main_graph": main_graph,              # torch_geometric.data.Data
            "neighbor_graphs": neighbor_datas,          # List[Data]
            "num_neighbors": len(neighbor_datas)
        }

        torch.save(save_dict, save_path)
        print(f"[SAVE] {address} → {len(neighbor_datas)} 个邻居图")
        return True

    except Exception as e:
        print(f"[ERROR] {address} 构建失败: {e}")
        return False


def build_and_cache_dataset(address_label_csv: str,
                            max_neighbors=MAX_NEIGHBORS,
                            max_txs=MAX_TXS_PER_NEIGHBOR):

    df = pd.read_csv(address_label_csv)
    print(f"读取 CSV 成功，共 {len(df)} 个地址")
    print(f"原始列名: {list(df.columns)}")

    normalized_columns = {col.strip().lower().replace(' ', '').replace('_', ''): col for col in df.columns}

    addr_candidates = ['address', 'addr', 'contract', 'contractaddress', 'targetaddress']
    addr_col = None
    for cand in addr_candidates:
        if cand in normalized_columns:
            addr_col = normalized_columns[cand]
            break
    if addr_col is None:
        first_col = df.columns[0]
        sample_val = str(df.iloc[0, 0]).lower()
        if sample_val.startswith('0x') and len(sample_val) == 42:
            addr_col = first_col
            print(f"自动推断 address 列为: {first_col}")

    label_candidates = ['label', 'verifiedlabel', 'isponzi', 'target', 'y', 'class', 'flag']
    label_col = None
    for cand in label_candidates:
        if cand in normalized_columns:
            label_col = normalized_columns[cand]
            break
    if label_col is None:
        label_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
        print(f"自动推断 label 列为: {label_col}")

    if addr_col is None or label_col is None:
        raise ValueError(f"无法识别 address/label 列，可用列: {list(df.columns)}")

    print(f"最终使用 → address 列: '{addr_col}' | label 列: '{label_col}'")

    success = failed = 0
    pbar = tqdm(df.iterrows(), total=len(df), desc="构图中")

    for idx, row in pbar:
        try:
            raw_addr = row[addr_col]
            address = str(raw_addr).strip().lower()
            if not (address.startswith('0x') and len(address) == 42):
                print(f"\n[无效地址] {address}")
                failed += 1
                continue

            label = int(row[label_col])

            pbar.set_description(f"{address[:10]}... | Success:{success} Failed:{failed}")

            if save_multigraph_data(address, label, max_neighbors, max_txs):
                success += 1
            else:
                failed += 1

        except Exception as e:
            print(f"\n[处理异常] 第 {idx} 行: {e}")
            failed += 1

    print("\n" + "="*60)
    print("批量构图全部完成！")
    print(f"成功: {success} 个")
    print(f"失败: {failed} 个")
    print(f"缓存目录: {CACHE_DIR}")
    print("="*60)


if __name__ == "__main__":
    CSV_PATH = f"./Dataset/Ponzi/{DATASET_NAME}.csv"
    
    if not os.path.exists(CSV_PATH):
        print(f"错误: CSV文件不存在 - {CSV_PATH}")
        exit(1)

    build_and_cache_dataset(
        address_label_csv=CSV_PATH,
        max_neighbors=MAX_NEIGHBORS,
        max_txs=MAX_TXS_PER_NEIGHBOR
    )