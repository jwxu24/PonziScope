import os
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup

import pandas as pd
import re
import tqdm
import time
from Utils import *
import json
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
from collections import defaultdict

TX_TYPES = ['Normal', 'Internal', 'ERC20']


def get_all_datasets():
    """
    获取 Ponzi 文件夹下所有的数据集文件名
    
    返回:
        list: 数据集文件名列表（不含扩展名）
    """
    ponzi_dir = "Dataset/Ponzi"
    datasets = []
    
    if os.path.exists(ponzi_dir):
        for filename in os.listdir(ponzi_dir):
            if filename.endswith('.csv'):
                datasets.append(filename[:-4])  # 去掉 .csv 扩展名
    
    datasets.sort()
    return datasets


def GetAddressRelatedTransactions(dataset=None):
    """
    获取与地址相关的交易数据
    
    从Ponzi文件夹读取地址列表（支持单个数据集或全部数据集），然后为每个地址获取三种类型的交易数据：
    普通交易(Normal)、内部交易(Internal)和ERC20代币交易(ERC20)。
    所有交易数据保存在Dataset/PonziCombine/RelatedTransactions/目录下的相应子文件夹中。
    如果某地址的交易数据文件已存在，则跳过该地址以避免重复获取。
    
    参数:
        dataset: 数据集名称（可选，默认为None表示处理所有数据集）
    
    返回值:
        无，结果直接保存到对应的CSV文件中
    """
    relatedTransactionPath = "Dataset/PonziCombine/RelatedTransactions/"
    
    data = DataSource()
    
    if not os.path.exists(relatedTransactionPath):
        os.makedirs(relatedTransactionPath)
    
    for directory in ['Normal', 'Internal', 'ERC20']:
        path = os.path.join(relatedTransactionPath, directory)
        if not os.path.exists(path):
            os.makedirs(path)
            
    # 获取要处理的数据集列表
    if dataset:
        datasets = [dataset]
    else:
        datasets = get_all_datasets()
    
    print(f"处理数据集: {datasets}")
    
    # 读取所有数据集文件中的地址（保留标签信息）
    all_addresses = []
    for ds in datasets:
        csv_path = f"Dataset/Ponzi/{ds}.csv"
        if os.path.exists(csv_path):
            ponzi_df = pd.read_csv(csv_path)
            all_addresses.extend([(row['Address'], row['VerifiedLabel']) for _, row in ponzi_df.iterrows()])
            print(f"从 {ds}.csv 加载了 {len(ponzi_df)} 个地址")
    
    # 去重，保留唯一地址（保留第一个出现的标签）
    seen = set()
    addresses_with_labels = []
    for addr, label in all_addresses:
        if addr not in seen:
            seen.add(addr)
            addresses_with_labels.append((addr, label))
    
    addresses = [addr for addr, _ in addresses_with_labels]
    print(f"去重后共 {len(addresses)} 个地址")
    
    # 保存地址列表到 addresses.csv（供后续函数使用）
    addresses_df = pd.DataFrame(addresses_with_labels, columns=['Address', 'VerifiedLabel'])
    addresses_df.to_csv("Dataset/PonziCombine/addresses.csv", index=False)
    print(f"地址列表已保存到 Dataset/PonziCombine/addresses.csv")
    
    for address in tqdm.tqdm(addresses, desc="Processing addresses"):
        for tt in ['Normal/', 'Internal/', 'ERC20/']:
            # 检查该地址的交易文件是否已经存在
            file_path = f"{relatedTransactionPath}{tt}{address}.csv"
            if not os.path.exists(file_path):
                # 如果文件不存在，则获取该地址的交易数据
                try:
                    data.getTotalDatafromScan(address, tt, f"{relatedTransactionPath}{tt}")
                    time.sleep(0.1)  # 添加延时，避免API请求过于频繁
                except Exception as e:
                    print(f"Error processing {address} for {tt}: {e}")

def GetRelatedAddressLabel(reverse=False):
    relatedTransactionPath = "Dataset/PonziCombine/RelatedTransactions/"
    
    # 根据reverse参数选择不同的保存位置
    if reverse:
        labels_json_path = "Dataset/PonziCombine/address_labels_reverse.json"
        skipped_addresses_path = "Dataset/PonziCombine/skipped_addresses_reverse.json"
        print("使用反向模式，数据将保存到", labels_json_path)
    else:
        labels_json_path = "Dataset/PonziCombine/address_labels.json"
        skipped_addresses_path = "Dataset/PonziCombine/skipped_addresses.json"
    
    # 加载已有的标签数据，如果文件不存在则创建空字典
    if os.path.exists(labels_json_path):
        with open(labels_json_path, 'r') as f:
            address_labels = json.load(f)
    else:
        address_labels = {}
    
    # 加载已跳过的地址，如果文件不存在则创建空字典
    if os.path.exists(skipped_addresses_path):
        with open(skipped_addresses_path, 'r') as f:
            skipped_addresses = json.load(f)
    else:
        skipped_addresses = {}
    
    # 读取 Ponzi 文件夹下所有数据集
    datasets = get_all_datasets()
    print(f"处理数据集: {datasets}")
    
    # 合并所有数据集
    dfs = []
    for ds in datasets:
        csv_path = f"Dataset/Ponzi/{ds}.csv"
        if os.path.exists(csv_path):
            temp_df = pd.read_csv(csv_path)
            dfs.append(temp_df)
            print(f"从 {ds}.csv 加载了 {len(temp_df)} 条记录")
    
    df = pd.concat(dfs, ignore_index=True)
    print(f"合并后共 {len(df)} 条记录")
    
    # 根据reverse参数决定处理顺序
    if reverse:
        df = df.iloc[::-1]
    
    for index, row in tqdm.tqdm(df.iterrows(), total=len(df)):
        address = row['Address']
        label = row['VerifiedLabel']
        
        # 构建相关地址集合，排除当前处理的地址本身
        related_addresses = set()
        
        transaction_counts = {'Normal': 0, 'Internal': 0, 'ERC20': 0}
        for tt in ['Normal/', 'Internal/', 'ERC20/']:
            file_path = f"{relatedTransactionPath}{tt}{address}.csv"
            if os.path.exists(file_path):
                transaction_df = pd.read_csv(file_path)
                related_addresses.update([addr for addr in transaction_df['from'].tolist() if isinstance(addr, str) and addr.lower() != address.lower()])
                related_addresses.update([addr for addr in transaction_df['to'].tolist() if isinstance(addr, str) and addr.lower() != address.lower()])
                transaction_counts[tt.replace('/', '')] = len(transaction_df)
         
        num_related = len(related_addresses)
        tqdm.tqdm.write(f"地址 {address} (Label={label}): {num_related} 个相关地址, Normal: {transaction_counts['Normal']}, Internal: {transaction_counts['Internal']}, ERC20: {transaction_counts['ERC20']} 条交易")
        
        if num_related > 1000:
            skipped_addresses[address] = num_related
            # 保存跳过的地址
            with open(skipped_addresses_path, 'w') as f:
                json.dump(skipped_addresses, f)
            tqdm.tqdm.write(f"跳过 {address}: 相关地址数 > 1000")
            continue
        
        # 遍历相关地址获取标签
        new_labels_added = False
        for related_addr in tqdm.tqdm(related_addresses, leave=False):
            # 检查是否已经有该地址的标签数据
            if related_addr.lower() not in address_labels:
                # 如果没有，则从网页获取标签
                span_label, title_label, label_text = getAddressLabelFromEthereumPage(related_addr)
                # 保存到字典中
                address_labels[related_addr.lower()] = {
                    "span_label": span_label,
                    "title_label": title_label,
                    "label_text": label_text
                }
                new_labels_added = True
        
        # 只有在新增了标签数据时才保存
        if new_labels_added:
            with open(labels_json_path, 'w') as f:
                json.dump(address_labels, f)
    

def GetRelatedAddressTransactions(reverse=False, use_reverse_file=False):
    """
    获取相关地址的交易数据
    
    从address_labels.json或address_labels_reverse.json读取已收集的相关地址，为每个地址获取三种类型的交易数据：
    普通交易(Normal)、内部交易(Internal)和ERC20代币交易(ERC20)。
    数据保存在Dataset/PonziCombine/RelatedAddressTransactions/目录下。
    如果任意一类交易文件已存在，则跳过该地址。
    
    参数:
        reverse (bool): 是否反向处理地址列表，默认为False
        use_reverse_file (bool): 是否使用address_labels_reverse.json文件，默认为False
    """
    relatedAddressTransactionPath = "Dataset/PonziCombine/RelatedAddressTransactions/"
    if use_reverse_file:
        labels_json_path = "Dataset/PonziCombine/address_labels_reverse.json"
    else:
        labels_json_path = "Dataset/PonziCombine/address_labels.json"
    
    data = DataSource()
    
    # 确保目录存在
    if not os.path.exists(relatedAddressTransactionPath):
        os.makedirs(relatedAddressTransactionPath)
    
    for directory in ['Normal', 'Internal', 'ERC20']:
        path = os.path.join(relatedAddressTransactionPath, directory)
        if not os.path.exists(path):
            os.makedirs(path)
    
    # 加载已有的标签数据
    if not os.path.exists(labels_json_path):
        return
    
    with open(labels_json_path, 'r') as f:
        address_labels = json.load(f)
    
    # 获取所有相关地址
    related_addresses = list(address_labels.keys())
    if reverse:
        related_addresses = related_addresses[::-1]  # 反向处理地址列表
        print(f"共有 {len(related_addresses)} 个相关地址需要获取交易数据 (反向处理)")
    else:
        print(f"共有 {len(related_addresses)} 个相关地址需要获取交易数据")
    
    for address in tqdm.tqdm(related_addresses):
        # 检查该地址的三类交易文件是否存在任意一个
        transaction_types = ['Normal/', 'Internal/', 'ERC20/']
        existing_files = [tt for tt in transaction_types if os.path.exists(f"{relatedAddressTransactionPath}{tt}{address}.csv")]
        
        if existing_files:
            # tqdm.tqdm.write(f"跳过地址 {address}: 已存在交易文件 {', '.join([t.replace('/', '') for t in existing_files])}")
            
            continue
            
        # tqdm.tqdm.write(f"处理地址 {address}: 无现有交易文件")
        # 获取交易数据
        for tt in transaction_types:
            try:
                data.getTotalDatafromScan(address, tt, f"{relatedAddressTransactionPath}{tt}")
            except Exception as e:
                print(f"获取 {address} 的 {tt} 交易失败: {e}")

# 特征提取功能
def extract_node_features(addr, is_center, tx_types, cache_dict=None):
    """
    Extract node features from transaction data
    
    Args:
        addr: Address to extract features for
        is_center: Whether this is a center node (bool or float)
        tx_types: List of transaction types to process
        cache_dict: Optional cache dictionary
        
    Returns:
        Feature tensor
    """
    addr_lower = addr.lower()
    is_center_bool = bool(is_center)
    
    # Create cache key
    cache_key = f"{addr_lower}_{1 if is_center_bool else 0}"
    
    # Return from cache if available
    if cache_dict is not None and cache_key in cache_dict:
        return cache_dict[cache_key]
    
    # Select appropriate directory based on node type
    tx_dir = os.path.join('Dataset', 'PonziCombine', 
                         'RelatedTransactions' if is_center_bool else 'RelatedAddressTransactions')
    
    # Pre-allocate statistics arrays
    tx_counts = np.zeros(len(tx_types), dtype=np.float32)
    tx_values = np.zeros(len(tx_types), dtype=np.float32)
    tx_age = np.zeros(len(tx_types), dtype=np.float32)  # Latest transaction timestamp
    
    # Only read these columns to improve efficiency
    columns_to_read = ['value', 'timeStamp']
    max_rows = 20000  # Limit to 20,000 rows per file
    
    # Process all transaction types
    for i, tx_type in enumerate(tx_types):
        tx_file = os.path.join(tx_dir, tx_type, f"{addr}.csv")
        
        if not os.path.exists(tx_file) or os.path.getsize(tx_file) == 0:
            continue
        
        try:
            # Check file header for available columns
            with open(tx_file, 'r') as f:
                header = f.readline().strip().split(',')
                valid_cols = [col for col in columns_to_read if col in header]
                
                if not valid_cols:
                    continue
            
            # Efficiently read data with specified types
            tx_df = pd.read_csv(
                tx_file, 
                usecols=valid_cols,
                nrows=max_rows,
                engine='c',      # Use C engine for faster processing
                dtype={          # Specify data types to avoid inference
                    'value': np.float64,
                    'timeStamp': np.float64
                }
            )
            
            # Calculate transaction count
            tx_count = len(tx_df)
            tx_counts[i] = tx_count
            
            # Calculate transaction value and latest timestamp
            if 'value' in tx_df.columns and tx_count > 0:
                # Ensure valid numerical values
                values = pd.to_numeric(tx_df['value'], errors='coerce')
                tx_values[i] = values.sum(skipna=True)
                
            if 'timeStamp' in tx_df.columns and tx_count > 0:
                # Ensure valid timestamps
                timestamps = pd.to_numeric(tx_df['timeStamp'], errors='coerce')
                if not timestamps.isna().all():
                    tx_age[i] = timestamps.max(skipna=True)
                
        except Exception as e:
            print(f"Error reading transaction file {tx_file}: {e}")
    
    # Create derived features
    total_tx_count = np.sum(tx_counts)
    total_value = np.sum(tx_values)
    tx_count_ratios = tx_counts / (total_tx_count + 1e-8)  # Avoid division by zero
    
    # Combine all features
    all_features = np.concatenate([
        tx_counts,            # Transaction counts by type
        tx_count_ratios,      # Transaction count ratios
        tx_values,            # Transaction values by type
        [total_tx_count],     # Total transaction count
        [total_value],        # Total transaction value
        tx_age                # Latest transaction timestamps by type
    ])
    
    # Normalize large values using vectorized operation
    mask = all_features > 1000000
    all_features[mask] = np.log1p(all_features[mask])
    
    # Convert to tensor
    features_tensor = torch.tensor(all_features, dtype=torch.float)
    
    # Store in cache if provided
    if cache_dict is not None:
        cache_dict[cache_key] = features_tensor
    
    return features_tensor

def process_address_features(addr, feature_cache_dir, is_center):
    """Process feature extraction for a single address
    
    Args:
        addr: Address to extract features for
        feature_cache_dir: Feature cache directory
        is_center: Whether this is a center node
        
    Returns:
        Tuple of (address, is_center_flag, features_tensor)
    """
    # Check for transaction files before processing
    tx_dir = os.path.join('Dataset', 'PonziCombine', 
                         'RelatedTransactions' if is_center else 'RelatedAddressTransactions')
    
    # Quick check for any transaction files
    has_tx_files = False
    for tx_type in TX_TYPES:
        tx_file = os.path.join(tx_dir, tx_type, f"{addr}.csv")
        if os.path.exists(tx_file) and os.path.getsize(tx_file) > 0:
            has_tx_files = True
            break
    
    if not has_tx_files:
        # Return zero features if no transaction files exist
        feature_size = len(TX_TYPES) * 3 + 2 + len(TX_TYPES)  # Calculated size based on feature extraction logic
        features = torch.zeros(feature_size, dtype=torch.float)
        return addr, is_center, features
    
    # Extract features if transaction files exist
    features = extract_node_features(addr, is_center, TX_TYPES)
    return addr, is_center, features

def extract_and_save_features(is_center=True, batch_size=200, num_workers=None):
    """
    Extract and save node features in parallel to CSV files
    
    Args:
        is_center: Whether to extract center node features (True) or neighbor node features (False)
        batch_size: Number of addresses to process in each batch
        num_workers: Number of threads to use, defaults to auto-determination
    """
    # Create feature cache directory
    feature_cache_dir = os.path.join('Dataset', 'PonziCombine', 'feature_cache')
    os.makedirs(feature_cache_dir, exist_ok=True)
    
    # Set node features CSV path
    node_features_file = os.path.join(feature_cache_dir, 'node_features.csv')
    
    # Initialize node features cache
    node_features_cache = {}
    
    # Load existing features if available
    if os.path.exists(node_features_file):
        try:
            node_df = pd.read_csv(node_features_file)
            print(f"Loaded {len(node_df)} cached node features")
            
            # Build node features cache dictionary - only for relevant node type
            for _, row in node_df.iterrows():
                if bool(row['is_center']) == is_center:  # Only load features for requested node type
                    addr = row['address']
                    # Convert feature string to tensor
                    features = torch.tensor([float(x) for x in row['features'].split(',')], dtype=torch.float)
                    # Store in cache
                    node_features_cache[f"{addr}_{int(is_center)}"] = features
        except Exception as e:
            print(f"Error loading node features cache: {e}")
    
    # Get addresses to process based on node type
    if is_center:
        print("Processing center node features")
        # For center nodes, get from dataset file and filter by transaction existence
        df = pd.read_csv("Dataset/PonziCombine/addresses.csv")
        addresses = [addr for addr in df['Address'].tolist() if any(
            os.path.exists(os.path.join('Dataset', 'PonziCombine', 'RelatedTransactions', tx_type, f"{addr}.csv"))
            for tx_type in TX_TYPES
        )]
    else:
        print("Processing neighbor node features")
        # For neighbor nodes, try loading from neighborhood.json first
        neighborhood_file = os.path.join(feature_cache_dir, 'neighborhood.json')
        
        if os.path.exists(neighborhood_file):
            try:
                print(f"Loading neighborhood information from {neighborhood_file}")
                with open(neighborhood_file, 'r') as f:
                    neighborhood_data = json.load(f)
                
                # Extract unique neighbor addresses
                neighbor_addresses = set()
                for _, neighbors in neighborhood_data.items():
                    neighbor_addresses.update(neighbors)
                
                addresses = list(neighbor_addresses)
                print(f"Found {len(addresses)} unique neighbor addresses from neighborhood.json")
            except Exception as e:
                print(f"Error loading neighborhood data: {e}, falling back to directory scanning")
                addresses = []
        else:
            print("No neighborhood.json found, scanning directories for neighbor addresses")
            addresses = []
        
        # Fallback to directory scanning if needed
        if not addresses:
            neighbor_addresses = set()
            for tx_type in TX_TYPES:
                tx_dir = os.path.join('Dataset', 'PonziCombine', 'RelatedAddressTransactions', tx_type)
                if os.path.exists(tx_dir):
                    # Extract addresses from filenames
                    addrs = [os.path.splitext(f)[0] for f in os.listdir(tx_dir) if f.endswith('.csv')]
                    neighbor_addresses.update(addrs)
            
            addresses = list(neighbor_addresses)
            print(f"Found {len(addresses)} neighbor addresses from directory scanning")
        
        # For neighbor nodes, pre-filter those with actual transaction files
        if len(addresses) > 100:  # Only filter when needed
            print("Pre-filtering addresses with transaction files...")
            tx_base_dir = 'Dataset/PonziCombine/RelatedAddressTransactions'
            
            # Set up parallel filtering
            check_workers = min(16, multiprocessing.cpu_count()) if num_workers is None else num_workers
            
            def has_tx_files(addr):
                """Check if address has any valid transaction files"""
                return any(os.path.exists(os.path.join(tx_base_dir, tx_type, f"{addr}.csv")) and 
                          os.path.getsize(os.path.join(tx_base_dir, tx_type, f"{addr}.csv")) > 0
                          for tx_type in TX_TYPES)
            
            # Parallel check for transaction files
            valid_addresses = []
            with ThreadPoolExecutor(max_workers=check_workers) as executor:
                futures = [executor.submit(has_tx_files, addr) for addr in addresses]
                for i, future in enumerate(tqdm.tqdm(as_completed(futures), total=len(futures), 
                                           desc="Checking transaction files")):
                    if future.result():
                        valid_addresses.append(addresses[i])
            
            print(f"Filtered down to {len(valid_addresses)} addresses with transaction files")
            addresses = valid_addresses

            # with ThreadPoolExecutor(max_workers=check_workers) as executor:
            #     futures = [executor.submit(has_tx_files, addr) for addr in addresses]
                
            #     # 添加调试：检查前几个地址的结果
            #     debug_checked = 0
            #     for i, future in enumerate(tqdm.tqdm(as_completed(futures), total=len(futures), 
            #                                     desc="Checking transaction files")):
            #         result = future.result()
                    
            #         # 调试：打印前5个地址的检查结果
            #         if debug_checked < 5:
            #             print(f"DEBUG: Address {addresses[i]} - has_tx_files result: {result}")
            #             if not result:
            #                 # 详细检查为什么返回False
            #                 debug_addr = addresses[i]
            #                 print(f"DEBUG: Checking why {debug_addr} failed:")
            #                 for tx_type in TX_TYPES:
            #                     file_path = os.path.join(tx_base_dir, tx_type, f"{debug_addr}.csv")
            #                     exists = os.path.exists(file_path)
            #                     size = os.path.getsize(file_path) if exists else 0
            #                     print(f"  {tx_type}: {file_path} | exists: {exists} | size: {size}")
            #             debug_checked += 1
                    
            #         if result:
            #             valid_addresses.append(addresses[i])

            # print(f"Filtered down to {len(valid_addresses)} addresses with transaction files")
            # addresses = valid_addresses

    # Identify addresses already processed
    if os.path.exists(node_features_file):
        node_df = pd.read_csv(node_features_file)
        processed_addresses = set(row['address'] for _, row in node_df.iterrows() 
                                if bool(row['is_center']) == is_center)
    else:
        processed_addresses = set()
    
    # Filter out already processed addresses
    addresses_to_process = [addr for addr in addresses if addr not in processed_addresses]
    
    print(f"Total addresses: {len(addresses)}")
    print(f"Already processed: {len(processed_addresses)}")
    print(f"To process: {len(addresses_to_process)}")
    
    if not addresses_to_process:
        print("No new addresses to process")
        return
    
    # Setup parallel processing
    num_workers = min(16, multiprocessing.cpu_count()) if num_workers is None else num_workers
    print(f"Using {num_workers} workers for feature extraction")
    
    # Create lock for thread-safe operations
    lock = multiprocessing.Manager().Lock()
    
    # Process in batches
    total_addresses = len(addresses_to_process)
    new_features = []
    
    for batch_start in range(0, total_addresses, batch_size):
        batch_end = min(batch_start + batch_size, total_addresses)
        current_batch = addresses_to_process[batch_start:batch_end]
        batch_features = []
        
        # Process current batch in parallel
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_address_features, addr, feature_cache_dir, is_center) 
                      for addr in current_batch]
            
            # Collect results
            for future in tqdm.tqdm(as_completed(futures), total=len(futures), 
                                   desc=f"Batch {batch_start//batch_size + 1}/{(total_addresses+batch_size-1)//batch_size}"):
                try:
                    addr, is_c, features = future.result()
                    # Convert features to string for CSV storage
                    features_str = ','.join(map(str, features.numpy().tolist()))
                    
                    batch_features.append({
                        'address': addr,
                        'is_center': int(is_c),
                        'features': features_str
                    })
                    
                    # Update memory cache
                    node_features_cache[f"{addr}_{1 if is_c else 0}"] = features
                except Exception as e:
                    print(f"Error processing address: {e}")
        
        # Add batch results to the full results list
        new_features.extend(batch_features)
        
        # Save results after each batch
        with lock:
            # Create DataFrame for new features
            new_df = pd.DataFrame(new_features)
            
            if os.path.exists(node_features_file):
                try:
                    # Append to existing file
                    old_df = pd.read_csv(node_features_file)
                    combined_df = pd.concat([old_df, new_df], ignore_index=True)
                    combined_df.drop_duplicates(subset=['address', 'is_center'], inplace=True)
                    combined_df.to_csv(node_features_file, index=False)
                except Exception:
                    # Fallback if file is corrupted
                    new_df.to_csv(node_features_file, index=False)
            else:
                # Create new file
                new_df.to_csv(node_features_file, index=False)
            
            # Clear batch memory
            new_features = []
    
    print(f"Feature extraction completed for {'center' if is_center else 'neighbor'} nodes")
    print(f"Features saved to {node_features_file}")

def extract_and_save_neighborhood_information():
    """Extract and save neighborhood relationships for each center node
    
    Read transaction data and collect all neighbor nodes for each center address,
    saving the results to a JSON file. This allows for directly loading neighbor
    information when building graphs, eliminating the need to re-analyze transactions.
    """
    print("Extracting center node to neighbor relationships...")
    
    # Create save path
    feature_cache_dir = os.path.join('Dataset', 'PonziCombine', 'feature_cache')
    os.makedirs(feature_cache_dir, exist_ok=True)
    neighborhood_file = os.path.join(feature_cache_dir, 'neighborhood.json')
    
    # Check for existing neighborhood data
    if os.path.exists(neighborhood_file):
        try:
            with open(neighborhood_file, 'r') as f:
                neighborhood_data = json.load(f)
                print(f"Loaded existing neighborhood data for {len(neighborhood_data)} center nodes")
        except Exception as e:
            print(f"Error reading existing neighborhood file: {e}")
            neighborhood_data = {}
    else:
        neighborhood_data = {}
    
    # Get center node list
    df = pd.read_csv("Dataset/PonziCombine/addresses.csv")
    addresses = df['Address'].tolist()
    
    # Filter to include only addresses with transaction data
    valid_addresses = [
        address for address in addresses if any(
            os.path.exists(os.path.join('Dataset', 'PonziCombine', 'RelatedTransactions', tx_type, f"{address}.csv")) 
            for tx_type in TX_TYPES
        )
    ]
    
    print(f"Found {len(valid_addresses)} valid center nodes that need neighbor extraction")
    
    # Find addresses not yet processed
    addresses_to_process = [addr for addr in valid_addresses if addr not in neighborhood_data]
    print(f"Of which {len(addresses_to_process)} nodes need new neighbor extraction")
    
    if not addresses_to_process:
        print("No new addresses to process")
        return neighborhood_data
        
    # Collect neighbors for each address in parallel
    num_workers = min(16, multiprocessing.cpu_count())
    batch_size = 100  # Number of addresses per batch
    
    # Create shared lock
    lock = multiprocessing.Manager().Lock()
    
    for batch_start in range(0, len(addresses_to_process), batch_size):
        batch_end = min(batch_start + batch_size, len(addresses_to_process))
        current_batch = addresses_to_process[batch_start:batch_end]
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(extract_address_neighbors, address) for address in current_batch]
                
            new_neighbors = {}
            for future in tqdm.tqdm(
                as_completed(futures), 
                total=len(futures), 
                desc=f"Batch {batch_start//batch_size + 1}/{(len(addresses_to_process)+batch_size-1)//batch_size}"
            ):
                try:
                    address, neighbors = future.result()
                    new_neighbors[address] = list(neighbors)  # Convert to list for JSON serialization
                except Exception as e:
                    print(f"Error processing address neighbors: {e}")
            
            # Update data and save after each batch
            with lock:
                neighborhood_data.update(new_neighbors)
                with open(neighborhood_file, 'w') as f:
                    json.dump(neighborhood_data, f)
                    
    print(f"Neighborhood extraction complete, data saved to {neighborhood_file}")
    return neighborhood_data

def extract_address_neighbors(address):
    """Extract neighbors for a single address
    
    Args:
        address: Center address to find neighbors for
        
    Returns:
        Tuple of (address, neighbors_set)
    """
    neighbors = set()
    address_lower = address.lower()
    
    # Check all three transaction types
    for tx_type in TX_TYPES:
        tx_file = os.path.join(
            'Dataset', 'PonziCombine', 'RelatedTransactions', 
            tx_type, f"{address}.csv"
        )
        
        if not os.path.exists(tx_file) or os.path.getsize(tx_file) == 0:
            continue
            
        try:
            # Read transaction data, only get from and to columns
            tx_df = pd.read_csv(
                tx_file, 
                usecols=['from', 'to'] if 'from' in pd.read_csv(tx_file, nrows=0).columns else None,
                engine='c',
                dtype={'from': 'str', 'to': 'str'}  # Explicitly specify string type
            )
            
            if 'from' in tx_df.columns and 'to' in tx_df.columns:
                # Process outgoing transaction neighbors
                outgoing_mask = tx_df['from'].str.lower() == address_lower
                outgoing_neighbors = tx_df.loc[outgoing_mask, 'to'].dropna().tolist()
                neighbors.update(outgoing_neighbors)
                
                # Process incoming transaction neighbors
                incoming_mask = tx_df['to'].str.lower() == address_lower
                incoming_neighbors = tx_df.loc[incoming_mask, 'from'].dropna().tolist()
                neighbors.update(incoming_neighbors)
                
        except Exception as e:
            print(f"Error reading transaction file {tx_file}: {e}")
    
    return address, neighbors

if __name__ == "__main__":
    import argparse
    
    # 创建参数解析器
    parser = argparse.ArgumentParser(description="数据处理工具")
    parser.add_argument("--get_transactions", action="store_true", help="获取相关交易（自动处理Ponzi文件夹下所有CSV文件）")
    parser.add_argument("--get_labels", action="store_true", help="获取相关地址标签")
    parser.add_argument("--get_address_transactions", action="store_true", help="获取相关地址的交易数据")
    parser.add_argument("--extract_center_features", action="store_true", help="提取中心节点特征")
    parser.add_argument("--extract_neighbor_features", action="store_true", help="提取邻居节点特征")
    parser.add_argument("--extract_neighbors", action="store_true", help="提取并保存邻居节点关系")
    parser.add_argument("--workers", type=int, default=None, help="使用的工作线程数")
    parser.add_argument("--batch_size", type=int, default=200, help="每批处理的地址数")
    parser.add_argument("--reverse", action="store_true", help="反向处理地址列表")
    parser.add_argument("--use_reverse_file", action="store_true", help="使用address_labels_reverse.json文件")
    
    args = parser.parse_args()
    
    if args.get_transactions:
        GetAddressRelatedTransactions()  # 自动处理所有数据集
    if args.get_labels:
        GetRelatedAddressLabel(args.reverse)
    if args.get_address_transactions:
        GetRelatedAddressTransactions(args.reverse, args.use_reverse_file)
    if args.extract_center_features:
        extract_and_save_features(is_center=True, batch_size=args.batch_size, num_workers=args.workers)
    if args.extract_neighbor_features:
        extract_and_save_features(is_center=False, batch_size=args.batch_size, num_workers=args.workers)
    if args.extract_neighbors:
        extract_and_save_neighborhood_information()