import pandas as pd
import os
import torch
import numpy as np
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
import warnings
import time
warnings.filterwarnings("ignore")

def _load_all_txs_for_address(address, is_target=True):
    all_txs_dfs = []
    core_columns = {
        'blockNumber': 'str', 'from': 'str', 'to': 'str', 'value': 'str',
        'gasUsed': 'str', 'gasPrice': 'str', 'timeStamp': 'str',
        'contractAddress': 'str', 'tokenDecimal': 'str', 'tokenSymbol': 'str'
    }
    tx_type_mapping = {'Normal': 0, 'Internal': 1, 'ERC20': 2}

    base_dir = 'Dataset/PonziCombine/RelatedTransactions' if is_target else 'Dataset/PonziCombine/RelatedAddressTransactions'

    for tx_type in ['Normal', 'Internal', 'ERC20']:
        file_path = f'{base_dir}/{tx_type}/{address}.csv'
        if not os.path.exists(file_path):
            continue
        
        address_tx_df = pd.read_csv(file_path, dtype=str, engine='c')
        standard_df = pd.DataFrame()

        metadata_columns = {'contractAddress', 'tokenDecimal', 'tokenSymbol'}
        for col in core_columns.keys():
            if col in address_tx_df.columns:
                standard_df[col] = address_tx_df[col]
            else:
                standard_df[col] = '' if col in metadata_columns else '0'
        
        standard_df['tx_type'] = tx_type_mapping[tx_type]
        all_txs_dfs.append(standard_df)

    if not all_txs_dfs:
        return None

    combined_tx_df = pd.concat(all_txs_dfs, ignore_index=True)
    combined_tx_df['blockNumber'] = pd.to_numeric(combined_tx_df['blockNumber'], errors='coerce')
    combined_tx_df['timeStamp'] = pd.to_numeric(combined_tx_df['timeStamp'], errors='coerce')
    combined_tx_df.dropna(subset=['blockNumber', 'timeStamp'], inplace=True)
    return _add_unit_safe_values(combined_tx_df)


def _add_unit_safe_values(df):
    """Add native-ETH and type-aware amount columns without mixing units.

    Normal and internal transaction values are denominated in Wei. ERC20
    values are denominated in token-specific smallest units and are normalized
    with tokenDecimal when that metadata is available. Legacy ERC20 CSV files
    without tokenDecimal receive a zero amount; treating those values as ETH
    would be materially incorrect. ERC20 occurrence/type features remain usable.
    """
    result = df.copy()
    raw_value = pd.to_numeric(result['value'], errors='coerce').fillna(0.0)
    is_erc20 = result['tx_type'].eq(2)

    # NaN (rather than zero) makes pandas amount averages ignore ERC20 rows;
    # sums still resolve to zero for addresses with no native-ETH transfers.
    result['value_eth'] = np.nan
    result.loc[~is_erc20, 'value_eth'] = raw_value.loc[~is_erc20] / 1e18

    if 'tokenDecimal' in result.columns:
        decimals = pd.to_numeric(result['tokenDecimal'], errors='coerce')
    else:
        decimals = pd.Series(np.nan, index=result.index, dtype=float)
    valid_decimals = is_erc20 & decimals.notna() & decimals.between(0, 36)
    result['value_token'] = 0.0
    if valid_decimals.any():
        result.loc[valid_decimals, 'value_token'] = (
            raw_value.loc[valid_decimals]
            / np.power(10.0, decimals.loc[valid_decimals].astype(float))
        )

    # This edge feature is type-aware: ETH for native transfers and token units
    # for ERC20 transfers. tx_type is included as a separate edge feature.
    result['value_normalized'] = result['value_eth']
    result.loc[is_erc20, 'value_normalized'] = result.loc[is_erc20, 'value_token']
    return result

def _build_graph_from_df(df, center_address=None): 
    if df is None or df.empty:
        return None, None

    all_nodes = pd.unique(df[['from', 'to']].values.ravel('K'))
    addr_to_index = {addr: i for i, addr in enumerate(all_nodes)}

    edge_index, edge_attr = extract_edge_features(df, addr_to_index, center_address)
    node_feature_df = extract_node_features(df)

    node_feature_df = node_feature_df.set_index('address')
    ordered_feature_df = node_feature_df.reindex(all_nodes).fillna(0)

    center_flag = torch.zeros(len(all_nodes), 1)
    if center_address in addr_to_index:
        center_flag[addr_to_index[center_address]] = 1.0

    scaler = StandardScaler()
    feature_cols = ordered_feature_df.columns
    processed = ordered_feature_df.copy()
    for col in feature_cols:
        processed[col] = np.nan_to_num(processed[col], nan=0.0, posinf=1e6, neginf=-1e6)
        if processed[col].nunique() > 1:
            positive = processed[col] > 0
            if positive.any():
                processed.loc[positive, col] = np.log1p(processed.loc[positive, col])
        processed[col] = np.clip(processed[col], -1e6, 1e6)

    non_const = [c for c in feature_cols if processed[c].nunique() > 1]
    if non_const:
        processed[non_const] = scaler.fit_transform(processed[non_const])
    processed = np.nan_to_num(processed.values, nan=0.0)
    processed = np.clip(processed, -10, 10)

    x = torch.tensor(processed, dtype=torch.float32)
    x = torch.cat([x, center_flag], dim=1)  

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data, addr_to_index

def extract_edge_features(df, addr_to_index, center_address=None):
    edges_df = df.copy()
    edges_df['from_idx'] = edges_df['from'].map(addr_to_index)
    edges_df['to_idx'] = edges_df['to'].map(addr_to_index)
    edges_df.dropna(subset=['from_idx', 'to_idx'], inplace=True)

    edge_index = torch.tensor([edges_df['from_idx'].values, edges_df['to_idx'].values], dtype=torch.long)

    if 'value_normalized' not in edges_df.columns:
        edges_df = _add_unit_safe_values(edges_df)
    edges_df['gasUsed'] = pd.to_numeric(edges_df['gasUsed'], errors='coerce').fillna(0)
    edges_df['gasPrice'] = pd.to_numeric(edges_df['gasPrice'], errors='coerce').fillna(0) / 1e9
    edges_df['timeStamp'] = pd.to_numeric(edges_df['timeStamp'], errors='coerce').fillna(0)

    features = edges_df[['value_normalized', 'gasUsed', 'gasPrice', 'timeStamp', 'tx_type']].copy()
    features.rename(columns={'value_normalized': 'value'}, inplace=True)
    for col in ['value', 'gasUsed', 'gasPrice', 'timeStamp']:
        features[col] = np.nan_to_num(features[col])
        features[col] = np.log1p(features[col].clip(0))
        features[col] = np.clip(features[col], -1e6, 1e6)

    scaler = StandardScaler()
    num_cols = ['value', 'gasUsed', 'gasPrice', 'timeStamp']
    features[num_cols] = scaler.fit_transform(features[num_cols])
    features[num_cols] = np.clip(features[num_cols], -10, 10)

    is_center_edge = torch.zeros(len(edges_df), 1)
    if center_address:
        center_idx = addr_to_index.get(center_address, -1)
        if center_idx != -1:
            mask_from = edges_df['from_idx'] == center_idx
            mask_to = edges_df['to_idx'] == center_idx
            is_center_edge[mask_from | mask_to] = 1.0

    edge_attr = torch.tensor(features.values, dtype=torch.float32)
    edge_attr = torch.cat([edge_attr, is_center_edge], dim=1)

    return edge_index, edge_attr

def build_two_layer_multigraph(address, max_neighbors=200, center_address=None, max_txs=200):
    target_tx_df = _load_all_txs_for_address(address, is_target=True)
    if target_tx_df is None:
        return None, None, []
 
    deposit_addresses = []
    withdrawal_addresses = []
  
    deposit_amounts = {}
    withdrawal_amounts = {}
    
    for _, row in target_tx_df.iterrows():
        from_addr = row['from']
        to_addr = row['to']
        # Behavioral tiers model native-ETH redistribution. ERC20 quantities
        # are token-specific and must not be compared or summed with ETH.
        value = float(row['value_eth'])
        
        if to_addr == address and value > 0:
            deposit_addresses.append(from_addr)
            deposit_amounts[from_addr] = deposit_amounts.get(from_addr, 0) + value
        
        elif from_addr == address and value > 0:
            withdrawal_addresses.append(to_addr)
            withdrawal_amounts[to_addr] = withdrawal_amounts.get(to_addr, 0) + value
    
    from collections import Counter
    deposit_counter = Counter(deposit_addresses)
    withdrawal_counter = Counter(withdrawal_addresses)
    
    important_addresses = set()
    
    both_behavior_addrs = set(deposit_counter.keys()) & set(withdrawal_counter.keys())
    for addr in both_behavior_addrs:
        if deposit_counter[addr] >= 2 and withdrawal_counter[addr] >= 1:
            important_addresses.add(addr)
    
    only_withdrawal_addrs = set(withdrawal_counter.keys()) - set(deposit_counter.keys())
    for addr in only_withdrawal_addrs:
        if withdrawal_counter[addr] >= 2 or withdrawal_amounts.get(addr, 0) > 0:
            important_addresses.add(addr)
    
    large_depositors = []
    for addr, amount in deposit_amounts.items():
        if amount > np.percentile(list(deposit_amounts.values()), 80):
            large_depositors.append(addr)
    
    all_important = list(important_addresses) + large_depositors
    if len(all_important) > max_neighbors:
        sorted_addrs = []
        
        both_behavior = [addr for addr in all_important if addr in both_behavior_addrs]
        sorted_addrs.extend(both_behavior)
        
        only_withdrawal = [addr for addr in all_important if addr in only_withdrawal_addrs]
        sorted_addrs.extend(only_withdrawal)
        
        large_deposit_only = [addr for addr in all_important 
                            if addr not in both_behavior_addrs and addr not in only_withdrawal_addrs]
        sorted_addrs.extend(large_deposit_only)
        
        all_neighbors = sorted_addrs[:max_neighbors]
    else:
        all_neighbors = all_important
  
    neighbor_graphs = []
    neighbor_dfs = []
    
    for neighbor in all_neighbors:
        neighbor_tx_df = _load_all_txs_for_address(neighbor, is_target=False)
        if neighbor_tx_df is None:
            continue
        
        if len(neighbor_tx_df) > max_txs:
            neighbor_tx_df = neighbor_tx_df.nlargest(max_txs, 'timeStamp', keep='first')
        
        neighbor_dfs.append(neighbor_tx_df)
        
        neighbor_graph, neighbor_addr_to_index = _build_graph_from_df(neighbor_tx_df, center_address=None)
        if neighbor_graph is not None:
            neighbor_graphs.append((neighbor, neighbor_graph, neighbor_addr_to_index))
    
    if neighbor_dfs:
        combined_df = pd.concat([target_tx_df] + neighbor_dfs, ignore_index=True)
    else:
        combined_df = target_tx_df
    
    main_graph, main_addr_to_index = _build_graph_from_df(combined_df, center_address=address)

    if max_neighbors == 0:
        neighbor_graphs = []

    return main_graph, main_addr_to_index, neighbor_graphs

def extract_node_features(df):
    if 'value_eth' not in df.columns or 'value_normalized' not in df.columns:
        df = _add_unit_safe_values(df)
    df['gasUsed'] = pd.to_numeric(df['gasUsed'], errors='coerce').fillna(0)
    df['gasPrice_gwei'] = pd.to_numeric(df['gasPrice'], errors='coerce').fillna(0) / 1e9
    df['timeStamp'] = pd.to_numeric(df['timeStamp'], errors='coerce').fillna(0)

    all_possible_columns = [
        'out_degree', 'unique_to_addresses', 'daily_tx_frequency',
        'total_ether_sent', 'avg_ether_sent', 'max_ether_sent', 'min_ether_sent', 'std_ether_sent',
        'avg_gas_price', 'total_gas_used', 'avg_gas_used',
        'normal_tx_count', 'internal_tx_count', 'erc20_tx_count',
        'normal_tx_ratio', 'internal_tx_ratio', 'erc20_tx_ratio',
        'in_degree', 'unique_from_addresses',
        'total_ether_received', 'avg_ether_received', 'max_ether_received', 'min_ether_received', 'std_ether_received',
        'lifetime_blocks', 'lifetime_days', 'tx_count',
        'net_flow', 'in_out_ratio',
        'counterparty_diversity',
        'large_tx_out_count', 'large_tx_out_ratio', 'large_tx_in_count', 'large_tx_in_ratio',
        'tx_intensity', 'economic_efficiency', 'concentration_ratio'
    ]

    out_features = df.groupby('from').agg(
        out_degree=('to', 'count'),
        unique_to_addresses=('to', 'nunique'),
        daily_tx_frequency=('timeStamp', lambda x: len(x) / (max((x.max() - x.min()) / 86400, 1))),
        
        total_ether_sent=('value_eth', 'sum'),
        avg_ether_sent=('value_eth', 'mean'),
        max_ether_sent=('value_eth', 'max'),
        min_ether_sent=('value_eth', 'min'),
        std_ether_sent=('value_eth', 'std'),
        
        avg_gas_price=('gasPrice_gwei', 'mean'),
        total_gas_used=('gasUsed', 'sum'),
        avg_gas_used=('gasUsed', 'mean'),
        
        normal_tx_count=('tx_type', lambda x: (x == 0).sum()),
        internal_tx_count=('tx_type', lambda x: (x == 1).sum()),
        erc20_tx_count=('tx_type', lambda x: (x == 2).sum())
    ).rename_axis('address')

    total_tx = out_features['out_degree']
    out_features['normal_tx_ratio'] = out_features['normal_tx_count'] / total_tx.replace(0, 1)
    out_features['internal_tx_ratio'] = out_features['internal_tx_count'] / total_tx.replace(0, 1)
    out_features['erc20_tx_ratio'] = out_features['erc20_tx_count'] / total_tx.replace(0, 1)

    in_features = df.groupby('to').agg(
        in_degree=('from', 'count'),
        unique_from_addresses=('from', 'nunique'),
        total_ether_received=('value_eth', 'sum'),
        avg_ether_received=('value_eth', 'mean'),
        max_ether_received=('value_eth', 'max'),
        min_ether_received=('value_eth', 'min'),
        std_ether_received=('value_eth', 'std'),
    ).rename_axis('address')

    from_blocks = df[['from', 'blockNumber', 'timeStamp']].rename(columns={'from': 'address'})
    to_blocks = df[['to', 'blockNumber', 'timeStamp']].rename(columns={'to': 'address'})
    all_blocks = pd.concat([from_blocks, to_blocks])
    
    lifetime_features = all_blocks.groupby('address').agg(
        min_block=('blockNumber', 'min'),
        max_block=('blockNumber', 'max'),
        min_timestamp=('timeStamp', 'min'),
        max_timestamp=('timeStamp', 'max'),
        tx_count=('blockNumber', 'count')
    )
    
    lifetime_features['lifetime_blocks'] = lifetime_features['max_block'] - lifetime_features['min_block']
    lifetime_features['lifetime_seconds'] = lifetime_features['max_timestamp'] - lifetime_features['min_timestamp']
    lifetime_features['lifetime_days'] = lifetime_features['lifetime_seconds'] / max(86400, 1)
    
    financial_features = pd.DataFrame(index=pd.unique(df[['from', 'to']].values.ravel('K')))
    financial_features['net_flow'] = 0
    financial_features['in_out_ratio'] = 0
    
    sent_amounts = df.groupby('from')['value_eth'].sum()
    received_amounts = df.groupby('to')['value_eth'].sum()
    
    financial_features.loc[sent_amounts.index, 'net_flow'] -= sent_amounts
    financial_features.loc[received_amounts.index, 'net_flow'] += received_amounts
    
    out_degree = df.groupby('from').size()
    in_degree = df.groupby('to').size()
    
    financial_features['in_out_ratio'] = in_degree / (out_degree + 1)
    financial_features['in_out_ratio'] = financial_features['in_out_ratio'].fillna(0)
    
    counterparty_features = pd.DataFrame(index=pd.unique(df[['from', 'to']].values.ravel('K')))
    
    from_counterparties = df.groupby('from')['to'].apply(lambda x: x.nunique())
    to_counterparties = df.groupby('to')['from'].apply(lambda x: x.nunique())
    
    counterparty_features['counterparty_diversity'] = 0
    counterparty_features.loc[from_counterparties.index, 'counterparty_diversity'] += from_counterparties
    counterparty_features.loc[to_counterparties.index, 'counterparty_diversity'] += to_counterparties
    
    large_tx_threshold = df['value_eth'].quantile(0.9) if len(df) > 0 else 0
    
    large_out_tx = pd.DataFrame(index=pd.unique(df[['from', 'to']].values.ravel('K')))
    large_in_tx = pd.DataFrame(index=pd.unique(df[['from', 'to']].values.ravel('K')))
    
    large_out_tx['large_tx_out_count'] = 0
    large_out_tx['large_tx_out_ratio'] = 0
    large_in_tx['large_tx_in_count'] = 0
    large_in_tx['large_tx_in_ratio'] = 0
    
    if large_tx_threshold > 0:
        large_out_df = df[df['value_eth'] > large_tx_threshold].groupby('from').agg(
            large_tx_out_count=('value_eth', 'count'),
            large_tx_out_ratio=('value_eth', lambda x: len(x) / max(len(df[df['from'] == x.name]), 1))
        )
        
        large_in_df = df[df['value_eth'] > large_tx_threshold].groupby('to').agg(
            large_tx_in_count=('value_eth', 'count'),
            large_tx_in_ratio=('value_eth', lambda x: len(x) / max(len(df[df['to'] == x.name]), 1))
        )
        
        if not large_out_df.empty:
            large_out_tx = large_out_tx.combine_first(large_out_df)
        if not large_in_df.empty:
            large_in_tx = large_in_tx.combine_first(large_in_df)
    
    all_features_list = []
    
    feature_groups = [
        (in_features, ''),
        (out_features, ''),
        (lifetime_features[['lifetime_blocks', 'lifetime_days', 'tx_count']], ''),
        (financial_features, ''),
        (counterparty_features, ''),
        (large_out_tx, ''),
        (large_in_tx, '')
    ]
    
    for features, prefix in feature_groups:
        if not features.empty:
            if prefix:
                features_prefixed = features.add_prefix(prefix)
            else:
                features_prefixed = features
            all_features_list.append(features_prefixed)
    
    if all_features_list:
        all_features = pd.concat(all_features_list, axis=1)
    else:
        all_features = pd.DataFrame(index=pd.unique(df[['from', 'to']].values.ravel('K')))
    
    all_features = all_features.fillna(0)
    all_features = all_features.replace([np.inf, -np.inf], 0)
    
    all_features['tx_intensity'] = all_features.get('tx_count', 0) / (all_features.get('lifetime_days', 0) + 1)
    
    total_gas = all_features.get('total_gas_used', 0)
    total_eth_received = all_features.get('total_ether_received', 0)
    total_eth_sent = all_features.get('total_ether_sent', 0)
    all_features['economic_efficiency'] = (total_eth_received + total_eth_sent) / (total_gas + 1)
    
    counterparty_div = all_features.get('counterparty_diversity', 0)
    tx_count = all_features.get('tx_count', 0)
    all_features['concentration_ratio'] = counterparty_div / (tx_count + 1)
    all_features = all_features.fillna(0)
    all_features = all_features.replace([np.inf, -np.inf], 0)
    
    for col in all_possible_columns:
        if col not in all_features.columns:
            all_features[col] = 0
    all_features = all_features.reindex(columns=all_possible_columns, fill_value=0)
    
    all_nodes = pd.unique(df[['from', 'to']].values.ravel('K'))
    feature_df = all_features.reindex(all_nodes).fillna(0)
    
    return feature_df.reset_index().rename(columns={'index': 'address'})

if __name__ == '__main__':
    start_time = time.time()
    main_graph, main_addr_to_index, neighbor_graphs = build_two_layer_multigraph('0x8f13a1d43408b6434dd10e161361386f3952d665', max_neighbors=200, max_txs=200)
    if main_graph is not None:
        print("Main two-layer graph constructed successfully!")
        print(f"Number of nodes in main graph: {main_graph.x.shape[0]}")
        print(f"Number of edges in main graph: {main_graph.edge_index.shape[1]}")
        print(f"Node feature dimension: {main_graph.x.shape[1]}")
        print(f"Edge feature dimension: {main_graph.edge_attr.shape[1]}")
        print(f"Number of neighbor graphs: {len(neighbor_graphs)}")
        for neighbor, graph, idx in neighbor_graphs:
            print(f"Neighbor {neighbor}: nodes={graph.x.shape[0]}, edges={graph.edge_index.shape[1]}")
        print(f"Time taken: {time.time() - start_time} seconds")
    else: 
        print("Failed to construct graph")
