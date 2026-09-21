import pandas as pd
import os
import torch
import numpy as np
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
import warnings
import time
from pathlib import Path
from Data_protocol import ROOT, END_BLOCK

def deduplicate_transactions(df):
    result = df.copy()
    identities = []
    for row in result.to_dict('records'):
        tx_hash = str(row.get('hash', '')).strip().lower()
        tx_type = int(row['tx_type'])
        identity = None
        if tx_hash and tx_hash != 'nan':
            if tx_type == 0:
                identity = f'0:{tx_hash}'
            else:
                discriminator = str(row.get('traceId' if tx_type == 1 else 'logIndex', '')).strip()
                if discriminator and discriminator != 'nan':
                    identity = f'{tx_type}:{tx_hash}:{discriminator}'
        identities.append(identity)
    keys = pd.Series(identities, index=result.index, dtype=object)
    return result.loc[keys.isna() | ~keys.duplicated()].reset_index(drop=True)


def _load_all_txs_for_address(address, is_target=True, end_block=END_BLOCK,
                              cutoff_timestamp=None, data_root=None):
    address = address.strip().lower()
    data_root = Path(data_root) if data_root is not None else ROOT / 'Dataset' / 'PonziCombine'
    base_dir = data_root / ('RelatedTransactions' if is_target else 'RelatedAddressTransactions')
    parts = []
    if not any((base_dir / name / f'{address}.csv').exists() for name in ['Normal', 'Internal', 'ERC20']):
        raise FileNotFoundError(f'No transaction files for {address} in {base_dir}')
    metadata = ['contractAddress', 'tokenDecimal', 'tokenSymbol', 'hash', 'traceId',
                'logIndex', 'transactionIndex', 'isError', 'txreceipt_status']
    for tx_type, name in enumerate(['Normal', 'Internal', 'ERC20']):
        path = base_dir / name / f'{address}.csv'
        if not path.exists():
            continue
        try:
            part = pd.read_csv(path, dtype=str).fillna('')
        except pd.errors.EmptyDataError:
            continue
        if part.empty:
            continue
        required = {'blockNumber', 'timeStamp', 'from', 'to', 'value'}
        if not required.issubset(part.columns):
            raise ValueError(f'Missing transaction columns in {path}: {sorted(required - set(part.columns))}')
        for name in metadata:
            if name not in part:
                part[name] = ''
        for name in ['gasUsed', 'gasPrice']:
            if name not in part:
                part[name] = '0'
        part['tx_type'] = tx_type
        for name in ['from', 'to', 'contractAddress', 'hash']:
            part[name] = part[name].astype(str).str.strip().str.lower()
        if tx_type == 0:
            creation = part['to'].eq('')
            part.loc[creation, 'to'] = part.loc[creation, 'contractAddress']
        valid = part['from'].str.fullmatch(r'0x[0-9a-f]{40}') & part['to'].str.fullmatch(r'0x[0-9a-f]{40}')
        valid &= part['from'].eq(address) | part['to'].eq(address)
        valid &= ~part['isError'].isin(['1', 'true']) & ~part['txreceipt_status'].eq('0')
        for name in ['blockNumber', 'timeStamp']:
            part[name] = pd.to_numeric(part[name], errors='coerce')
            valid &= part[name].notna() & part[name].ge(0)
        valid &= part['blockNumber'].le(end_block)
        if cutoff_timestamp is not None:
            valid &= part['timeStamp'].le(cutoff_timestamp)
        part = part.loc[valid].copy()
        if not part.empty:
            parts.append(part)
    if not parts:
        return None
    combined = deduplicate_transactions(pd.concat(parts, ignore_index=True))
    combined = combined.sort_values(['timeStamp', 'blockNumber', 'hash', 'traceId', 'logIndex'], kind='stable')
    return _add_unit_safe_values(combined.reset_index(drop=True))


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

    all_nodes = sorted(pd.unique(df[['from', 'to']].values.ravel('K')))
    addr_to_index = {addr: i for i, addr in enumerate(all_nodes)}

    edge_index, edge_attr = extract_edge_features(df, addr_to_index, center_address)
    node_feature_df = extract_node_features(df)

    node_feature_df = node_feature_df.set_index('address')
    ordered_feature_df = node_feature_df.reindex(all_nodes).fillna(0)

    center_flag = torch.zeros(len(all_nodes), 1)
    if center_address in addr_to_index:
        center_flag[addr_to_index[center_address]] = 1.0

    processed = np.nan_to_num(ordered_feature_df.to_numpy(dtype=float), nan=0.0, posinf=1e6, neginf=-1e6)
    processed = np.sign(processed) * np.log1p(np.abs(processed))
    processed = np.clip(StandardScaler().fit_transform(processed), -10, 10)

    x = torch.tensor(processed, dtype=torch.float32)
    x = torch.cat([x, center_flag], dim=1)  

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data, addr_to_index

def extract_edge_features(df, addr_to_index, center_address=None):
    edges_df = df.copy()
    edges_df['from_idx'] = edges_df['from'].map(addr_to_index)
    edges_df['to_idx'] = edges_df['to'].map(addr_to_index)
    edges_df.dropna(subset=['from_idx', 'to_idx'], inplace=True)

    edge_index = torch.tensor(np.stack([edges_df['from_idx'].values, edges_df['to_idx'].values]), dtype=torch.long)

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
            is_center_edge[torch.tensor((mask_from | mask_to).to_numpy(), dtype=torch.bool)] = 1.0

    edge_attr = torch.tensor(features.values, dtype=torch.float32)
    edge_attr = torch.cat([edge_attr, is_center_edge], dim=1)

    return edge_index, edge_attr

def select_neighbors(target_tx_df, address, max_neighbors=200):
    if max_neighbors < 0:
        raise ValueError('max_neighbors must be nonnegative')
    address = address.strip().lower()
    positive = target_tx_df[target_tx_df['value_eth'].gt(0)]
    deposits = positive[positive['to'].eq(address) & ~positive['from'].eq(address)]
    withdrawals = positive[positive['from'].eq(address) & ~positive['to'].eq(address)]
    deposited = deposits.groupby('from')['value_eth'].sum()
    withdrawn = withdrawals.groupby('to')['value_eth'].sum()
    withdrawal_counts = withdrawals.groupby('to').size()
    shared = deposited.index.intersection(withdrawn.index)
    profitable = set(shared[withdrawn.reindex(shared).to_numpy() > deposited.reindex(shared).to_numpy()])
    withdrawal_only = set(withdrawal_counts[withdrawal_counts.ge(2)].index) - set(deposited.index)
    large = set(deposited[deposited.gt(deposited.quantile(0.8))].index) if not deposited.empty else set()
    candidates = profitable | withdrawal_only | large
    interactions = target_tx_df[target_tx_df['from'].eq(address) | target_tx_df['to'].eq(address)]
    outgoing = interactions[['to', 'timeStamp']].rename(columns={'to': 'address'})
    incoming = interactions[['from', 'timeStamp']].rename(columns={'from': 'address'})
    last_seen = pd.concat([outgoing, incoming]).groupby('address')['timeStamp'].max().to_dict()
    return sorted(candidates, key=lambda node: (-last_seen[node], node))[:max_neighbors]


def build_two_layer_multigraph(address, max_neighbors=200, center_address=None, max_txs=200,
                               end_block=END_BLOCK, cutoff_timestamp=None,
                               history_fraction=1.0, data_root=None):
    if max_neighbors < 0 or max_txs < 1 or not 0 < history_fraction <= 1:
        raise ValueError('Require K >= 0, M >= 1 and 0 < history_fraction <= 1')
    address = address.strip().lower()
    target_tx_df = _load_all_txs_for_address(address, True, end_block, cutoff_timestamp, data_root)
    if target_tx_df is None or target_tx_df.empty:
        return None, None, []
    if history_fraction < 1:
        observed_count = max(1, int(np.ceil(len(target_tx_df) * history_fraction)))
        fraction_cutoff = float(target_tx_df.iloc[observed_count - 1]['timeStamp'])
        cutoff_timestamp = fraction_cutoff if cutoff_timestamp is None else min(cutoff_timestamp, fraction_cutoff)
        target_tx_df = target_tx_df[target_tx_df['timeStamp'].le(cutoff_timestamp)].copy()
    neighbors = select_neighbors(target_tx_df, address, max_neighbors)
    neighbor_graphs = []
    neighbor_dfs = []
    for neighbor in neighbors:
        neighbor_tx_df = _load_all_txs_for_address(neighbor, False, end_block, cutoff_timestamp, data_root)
        if neighbor_tx_df is None or neighbor_tx_df.empty:
            continue
        neighbor_dfs.append(neighbor_tx_df.tail(max_txs))
        graph, indices = _build_graph_from_df(neighbor_tx_df, center_address=neighbor)
        if graph is not None:
            neighbor_graphs.append((neighbor, graph, indices))
    combined_df = deduplicate_transactions(pd.concat([target_tx_df] + neighbor_dfs, ignore_index=True))
    main_graph, main_indices = _build_graph_from_df(combined_df, center_address=address)
    return main_graph, main_indices, neighbor_graphs


def extract_node_features(df):
    df = df.copy()
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
    financial_features['net_flow'] = 0.0
    financial_features['in_out_ratio'] = 0
    
    sent_amounts = df.groupby('from')['value_eth'].sum()
    received_amounts = df.groupby('to')['value_eth'].sum()
    
    financial_features.loc[sent_amounts.index, 'net_flow'] -= sent_amounts
    financial_features.loc[received_amounts.index, 'net_flow'] += received_amounts
    
    out_degree = df.groupby('from').size()
    in_degree = df.groupby('to').size()
    
    financial_features['in_out_ratio'] = in_degree.reindex(financial_features.index, fill_value=0) / (out_degree.reindex(financial_features.index, fill_value=0) + 1)
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
    
    if pd.notna(large_tx_threshold) and large_tx_threshold > 0:
        large_rows = df[df['value_eth'].gt(large_tx_threshold)]
        large_out_tx['large_tx_out_count'] = large_rows.groupby('from').size().reindex(large_out_tx.index, fill_value=0)
        large_in_tx['large_tx_in_count'] = large_rows.groupby('to').size().reindex(large_in_tx.index, fill_value=0)
        large_out_tx['large_tx_out_ratio'] = large_out_tx['large_tx_out_count'] / out_degree.reindex(large_out_tx.index, fill_value=0).clip(lower=1)
        large_in_tx['large_tx_in_ratio'] = large_in_tx['large_tx_in_count'] / in_degree.reindex(large_in_tx.index, fill_value=0).clip(lower=1)

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
