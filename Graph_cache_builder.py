import argparse
import hashlib
import json
from pathlib import Path

import torch
from tqdm import tqdm

from Data_protocol import ROOT, END_BLOCK, default_cache_dir, file_digest, graph_config, load_labels, write_json
from Graph_construction import build_two_layer_multigraph


def source_digest(data_root):
    files = sorted(path for folder in ['RelatedTransactions', 'RelatedAddressTransactions'] for path in (data_root / folder).rglob('*.csv'))
    if not files:
        raise FileNotFoundError(f'No transaction CSV files under {data_root}')
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(data_root)).replace('\\', '/').encode())
        digest.update(file_digest(path).encode())
    return digest.hexdigest()


def build_and_cache_dataset(address_label_csv, cache_dir, max_neighbors=200, max_txs=200,
                            end_block=END_BLOCK, history_fraction=1.0, data_root=None):
    records = load_labels(address_label_csv)
    if records.empty:
        raise ValueError('Empty dataset')
    data_root = Path(data_root) if data_root is not None else ROOT / 'Dataset' / 'PonziCombine'
    fingerprint = source_digest(data_root)
    config = graph_config(max_neighbors, max_txs, end_block, history_fraction)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / 'manifest.json'
    if manifest_path.exists():
        old_manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if old_manifest['config'] != config:
            raise ValueError('Cache directory belongs to a different graph configuration; choose another directory')
    samples = []
    excluded_count = 0
    for row in tqdm(records.itertuples(index=False), total=len(records), desc='Building graphs'):
        address, label = row.Address, int(row.VerifiedLabel)
        path = cache_dir / f'{address}.pt'
        data = None
        if path.exists():
            cached = torch.load(path, map_location='cpu', weights_only=False)
            if (cached.get('config') == config and cached.get('source_sha256') == fingerprint
                    and cached.get('address') == address and int(cached['label']) == label):
                data = cached
        if data is None:
            main_graph, _, neighbors = build_two_layer_multigraph(
                address, max_neighbors=max_neighbors, max_txs=max_txs, end_block=end_block,
                history_fraction=history_fraction, data_root=data_root,
            )
            if main_graph is None:
                excluded_count += 1
                print(f'Excluded {address}: no successful transactions within the observation window')
                continue
            data = {'address': address, 'label': torch.tensor(label, dtype=torch.long),
                    'main_graph': main_graph, 'neighbor_graphs': [graph for _, graph, _ in neighbors],
                    'neighbor_addresses': [node for node, _, _ in neighbors], 'num_neighbors': len(neighbors),
                    'config': config, 'source_sha256': fingerprint}
            temporary = path.with_suffix('.pt.tmp')
            torch.save(data, temporary)
            temporary.replace(path)
        samples.append({'address': address, 'label': label, 'sha256': file_digest(path)})
    if not samples:
        raise ValueError('No usable graphs were produced')
    manifest = {'config': config, 'dataset_sha256': file_digest(address_label_csv),
                'source_sha256': fingerprint, 'source_root': str(data_root.resolve()),
                'unique_input_count': len(records), 'excluded_count': excluded_count, 'samples': samples}
    write_json(manifest_path, manifest)
    print(f'Cached {len(samples)} graphs; excluded {excluded_count}; directory: {cache_dir}')
    return manifest


def main():
    parser = argparse.ArgumentParser(description='Build versioned PonziScope graph caches')
    parser.add_argument('--dataset', choices=['Dataset1', 'Dataset2'], default='Dataset1')
    parser.add_argument('--dataset_csv')
    parser.add_argument('--cache_dir')
    parser.add_argument('--data_root')
    parser.add_argument('--max_neighbors', type=int, default=200)
    parser.add_argument('--max_txs', type=int, default=200)
    parser.add_argument('--end_block', type=int, default=END_BLOCK)
    parser.add_argument('--history_fraction', type=float, default=1.0)
    args = parser.parse_args()
    config = graph_config(args.max_neighbors, args.max_txs, args.end_block, args.history_fraction)
    csv_path = Path(args.dataset_csv) if args.dataset_csv else ROOT / 'Dataset' / 'Ponzi' / f'{args.dataset}.csv'
    cache_dir = Path(args.cache_dir) if args.cache_dir else default_cache_dir(args.dataset, config)
    build_and_cache_dataset(csv_path, cache_dir, args.max_neighbors, args.max_txs,
                            args.end_block, args.history_fraction, args.data_root)


if __name__ == '__main__':
    main()
