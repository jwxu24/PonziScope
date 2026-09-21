import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
END_BLOCK = 24996367
CACHE_VERSION = 2
NODE_DIM = 38
EDGE_DIM = 6


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_labels(path):
    frame = pd.read_csv(path, dtype=str)
    columns = {name.strip().lower().replace('_', '').replace(' ', ''): name for name in frame}
    address_col = next((columns[k] for k in ['address', 'addr', 'contractaddress'] if k in columns), None)
    label_col = next((columns[k] for k in ['verifiedlabel', 'label', 'isponzi', 'target', 'y'] if k in columns), None)
    if address_col is None or label_col is None:
        raise ValueError('Dataset must contain address and binary label columns')
    labels = pd.to_numeric(frame[label_col], errors='coerce')
    addresses = frame[address_col].fillna('').str.strip().str.lower()
    invalid = ~addresses.str.fullmatch(r'0x[0-9a-f]{40}') | ~labels.isin([0, 1])
    if invalid.any():
        raise ValueError(f'Invalid addresses or labels at CSV rows {(frame.index[invalid] + 2).tolist()}')
    records = pd.DataFrame({'Address': addresses, 'VerifiedLabel': labels.astype(int)})
    conflicts = records.groupby('Address')['VerifiedLabel'].nunique()
    conflicts = conflicts[conflicts > 1].index.tolist()
    if conflicts:
        raise ValueError(f'Conflicting labels for {len(conflicts)} addresses; resolve the source labels first: {conflicts}')
    return records.drop_duplicates('Address').sort_values('Address').reset_index(drop=True)


def graph_config(max_neighbors=200, max_txs=200, end_block=END_BLOCK, history_fraction=1.0):
    if max_neighbors < 0 or max_txs < 1 or end_block < 0 or not 0 < history_fraction <= 1:
        raise ValueError('Require K >= 0, M >= 1, end_block >= 0 and 0 < history_fraction <= 1')
    return {
        'cache_version': CACHE_VERSION,
        'max_neighbors': max_neighbors,
        'max_txs': max_txs,
        'end_block': end_block,
        'history_fraction': history_fraction,
        'node_dim': NODE_DIM,
        'edge_dim': EDGE_DIM,
        'graph_code_sha256': file_digest(ROOT / 'Graph_construction.py'),
    }


def default_cache_dir(dataset, config):
    suffix = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
    return ROOT / 'Dataset' / f'graph_cache_multigraph_{dataset}_v{CACHE_VERSION}_{suffix}'


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)
