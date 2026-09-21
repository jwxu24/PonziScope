import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Subset
from torch_geometric.data import Batch
from torch_geometric.nn import GATConv, global_mean_pool

from Data_protocol import ROOT, END_BLOCK, NODE_DIM, EDGE_DIM, default_cache_dir, file_digest, graph_config, load_labels, write_json


class MultiGraphDataset(torch.utils.data.Dataset):
    def __init__(self, cache_dir, expected_config, csv_path):
        self.cache_dir = Path(cache_dir)
        manifest_path = self.cache_dir / 'manifest.json'
        if not manifest_path.exists():
            raise ValueError('Graph manifest missing; rebuild with Graph_cache_builder.py')
        self.manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if self.manifest['config'] != expected_config:
            raise ValueError('Graph configuration or code changed; rebuild the cache')
        if self.manifest['dataset_sha256'] != file_digest(csv_path):
            raise ValueError('Dataset changed; rebuild the cache')
        labels = load_labels(csv_path).set_index('Address')['VerifiedLabel'].to_dict()
        self.samples = sorted(self.manifest['samples'], key=lambda row: row['address'])
        addresses = [row['address'] for row in self.samples]
        if not addresses or len(addresses) != len(set(addresses)):
            raise ValueError('Manifest must contain nonempty, unique target addresses')
        for sample in self.samples:
            if labels.get(sample['address']) != sample['label']:
                raise ValueError(f'Label mismatch for {sample["address"]}')
            path = self.cache_dir / f'{sample["address"]}.pt'
            if not path.exists() or file_digest(path) != sample['sha256']:
                raise ValueError(f'Missing or changed graph: {path}')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        data = torch.load(self.cache_dir / f'{sample["address"]}.pt', map_location='cpu', weights_only=False)
        if data['address'] != sample['address'] or int(data['label']) != sample['label']:
            raise ValueError('Graph identity does not match manifest')
        if data['config'] != self.manifest['config']:
            raise ValueError('Graph configuration does not match manifest')
        return data

    @staticmethod
    def collate_fn(samples):
        return {
            'main_batch': Batch.from_data_list([s['main_graph'] for s in samples]),
            'neighbor_batches': [Batch.from_data_list(s['neighbor_graphs']) if s['neighbor_graphs'] else None for s in samples],
            'labels': torch.stack([s['label'] for s in samples]),
            'addresses': [s['address'] for s in samples],
        }


class SingleGraphEncoder(nn.Module):
    def __init__(self, node_dim=NODE_DIM, edge_dim=EDGE_DIM, hidden_dim=128, out_dim=128):
        super().__init__()
        self.conv1 = GATConv(node_dim, hidden_dim, heads=4, concat=True, edge_dim=edge_dim)
        self.conv2 = GATConv(hidden_dim * 4, out_dim, heads=1, concat=False, edge_dim=edge_dim)
        self.norm1 = nn.LayerNorm(hidden_dim * 4)
        self.norm2 = nn.LayerNorm(out_dim)

    def forward(self, data):
        x = F.elu(self.norm1(self.conv1(data.x, data.edge_index, data.edge_attr)))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.elu(self.norm2(self.conv2(x, data.edge_index, data.edge_attr)))
        return global_mean_pool(x, data.batch)


class MultiGraphClassifier(nn.Module):
    def __init__(self, emb_dim=128, num_heads=8):
        super().__init__()
        self.encoder = SingleGraphEncoder(out_dim=emb_dim)
        self.attn = nn.MultiheadAttention(emb_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(emb_dim)
        self.neighbor_gate = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.Sigmoid())
        self.classifier = nn.Sequential(
            nn.Linear(emb_dim, 128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 2),
        )

    def forward(self, main_batch, neighbor_batches):
        main_emb = self.encoder(main_batch)
        sequences = []
        for i, neighbors in enumerate(neighbor_batches):
            sequence = main_emb[i:i + 1]
            if neighbors is not None and neighbors.num_nodes > 0:
                embedding = self.encoder(neighbors)
                sequence = torch.cat([sequence, embedding * self.neighbor_gate(embedding)], dim=0)
            sequences.append(sequence)
        lengths = torch.tensor([len(s) for s in sequences], device=main_emb.device)
        padded = nn.utils.rnn.pad_sequence(sequences, batch_first=True)
        mask = torch.arange(padded.size(1), device=padded.device)[None, :] >= lengths[:, None]
        fused, _ = self.attn(padded, padded, padded, key_padding_mask=mask, need_weights=False)
        return self.classifier(self.norm(fused[:, 0, :]))


class ClassBalancedFocalLoss(nn.Module):
    def __init__(self, samples_per_class, beta=0.999, gamma=3.0):
        super().__init__()
        counts = np.asarray(samples_per_class, dtype=float)
        if counts.shape != (2,) or np.any(counts <= 0) or not 0 <= beta < 1 or gamma < 0:
            raise ValueError('Require both classes, 0 <= beta < 1 and gamma >= 0')
        weights = (1 - beta) / (1 - np.power(beta, counts))
        self.register_buffer('class_weights', torch.tensor(weights, dtype=torch.float32))
        self.gamma = gamma

    def forward(self, logits, targets):
        log_p = F.log_softmax(logits, dim=1).gather(1, targets[:, None]).squeeze(1)
        return (-self.class_weights[targets] * (1 - log_p.exp()) ** self.gamma * log_p).mean()


def find_best_threshold(y_true, y_prob):
    candidates = [(f1_score(y_true, np.asarray(y_prob) >= t, zero_division=0), -abs(t - 0.5), -t, t) for t in np.linspace(0.05, 0.95, 91)]
    best = max(candidates)
    return float(best[3]), float(best[0])


def score_predictions(y_true, y_prob, threshold):
    prediction = (np.asarray(y_prob) >= threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_true, prediction, average='binary', zero_division=0)
    both = len(np.unique(y_true)) == 2
    return {'Precision': float(p), 'Recall': float(r), 'F1': float(f1),
            'AUC': float(roc_auc_score(y_true, y_prob)) if both else None,
            'AveragePrecision': float(average_precision_score(y_true, y_prob)) if both else None,
            'Accuracy': float(accuracy_score(y_true, prediction)), 'Threshold': float(threshold)}


def predict(model, loader, device, criterion=None):
    model.eval()
    labels, probabilities, addresses, total_loss = [], [], [], 0.0
    with torch.no_grad():
        for batch in loader:
            logits = model(batch['main_batch'].to(device), [nb.to(device) if nb is not None else None for nb in batch['neighbor_batches']])
            targets = batch['labels'].to(device)
            if criterion is not None:
                total_loss += criterion(logits, targets).item() * len(targets)
            labels.extend(targets.cpu().tolist())
            probabilities.extend(logits.softmax(dim=1)[:, 1].cpu().tolist())
            addresses.extend(batch['addresses'])
    if not labels:
        raise ValueError('Cannot evaluate an empty split')
    return np.array(labels), np.array(probabilities), addresses, total_loss / len(labels)


def make_splits(labels, num_folds=5, validation_fraction=0.2, seed=24):
    labels = np.asarray(labels)
    if len(np.unique(labels)) != 2 or np.bincount(labels).min() < num_folds:
        raise ValueError('Each class must contain at least num_folds samples')
    if not 0 < validation_fraction < 1:
        raise ValueError('validation_fraction must lie between 0 and 1')
    splits = []
    outer = StratifiedKFold(num_folds, shuffle=True, random_state=seed)
    for fold, (development, test) in enumerate(outer.split(np.zeros(len(labels)), labels), 1):
        train, validation = train_test_split(development, test_size=validation_fraction, random_state=seed + fold, stratify=labels[development])
        splits.append((np.sort(train), np.sort(validation), np.sort(test)))
    return splits


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_fold(dataset, split, fold, args, device):
    seed_everything(args.seed + fold)
    train_indices, validation_indices, test_indices = split
    fold_dir = Path(args.output_dir) / f'fold_{fold}'
    fold_dir.mkdir(parents=True, exist_ok=True)
    write_json(fold_dir / 'split.json', {name: [dataset.samples[int(i)]['address'] for i in indices] for name, indices in zip(['train', 'validation', 'test'], split)})
    train_loader, validation_loader, test_loader = [DataLoader(Subset(dataset, indices.tolist()), batch_size=args.batch_size, shuffle=i == 0, collate_fn=MultiGraphDataset.collate_fn) for i, indices in enumerate(split)]
    model = MultiGraphClassifier().to(device)
    counts = np.bincount([dataset.samples[int(i)]['label'] for i in train_indices], minlength=2)
    criterion = ClassBalancedFocalLoss(counts, args.beta, args.gamma).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    best_key, best_state, stale_epochs = (-1.0, -float('inf')), None, 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch['main_batch'].to(device), [nb.to(device) if nb is not None else None for nb in batch['neighbor_batches']])
            loss = criterion(logits, batch['labels'].to(device))
            if args.l2_lambda:
                loss = loss + args.l2_lambda * sum(p.square().sum() for p in model.parameters())
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite training loss')
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(batch['labels'])
        labels, probabilities, _, validation_loss = predict(model, validation_loader, device, criterion)
        threshold, validation_f1 = find_best_threshold(labels, probabilities)
        history.append({'Epoch': epoch, 'TrainLoss': train_loss / len(train_indices), 'ValidationLoss': validation_loss, 'ValidationF1': validation_f1, 'Threshold': threshold})
        print(f'Fold {fold} | Epoch {epoch} | Validation F1={validation_f1:.4f} | Threshold={threshold:.2f}')
        key = (validation_f1, -validation_loss)
        if key > best_key:
            best_key = key
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            best_epoch, best_threshold, best_validation_loss = epoch, threshold, validation_loss
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break
    if best_state is None:
        raise RuntimeError('No valid checkpoint was produced')
    model.load_state_dict(best_state)
    torch.save({'model_state_dict': best_state, 'threshold': best_threshold, 'epoch': best_epoch, 'training_config': vars(args), 'graph_config': dataset.manifest['config']}, fold_dir / 'best_model.pt')
    pd.DataFrame(history).to_csv(fold_dir / 'training_history.csv', index=False)
    labels, probabilities, addresses, _ = predict(model, test_loader, device)
    metrics = score_predictions(labels, probabilities, best_threshold)
    pd.DataFrame({'Address': addresses, 'Label': labels, 'Probability': probabilities, 'Prediction': (probabilities >= best_threshold).astype(int), 'Threshold': best_threshold}).to_csv(fold_dir / 'test_predictions.csv', index=False)
    metrics.update({'Fold': fold, 'BestEpoch': best_epoch, 'ValidationF1': best_key[0], 'ValidationLoss': best_validation_loss})
    write_json(fold_dir / 'test_metrics.json', metrics)
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description='PonziScope held-out outer-fold evaluation')
    parser.add_argument('--dataset', choices=['Dataset1', 'Dataset2'], default='Dataset1')
    for name in ['dataset_csv', 'cache_dir', 'output_dir']:
        parser.add_argument('--' + name)
    for name, default in [('max_neighbors', 200), ('max_txs', 200), ('end_block', END_BLOCK), ('batch_size', 16), ('epochs', 250), ('patience', 30), ('seed', 24), ('num_folds', 5)]:
        parser.add_argument('--' + name, type=int, default=default)
    for name, default in [('history_fraction', 1.0), ('learning_rate', 1e-3), ('weight_decay', 1e-2), ('l2_lambda', 0.0), ('beta', 0.999), ('gamma', 3.0), ('validation_fraction', 0.2)]:
        parser.add_argument('--' + name, type=float, default=default)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


def main():
    args = parse_args()
    if min(args.epochs, args.patience, args.batch_size) < 1 or args.num_folds < 2 or args.learning_rate <= 0 or min(args.weight_decay, args.l2_lambda) < 0:
        raise ValueError('Invalid training configuration')
    config = graph_config(args.max_neighbors, args.max_txs, args.end_block, args.history_fraction)
    csv_path = Path(args.dataset_csv) if args.dataset_csv else ROOT / 'Dataset' / 'Ponzi' / f'{args.dataset}.csv'
    cache_dir = Path(args.cache_dir) if args.cache_dir else default_cache_dir(args.dataset, config)
    args.output_dir = str(Path(args.output_dir) if args.output_dir else ROOT / f'5fold_results_{args.dataset}_v2_h{args.history_fraction:g}')
    dataset = MultiGraphDataset(cache_dir, config, csv_path)
    splits = make_splits([s['label'] for s in dataset.samples], args.num_folds, args.validation_fraction, args.seed)
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Output directory is not empty; choose a new --output_dir')
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'run_config.json', {'training': vars(args), 'graph': config, 'dataset_sha256': file_digest(csv_path), 'main_sha256': file_digest(Path(__file__)), 'torch_version': torch.__version__})
    results = [train_fold(dataset, split, fold, args, torch.device(args.device)) for fold, split in enumerate(splits, 1)]
    frame = pd.DataFrame(results)
    frame.to_csv(output / '5fold_results.csv', index=False)
    frame[['Precision', 'Recall', 'F1', 'AUC', 'AveragePrecision', 'Accuracy']].agg(['mean', 'std']).to_csv(output / 'summary.csv')
    print(frame)


if __name__ == '__main__':
    main()
