import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Batch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    f1_score
)
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import warnings
import argparse

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="MultiGraph Classifier Training")
parser.add_argument("--dataset", type=str, default="Dataset1", 
                    choices=["Dataset1", "Dataset2"], help="选择数据集")
parser.add_argument("--output_dir", type=str, default=None, 
                    help="输出结果目录（默认自动生成）")
parser.add_argument("--cache_dir", type=str, default=None, 
                    help="图缓存目录（默认自动生成）")
parser.add_argument("--max_neighbors", type=int, default=200,
                    help="最大邻居数量（用于定位默认图缓存目录）")
parser.add_argument("--max_txs", type=int, default=200,
                    help="每个邻居的最大交易数量（用于定位默认图缓存目录）")
args = parser.parse_args()

DATASET_NAME = args.dataset
if args.cache_dir:
    CACHE_DIR = args.cache_dir
else:
    CACHE_DIR = (
        f"./Dataset/graph_cache_multigraph_{DATASET_NAME}_"
        f"maxneighbors{args.max_neighbors}_maxtxs{args.max_txs}"
    )

if args.output_dir:
    OUTPUT_ROOT = args.output_dir
else:
    OUTPUT_ROOT = f"5fold_results_{DATASET_NAME}"

NODE_DIM = 38
EDGE_DIM = 6
os.makedirs(OUTPUT_ROOT, exist_ok=True)

print(f"使用数据集: {DATASET_NAME}")
print(f"图缓存目录: {CACHE_DIR}")
print(f"输出结果目录: {OUTPUT_ROOT}")

config = {
    "batch_size": 16,
    "hidden_channels": 256,
    "embedding_dim": 128,
    "learning_rate": 1e-3,
    "lr_stage2": 1e-4,
    "warmup_epochs": 5,
    "epochs": 250,
    "patience": 30,
    "seed": 24,
    "num_folds": 5,
    "two_stage_epochs": 80
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(config["seed"])
np.random.seed(config["seed"])

class MultiGraphDataset(torch.utils.data.Dataset):
    def __init__(self, cache_dir=CACHE_DIR):
        self.files = [
            os.path.join(cache_dir, f)
            for f in os.listdir(cache_dir)
            if f.endswith(".pt")
        ]
        print(f"Loaded {len(self.files)} samples")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return torch.load(self.files[idx], map_location="cpu")

    @staticmethod
    def collate_fn(batch):
        main_graphs = [b["main_graph"] for b in batch]
        labels = torch.stack([b["label"] for b in batch])

        neighbor_batches = []
        for b in batch:
            neighs = b.get("neighbor_graphs", [])
            if len(neighs) == 0:
                empty = main_graphs[0].__class__()
                empty.x = torch.zeros((0, NODE_DIM))
                empty.edge_index = torch.zeros((2, 0), dtype=torch.long)
                empty.edge_attr = torch.zeros((0, EDGE_DIM))
                nb = Batch.from_data_list([empty])
            else:
                nb = Batch.from_data_list(neighs)
            neighbor_batches.append(nb)

        return {
            "main_batch": Batch.from_data_list(main_graphs),
            "neighbor_batches": neighbor_batches,
            "labels": labels
        }

class SingleGraphEncoder(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GATConv(node_dim, hidden_dim, heads=4, concat=True, edge_dim=edge_dim)
        self.conv2 = GATConv(hidden_dim * 4, out_dim, heads=1, concat=False, edge_dim=edge_dim)
        self.norm1 = nn.LayerNorm(hidden_dim * 4)
        self.norm2 = nn.LayerNorm(out_dim)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = data.batch
        x = F.elu(self.conv1(x, edge_index, edge_attr))
        x = self.norm1(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.elu(self.conv2(x, edge_index, edge_attr))
        x = self.norm2(x)
        return global_mean_pool(x, batch)

class MultiGraphClassifier(nn.Module):
    def __init__(self, emb_dim=128, num_heads=8):
        super().__init__()
        self.encoder = SingleGraphEncoder(NODE_DIM, EDGE_DIM, 256, emb_dim)
        self.attn = nn.MultiheadAttention(emb_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(emb_dim)

        self.neighbor_gate = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.Sigmoid()
        )

        self.classifier = nn.Sequential(
            nn.Linear(emb_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )

    def forward(self, main_batch, neighbor_batches):
        main_emb = self.encoder(main_batch)

        neighbor_embs = []
        for nb in neighbor_batches:
            if nb.num_graphs > 0:
                emb = self.encoder(nb)
                gate = self.neighbor_gate(emb)
                emb = emb * gate
            else:
                emb = torch.zeros(1, main_emb.size(-1), device=main_emb.device)
            neighbor_embs.append(emb)

        seqs = [
            torch.cat([main_emb[i:i+1], neighbor_embs[i]], dim=0)
            for i in range(main_emb.size(0))
        ]

        seq_padded = nn.utils.rnn.pad_sequence(seqs, batch_first=True)
        mask = (seq_padded.abs().sum(-1) == 0)

        attn_out, _ = self.attn(seq_padded, seq_padded, seq_padded, key_padding_mask=mask)
        final = self.norm(attn_out[:, 0, :])
        return self.classifier(final)

class ClassBalancedFocalLoss(nn.Module):
    def __init__(self, samples_per_class, beta=0.999, gamma=3.0):
        super().__init__()
        effective_num = 1.0 - np.power(beta, samples_per_class)
        weights = (1.0 - beta) / effective_num
        weights = weights / weights.sum() * len(samples_per_class)
        self.class_weights = torch.tensor(weights, dtype=torch.float)
        self.gamma = gamma

    def forward(self, logits, targets):
        if self.class_weights.device != logits.device:
            self.class_weights = self.class_weights.to(logits.device)

        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)
        targets_oh = F.one_hot(targets, num_classes=2).float()

        loss = -targets_oh * ((1 - probs) ** self.gamma) * log_probs
        loss = loss * self.class_weights.unsqueeze(0)
        return loss.sum(dim=1).mean()

def find_best_threshold(y_true, y_prob):
    best_t, best_f1 = 0.5, 0.0
    for t in np.linspace(0.05, 0.95, 91):
        preds = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1

def evaluate(model, loader):
    model.eval()
    trues, probs = [], []

    with torch.no_grad():
        for batch in loader:
            logits = model(
                batch["main_batch"].to(DEVICE),
                [nb.to(DEVICE) for nb in batch["neighbor_batches"]]
            )
            prob = F.softmax(logits, dim=1)[:, 1]
            trues.extend(batch["labels"].tolist())
            probs.extend(prob.cpu().tolist())

    trues = np.array(trues)
    probs = np.array(probs)

    best_t, _ = find_best_threshold(trues, probs)
    preds = (probs >= best_t).astype(int)

    p, r, f1, _ = precision_recall_fscore_support(trues, preds, average="binary", zero_division=0)
    if len(np.unique(trues)) > 1:
        auc = roc_auc_score(trues, probs)
    else:
        auc = float('nan')
    acc = accuracy_score(trues, preds)

    return p, r, f1, auc, best_t

if __name__ == "__main__":
    dataset = MultiGraphDataset()
    labels = [torch.load(f)["label"].item() for f in dataset.files]

    skf = StratifiedKFold(n_splits=config["num_folds"], shuffle=True, random_state=config["seed"])
    results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels), 1):
        print(f"\n===== Fold {fold} =====")

        train_loader = DataLoader(
            Subset(dataset, train_idx),
            batch_size=config["batch_size"],
            shuffle=True,
            collate_fn=MultiGraphDataset.collate_fn
        )
        val_loader = DataLoader(
            Subset(dataset, val_idx),
            batch_size=config["batch_size"],
            shuffle=False,
            collate_fn=MultiGraphDataset.collate_fn
        )

        model = MultiGraphClassifier().to(DEVICE)

        class_counts = np.bincount([labels[i] for i in train_idx], minlength=2)
        criterion = ClassBalancedFocalLoss(class_counts)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])

        best_train_loss = float('inf')     
        best_state = None
        patience = 0

        for epoch in range(config["epochs"]):
            model.train()
            train_loss = 0.0
            num_batches = 0

            for batch in train_loader:
                optimizer.zero_grad()
                logits = model(
                    batch["main_batch"].to(DEVICE),
                    [nb.to(DEVICE) for nb in batch["neighbor_batches"]]
                )
                loss = criterion(logits, batch["labels"].to(DEVICE))
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                num_batches += 1

            model.eval()
            p, r, f1, auc, t = evaluate(model, val_loader)

            print(f"Epoch {epoch:03d} | Train Loss={train_loss:.4f} | "
                  f"F1={f1:.4f} AUC={auc:.4f} T={t:.2f}")

            if train_loss < best_train_loss:
                best_train_loss = train_loss
                best_state = model.state_dict().copy()
                patience = 0
                best_p, best_r, best_f1, best_auc, best_t = p, r, f1, auc, t
            else:
                patience += 1
                if patience >= config["patience"]:
                    print(f"Early stopping at epoch {epoch}")
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        p, r, f1, auc, t = evaluate(model, val_loader)

        results.append({
            "Fold": fold,
            "Precision": round(p, 4),
            "Recall": round(r, 4),
            "F1": round(f1, 4),
            "AUC": round(auc, 4),
            "Threshold": round(t, 2),
            "Best_Val_Loss": round(best_train_loss, 4) 
        })

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUTPUT_ROOT, "5fold_results.csv"), index=False)
    print(df)
    print("\nDone.")
