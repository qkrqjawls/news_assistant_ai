import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder

def user_item_GCN_embedding(
    user_ids: list,
    item_ids: list,
    raw_edges: list,
    user_feats: dict,
    item_feats: dict,
    emb_dim: int = 128,
    hidden_dim: int = 32,
    out_dim: int = 64,
    bpr_epochs: int = 200,
    bpr_neg_ratio: int = 1,
    batch_size: int = 128,
    lr: float = 0.01
):
    """
    Computes GCN-based user and item embeddings with BPR loss.

    Args:
        user_ids: List of original user IDs.
        item_ids: List of original item IDs.
        raw_edges: List of (user_id, item_id) interactions.
        user_feats: Dict mapping user_id -> np.ndarray of shape (emb_dim,).
        item_feats: Dict mapping item_id -> np.ndarray of shape (emb_dim,).
        emb_dim: Dimension of initial node features (defaults from ENV).
        hidden_dim: Hidden channels for GCN layers (default 32).
        out_dim: Output embedding dimension (defaults to emb_dim).
        bpr_epochs: Number of BPR training epochs.
        bpr_neg_ratio: Number of negative samples per positive.
        batch_size: Batch size for BPR training.
        lr: Learning rate for optimizer.

    Returns:
        Tuple of two dicts: (user_id -> embedding ndarray, item_id -> embedding ndarray)
    """
    # 1) Dimensions from environment or arguments
    EMB_DIM = int(os.environ.get("EMB_DIM", 128)) if emb_dim is None else emb_dim
    HIDDEN_DIM = 32 if hidden_dim is None else hidden_dim
    OUT_DIM = EMB_DIM if out_dim is None else out_dim

    # 2) Encode IDs
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()
    user_idx = user_encoder.fit_transform(user_ids)
    item_idx = item_encoder.fit_transform(item_ids) + len(user_ids)
    user_to_enc = dict(zip(user_ids, user_idx))
    item_to_enc = dict(zip(item_ids, item_idx))
    inv_user = {enc: uid for uid, enc in user_to_enc.items()}
    inv_item = {enc: iid for iid, enc in item_to_enc.items()}

    num_users, num_items = len(user_ids), len(item_ids)
    num_nodes = num_users + num_items

    # 3) Build bidirectional edge_index
    src = [user_to_enc[u] for u, i in raw_edges]
    dst = [item_to_enc[i] for u, i in raw_edges]
    edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)

    # 4) Initialize node features from provided dicts
    x = torch.zeros((num_nodes, EMB_DIM), dtype=torch.float32)
    for uid, enc in user_to_enc.items():
        feat = user_feats.get(uid, np.zeros(EMB_DIM))
        x[enc] = torch.from_numpy(feat).float()
    for iid, enc in item_to_enc.items():
        feat = item_feats.get(iid, np.zeros(EMB_DIM))
        x[enc] = torch.from_numpy(feat).float()

    data = Data(x=x, edge_index=edge_index)

    # 5) Define GCN model
    class SimpleGCN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = GCNConv(EMB_DIM, HIDDEN_DIM)
            self.conv2 = GCNConv(HIDDEN_DIM, OUT_DIM)

        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = self.conv2(x, edge_index)
            return x

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleGCN().to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 6) Prepare BPR dataset and loader
    pos_map = {}
    for u_enc, i_enc in zip(src, dst):
        pos_map.setdefault(u_enc, []).append(i_enc - num_users)

    class BPRDataset(Dataset):
        def __init__(self, pos_map, num_items, neg_ratio):
            self.pairs = [(u, pos) for u, items in pos_map.items() for pos in items]
            self.num_items = num_items
            self.neg = neg_ratio

        def __len__(self):
            return len(self.pairs)

        def __getitem__(self, idx):
            u, pos = self.pairs[idx]
            # Negative sampling
            neg = random.randrange(self.num_items)
            while neg in pos_map.get(u, []):
                neg = random.randrange(self.num_items)
            return u, pos, neg

    dataset = BPRDataset(pos_map, num_items, bpr_neg_ratio)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 7) Train with BPR loss
    model.train()
    for epoch in range(1, bpr_epochs + 1):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        z_user = out[:num_users]
        z_item = out[num_users:]
        total_loss = 0.0
        for u, pos, neg in loader:
            u, pos, neg = u.to(device), pos.to(device), neg.to(device)
            u_emb = z_user[u]
            pos_emb = z_item[pos]
            neg_emb = z_item[neg]
            loss = -F.logsigmoid((u_emb * pos_emb).sum(-1) - (u_emb * neg_emb).sum(-1)).mean()
            total_loss += loss
        total_loss = total_loss / len(loader)
        total_loss.backward()
        optimizer.step()

    # 8) Compute final embeddings
    model.eval()
    with torch.no_grad():
        final_out = model(data.x, data.edge_index).cpu()
    users_emb = final_out[:num_users]
    items_emb = final_out[num_users:]

    # 9) Map back to original IDs
    user_id_to_vec = {inv_user[i]: users_emb[i].numpy() for i in range(num_users)}
    item_id_to_vec = {inv_item[i + num_users]: items_emb[i].numpy() for i in range(num_items)}

    return user_id_to_vec, item_id_to_vec
