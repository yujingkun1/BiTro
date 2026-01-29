#!/usr/bin/env python3
"""
Simple trainer for Xenium cell-level supervision.
Reuses the model from spitial_model.models.transformer and attaches a node-level readout head.
"""
import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from Cell2Gene.spitial_model.models.transformer import StaticGraphTransformerPredictor

class NodeReadout(nn.Module):
    def __init__(self, embed_dim, out_dim, dropout=0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, out_dim)
        )
    def forward(self, H):
        return self.head(H)

def collate_fn(entries):
    # entries: list of dicts with keys 'graph' and 'targets'
    graphs = [e['graph'] for e in entries]
    targets = [torch.from_numpy(e['targets']) for e in entries]
    return {'graphs': graphs, 'targets': targets}

def train_xenium(model, node_head, train_loader, val_loader, device='cuda', num_epochs=50, lr=1e-3, save_path='xenium_best.pt', accum_steps=4):
    model.to(device)
    node_head.to(device)
    params = list(model.parameters()) + list(node_head.parameters())
    optimizer = optim.AdamW(params, lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()

    best_val = float('inf')
    for epoch in range(num_epochs):
        model.train()
        node_head.train()
        running = 0.0
        n_batches = 0
        pbar = tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{num_epochs}", leave=False)
        for batch in pbar:
            graphs = batch['graphs']
            targets = batch['targets']
            # gradient accumulation support
            # zero grads only at accumulation boundary
            if n_batches % accum_steps == 0:
                optimizer.zero_grad()

            # forward through model to obtain node embeddings
            out = model(graphs, return_node_embeddings=True)
            if isinstance(out, tuple) and len(out) >= 2:
                _, node_embeddings_list, processed_indices = out
            else:
                node_embeddings_list, processed_indices = [], []

            loss = torch.tensor(0.0, device=device)
            count = 0
            for emb, idx, tgt in zip(node_embeddings_list, processed_indices, targets):
                # emb: [S, E], tgt: [S, T]
                emb = emb.to(device)
                tgt = tgt.to(device)
                pred = node_head(emb)  # [S, T]
                loss = loss + criterion(pred, tgt)
                count += 1

            if count == 0:
                # even when no valid nodes, increment batch counter to keep accumulation synced
                n_batches += 1
                continue

            loss = loss / count
            # scale loss by accumulation steps
            loss = loss / float(accum_steps)
            loss.backward()

            # step optimizer at accumulation boundary
            if (n_batches + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(params, 5.0)
                optimizer.step()

            running += (loss.item() * accum_steps)
            n_batches += 1
            pbar.set_postfix({'loss': f'{(running/n_batches):.4f}'})
        pbar.close()

        # validation
        model.eval(); node_head.eval()
        val_loss = 0.0; vcnt = 0
        with torch.no_grad():
            for batch in val_loader:
                graphs = batch['graphs']
                targets = batch['targets']
                out = model(graphs, return_node_embeddings=True)
                if isinstance(out, tuple) and len(out) >= 2:
                    _, node_embeddings_list, processed_indices = out
                else:
                    node_embeddings_list, processed_indices = [], []
                for emb, idx, tgt in zip(node_embeddings_list, processed_indices, targets):
                    emb = emb.to(device); tgt = tgt.to(device)
                    pred = node_head(emb)
                    val_loss += float(nn.functional.mse_loss(pred, tgt).item())
                    vcnt += 1
        if vcnt > 0:
            val_loss = val_loss / vcnt
        print(f"Epoch {epoch+1}: Train Loss={(running/n_batches if n_batches else 0):.6f}, Val Loss={val_loss:.6f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save({'model': model.state_dict(), 'node_head': node_head.state_dict()}, save_path)
            print(f"  Saved best model to {save_path}")

    return best_val


