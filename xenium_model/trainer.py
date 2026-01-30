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
from scipy.stats import pearsonr

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

def train_xenium(model, node_head, train_loader, val_loader, device='cuda', num_epochs=50, lr=1e-3, save_path='xenium_best.pt', accum_steps=4, early_stopping_patience=10, early_stopping_min_delta=1e-4):
    model.to(device)
    node_head.to(device)
    params = list(model.parameters()) + list(node_head.parameters())
    optimizer = optim.AdamW(params, lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()

    best_val = float('inf')
    best_overall_pearson = -1.0
    best_mean_gene_pearson = -1.0
    epoch_overall_pearsons = []
    epoch_mean_gene_pearsons = []
    early_stop_counter = 0
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
                # targets may be numpy arrays; convert to tensor if needed
                if isinstance(tgt, np.ndarray):
                    tgt = torch.from_numpy(tgt).to(device)
                else:
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
        all_preds = []
        all_tgts = []
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
                    emb = emb.to(device)
                    # convert numpy targets to tensor if necessary
                    if isinstance(tgt, np.ndarray):
                        tgt = torch.from_numpy(tgt).to(device)
                    else:
                        tgt = tgt.to(device)
                    pred = node_head(emb)
                    val_loss += float(nn.functional.mse_loss(pred, tgt).item())
                    # collect for pearson metrics
                    all_preds.append(pred.cpu().numpy())
                    all_tgts.append(tgt.cpu().numpy())
                    vcnt += 1
        if vcnt > 0:
            val_loss = val_loss / vcnt
        print(f"Epoch {epoch+1}: Train Loss={(running/n_batches if n_batches else 0):.6f}, Val Loss={val_loss:.6f}")
        # compute pearson metrics for this epoch
        if all_preds:
            all_preds = np.vstack(all_preds)
            all_tgts = np.vstack(all_tgts)
            try:
                overall_corr, overall_p = pearsonr(all_tgts.flatten(), all_preds.flatten())
            except Exception:
                overall_corr, overall_p = 0.0, 1.0
            gene_corrs = []
            for gi in range(all_preds.shape[1]):
                t = all_tgts[:, gi]
                p = all_preds[:, gi]
                if np.var(t) > 1e-8 and np.var(p) > 1e-8:
                    try:
                        corr, _ = pearsonr(t, p)
                        if not np.isnan(corr):
                            gene_corrs.append(corr)
                    except Exception:
                        continue
            mean_gene_corr = float(np.mean(gene_corrs)) if gene_corrs else 0.0
        else:
            overall_corr, overall_p, mean_gene_corr = 0.0, 1.0, 0.0

        epoch_overall_pearsons.append(overall_corr)
        epoch_mean_gene_pearsons.append(mean_gene_corr)
        print(f"  Mean Gene Pearson: {mean_gene_corr:.6f}")
        print(f"  Overall Pearson: {overall_corr:.6f} (p={overall_p:.2e})")

        # save best by overall pearson
        if overall_corr > best_overall_pearson + early_stopping_min_delta:
            best_overall_pearson = overall_corr
            best_mean_gene_pearson = mean_gene_corr
            save_dir = os.path.dirname(save_path) if os.path.dirname(save_path) else None
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            torch.save({'model': model.state_dict(), 'node_head': node_head.state_dict()}, save_path)
            print(f"  *** Saving best model by overall Pearson: {best_overall_pearson:.6f} (Epoch {epoch+1}) ***")
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        # early stopping by overall pearson
        if early_stop_counter >= early_stopping_patience:
            print(f"*** Early stopping: no overall Pearson improvement for {early_stopping_patience} epochs ***")
            print(f"Best overall Pearson: {best_overall_pearson:.6f}, Best mean gene Pearson: {best_mean_gene_pearson:.6f}")
            break

    return best_val


