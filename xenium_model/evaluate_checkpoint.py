#!/usr/bin/env python3
"""
Evaluate a saved xenium checkpoint (model + node_head) on the xenium dataset
and report overall Pearson and mean per-gene Pearson.
"""
import os
import argparse
import numpy as np
import torch
from Cell2Gene.xenium_model.dataset import XeniumGraphDataset
from Cell2Gene.spitial_model.trainer import setup_model
from Cell2Gene.xenium_model.trainer import NodeReadout
from scipy.stats import pearsonr

def load_checkpoint(ckpt_path, device):
    data = torch.load(ckpt_path, map_location=device)
    return data

def evaluate_checkpoint(ckpt_path, graphs_pkl, processed_pkl, features_h5, device='cpu', max_entries=None):
    ds = XeniumGraphDataset(graphs_pkl, processed_pkl, features_h5)
    entries = ds.get_entries()
    if max_entries:
        entries = entries[:max_entries]

    # infer dims
    sample = entries[0]
    target_dim = sample['targets'].shape[1]

    model = setup_model(feature_dim=128, num_genes=1, device=device)  # num_genes unused
    node_head = NodeReadout(embed_dim=256, out_dim=target_dim)

    ck = load_checkpoint(ckpt_path, device)
    model.load_state_dict(ck.get('model', {}), strict=False)
    node_head.load_state_dict(ck.get('node_head', {}), strict=False)
    model.to(device).eval()
    node_head.to(device).eval()

    all_preds = []
    all_tgts = []

    with torch.no_grad():
        for entry in entries:
            g = entry['graph']
            targets = entry['targets']  # numpy [S, T]
            out = model([g], return_node_embeddings=True)
            if isinstance(out, tuple) and len(out) >= 2:
                _, node_embeddings_list, processed_indices = out
            else:
                node_embeddings_list = []
            # usually one graph
            if not node_embeddings_list:
                continue
            emb = node_embeddings_list[0]  # [S, E]
            emb = emb.to(device)
            pred = node_head(emb)  # [S, T]
            pred_np = pred.cpu().numpy()
            all_preds.append(pred_np)
            all_tgts.append(targets)

    if not all_preds:
        print("No predictions generated.")
        return

    all_preds = np.vstack(all_preds)  # [Ncells, T]
    all_tgts = np.vstack(all_tgts)

    # overall pearson
    overall_corr, overall_p = pearsonr(all_tgts.flatten(), all_preds.flatten())
    # per-gene pearson
    gene_corrs = []
    for gi in range(all_preds.shape[1]):
        t = all_tgts[:, gi]
        p = all_preds[:, gi]
        if np.var(t) > 1e-8 and np.var(p) > 1e-8:
            corr, _ = pearsonr(t, p)
            if not np.isnan(corr):
                gene_corrs.append(corr)
    mean_gene_corr = float(np.mean(gene_corrs)) if gene_corrs else 0.0

    print(f"Overall Pearson: {overall_corr:.6f} (p={overall_p:.2e})")
    print(f"Mean per-gene Pearson: {mean_gene_corr:.6f}")
    return overall_corr, mean_gene_corr

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--graphs_pkl', default='/data/yujk/hovernet2feature/xenium_graphs/xenium_intra_spot_graphs.pkl')
    parser.add_argument('--processed_pkl', default='/data/yujk/hovernet2feature/xenium_graphs/xenium_processed_data.pkl')
    parser.add_argument('--features_h5', default='/data/yujk/hovernet2feature/xenium/outs/cell_feature_matrix.h5')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--max_entries', type=int, default=None)
    args = parser.parse_args()

    evaluate_checkpoint(args.ckpt, args.graphs_pkl, args.processed_pkl, args.features_h5, device=args.device, max_entries=args.max_entries)

if __name__ == "__main__":
    main()


