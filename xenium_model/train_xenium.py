#!/usr/bin/env python3
"""
Entry script to train Xenium cell-level model.
"""
import os
import sys
import argparse
import numpy as np
from torch.utils.data import DataLoader

# Ensure project root is on sys.path so absolute imports of Cell2Gene work when run as script
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Cell2Gene.xenium_model.dataset import XeniumGraphDataset
from Cell2Gene.xenium_model.trainer import NodeReadout, train_xenium

def build_data_loaders(graphs_pkl, processed_pkl, features_h5, batch_size=1, val_fraction=0.2, n_splits=5, num_workers=0, pin_memory=False):
    ds = XeniumGraphDataset(graphs_pkl, processed_pkl, features_h5)
    entries = ds.get_entries()
    N = len(entries)
    idxs = np.arange(N)
    np.random.shuffle(idxs)
    split = int(N * (1 - val_fraction))
    train_idx = idxs[:split]
    val_idx = idxs[split:]
    train_entries = [entries[i] for i in train_idx]
    val_entries = [entries[i] for i in val_idx]

    collate = lambda b: {'graphs': [x['graph'] for x in b], 'targets': [x['targets'] for x in b]}
    train_loader = DataLoader(train_entries, batch_size=batch_size, shuffle=True, collate_fn=collate, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_entries, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, val_loader, ds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graphs_pkl', default='/data/yujk/hovernet2feature/xenium_graphs/xenium_intra_spot_graphs.pkl')
    parser.add_argument('--processed_pkl', default='/data/yujk/hovernet2feature/xenium_graphs/xenium_processed_data.pkl')
    parser.add_argument('--features_h5', default='/data/yujk/hovernet2feature/xenium/outs/cell_feature_matrix.h5')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--accum_steps', type=int, default=4, help='gradient accumulation steps')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--out', default='/data/yujk/hovernet2feature/xenium_model/checkpoint.pt')
    args = parser.parse_args()

    train_loader, val_loader, ds = build_data_loaders(args.graphs_pkl, args.processed_pkl, args.features_h5, batch_size=args.batch_size, num_workers=0, pin_memory=False)

    # infer feature dims
    sample_entry = ds.get_entries()[0]
    embed_dim = 256  # default used by spatial model
    target_dim = sample_entry['targets'].shape[1]

    # create model (reuse spitial model)
    from Cell2Gene.spitial_model.trainer import setup_model
    device = 'cuda' if os.environ.get('CUDA_VISIBLE_DEVICES', None) else 'cpu'
    model = setup_model(feature_dim=128, num_genes=1, device=device)  # num_genes unused for node head here

    node_head = NodeReadout(embed_dim=256, out_dim=target_dim)

    best = train_xenium(model, node_head, train_loader, val_loader, device=device, num_epochs=args.epochs, save_path=args.out, accum_steps=args.accum_steps)
    print("Best val loss:", best)

if __name__ == "__main__":
    main()


