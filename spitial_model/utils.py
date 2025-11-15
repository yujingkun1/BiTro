#!/usr/bin/env python3
"""
Utility Functions for Cell2Gene

author: Jingkun Yu
"""

import os
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import pickle
try:
    from torch_geometric.data import Data
except Exception:
    Data = None


def get_fold_samples(fold_idx, all_samples=None):
    """
    Get train/test splits for 10-fold cross validation (fixed splits matching original implementation)
    """
    fold_splits = {
        0: ['TENX152', 'MISC73', 'MISC72', 'MISC71', 'MISC70'],
        1: ['MISC69', 'MISC68', 'MISC67', 'MISC66', 'MISC65'],
        2: ['MISC64', 'MISC63', 'MISC62', 'MISC58', 'MISC57'],
        3: ['MISC56', 'MISC51', 'MISC50', 'MISC49', 'MISC48'],
        4: ['MISC47', 'MISC46', 'MISC45', 'MISC44', 'MISC43'],
        5: ['MISC42', 'MISC41', 'MISC40', 'MISC39', 'MISC38'],
        6: ['MISC37', 'MISC36', 'MISC35', 'MISC34', 'MISC33'],
        7: ['TENX92', 'TENX91', 'TENX90', 'TENX89', 'TENX49'],
        8: ['TENX29', 'ZEN47', 'ZEN46', 'ZEN45', 'ZEN44'],
        9: ['ZEN43', 'ZEN42', 'ZEN39', 'ZEN38']
    }
    fold_splits = {
        0:['SPA154', 'SPA153', 'SPA152', 'SPA151', 'SPA150', 'SPA149'],
        1:['SPA148', 'SPA147', 'SPA146', 'SPA145', 'SPA144', 'SPA143'],
        2:['SPA142', 'SPA141', 'SPA140', 'SPA139', 'SPA138', 'SPA137'],
        3:['SPA136', 'SPA135', 'SPA134', 'SPA133', 'SPA132', 'SPA131'],
        4:['SPA130', 'SPA129', 'SPA128', 'SPA127', 'SPA126', 'SPA125'],
        5:['SPA124', 'SPA123', 'SPA122', 'SPA121', 'SPA120', 'SPA119'],
    }
    # fold_splits = {
    #     0:['NCBI770', 'NCBI769', 'NCBI768', 'NCBI767', 'NCBI766', 'NCBI765',
    #    'NCBI764', 'NCBI763', 'NCBI762', 'NCBI761', 'NCBI760', 'NCBI759'],
    # }

    test_samples = fold_splits[fold_idx]
    train_samples = []
    for fold, samples in fold_splits.items():
        if fold != fold_idx:
            train_samples.extend(samples)

    return train_samples, test_samples


def evaluate_model(model, data_loader, device):
    """
    Evaluate model on a dataset
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    criterion = nn.MSELoss()

    with torch.no_grad():
        for batch in data_loader:
            spot_expressions = batch["spot_expressions"].to(device)
            spot_graphs = batch["spot_graphs"]

            # Forward pass
            predictions = model(spot_graphs)

            # Targets are already log1p-transformed in the dataset.
            # Ensure we compare in the same space without extra log on predictions.
            loss = criterion(predictions, spot_expressions)

            if not (torch.isnan(loss) or torch.isinf(loss)):
                total_loss += loss.item()
                num_batches += 1

    return total_loss / max(num_batches, 1)


def evaluate_model_metrics(model, data_loader, device):
    """
    Comprehensive model evaluation with multiple metrics
    """
    model.eval()
    all_predictions = []
    all_targets = []

    print("=== Evaluating model ===")

    dataset = getattr(data_loader, 'dataset', None)
    gene_mean = None
    gene_std = None
    apply_normalization = False
    if dataset is not None and getattr(dataset, 'apply_gene_normalization', False):
        gene_mean = getattr(dataset, 'gene_mean_np', None)
        gene_std = getattr(dataset, 'gene_std_np', None)
        if gene_mean is not None and gene_std is not None:
            gene_mean = gene_mean.astype(np.float32)
            gene_std = gene_std.astype(np.float32)
            apply_normalization = True

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            spot_expressions = batch["spot_expressions"].to(device)
            spot_graphs = batch["spot_graphs"]

            # Forward pass
            predictions = model(spot_graphs)

            # Targets are already in log space; keep predictions in the same space
            predictions_np = predictions.cpu().numpy()
            targets_np = spot_expressions.cpu().numpy()

            if apply_normalization:
                predictions_np = predictions_np * gene_std + gene_mean
                targets_np = targets_np * gene_std + gene_mean

            all_predictions.append(predictions_np)
            all_targets.append(targets_np)

    # Merge all batch results
    all_predictions = np.vstack(all_predictions)  # [n_spots, n_genes]
    all_targets = np.vstack(all_targets)

    print(
        f"Evaluation samples: {all_predictions.shape[0]} spots × {all_predictions.shape[1]} genes")

    # Calculate metrics
    # 1. Overall MSE
    mse = mean_squared_error(all_targets.flatten(), all_predictions.flatten())

    # 2. Overall Pearson correlation
    overall_corr, overall_p = pearsonr(
        all_targets.flatten(), all_predictions.flatten())

    # 3. Per-gene metrics
    gene_correlations = []
    gene_mses = []

    for gene_idx in range(all_predictions.shape[1]):
        true_values = all_targets[:, gene_idx]
        pred_values = all_predictions[:, gene_idx]

        # Skip genes with zero variance
        if np.var(true_values) > 1e-8 and np.var(pred_values) > 1e-8:
            gene_corr, _ = pearsonr(true_values, pred_values)
            if not np.isnan(gene_corr):
                gene_correlations.append(gene_corr)

        gene_mse = mean_squared_error(true_values, pred_values)
        gene_mses.append(gene_mse)

    # 4. Per-spot metrics
    spot_correlations = []
    spot_mses = []

    for spot_idx in range(all_predictions.shape[0]):
        true_values = all_targets[spot_idx, :]
        pred_values = all_predictions[spot_idx, :]

        # Skip spots with zero variance
        if np.var(true_values) > 1e-8 and np.var(pred_values) > 1e-8:
            spot_corr, _ = pearsonr(true_values, pred_values)
            if not np.isnan(spot_corr):
                spot_correlations.append(spot_corr)

        spot_mse = mean_squared_error(true_values, pred_values)
        spot_mses.append(spot_mse)

    # Summary statistics
    results = {
        'overall_mse': mse,
        'overall_correlation': overall_corr,
        'overall_correlation_pval': overall_p,
        'mean_gene_correlation': np.mean(gene_correlations) if gene_correlations else 0,
        'median_gene_correlation': np.median(gene_correlations) if gene_correlations else 0,
        'mean_spot_correlation': np.mean(spot_correlations) if spot_correlations else 0,
        'median_spot_correlation': np.median(spot_correlations) if spot_correlations else 0,
        'mean_gene_mse': np.mean(gene_mses),
        'mean_spot_mse': np.mean(spot_mses),
        'gene_correlations': gene_correlations,
        'spot_correlations': spot_correlations,
        'gene_mses': gene_mses,
        'spot_mses': spot_mses
    }

    # Print results
    print(f"\n=== Evaluation Results ===")
    print(f"Overall MSE: {results['overall_mse']:.6f}")
    print(
        f"Overall Correlation: {results['overall_correlation']:.6f} (p={results['overall_correlation_pval']:.2e})")
    print(f"Mean Gene Correlation: {results['mean_gene_correlation']:.6f}")
    print(f"Median Gene Correlation: {results['median_gene_correlation']:.6f}")
    print(f"Mean Spot Correlation: {results['mean_spot_correlation']:.6f}")
    print(f"Median Spot Correlation: {results['median_spot_correlation']:.6f}")
    print(f"Mean Gene MSE: {results['mean_gene_mse']:.6f}")
    print(f"Mean Spot MSE: {results['mean_spot_mse']:.6f}")

    return results, all_predictions, all_targets


# Pearson correlation loss removed: training now uses pure MSE. Pearson metrics are computed only for evaluation.


def save_evaluation_results(results, predictions, targets, fold_idx, save_dir="./logs"):
    """
    Save evaluation results to files
    """
    os.makedirs(save_dir, exist_ok=True)

    # Save metrics
    metrics_file = os.path.join(save_dir, f"fold_{fold_idx}_metrics.txt")
    with open(metrics_file, 'w') as f:
        f.write("=== Evaluation Results ===\n")
        f.write(f"Overall MSE: {results['overall_mse']:.6f}\n")
        f.write(f"Overall Correlation: {results['overall_correlation']:.6f}\n")
        f.write(
            f"Mean Gene Correlation: {results['mean_gene_correlation']:.6f}\n")
        f.write(
            f"Median Gene Correlation: {results['median_gene_correlation']:.6f}\n")
        f.write(
            f"Mean Spot Correlation: {results['mean_spot_correlation']:.6f}\n")
        f.write(
            f"Median Spot Correlation: {results['median_spot_correlation']:.6f}\n")

    # Save predictions and targets
    np.save(os.path.join(
        save_dir, f"fold_{fold_idx}_predictions.npy"), predictions)
    np.save(os.path.join(save_dir, f"fold_{fold_idx}_targets.npy"), targets)

    print(f"Results saved to {save_dir}")


def save_epoch_metrics(train_losses, test_losses, epoch_mean_gene_corrs, epoch_overall_corrs, fold_idx, save_dir="./logs"):
    """
    Save per-epoch detailed metrics (losses and Pearson correlations) to JSON file.
    
    Args:
        train_losses: List of training losses per epoch
        test_losses: List of test losses per epoch
        epoch_mean_gene_corrs: List of mean gene Pearson correlations per epoch
        epoch_overall_corrs: List of overall Pearson correlations per epoch
        fold_idx: Fold index
        save_dir: Directory to save the metrics file
    
    Returns:
        Path to the saved metrics file
    """
    import json
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert numpy types to Python native types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    # Prepare epoch details
    epoch_details = []
    num_epochs_trained = len(train_losses)
    
    for epoch_idx in range(num_epochs_trained):
        epoch_detail = {
            'epoch': epoch_idx + 1,
            'train_loss': float(train_losses[epoch_idx]) if epoch_idx < len(train_losses) else None,
            'test_loss': float(test_losses[epoch_idx]) if epoch_idx < len(test_losses) else None,
            'mean_gene_correlation': float(epoch_mean_gene_corrs[epoch_idx]) if epoch_idx < len(epoch_mean_gene_corrs) else None,
            'overall_correlation': float(epoch_overall_corrs[epoch_idx]) if epoch_idx < len(epoch_overall_corrs) else None
        }
        epoch_details.append(epoch_detail)
    
    # Save per-epoch metrics to JSON file
    epoch_metrics_file = os.path.join(save_dir, f"fold_{fold_idx}_epoch_metrics.json")
    with open(epoch_metrics_file, 'w') as f:
        json.dump({
            'fold_idx': fold_idx,
            'num_epochs': num_epochs_trained,
            'epoch_details': epoch_details
        }, f, indent=2, default=convert_numpy_types)
    
    print(f"✓ Saved per-epoch metrics to {epoch_metrics_file}")
    return epoch_metrics_file


def plot_training_curves(train_losses, test_losses, fold_idx, save_dir="./logs", epoch_mean_gene_corrs=None, epoch_overall_corrs=None):
    """
    Plot training and test loss curves
    """
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    ax1 = plt.gca()
    ax1.plot(train_losses, label='Train Loss', color='blue')
    ax1.plot(test_losses, label='Test Loss', color='red')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'Training Curves - Fold {fold_idx}')
    ax1.grid(True)

    have_right_axis = False
    if epoch_mean_gene_corrs is not None and len(epoch_mean_gene_corrs) > 0:
        ax2 = ax1.twinx()
        ax2.plot(epoch_mean_gene_corrs, label='Mean Gene Pearson', color='green')
        ax2.set_ylabel('Pearson')
        have_right_axis = True
        if epoch_overall_corrs is not None and len(epoch_overall_corrs) > 0:
            ax2.plot(epoch_overall_corrs, label='Overall Pearson', color='orange')
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
    elif epoch_overall_corrs is not None and len(epoch_overall_corrs) > 0:
        ax2 = ax1.twinx()
        ax2.plot(epoch_overall_corrs, label='Overall Pearson', color='orange')
        ax2.set_ylabel('Pearson')
        have_right_axis = True
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
    else:
        ax1.legend(loc='best')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"fold_{fold_idx}_training_curves.png"))
    plt.close()


def plot_fold_gene_correlation_distribution(gene_correlations, fold_idx, save_dir="./logs"):
    """
    Plot distribution of per-gene Pearson correlations for a fold.
    """
    os.makedirs(save_dir, exist_ok=True)
    if gene_correlations is None or len(gene_correlations) == 0:
        return
    plt.figure(figsize=(8, 5))
    sns.histplot(gene_correlations, bins=30, kde=True, color='purple')
    plt.xlabel('Per-Gene Pearson Correlation')
    plt.ylabel('Count')
    plt.title(f'Gene Correlation Distribution - Fold {fold_idx}')
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"fold_{fold_idx}_gene_correlation_hist.png"))
    plt.close()


def plot_metric_across_folds(values, title, ylabel, filename, save_dir="./logs"):
    """
    Plot a simple bar chart of a metric across folds.
    """
    os.makedirs(save_dir, exist_ok=True)
    if values is None or len(values) == 0:
        return
    plt.figure(figsize=(10, 5))
    xs = list(range(1, len(values) + 1))
    sns.barplot(x=xs, y=values, color='steelblue')
    plt.xlabel('Fold')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()


def setup_device(device_id=0):
    """
    Setup CUDA device
    """
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{device_id}")
        print(f"Using device: {device}")
        print(f"GPU: {torch.cuda.get_device_name(device_id)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    return device


def count_parameters(model):
    """
    Count model parameters
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    return total_params, trainable_params


def scan_cells_and_graphs(hest_data_dir, graph_dir, features_dir, sample_ids):
    """
    读取给定样本的所有细胞，并判断每个 spot 是否有图：
    - 若 `hest_intra_spot_graphs.pkl` 中包含 (sample_id, spot_idx) → 视为有图
    - 否则尝试从 features_dir 的 `{sample_id}_combined_features.npz` 中构造特征（无图但有特征）

    返回：list[dict]
      - sample_id, spot_idx, has_graph(bool), has_features(bool), num_cells(int)
    """
    results = []
    aggregated_intra_path = os.path.join(
        graph_dir, "hest_intra_spot_graphs.pkl") if graph_dir else None
    intra = None
    if aggregated_intra_path and os.path.exists(aggregated_intra_path):
        try:
            with open(aggregated_intra_path, 'rb') as f:
                intra = pickle.load(f)
        except Exception:
            intra = None

    # 遍历 AnnData 以确定 spot 数量
    import scanpy as sc
    for sample_id in (sample_ids if isinstance(sample_ids, list) else [sample_ids]):
        st_file = os.path.join(hest_data_dir, "st", f"{sample_id}.h5ad")
        if not os.path.exists(st_file):
            continue
        adata = sc.read_h5ad(st_file)
        n_spots = adata.n_obs

        # 载入 features
        per_spot_map = {}
        npz_path = os.path.join(
            features_dir, f"{sample_id}_combined_features.npz") if features_dir else None
        if npz_path and os.path.exists(npz_path):
            import numpy as np
            try:
                npz = np.load(npz_path, allow_pickle=True)
                keys = list(npz.keys())
                per_spot = None
                if 'per_spot' in keys:
                    try:
                        per_spot = npz['per_spot'].item()
                    except Exception:
                        per_spot = None
                if isinstance(per_spot, dict):
                    for si, v in per_spot.items():
                        x = v.get('x') if isinstance(v, dict) else None
                        pos = v.get('pos') if isinstance(v, dict) else None
                        if x is not None and pos is not None and len(x) == len(pos):
                            per_spot_map[int(si)] = len(x)
                else:
                    feats = npz['features'] if 'features' in keys else None
                    poss = npz['positions'] if 'positions' in keys else None
                    spot_ptr = npz['spot_ptr'] if 'spot_ptr' in keys else None
                    spot_index = npz['spot_index'] if 'spot_index' in keys else None
                    if feats is not None and poss is not None:
                        if spot_ptr is not None:
                            for si in range(len(spot_ptr) - 1):
                                s, e = int(spot_ptr[si]), int(spot_ptr[si+1])
                                if e > s:
                                    per_spot_map[si] = e - s
                        elif spot_index is not None:
                            from collections import Counter
                            per_spot_map = dict(
                                Counter([int(i) for i in spot_index.tolist()]))
            except Exception:
                pass

        for spot_idx in range(n_spots):
            has_graph = bool(intra and isinstance(
                intra.get(sample_id), dict) and spot_idx in intra[sample_id])
            num_cells = int(per_spot_map.get(spot_idx, 0)
                            ) if per_spot_map else 0
            has_features = num_cells > 0
            results.append({
                'sample_id': sample_id,
                'spot_idx': spot_idx,
                'has_graph': has_graph,
                'has_features': has_features,
                'num_cells': num_cells
            })

    return results
