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
        0: ['TENX152']
    }
    
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
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            spot_expressions = batch["spot_expressions"].to(device)
            spot_graphs = batch["spot_graphs"]
            
            # Forward pass
            predictions = model(spot_graphs)
            
            # Targets are already in log space; keep predictions in the same space
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(spot_expressions.cpu().numpy())
    
    # Merge all batch results
    all_predictions = np.vstack(all_predictions)  # [n_spots, n_genes]
    all_targets = np.vstack(all_targets)
    
    print(f"Evaluation samples: {all_predictions.shape[0]} spots × {all_predictions.shape[1]} genes")
    
    # Calculate metrics
    # 1. Overall MSE
    mse = mean_squared_error(all_targets.flatten(), all_predictions.flatten())
    
    # 2. Overall Pearson correlation
    overall_corr, overall_p = pearsonr(all_targets.flatten(), all_predictions.flatten())
    
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
    print(f"Overall Correlation: {results['overall_correlation']:.6f} (p={results['overall_correlation_pval']:.2e})")
    print(f"Mean Gene Correlation: {results['mean_gene_correlation']:.6f}")
    print(f"Median Gene Correlation: {results['median_gene_correlation']:.6f}")
    print(f"Mean Spot Correlation: {results['mean_spot_correlation']:.6f}")
    print(f"Median Spot Correlation: {results['median_spot_correlation']:.6f}")
    print(f"Mean Gene MSE: {results['mean_gene_mse']:.6f}")
    print(f"Mean Spot MSE: {results['mean_spot_mse']:.6f}")
    
    return results, all_predictions, all_targets


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
        f.write(f"Mean Gene Correlation: {results['mean_gene_correlation']:.6f}\n")
        f.write(f"Median Gene Correlation: {results['median_gene_correlation']:.6f}\n")
        f.write(f"Mean Spot Correlation: {results['mean_spot_correlation']:.6f}\n")
        f.write(f"Median Spot Correlation: {results['median_spot_correlation']:.6f}\n")
    
    # Save predictions and targets
    np.save(os.path.join(save_dir, f"fold_{fold_idx}_predictions.npy"), predictions)
    np.save(os.path.join(save_dir, f"fold_{fold_idx}_targets.npy"), targets)
    
    print(f"Results saved to {save_dir}")


def plot_training_curves(train_losses, test_losses, fold_idx, save_dir="./logs"):
    """
    Plot training and test loss curves
    """
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(test_losses, label='Test Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Curves - Fold {fold_idx}')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(os.path.join(save_dir, f"fold_{fold_idx}_training_curves.png"))
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
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return total_params, trainable_params