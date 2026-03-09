#!/usr/bin/env python3
"""
Main training script for BiTro HEST spatial gene expression prediction.

author: Jingkun Yu
"""
# fmt: off
# pylint: disable=wrong-import-order
import os
import sys
import json
import warnings
import argparse

# Keep numba cache writable when downstream imports pull in scanpy/numba.
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")

# Add parent directory to sys.path BEFORE any local imports
_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_current_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

import numpy as np
import torch
from torch.utils.data import DataLoader

# Import from local spitial_model submodules
from spitial_model.dataset import HESTSpatialDataset, collate_fn_hest_graph
from spitial_model.trainer import train_hest_graph_model, setup_optimizer_and_scheduler, setup_model
from spitial_model.utils import evaluate_model_metrics, save_evaluation_results, plot_training_curves, setup_device, plot_fold_gene_correlation_distribution, plot_metric_across_folds, save_epoch_metrics
# fmt: on

warnings.filterwarnings("ignore")


def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(
        f'Object of type {obj.__class__.__name__} is not JSON serializable')


def parse_sample_ids(sample_ids_env):
    """Parse a comma-separated sample id list from the environment."""
    if not sample_ids_env:
        return None

    sample_ids = []
    seen = set()
    for sample_id in sample_ids_env.split(","):
        sample_id = sample_id.strip()
        if sample_id and sample_id not in seen:
            sample_ids.append(sample_id)
            seen.add(sample_id)
    return sample_ids or None


def discover_sample_ids(hest_data_dir, explicit_sample_ids=None):
    """Resolve the ordered sample id list used for CV planning."""
    if explicit_sample_ids:
        return explicit_sample_ids

    st_dir = os.path.join(hest_data_dir, "st")
    if not os.path.isdir(st_dir):
        raise FileNotFoundError(f"ST directory not found: {st_dir}")

    sample_ids = sorted(
        os.path.splitext(name)[0]
        for name in os.listdir(st_dir)
        if name.endswith(".h5ad")
    )
    if not sample_ids:
        raise RuntimeError(f"No .h5ad samples found under: {st_dir}")
    return sample_ids


def build_cv_plan(sample_ids, cv_mode, all_fold_results, start_fold, heldouts_env=None):
    """Build fold tuples as (fold_idx, train_samples, test_samples)."""
    if cv_mode == "loo":
        if heldouts_env:
            requested = parse_sample_ids(heldouts_env) or []
            missing = [sample_id for sample_id in requested if sample_id not in sample_ids]
            if missing:
                raise ValueError(
                    f"Held-out sample(s) not found: {missing}. Available: {sorted(sample_ids)}"
                )
            heldout_samples = requested
            print(f"✓ LOO held-out specified: {heldout_samples}")
        else:
            heldout_samples = list(sample_ids)

        fold_plan = []
        for fold_idx, heldout in enumerate(heldout_samples):
            if fold_idx in all_fold_results:
                continue
            train_samples = [sample_id for sample_id in sample_ids if sample_id != heldout]
            fold_plan.append((fold_idx, train_samples, [heldout]))
        return fold_plan

    if cv_mode != "kfold":
        raise ValueError(f"Unknown CV_MODE: {cv_mode}")

    if not sample_ids:
        raise RuntimeError("No samples available for k-fold CV")

    num_folds = min(10, len(sample_ids))
    if num_folds == 1:
        print("Warning: only one sample available; k-fold will reuse it for both train and test")
        return [(0, list(sample_ids), list(sample_ids))]

    fold_chunks = [
        list(chunk)
        for chunk in np.array_split(np.array(sample_ids, dtype=object), num_folds)
        if len(chunk) > 0
    ]
    fold_plan = []
    for fold_idx, test_samples in enumerate(fold_chunks[start_fold:], start=start_fold):
        if fold_idx in all_fold_results:
            continue
        train_samples = [
            sample_id for idx, chunk in enumerate(fold_chunks)
            if idx != fold_idx for sample_id in chunk
        ]
        fold_plan.append((fold_idx, train_samples, list(test_samples)))
    return fold_plan


def main():
    """Main training workflow - supports 10-fold and leave-one-out cross validation"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train BiTro HEST Spatial Model')
    parser.add_argument('--output_dir', type=str, 
                       default=os.environ.get('OUTPUT_DIR', './log_normalized'),
                       help='Output directory for logs, results, and checkpoints (default: ./log_normalized or from OUTPUT_DIR env var)')
    parser.add_argument('--hest_data_dir', type=str,
                        default=os.environ.get('HEST_DATA_DIR', '/data/yujk/hovernet2feature/HEST/hest_data'),
                        help='HEST dataset root directory')
    parser.add_argument('--graph_dir', type=str,
                        default=os.environ.get('SPATIAL_GRAPH_DIR', '/data/yujk/hovernet2feature/hest_graphs_dinov3_other_cancer'),
                        help='Spatial graph directory')
    parser.add_argument('--features_dir', type=str,
                        default=os.environ.get('SPATIAL_FEATURE_DIR', '/data/yujk/hovernet2feature/hest_dinov3_other_cancer'),
                        help='Spatial feature directory')
    parser.add_argument('--gene_file', type=str,
                        default=os.environ.get('GENE_FILE', '/data/yujk/hovernet2feature/HEST-Bench/HCC/mean_50genes.txt'),
                        help='Gene list file')
    parser.add_argument('--sample_ids', type=str,
                        default=os.environ.get('SAMPLE_IDS'),
                        help='Comma-separated sample ids to use')
    parser.add_argument('--cv_mode', type=str,
                        default=os.environ.get('CV_MODE', 'loo'),
                        choices=['loo', 'kfold'],
                        help='Cross-validation mode')
    parser.add_argument('--cv_heldout', type=str,
                        default=os.environ.get('CV_HELDOUT') or os.environ.get('LOO_HELDOUT'),
                        help='Comma-separated held-out sample ids for leave-one-out mode')
    parser.add_argument('--cuda_device_id', type=int,
                        default=int(os.environ.get('CUDA_DEVICE_ID', '0')),
                        help='CUDA device id')
    parser.add_argument('--use_transfer_learning', type=str,
                        default=os.environ.get('USE_TRANSFER_LEARNING', 'false'),
                        choices=['true', 'false'],
                        help='Whether to use transfer learning')
    parser.add_argument('--freeze_backbone', type=str,
                        default=os.environ.get('FREEZE_BACKBONE', 'false'),
                        choices=['true', 'false'],
                        help='Whether to freeze backbone layers')
    parser.add_argument('--bulk_model_path', type=str,
                        default=os.environ.get('BULK_MODEL_PATH', '/data/yujk/hovernet2feature/best_BRCA_lora_model_cluster.pt'),
                        help='Bulk model checkpoint path for transfer learning')
    parser.add_argument('--numba_cache_dir', type=str,
                        default=os.environ.get('NUMBA_CACHE_DIR', '/tmp/numba_cache'),
                        help='Writable numba cache directory')
    args = parser.parse_args()

    os.environ["NUMBA_CACHE_DIR"] = args.numba_cache_dir
    
    # Set output directory
    output_dir = args.output_dir
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    
    # Configuration parameters
    hest_data_dir = args.hest_data_dir
    graph_dir = args.graph_dir
    features_dir = args.features_dir

    # Specify gene file
    gene_file = args.gene_file

    batch_size = 128
    num_epochs = 10
    learning_rate = 1e-4
    weight_decay = 1e-5   # Use a standard, non-zero weight decay.
    feature_dim = 128

    # Transfer learning configuration.
    use_transfer_learning = args.use_transfer_learning.lower() == "true"
    bulk_model_path = args.bulk_model_path
    freeze_backbone = args.freeze_backbone.lower() == "true"

    # Early stopping parameters
    patience = 5
    min_delta = 1e-6

    # Cross-validation mode
    # options: "kfold" (existing 10-fold) or "loo" (leave-one-out)
    cv_mode = args.cv_mode
    start_fold = 0  # only used for kfold
    device_id = args.cuda_device_id
    explicit_sample_ids = parse_sample_ids(args.sample_ids)
    sample_ids = discover_sample_ids(hest_data_dir, explicit_sample_ids)
    heldouts_env = args.cv_heldout

    # Gene variance normalization control
    # Set to True to apply per-gene variance normalization, False to disable
    use_gene_normalization = True

    print("=== HEST Spatial Supervised Training ===")
    print("✓ Using direct file reading (no HEST API required)")
    print(f"✓ Samples discovered: {sample_ids}")
    print(f"✓ CV mode: {cv_mode}")
    print(f"✓ HEST data directory: {hest_data_dir}")
    print(f"✓ Graph directory: {graph_dir}")
    print(f"✓ Feature directory: {features_dir}")
    print(f"✓ Gene file: {gene_file}")
    print(f"✓ Explicit sample ids arg: {explicit_sample_ids}")
    print(f"✓ CUDA device id: {device_id}")
    print(f"✓ NUMBA cache dir: {args.numba_cache_dir}")
    print(f"✓ Gene variance normalization: {'Enabled' if use_gene_normalization else 'Disabled'}")
    if use_transfer_learning:
        print(f"✓ Transfer Learning enabled from bulkmodel: {bulk_model_path}")
        print(f"✓ Freeze backbone: {freeze_backbone}")
    else:
        print("✓ Transfer Learning disabled - training from scratch")
    if cv_mode == "kfold":
        print(f"✓ Starting from Fold {start_fold + 1}")

    print(f"✓ Loss: MSE + optional cluster regularizer")

    # Setup device
    device = setup_device(device_id=device_id)

    # Math/precision optimizations
    try:
        torch.set_float32_matmul_precision('high')
    except Exception:
        pass
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass

    # Create results directory
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print(f"✓ Output directory: {output_dir}")
    print(f"✓ Checkpoint directory: {checkpoint_dir}")

    # Store all fold results
    all_fold_results = {}

    # Try to load temporary results
    temp_results_file = os.path.join(output_dir, "temp_fold_results.json")
    if os.path.exists(temp_results_file):
        try:
            with open(temp_results_file, 'r') as f:
                all_fold_results = json.load(f)
                # Convert string keys back to int
                all_fold_results = {
                    int(k): v for k, v in all_fold_results.items()}
                print(
                    f"✓ Loaded temporary results: {len(all_fold_results)} folds")
                print(f"  Completed folds: {sorted(all_fold_results.keys())}")
        except Exception as e:
            print(f"Warning: Could not load temporary results file: {e}")

    fold_plan = build_cv_plan(
        sample_ids=sample_ids,
        cv_mode=cv_mode,
        all_fold_results=all_fold_results,
        start_fold=start_fold,
        heldouts_env=heldouts_env,
    )
    if not fold_plan:
        print("✓ No remaining folds to process")
        return

    # Show summary of folds to be processed
    if all_fold_results:
        print(f"✓ Resuming from previous run: {len(all_fold_results)} folds already completed")
        completed_folds = sorted(all_fold_results.keys())
        print(f"  Already completed folds: {completed_folds}")

    for fold_tuple in fold_plan:
        fold_idx, train_samples, test_samples = fold_tuple
        total_folds = len(fold_plan)
        print(f"\n{'='*50}")
        print(f"Starting Fold {fold_idx + 1} (remaining: {total_folds})")
        print(f"{'='*50}")

        # Get current fold train and test samples
        print(f"Training samples ({len(train_samples)}): {train_samples}")
        print(f"Test samples ({len(test_samples)}): {test_samples}")

        # Create datasets (only load required samples)
        train_dataset = HESTSpatialDataset(
            hest_data_dir=hest_data_dir,
            graph_dir=graph_dir,
            features_dir=features_dir,
            sample_ids=train_samples,  # Only pass training samples
            feature_dim=feature_dim,
            mode='train',
            gene_file=gene_file,
            apply_gene_normalization=use_gene_normalization
        )

        normalization_stats = train_dataset.get_normalization_stats()

        test_dataset = HESTSpatialDataset(
            hest_data_dir=hest_data_dir,
            graph_dir=graph_dir,
            features_dir=features_dir,
            sample_ids=test_samples,   # Only pass test samples
            feature_dim=feature_dim,
            mode='test',
            gene_file=gene_file,
            apply_gene_normalization=train_dataset.apply_gene_normalization,
            normalization_stats=normalization_stats
        )

        # Create data loaders (defaults; simpler for stability)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn_hest_graph,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn_hest_graph,
        )

        num_genes = train_dataset.num_genes

        print(f"\n=== Fold {fold_idx + 1} Dataset Info ===")
        print(f"Device: {device}")
        print(f"Gene count: {num_genes}")
        print(f"Training spots: {len(train_dataset)}")
        print(f"Test spots: {len(test_dataset)}")

        # Create model
        model = setup_model(feature_dim, num_genes, device, use_transfer_learning=use_transfer_learning,
                            bulk_model_path=bulk_model_path, freeze_backbone=freeze_backbone)
        if model is None:
            print("Failed to setup model, skipping this fold")
            continue

        # Setup optimizer and scheduler
        optimizer, scheduler = setup_optimizer_and_scheduler(
            model, learning_rate, weight_decay, num_epochs
        )

        print(f"\n=== Fold {fold_idx + 1} Training Configuration ===")
        print(f"Model: StaticGraphTransformerPredictor + GAT")
        print(f"Optimizer: AdamW, Learning rate: {learning_rate}")
        print(f"Batch size: {batch_size}, Epochs: {num_epochs}")
        print(f"Early stopping: patience={patience}, min_delta={min_delta}")

        print(f"Loss: MSE + cluster")

        # Define checkpoint path for this fold
        best_model_path = os.path.join(output_dir, "best_hest_graph_model.pt")

        # Train model
        train_losses, test_losses, epoch_mean_gene_corrs, epoch_overall_corrs = train_hest_graph_model(
            model, train_loader, test_loader, optimizer, scheduler,
            num_epochs=num_epochs, device=device, patience=patience, min_delta=min_delta, fold_idx=fold_idx,
            cluster_loss_weight=0.1, checkpoint_path=best_model_path
        )

        # Load best model for evaluation
        if os.path.exists(best_model_path):
            model.load_state_dict(torch.load(
                best_model_path, map_location=device))
            print("Loaded best model weights for evaluation")

        # Evaluate model performance
        try:
            eval_results, predictions, targets = evaluate_model_metrics(
                model, test_loader, device)
        except Exception as e:
            print(f"Warning: Evaluation failed for fold {fold_idx + 1}: {e}")
            print("Skipping this fold and continuing with zero metrics...")
            # Return zero metrics when evaluation fails
            eval_results = {
                'overall_mse': 0.0,
                'overall_correlation': 0.0,
                'overall_correlation_pval': 1.0,
                'mean_gene_correlation': 0.0,
                'median_gene_correlation': 0.0,
                'std_gene_correlation': 0.0,
                'mean_spot_correlation': 0.0,
                'median_spot_correlation': 0.0,
                'std_spot_correlation': 0.0,
                'gene_correlations': [],
                'spot_correlations': [],
                'gene_mses': [],
                'spot_mses': []
            }
            predictions = None
            targets = None

        # Compute best pearsons during training epochs
        best_overall_pearson = float(
            max(epoch_overall_corrs) if epoch_overall_corrs else eval_results.get(
                'overall_correlation', 0.0)
        )
        best_gene_pearson = float(
            max(epoch_mean_gene_corrs) if epoch_mean_gene_corrs else eval_results.get(
                'mean_gene_correlation', 0.0)
        )

        # Save current fold's best model
        fold_model_path = os.path.join(checkpoint_dir, f"best_hest_graph_model_fold_{fold_idx}.pt")
        if os.path.exists(best_model_path):
            os.rename(best_model_path, fold_model_path)

        # Save current fold's complete evaluation metrics
        fold_evaluation_results = {
            'fold_idx': fold_idx,
            'train_samples': train_samples,
            'test_samples': test_samples,
            'num_train_spots': len(train_dataset),
            'num_test_spots': len(test_dataset),
            'num_genes': num_genes,
            'train_losses': train_losses,
            'test_losses': test_losses,
            'final_train_loss': train_losses[-1] if train_losses else None,
            'final_test_loss': test_losses[-1] if test_losses else None,
            'eval_results': eval_results,
            'best_overall_pearson': best_overall_pearson,
            'best_gene_pearson': best_gene_pearson
        }

        # Store in all results
        all_fold_results[fold_idx] = fold_evaluation_results

        # Save results
        save_evaluation_results(eval_results, predictions,
                                targets, fold_idx, output_dir)
        plot_training_curves(train_losses, test_losses,
                             fold_idx, output_dir, epoch_mean_gene_corrs=epoch_mean_gene_corrs, epoch_overall_corrs=epoch_overall_corrs)

        # Plot fold-level per-gene Pearson distribution
        plot_fold_gene_correlation_distribution(
            eval_results.get('gene_correlations'), fold_idx, output_dir)

        # Save per-epoch detailed metrics
        save_epoch_metrics(train_losses, test_losses, epoch_mean_gene_corrs, epoch_overall_corrs, fold_idx, output_dir)

        # Persist per-fold best pearsons into a cumulative file
        try:
            best_metrics_file = os.path.join(output_dir, "fold_best_pearsons.json")
            if os.path.exists(best_metrics_file):
                with open(best_metrics_file, 'r') as f:
                    cumulative_best = json.load(f)
            else:
                cumulative_best = {}
            cumulative_best[str(fold_idx)] = {
                'best_overall_pearson': best_overall_pearson,
                'best_gene_pearson': best_gene_pearson
            }
            with open(best_metrics_file, 'w') as f:
                json.dump(cumulative_best, f, indent=2,
                          default=convert_numpy_types)
        except Exception as e:
            print(f"Warning: could not update fold_best_pearsons.json: {e}")

        # Save temporary results (in case of interruption)
        with open(temp_results_file, 'w') as f:
            json.dump(all_fold_results, f, indent=2,
                      default=convert_numpy_types)

        print(f"\n=== Fold {fold_idx + 1} Completed ===")
        print(f"Final test loss: {test_losses[-1] if test_losses else 'N/A'}")
        print(
            f"Overall correlation: {eval_results['overall_correlation']:.6f}")
        print(
            f"Mean gene correlation: {eval_results['mean_gene_correlation']:.6f}")
        print(
            f"Std gene correlation: {eval_results.get('std_gene_correlation', 0.0):.6f}")

        # Clean up memory
        del model, train_dataset, test_dataset, train_loader, test_loader
        torch.cuda.empty_cache()

    # Final summary
    print(f"\n{'='*60}")
    if cv_mode == "kfold":
        print(f"10-FOLD CROSS VALIDATION COMPLETED")
    else:
        print(f"LEAVE-ONE-OUT CROSS VALIDATION COMPLETED")
    print(f"{'='*60}")

    # Calculate overall statistics
    overall_correlations = []
    mean_gene_correlations = []
    final_test_losses = []
    best_overall_list = []
    best_gene_list = []

    for fold_idx, results in all_fold_results.items():
        eval_results = results['eval_results']
        overall_correlations.append(eval_results['overall_correlation'])
        mean_gene_correlations.append(eval_results['mean_gene_correlation'])
        final_test_losses.append(results['final_test_loss'])
        best_overall_list.append(results.get(
            'best_overall_pearson', eval_results['overall_correlation']))
        best_gene_list.append(results.get(
            'best_gene_pearson', eval_results['mean_gene_correlation']))

    print(
        f"Average overall correlation: {np.mean(overall_correlations):.6f} ± {np.std(overall_correlations):.6f}")
    print(
        f"Average gene correlation: {np.mean(mean_gene_correlations):.6f} ± {np.std(mean_gene_correlations):.6f}")
    print(
        f"Average final test loss: {np.mean(final_test_losses):.6f} ± {np.std(final_test_losses):.6f}")

    # Report averages of best pearsons across folds
    if best_overall_list and best_gene_list:
        print(
            f"Average BEST overall correlation: {np.mean(best_overall_list):.6f} ± {np.std(best_overall_list):.6f}")
        print(
            f"Average BEST gene correlation: {np.mean(best_gene_list):.6f} ± {np.std(best_gene_list):.6f}")

    # Save final results
    final_results_file = os.path.join(output_dir, "final_10fold_results.json" if cv_mode == "kfold" else "final_loo_results.json")
    with open(final_results_file, 'w') as f:
        json.dump(all_fold_results, f, indent=2, default=convert_numpy_types)

    # Save final best pearson summary and plots
    final_best_file = os.path.join(output_dir, "final_10fold_best_pearsons.json" if cv_mode == "kfold" else "final_loo_best_pearsons.json")
    try:
        with open(final_best_file, 'w') as f:
            json.dump({
                # Per-fold data
                'per_fold_best_overall_pearson': best_overall_list,
                'per_fold_best_gene_pearson': best_gene_list,
                'per_fold_final_overall_pearson': overall_correlations,
                'per_fold_final_gene_pearson': mean_gene_correlations,
                'per_fold_final_test_loss': final_test_losses,
                # Best model statistics (from training checkpoints)
                'average_best_overall_pearson': float(np.mean(best_overall_list)) if best_overall_list else None,
                'std_best_overall_pearson': float(np.std(best_overall_list)) if best_overall_list else None,
                'average_best_gene_pearson': float(np.mean(best_gene_list)) if best_gene_list else None,
                'std_best_gene_pearson': float(np.std(best_gene_list)) if best_gene_list else None,
                # Final evaluation statistics (based on best model loaded from checkpoint)
                'average_final_overall_pearson': float(np.mean(overall_correlations)) if overall_correlations else None,
                'std_final_overall_pearson': float(np.std(overall_correlations)) if overall_correlations else None,
                'average_final_gene_pearson': float(np.mean(mean_gene_correlations)) if mean_gene_correlations else None,
                'std_final_gene_pearson': float(np.std(mean_gene_correlations)) if mean_gene_correlations else None,
                'average_final_test_loss': float(np.mean(final_test_losses)) if final_test_losses else None,
                'std_final_test_loss': float(np.std(final_test_losses)) if final_test_losses else None,
            }, f, indent=2, default=convert_numpy_types)
    except Exception as e:
        print(f"Warning: could not save final best pearsons summary: {e}")

    # Plots across folds
    try:
        plot_metric_across_folds(best_overall_list,
                                 title='Best Overall Pearson Across Folds',
                                 ylabel='Best Overall Pearson',
                                 filename='best_overall_pearson_across_folds.png',
                                 save_dir=output_dir)
        plot_metric_across_folds(best_gene_list,
                                 title='Best Mean Gene Pearson Across Folds',
                                 ylabel='Best Mean Gene Pearson',
                                 filename='best_gene_pearson_across_folds.png',
                                 save_dir=output_dir)
    except Exception as e:
        print(f"Warning: could not generate across-fold Pearson plots: {e}")

    print(f"\nFinal results saved to: {final_results_file}")
    if os.path.exists(final_best_file):
        print(f"Best pearson summary saved to: {final_best_file}")
    print("Training completed successfully!")


if __name__ == "__main__":
    main()
