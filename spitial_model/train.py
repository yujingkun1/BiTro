#!/usr/bin/env python3
"""
Main Training Script for Cell2Gene HEST Spatial Gene Expression Prediction

author: Jingkun Yu
"""
# fmt: off
# pylint: disable=wrong-import-order
# add cluster
import os
import sys
import json
import warnings
import argparse

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
from spitial_model.utils import get_fold_samples, evaluate_model_metrics, save_evaluation_results, plot_training_curves, setup_device, plot_fold_gene_correlation_distribution, plot_metric_across_folds, save_epoch_metrics
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


def main():
    """Main training workflow - supports 10-fold and leave-one-out cross validation"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Cell2Gene HEST Spatial Model')
    parser.add_argument('--output_dir', type=str, 
                       default=os.environ.get('OUTPUT_DIR', './log_normalized'),
                       help='Output directory for logs, results, and checkpoints (default: ./log_normalized or from OUTPUT_DIR env var)')
    args = parser.parse_args()
    
    # Set output directory
    output_dir = args.output_dir
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    
    # Configuration parameters
    hest_data_dir = "/data/yujk/hovernet2feature/HEST/hest_data"
    graph_dir = "/data/yujk/hovernet2feature/hest_graphs_dinov3_other_cancer"
    features_dir = "/data/yujk/hovernet2feature/hest_dinov3_other_cancer"

    # Specify gene file
    gene_file = "/data/yujk/hovernet2feature/selected_hvg_genes_hest1k_100.txt"

    batch_size = 128
    num_epochs = 60
    learning_rate = 1e-4
    weight_decay = 1e-5   # 恢复正常weight_decay
    feature_dim = 128

    # 迁移学习配置
    use_transfer_learning = os.environ.get(
        "USE_TRANSFER_LEARNING", "false").lower() == "true"
    bulk_model_path = "/data/yujk/hovernet2feature/best_BRCA_lora_model_cluster.pt"
    freeze_backbone = os.environ.get(
        "FREEZE_BACKBONE", "false").lower() == "true"

    # Early stopping parameters
    patience = 5
    min_delta = 1e-6

    # Cross-validation mode
    # options: "kfold" (existing 10-fold) or "loo" (leave-one-out)
    cv_mode = os.environ.get("CV_MODE", "kfold")
    start_fold = 0  # only used for kfold

    # Gene variance normalization control
    # Set to True to apply per-gene variance normalization, False to disable
    use_gene_normalization = True

    print("=== HEST Spatial Supervised Training ===")
    print("✓ Using direct file reading (no HEST API required)")
    print("✓ Using 897 intersection genes")
    print(f"✓ CV mode: {cv_mode}")
    print(f"✓ Gene file: {gene_file}")
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
    device = setup_device(device_id=1)

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

    if cv_mode == "kfold":
        # Execute 10-fold cross validation (starting from specified fold)
        # Skip folds that have already been completed
        fold_plan = []
        for fi in range(start_fold, 10):
            if fi not in all_fold_results:
                fold_plan.append((fi,)+get_fold_samples(fi))
    elif cv_mode == "loo":
        # Build leave-one-out plan by scanning all samples from the fold definition union
        # Reuse get_fold_samples to enumerate all samples across folds
        all_samples = []
        for fi in range(0, 10):
            try:
                tr, te = get_fold_samples(fi)
                for s in tr + te:
                    if s not in all_samples:
                        all_samples.append(s)
            except Exception:
                # utils may define fewer than 10 folds (e.g., 6); stop when KeyError
                break
        if not all_samples:
            raise RuntimeError("No samples discovered for leave-one-out CV")

        # Support specifying held-out sample(s) via environment variable
        heldouts_env = os.environ.get(
            "CV_HELDOUT") or os.environ.get("LOO_HELDOUT")
        if heldouts_env:
            requested = [s.strip()
                         for s in heldouts_env.split(",") if s.strip()]
            # unique while preserving order
            seen = set()
            requested = [s for s in requested if not (
                s in seen or seen.add(s))]
            missing = [s for s in requested if s not in all_samples]
            if missing:
                raise ValueError(
                    f"Held-out sample(s) not found: {missing}. Available: {sorted(all_samples)}")
            print(f"✓ LOO held-out specified: {requested}")
            fold_plan = []
            for i, heldout in enumerate(requested):
                if i not in all_fold_results:
                    fold_plan.append((i, [s for s in all_samples if s not in {heldout}], [heldout]))
        else:
            # default: iterate all samples as held-out one by one
            # Skip folds that have already been completed
            fold_plan = []
            for i, heldout in enumerate(all_samples):
                if i not in all_fold_results:
                    fold_plan.append((i, [s for s in all_samples if s != heldout], [heldout]))
    else:
        raise ValueError(f"Unknown CV_MODE: {cv_mode}")

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
            num_epochs=num_epochs, device="cuda:1", patience=patience, min_delta=min_delta, fold_idx=fold_idx,
            cluster_loss_weight=0, checkpoint_path=best_model_path
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
