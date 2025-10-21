#!/usr/bin/env python3
"""
Main Training Script for Cell2Gene HEST Spatial Gene Expression Prediction

author: Jingkun Yu
"""

from spitial_model.utils import get_fold_samples, evaluate_model_metrics, save_evaluation_results, plot_training_curves, setup_device
from spitial_model.trainer import train_hest_graph_model, setup_optimizer_and_scheduler, setup_model
from spitial_model.dataset import HESTSpatialDataset, collate_fn_hest_graph
import numpy as np
import torch
from torch.utils.data import DataLoader
import os
import sys
import json
import warnings

# Add the Cell2Gene directory to sys.path BEFORE any local imports
_current_file_dir = os.path.dirname(
    os.path.abspath(__file__))  # spitial_model directory
_cell2gene_dir = os.path.dirname(_current_file_dir)  # Cell2Gene directory
if _cell2gene_dir not in sys.path:
    sys.path.insert(0, _cell2gene_dir)

# Now safe to import from spitial_model

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

    # Configuration parameters
    hest_data_dir = "/data/yujk/hovernet2feature/HEST/hest_data"
    graph_dir = "/data/yujk/hovernet2feature/hest_graphs_normalized_dinov3"
    features_dir = "/data/yujk/hovernet2feature/hest_normalized_dinov3"

    # Specify gene file
    gene_file = "/data/yujk/hovernet2feature/HEST/tutorials/SA_process/common_genes_misc_tenx_zen_897.txt"

    batch_size = 16
    num_epochs = 70
    learning_rate = 1e-6  # 大幅提高学习率
    weight_decay = 1e-5   # 恢复正常weight_decay
    feature_dim = 128

    # Early stopping parameters
    patience = 5
    min_delta = 1e-6

    # Cross-validation mode
    # options: "kfold" (existing 10-fold) or "loo" (leave-one-out)
    cv_mode = os.environ.get("CV_MODE", "kfold")
    start_fold = 0  # only used for kfold

    print("=== HEST Spatial Supervised Training ===")
    print("✓ Using direct file reading (no HEST API required)")
    print("✓ Using 897 intersection genes")
    print(f"✓ CV mode: {cv_mode}")
    print(f"✓ Gene file: {gene_file}")
    if cv_mode == "kfold":
        print(f"✓ Starting from Fold {start_fold + 1}")

    # Setup device
    device = setup_device(device_id=1)

    # Create results directory
    os.makedirs("./log_normalized", exist_ok=True)
    os.makedirs("./checkpoints", exist_ok=True)

    # Store all fold results
    all_fold_results = {}

    # Try to load temporary results
    temp_results_file = "./log_normalized/temp_fold_results.json"
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
        fold_plan = [(fi,)+get_fold_samples(fi)
                     for fi in range(start_fold, 10)]
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
            fold_plan = [
                (i, [s for s in all_samples if s not in {heldout}], [heldout])
                for i, heldout in enumerate(requested)
            ]
        else:
            # default: iterate all samples as held-out one by one
            fold_plan = [
                (i, [s for s in all_samples if s != heldout], [heldout])
                for i, heldout in enumerate(all_samples)
            ]
    else:
        raise ValueError(f"Unknown CV_MODE: {cv_mode}")

    for fold_tuple in fold_plan:
        fold_idx, train_samples, test_samples = fold_tuple
        total_folds = len(fold_plan)
        print(f"\n{'='*50}")
        print(f"Starting Fold {fold_idx + 1}/{total_folds}")
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
            gene_file=gene_file
        )

        test_dataset = HESTSpatialDataset(
            hest_data_dir=hest_data_dir,
            graph_dir=graph_dir,
            features_dir=features_dir,
            sample_ids=test_samples,   # Only pass test samples
            feature_dim=feature_dim,
            mode='test',
            gene_file=gene_file
        )

        # Create data loaders (reduce memory pressure)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn_hest_graph,
            num_workers=0,  # Avoid multi-process memory issues
            pin_memory=False,  # Reduce memory usage
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn_hest_graph,
            num_workers=0,  # Avoid multi-process memory issues
            pin_memory=False,  # Reduce memory usage
        )

        num_genes = train_dataset.num_genes

        print(f"\n=== Fold {fold_idx + 1} Dataset Info ===")
        print(f"Device: {device}")
        print(f"Gene count: {num_genes}")
        print(f"Training spots: {len(train_dataset)}")
        print(f"Test spots: {len(test_dataset)}")

        # Create model
        model = setup_model(feature_dim, num_genes, device)
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

        # Train model
        train_losses, test_losses = train_hest_graph_model(
            model, train_loader, test_loader, optimizer, scheduler,
            num_epochs=num_epochs, device=device, patience=patience, min_delta=min_delta, fold_idx=fold_idx
        )

        # Load best model for evaluation
        if os.path.exists("best_hest_graph_model.pt"):
            model.load_state_dict(torch.load(
                "best_hest_graph_model.pt", map_location=device))
            print("Loaded best model weights for evaluation")

        # Evaluate model performance
        eval_results, predictions, targets = evaluate_model_metrics(
            model, test_loader, device)

        # Save current fold's best model
        fold_model_path = f"./checkpoints/best_hest_graph_model_fold_{fold_idx}.pt"
        if os.path.exists("best_hest_graph_model.pt"):
            os.rename("best_hest_graph_model.pt", fold_model_path)

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
            'eval_results': eval_results
        }

        # Store in all results
        all_fold_results[fold_idx] = fold_evaluation_results

        # Save results
        save_evaluation_results(eval_results, predictions,
                                targets, fold_idx, "./log_normalized")
        plot_training_curves(train_losses, test_losses,
                             fold_idx, "./log_normalized")

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

    for fold_idx, results in all_fold_results.items():
        eval_results = results['eval_results']
        overall_correlations.append(eval_results['overall_correlation'])
        mean_gene_correlations.append(eval_results['mean_gene_correlation'])
        final_test_losses.append(results['final_test_loss'])

    print(
        f"Average overall correlation: {np.mean(overall_correlations):.6f} ± {np.std(overall_correlations):.6f}")
    print(
        f"Average gene correlation: {np.mean(mean_gene_correlations):.6f} ± {np.std(mean_gene_correlations):.6f}")
    print(
        f"Average final test loss: {np.mean(final_test_losses):.6f} ± {np.std(final_test_losses):.6f}")

    # Save final results
    final_results_file = "./log_normalized/final_10fold_results.json" if cv_mode == "kfold" else "./log_normalized/final_loo_results.json"
    with open(final_results_file, 'w') as f:
        json.dump(all_fold_results, f, indent=2, default=convert_numpy_types)

    print(f"\nFinal results saved to: {final_results_file}")
    print("Training completed successfully!")


if __name__ == "__main__":
    main()
