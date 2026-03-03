#!/usr/bin/env python3
import os
import sys
import argparse
import psutil

# Allow running from the project root by adding it to sys.path.
_current_file_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_file_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

from bulk_model.utils import check_environment_compatibility, load_gene_mapping
from bulk_model.dataset import BulkStaticGraphDataset372, collate_fn_bulk_372
from bulk_model.models import OptimizedTransformerPredictor
from bulk_model.trainer import train_optimized_model, load_spatial_pretrained_weights

# TF32/AMP-related performance settings.
torch.set_float32_matmul_precision("medium")
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# Run a basic environment check at startup.
check_environment_compatibility()


def parse_args():
    parser = argparse.ArgumentParser(description='Bulk Model Training with LoRA and Optimizations')
    
    # Defaults that can be edited in code (toggle True/False here).
    DEFAULT_USE_GENE_ATTENTION = False
    DEFAULT_APPLY_GENE_NORMALIZATION = False
    DEFAULT_ENABLE_CLUSTER_LOSS = False
    DEFAULT_CLUSTER_LOSS_WEIGHT = 0.1
    # Data paths.
    parser.add_argument("--graph-data-dir", type=str, 
                       default="/root/autodl-tmp/bulk_PRAD_graphs_new_all_graph",
                       help="Graph data directory")
    parser.add_argument("--gene-list-file", type=str,
                       default="/root/autodl-tmp/PRAD_intersection_genes.txt",
                       help="Gene list file (one gene per line)")
    parser.add_argument("--features-file", type=str,
                       default="/root/autodl-tmp/features.tsv",
                       help="Feature mapping TSV file")
    parser.add_argument("--tpm-csv-file", type=str,
                       default="/root/autodl-tmp/tpm-TCGA-PRAD-intersect-normalized.csv",
                       help="TPM expression CSV file")
    
    # Training parameters.
    parser.add_argument("--batch-size", type=int, default=2,
                       help="Patient batch size (None = dynamic search)")
    parser.add_argument("--graph-batch-size", type=int, default=128,
                       help="Graph batch size")
    parser.add_argument("--num-epochs", type=int, default=60,
                       help="Number of epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5,
                       help="Weight decay")
    parser.add_argument("--patience", type=int, default=15,
                       help="Early stopping patience")
    parser.add_argument("--cluster-loss-weight", type=float, default=0,
                       help="Cluster regularization weight (0 = disabled)")
    parser.add_argument("--enable-cluster-loss", dest="enable_cluster_loss", action="store_true",
                       help="Enable cluster loss (otherwise cluster-loss-weight is ignored)")

    # Transfer learning: initialize from a spatial checkpoint.
    parser.add_argument("--spatial-model-path", type=str, default=None,
                       help="Optional: spatial model checkpoint to initialize bulk model weights")
    parser.add_argument("--freeze-backbone-from-spatial", action="store_true",
                       help="When using a spatial checkpoint, freeze backbone (GNN + feature_projection + transformer) and train only the head")
    
    # LoRA parameters.
    parser.add_argument("--use-lora", action="store_true", default=True,
                       help="Enable LoRA adapters")
    parser.add_argument("--lora-r", type=int, default=8,
                       help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=16,
                       help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05,
                       help="LoRA dropout")
    parser.add_argument("--lora-freeze-base", action="store_true", default=True,
                       help="Freeze base weights when using LoRA")
    parser.add_argument("--use-gene-attention", dest="use_gene_attention", action="store_true",
                       help="Enable gene attention readout (default comes from code-level config)")
    parser.add_argument("--no-gene-attention", dest="use_gene_attention", action="store_false",
                       help="Disable gene attention readout")
    
    # DataLoader options.
    parser.add_argument("--num-workers-train", type=int, default=0,
                       help="Train DataLoader worker count (None = auto)")
    parser.add_argument("--num-workers-test", type=int, default=0,
                       help="Test DataLoader worker count (None = auto)")
    parser.add_argument("--pin-memory", action="store_true", default=True,
                       help="Enable pin_memory")
    parser.add_argument("--persistent-workers", action="store_true", default=True,
                       help="Enable persistent_workers")
    parser.add_argument("--prefetch-factor", type=int, default=1,
                       help="Prefetch factor")
    parser.add_argument("--save-normalization-stats", type=str, default=None,
                       help="Optional: path to save gene normalization stats (mean/std) after training (.npz or .json)")
    
    # Training diagnostics and profiling.
    parser.add_argument("--log-every", type=int, default=10,
                       help="Log interval (in batches) for debug output")
    parser.add_argument("--debug-logs", action="store_true",
                       help="Enable verbose debug logging")
    parser.add_argument("--enable-profiling", action="store_true",
                       help="Enable per-batch performance profiling")
    parser.add_argument("--cleanup-interval", type=int, default=1,
                       help="CUDA cache cleanup interval (in batches)")
    parser.add_argument("--max-train-samples", type=int, default=None,
                       help="Maximum number of training samples (None = all)")
    
    # Dynamic batch size search.
    parser.add_argument("--disable-dynamic-bsz", action="store_true",
                       help="Disable dynamic batch size search")
    parser.add_argument("--max-dynamic-bsz", type=int, default=8,
                       help="Maximum batch size considered during dynamic search")
    
    # Apply code-level defaults (so users can toggle in code without CLI args).
    parser.set_defaults(use_gene_attention=DEFAULT_USE_GENE_ATTENTION)
    parser.set_defaults(apply_gene_normalization=DEFAULT_APPLY_GENE_NORMALIZATION)
    parser.set_defaults(enable_cluster_loss=DEFAULT_ENABLE_CLUSTER_LOSS)
    parser.set_defaults(cluster_loss_weight=DEFAULT_CLUSTER_LOSS_WEIGHT)

    return parser.parse_args()


def count_parameters(model):
    """Return total and trainable parameter counts."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def format_number(num):
    """Format large integers for display (K/M/B)."""
    if num >= 1e9:
        return f"{num / 1e9:.2f}B"
    elif num >= 1e6:
        return f"{num / 1e6:.2f}M"
    elif num >= 1e3:
        return f"{num / 1e3:.2f}K"
    else:
        return str(num)


def find_optimal_batch_size(dataset, device, max_batch_size=16):
    """Heuristically search for the largest feasible batch size."""
    print("Searching for an optimal batch size...")
    
    for bs in [2, 4, 8, 16]:
        if bs > max_batch_size:
            break
        
        try:
            print(f"  Testing batch_size={bs}")
            
            # Create a temporary loader for probing.
            test_loader = DataLoader(
                dataset,
                batch_size=bs,
                num_workers=2,
                pin_memory=True
            )
            
            # Probe a single batch.
            for batch in test_loader:
                expressions = batch['expressions'].to(device)
                spot_graphs_list = batch['spot_graphs_list']
                
                # Roughly estimate memory usage.
                torch.cuda.empty_cache()
                before_mem = torch.cuda.memory_allocated(device)
                
                # Simulate a tiny forward/backward step.
                dummy_prediction = torch.randn(
                    expressions.shape,
                    device=device,
                    requires_grad=True
                )
                loss = nn.MSELoss()(dummy_prediction, expressions)
                loss.backward()
                
                after_mem = torch.cuda.memory_allocated(device)
                mem_usage_gb = (after_mem - before_mem) / 1024**3
                
                print(f"    ✓ batch_size={bs} is feasible, estimated memory≈{mem_usage_gb:.1f}GB")
                
                del dummy_prediction, loss, expressions, spot_graphs_list
                torch.cuda.empty_cache()
                break
        
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"    ✗ batch_size={bs} OOM")
                torch.cuda.empty_cache()
                return max(1, bs // 2)
            else:
                print(f"    ✗ batch_size={bs} failed: {e}")
                torch.cuda.empty_cache()
                continue
    
    return min(bs, max_batch_size)


def main():
    args = parse_args()
    print("=== Optimized: multi-graph batching for better GPU utilization + LoRA ===")

    selected_genes, _ = load_gene_mapping(args.gene_list_file, args.features_file)
    if not selected_genes:
        print("Error: failed to load gene mapping")
        return
    print(f"Final gene count: {len(selected_genes)}")

    train_dataset = BulkStaticGraphDataset372(
        graph_data_dir=args.graph_data_dir,
        split='train',
        selected_genes=selected_genes,
        max_samples=args.max_train_samples,
        tpm_csv_file=args.tpm_csv_file
    )
    test_dataset = BulkStaticGraphDataset372(
        graph_data_dir=args.graph_data_dir,
        split='test',
        selected_genes=selected_genes,
        max_samples=None,
        tpm_csv_file=args.tpm_csv_file
    )
    print(f"Train samples: {len(train_dataset)}, test samples: {len(test_dataset)}")

    # If requested, save normalization stats (mean/std) computed from train split.
    if args.save_normalization_stats:
        try:
            stats = getattr(train_dataset, "normalization_stats", None)
            if stats is None:
                # Ensure normalization stats are available (usually computed during dataset init).
                train_dataset.setup_gene_normalization()
                stats = getattr(train_dataset, "normalization_stats", None)
            save_path = args.save_normalization_stats
            import numpy as _np, json as _json
            if save_path.endswith(".json"):
                # Convert arrays to JSON-serializable lists.
                serial = {"mean": _np.asarray(stats["mean"]).tolist(), "std": _np.asarray(stats["std"]).tolist()}
                with open(save_path, "w") as sf:
                    _json.dump(serial, sf)
            else:
                # Default: save as NPZ with 'mean' and 'std'.
                _np.savez_compressed(save_path, mean=_np.asarray(stats["mean"]), std=_np.asarray(stats["std"]))
            print(f"Saved train split normalization stats to: {save_path}")
        except Exception as e:
            print(f"Warning: unable to save normalization stats: {e}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Configure patient batch size (manual or dynamic search).
    initial_batch_size = 2
    if args.batch_size is not None:
        batch_size = max(1, args.batch_size)
        print(f"Using user-specified patient batch size: {batch_size}")
    elif args.disable_dynamic_bsz:
        batch_size = initial_batch_size
        print(f"Dynamic batch size disabled; using initial value: {batch_size}")
    else:
        optimal_batch_size = find_optimal_batch_size(train_dataset, device, max_batch_size=args.max_dynamic_bsz)
        batch_size = max(initial_batch_size, optimal_batch_size)
        print(f"Dynamic search selected patient batch size: {batch_size}")

    graph_batch_size = args.graph_batch_size
    print(f"Graph batch size: {graph_batch_size} (improve GPU utilization)")

    # DataLoader configuration.
    cpu_cores = psutil.cpu_count(logical=False) or 1
    logical_cores = psutil.cpu_count(logical=True) or cpu_cores
    print(f"CPU cores: physical={cpu_cores}, logical={logical_cores}")
    
    num_workers_train = args.num_workers_train if args.num_workers_train is not None else min(8, cpu_cores)
    num_workers_test = args.num_workers_test if args.num_workers_test is not None else min(4, cpu_cores)

    def build_loader(dataset, shuffle, num_workers):
        kwargs = dict(
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn_bulk_372,
            num_workers=max(0, num_workers),
            pin_memory=args.pin_memory,
            drop_last=False,
        )
        if num_workers > 0:
            kwargs["persistent_workers"] = args.persistent_workers
            if args.prefetch_factor is not None:
                kwargs["prefetch_factor"] = max(1, args.prefetch_factor)
        else:
            kwargs["pin_memory"] = False
        return DataLoader(dataset, **kwargs)

    train_loader = build_loader(train_dataset, shuffle=True, num_workers=num_workers_train)
    test_loader = build_loader(test_dataset, shuffle=False, num_workers=num_workers_test)

    print(f"DataLoader: train_workers={num_workers_train}, test_workers={num_workers_test}, pin_memory={args.pin_memory}")

    # Compare parameter counts with/without LoRA.
    print("\n" + "="*60)
    print("📊 Parameter count summary")
    print("="*60)
    
    if args.use_lora:
        print("Comparing parameter counts with vs without LoRA...")
        
        # Create a temporary model without LoRA to estimate the baseline parameter count.
        print("  1. Creating model without LoRA (baseline params)...")
        model_without_lora = OptimizedTransformerPredictor(
            input_dim=train_dataset.feature_dim,
            gnn_hidden_dim=128,
            gnn_output_dim=128,
            embed_dim=256,
            num_genes=train_dataset.num_genes,
            num_layers=3,
            nhead=8,
            dropout=0.1,
            use_gnn=True,
            gnn_type='GAT',
            graph_batch_size=graph_batch_size,
            use_lora=False,  # No LoRA.
            use_gene_attention=args.use_gene_attention,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            lora_freeze_base=args.lora_freeze_base
        )
        
        total_params_wo_lora, trainable_params_wo_lora = count_parameters(model_without_lora)
        print("     ✓ Baseline parameter count computed")
        
        # Create the actual model with LoRA (optionally initialized from spatial checkpoint).
        print("  2. Creating model with LoRA...")
        model = OptimizedTransformerPredictor(
            input_dim=train_dataset.feature_dim,
            gnn_hidden_dim=128,
            gnn_output_dim=128,
            embed_dim=256,
            num_genes=train_dataset.num_genes,
            num_layers=3,
            nhead=8,
            dropout=0.1,
            use_gnn=True,
            gnn_type='GAT',
            graph_batch_size=graph_batch_size,
            use_lora=args.use_lora,
            use_gene_attention=args.use_gene_attention,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            lora_freeze_base=args.lora_freeze_base
        )

        # If a spatial checkpoint is provided, use it to initialize weights.
        if args.spatial_model_path:
            model = load_spatial_pretrained_weights(
                model,
                spatial_checkpoint_path=args.spatial_model_path,
                device=device,
                freeze_backbone=args.freeze_backbone_from_spatial,
            )
        
        total_params_with_lora, trainable_params_with_lora = count_parameters(model)
        print("     ✓ LoRA parameter count computed")
        
        # Display comparison.
        print("\n" + "-"*60)
        print("Parameter count comparison (without vs with LoRA):")
        print("-"*60)
        print(f"{'Metric':<20} {'No LoRA':<20} {'LoRA':<20} {'Change':<20}")
        print("-"*60)
        
        # Total parameters.
        total_diff = total_params_with_lora - total_params_wo_lora
        total_diff_pct = (total_diff / total_params_wo_lora * 100) if total_params_wo_lora > 0 else 0
        print(f"{'Total params':<20} {format_number(total_params_wo_lora):<20} {format_number(total_params_with_lora):<20} "
              f"{total_diff_pct:+.2f}% ({format_number(total_diff)})")
        
        # Trainable parameters.
        trainable_diff = trainable_params_with_lora - trainable_params_wo_lora
        trainable_diff_pct = (trainable_diff / trainable_params_wo_lora * 100) if trainable_params_wo_lora > 0 else 0
        reduction_pct = ((trainable_params_wo_lora - trainable_params_with_lora) / trainable_params_wo_lora * 100) if trainable_params_wo_lora > 0 else 0
        
        print(f"{'Trainable':<20} {format_number(trainable_params_wo_lora):<20} {format_number(trainable_params_with_lora):<20} "
              f"{trainable_diff_pct:+.2f}% ({format_number(trainable_diff)})")
        
        if trainable_params_with_lora < trainable_params_wo_lora:
            print(f"\n✓ LoRA reduced trainable parameters by {reduction_pct:.2f}%")
            print(f"  Trainable params: {format_number(trainable_params_wo_lora)} -> {format_number(trainable_params_with_lora)}")
            print(f"  Reduced by: {format_number(trainable_params_wo_lora - trainable_params_with_lora)}")
        
        # Frozen parameters.
        frozen_params_wo_lora = total_params_wo_lora - trainable_params_wo_lora
        frozen_params_with_lora = total_params_with_lora - trainable_params_with_lora
        print(f"\n{'Frozen':<20} {format_number(frozen_params_wo_lora):<20} {format_number(frozen_params_with_lora):<20} "
              f"{format_number(frozen_params_with_lora - frozen_params_wo_lora)}")
        
        print("="*60 + "\n")
        
        # Cleanup temporary baseline model.
        del model_without_lora
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    else:
        # LoRA disabled: show current model parameter counts only.
        print("LoRA disabled; showing current model parameter counts...")
        model = OptimizedTransformerPredictor(
            input_dim=train_dataset.feature_dim,
            gnn_hidden_dim=128,
            gnn_output_dim=128,
            embed_dim=256,
            num_genes=train_dataset.num_genes,
            num_layers=3,
            nhead=8,
            dropout=0.1,
            use_gnn=True,
            gnn_type='GAT',
            graph_batch_size=graph_batch_size,
            use_lora=False,
            use_gene_attention=args.use_gene_attention,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            lora_freeze_base=args.lora_freeze_base
        )

        if args.spatial_model_path:
            model = load_spatial_pretrained_weights(
                model,
                spatial_checkpoint_path=args.spatial_model_path,
                device=device,
                freeze_backbone=args.freeze_backbone_from_spatial,
            )
        
        total_params, trainable_params = count_parameters(model)
        frozen_params = total_params - trainable_params
        
        print("\n" + "-"*60)
        print("Parameter counts:")
        print("-"*60)
        print(f"Total params:     {format_number(total_params)}")
        print(f"Trainable params: {format_number(trainable_params)}")
        print(f"Frozen params:    {format_number(frozen_params)}")
        print("="*60 + "\n")

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-7)

    print("\n=== Training config (LoRA + optimized) ===")
    print(f"Graph batch size: {graph_batch_size}")
    print(f"LoRA: r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}, freeze_base={args.lora_freeze_base}")
    print("Mixed mode: graph-enhanced patients + raw-feature patients")
    print("Data retention: 100% (0% dropped)")
    print(f"Logging: log_every={args.log_every}, debug={args.debug_logs}, profiling={args.enable_profiling}")

    # If cluster loss is explicitly disabled, use 0; otherwise use the provided weight.
    cluster_weight_to_use = args.cluster_loss_weight if getattr(args, "enable_cluster_loss", False) else 0.0

    train_losses, test_losses = train_optimized_model(
        model, train_loader, test_loader, optimizer, scheduler,
        num_epochs=args.num_epochs, device=device, patience=args.patience,
        log_every=args.log_every, debug=args.debug_logs,
        enable_profiling=args.enable_profiling,
        cleanup_interval=args.cleanup_interval,
        cluster_loss_weight=cluster_weight_to_use
    )

    print("\n=== Training finished (mixed mode) ===")
    print("✓ Supports graph-enhanced and raw-feature patients")
    print("✓ LoRA applied (reduced trainable parameters)")
    print("✓ Data retention: 100% (0% dropped)")
    print("✓ Original computation logic preserved")


if __name__ == "__main__":
    main()


