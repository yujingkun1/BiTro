#!/usr/bin/env python3
"""
Training Module for Cell2Gene

author: Jingkun Yu
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


def train_hest_graph_model(model, train_loader, test_loader, optimizer, scheduler=None,
                           num_epochs=100, device="cuda", patience=10, min_delta=1e-6, fold_idx=None,
                           cluster_loss_weight: float = 0, checkpoint_path: str = "best_hest_graph_model.pt"):
    """
    Training function for HEST graph model with early stopping
    """
    model.to(device)
    criterion = nn.MSELoss()
    # Use correct GradScaler initialization to avoid API mismatch
    scaler = GradScaler('cuda')
    best_loss = float('inf')
    best_test_loss = float('inf')

    # Early stopping variables
    early_stopping_counter = 0
    best_epoch = 0
    # 最小启用早停的epoch阈值（每个fold在20个epoch之后才启动早停）
    min_early_stop_epoch = 12

    train_losses = []
    test_losses = []
    epoch_mean_gene_corrs = []  # per-epoch mean gene Pearson on test set
    epoch_overall_corrs = []    # per-epoch overall Pearson on test set

    print("=== Starting HEST Graph Training (with early stopping) ===")
    print(
        f"Early stopping settings: patience={patience}, min_delta={min_delta}")
    print(f"Early stopping active after epoch: {min_early_stop_epoch}")
    if fold_idx is not None:
        print(f"Current training: Fold {fold_idx + 1}")
    if cluster_loss_weight and cluster_loss_weight > 0:
        print(f"Using cluster regularization: weight={cluster_loss_weight}")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        num_batches = 0
        skipped_batches = 0

        pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                    desc=f"Epoch {epoch+1}/{num_epochs}",
                    ncols=100, leave=False)

        for batch_idx, batch in pbar:
            spot_expressions = batch["spot_expressions"].to(device, non_blocking=True)
            spot_graphs = batch["spot_graphs"]

            optimizer.zero_grad()

            # Flag for skipping batch
            skip_batch = False

            with autocast('cuda'):
                # Forward pass（返回节点嵌入用于聚类loss）
                out = model(spot_graphs, return_node_embeddings=True)
                if isinstance(out, tuple) and len(out) == 3:
                    predictions, node_embeddings_list, processed_indices = out
                else:
                    predictions = out
                    node_embeddings_list, processed_indices = [], list(range(len(spot_graphs)))

                # 添加调试信息
                if batch_idx == 0 and epoch == 0:
                    print(f"Debug - Batch {batch_idx}:")
                    print(
                        f"  Predictions shape: {predictions.shape}, range: [{predictions.min():.4f}, {predictions.max():.4f}]")
                    print(
                        f"  Targets shape: {spot_expressions.shape}, range: [{spot_expressions.min():.4f}, {spot_expressions.max():.4f}]")
                    print(
                        f"  predictions.requires_grad: {predictions.requires_grad}")

                # 对齐targets与有效预测（如有跳过的图）
                try:
                    if predictions.shape[0] != spot_expressions.shape[0]:
                        if processed_indices:
                            spot_expressions = spot_expressions.index_select(0, torch.as_tensor(processed_indices, device=spot_expressions.device))
                except Exception:
                    pass

                # Calculate loss in log space: pure MSE + 可选聚类loss
                recon_loss = criterion(predictions, spot_expressions)

                cluster_loss = torch.zeros((), device=predictions.device)
                if cluster_loss_weight and cluster_loss_weight > 0 and node_embeddings_list:
                    # 针对每个有效图，基于Transformer节点嵌入计算类内聚合损失
                    for emb, gi in zip(node_embeddings_list, processed_indices):
                        try:
                            g = spot_graphs[gi]
                            labels = getattr(g, 'cluster_labels', None)
                            if labels is None:
                                continue
                            labels = labels.to(emb.device)
                            if labels.dim() != 1 or labels.numel() != emb.size(0):
                                continue
                            # 对每个类别计算到类中心的平均距离
                            unique = torch.unique(labels)
                            for c in unique:
                                # 忽略负标签（如占位）
                                if c.item() < 0:
                                    continue
                                idx = (labels == c).nonzero(as_tuple=False).squeeze(1)
                                if idx.numel() > 1:
                                    cluster_feats = emb.index_select(0, idx)
                                    centroid = cluster_feats.mean(dim=0)
                                    distances = torch.norm(cluster_feats - centroid, dim=1)
                                    cluster_loss = cluster_loss + distances.mean()
                        except Exception:
                            continue

                loss = recon_loss + (cluster_loss_weight * cluster_loss if cluster_loss_weight and cluster_loss_weight > 0 else 0.0)

                # Check for anomalous values
                if torch.isnan(loss) or torch.isinf(loss):
                    print(
                        f"Warning: Batch {batch_idx} found anomalous loss {loss.item():.2f}, skipping batch")
                    skip_batch = True
                    skipped_batches += 1
                else:
                    # Backward pass
                    scaler.scale(loss).backward()

            if not skip_batch:
                # Gradient processing and optimizer update
                scaler.unscale_(optimizer)
                # 放宽梯度裁剪，允许更大的梯度更新
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item()
                num_batches += 1

                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        pbar.close()

        if num_batches == 0:
            print(f"⚠️  Epoch {epoch+1}: All batches were skipped")
            continue

        if skipped_batches > 0:
            total_batches = num_batches + skipped_batches
            fold_info = f"Fold {fold_idx + 1}, " if fold_idx is not None else ""
            print(
                f"\n⚠️  {fold_info}Epoch {epoch+1}: Skipped {skipped_batches}/{total_batches} batches")

        epoch_loss = running_loss / num_batches
        train_losses.append(epoch_loss)

        # Calculate test loss (pure MSE) and per-epoch Pearson (for monitoring only)
        from .utils import evaluate_model, evaluate_model_metrics
        test_loss = evaluate_model(model, test_loader, device)
        test_losses.append(test_loss)

        # Lightweight Pearson computation using existing metrics function
        try:
            metrics, _, _ = evaluate_model_metrics(model, test_loader, device)
            epoch_mean_gene_corr = float(metrics.get('mean_gene_correlation', 0.0))
            epoch_overall_corr = float(metrics.get('overall_correlation', 0.0))
        except Exception:
            epoch_mean_gene_corr = 0.0
            epoch_overall_corr = 0.0
        epoch_mean_gene_corrs.append(epoch_mean_gene_corr)
        epoch_overall_corrs.append(epoch_overall_corr)

        # Learning rate scheduling
        if scheduler is not None:
            scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {epoch_loss:.6f}, Test Loss: {test_loss:.6f}")
        print(f"  Mean Gene Pearson: {epoch_mean_gene_corr:.6f}")
        print(f"  Overall Pearson: {epoch_overall_corr:.6f}")
        print(f"  LR: {current_lr:.2e}")

        # 添加梯度范数监控
        total_grad_norm = 0.0
        param_count = 0
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_grad_norm += param_norm.item() ** 2
                param_count += 1
        total_grad_norm = total_grad_norm ** (1. / 2)
        print(f"  Grad Norm: {total_grad_norm:.6f}")

        # Early stopping logic
        if test_loss < best_test_loss - min_delta:
            # Test loss has significant improvement
            best_test_loss = test_loss
            best_epoch = epoch + 1
            early_stopping_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
            print(
                f"  *** Saving best model (Test Loss: {best_test_loss:.6f}, Epoch: {best_epoch}) ***")
        else:
            # Test loss has no significant improvement
            if (epoch + 1) >= min_early_stop_epoch:
                early_stopping_counter += 1
                print(
                    f"  Early stopping counter: {early_stopping_counter}/{patience} (active since epoch {min_early_stop_epoch})")

                if early_stopping_counter >= patience:
                    print(f"\n*** Early stopping triggered! ***")
                    print(
                        f"Test loss did not improve for {patience} epochs (min_delta={min_delta})")
                    print(
                        f"Best test loss: {best_test_loss:.6f} (Epoch {best_epoch})")
                    break
            else:
                print(f"  Early stopping inactive until epoch {min_early_stop_epoch}")

    print(f"\n=== Training completed ===")
    print(f"Best test loss: {best_test_loss:.6f} (Epoch {best_epoch})")
    print(f"Total epochs: {len(train_losses)}")

    return train_losses, test_losses, epoch_mean_gene_corrs, epoch_overall_corrs


def setup_optimizer_and_scheduler(model, learning_rate=1e-3, weight_decay=1e-5, num_epochs=60):
    """
    Setup optimizer and learning rate scheduler
    """
    # 大幅提高学习率到1e-3，这是更合理的范围
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6)

    return optimizer, scheduler


def setup_model(feature_dim, num_genes, device, use_transfer_learning=False, bulk_model_path=None, freeze_backbone=False):
    """
    Setup model with proper parameter initialization

    Args:
        feature_dim: Input feature dimension
        num_genes: Number of target genes
        device: Device to place model on
        use_transfer_learning: Whether to load pretrained weights from bulkmodel
        bulk_model_path: Path to bulkmodel weights
        freeze_backbone: Whether to freeze backbone layers during training
    """
    from .models import StaticGraphTransformerPredictor
    import os
    import os as _os

    # Create model
    # LoRA config via env (defaults enabled)
    use_lora_adapters = (_os.environ.get("USE_LORA", "true").lower() == "true")
    lora_r = int(_os.environ.get("LORA_R", "8"))
    lora_alpha = int(_os.environ.get("LORA_ALPHA", "16"))
    lora_dropout = float(_os.environ.get("LORA_DROPOUT", "0.05"))
    lora_freeze_base = (_os.environ.get("LORA_FREEZE_BASE", "true").lower() == "true")

    model = StaticGraphTransformerPredictor(
        input_dim=feature_dim,
        gnn_hidden_dim=128,
        gnn_output_dim=128,
        embed_dim=256,
        num_genes=num_genes,
        num_layers=2,
        nhead=8,
        dropout=0.3,
        use_gnn=True,
        gnn_type='GAT',
        n_pos=128,  # HIST2ST style positional encoding range
        use_lora=use_lora_adapters,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_freeze_base=lora_freeze_base,
    )

    # Optional: torch.compile for speedup on PyTorch 2.x
    try:
        use_compile = (_os.environ.get("USE_COMPILE", "false").lower() == "true")
        if use_compile and hasattr(torch, "compile"):
            model = torch.compile(model, mode="max-autotune")
            print("✓ torch.compile enabled (max-autotune)")
    except Exception as _e:
        print(f"torch.compile not enabled: {_e}")

    # Load pretrained weights from bulkmodel if transfer learning is enabled
    if use_transfer_learning and bulk_model_path and os.path.exists(bulk_model_path):
        print(f"\n=== Loading Pretrained Weights from Bulk Model ===")
        print(f"Bulk model path: {bulk_model_path}")
        try:
            bulk_state_dict = torch.load(bulk_model_path, map_location=device)
            print(f"Loaded state dict with {len(bulk_state_dict)} keys")

            # Load compatible weights and handle dimension mismatches
            model_state_dict = model.state_dict()
            loaded_keys = []
            skipped_keys = []
            mismatched_keys = []
            
            # Expected differences between bulk and spatial models:
            # - Bulk has pos_encoding, Spatial has x_embed/y_embed (spatial-specific)
            # - Spatial has gene_queries and gene_readout (spatial-specific)
            # - Both should share: gnn, feature_projection, transformer, output_projection (if compatible)
            
            # Categories for better logging
            lora_keys_loaded = []
            base_keys_loaded = []

            for key, value in bulk_state_dict.items():
                if key in model_state_dict:
                    if model_state_dict[key].shape == value.shape:
                        model_state_dict[key] = value
                        loaded_keys.append(key)
                        # Categorize keys
                        if 'lora' in key.lower():
                            lora_keys_loaded.append(key)
                        else:
                            base_keys_loaded.append(key)
                    else:
                        mismatched_keys.append(
                            (key, model_state_dict[key].shape, value.shape))
                else:
                    # Skip keys that are bulk-specific (e.g., pos_encoding) or spatial-specific
                    if 'pos_encoding' in key.lower():
                        # This is expected - bulk uses pos_encoding, spatial uses x_embed/y_embed
                        pass
                    else:
                        skipped_keys.append(key)

            model.load_state_dict(model_state_dict, strict=False)

            print(f"\n✓ Successfully loaded {len(loaded_keys)} weight layers from bulkmodel")
            if base_keys_loaded:
                print(f"  - Base weights: {len(base_keys_loaded)} layers")
                print(f"    Examples: {', '.join(base_keys_loaded[:3])}")
            if lora_keys_loaded:
                print(f"  - LoRA weights: {len(lora_keys_loaded)} layers")
                print(f"    Examples: {', '.join(lora_keys_loaded[:3])}")
            
            if mismatched_keys:
                print(f"\n⚠ Skipped {len(mismatched_keys)} layers due to shape mismatch:")
                # Show first 5
                for key, model_shape, bulk_shape in mismatched_keys[:5]:
                    print(f"    {key}: model {model_shape} vs bulk {bulk_shape}")
                if len(mismatched_keys) > 5:
                    print(f"    ... and {len(mismatched_keys) - 5} more")

            if skipped_keys:
                # Filter out expected differences
                unexpected_skips = [k for k in skipped_keys if 'pos_encoding' not in k.lower()]
                expected_skips = [k for k in skipped_keys if 'pos_encoding' in k.lower()]
                
                if expected_skips:
                    print(f"\n✓ Expected differences (bulk-specific layers):")
                    for key in expected_skips[:3]:
                        print(f"    {key} (bulk uses pos_encoding, spatial uses x_embed/y_embed)")
                
                if unexpected_skips:
                    print(f"\n⚠ Skipped {len(unexpected_skips)} layers not present in spatial model:")
                    for key in unexpected_skips[:5]:
                        print(f"    {key}")
                    if len(unexpected_skips) > 5:
                        print(f"    ... and {len(unexpected_skips) - 5} more")

            # Freeze backbone if requested
            if freeze_backbone:
                print(f"\n=== Freezing Backbone Layers ===")
                frozen_params = 0
                trainable_params = 0

                # Freeze GNN and feature projection layers (including LoRA adapters)
                if hasattr(model, 'gnn'):
                    for name, param in model.gnn.named_parameters():
                        param.requires_grad = False
                        frozen_params += param.numel()

                if hasattr(model, 'feature_projection'):
                    # Handle both base Linear and LoRALinear
                    for name, param in model.feature_projection.named_parameters():
                        param.requires_grad = False
                        frozen_params += param.numel()

                # Keep transformer, output projection, and spatial-specific layers trainable
                for param in model.transformer.parameters():
                    trainable_params += param.numel()

                for param in model.output_projection.parameters():
                    trainable_params += param.numel()
                
                # Spatial-specific layers (gene_queries, gene_readout, x_embed, y_embed)
                if hasattr(model, 'gene_queries'):
                    trainable_params += model.gene_queries.numel()
                if hasattr(model, 'gene_readout'):
                    for param in model.gene_readout.parameters():
                        trainable_params += param.numel()
                if hasattr(model, 'x_embed'):
                    for param in model.x_embed.parameters():
                        trainable_params += param.numel()
                if hasattr(model, 'y_embed'):
                    for param in model.y_embed.parameters():
                        trainable_params += param.numel()

                print(f"✓ Frozen {frozen_params:,} parameters (GNN + Feature Projection)")
                print(f"✓ Trainable {trainable_params:,} parameters (Transformer + Output + Spatial layers)")

        except Exception as e:
            print(f"❌ Error loading bulkmodel weights: {e}")
            print("Continuing with randomly initialized model...")
    else:
        if use_transfer_learning and not bulk_model_path:
            print("⚠ Transfer learning enabled but bulk_model_path not provided")
        elif use_transfer_learning and not os.path.exists(bulk_model_path):
            print(f"⚠ Bulk model file not found at {bulk_model_path}")
        print("Training spatial model from scratch with random initialization")

    # Check model parameter gradient settings
    print(f"\n=== Checking model parameter gradient settings ===")
    total_params = 0
    trainable_params = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
        else:
            print(f"Warning: Parameter {name} does not require grad!")

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    if trainable_params == 0:
        print("❌ Error: No trainable parameters!")
        return None

    # Ensure all trainable parameters require gradients (safety check)
    for param in model.parameters():
        if param.requires_grad == False and (not freeze_backbone or 'gnn' not in str(param)):
            param.requires_grad_(True)

    # Set model to training mode
    model.train()

    return model.to(device)
