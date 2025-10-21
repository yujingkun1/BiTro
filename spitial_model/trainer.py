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
                           num_epochs=100, device="cuda", patience=10, min_delta=1e-6, fold_idx=None):
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

    train_losses = []
    test_losses = []

    print("=== Starting HEST Graph Training (with early stopping) ===")
    print(
        f"Early stopping settings: patience={patience}, min_delta={min_delta}")
    if fold_idx is not None:
        print(f"Current training: Fold {fold_idx + 1}")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        num_batches = 0
        skipped_batches = 0

        pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                    desc=f"Epoch {epoch+1}/{num_epochs}",
                    ncols=100, leave=False)

        for batch_idx, batch in pbar:
            spot_expressions = batch["spot_expressions"].to(device)
            spot_graphs = batch["spot_graphs"]

            optimizer.zero_grad()

            # Flag for skipping batch
            skip_batch = False

            with autocast('cuda'):
                # Forward pass
                predictions = model(spot_graphs)

                # 添加调试信息
                if batch_idx == 0 and epoch == 0:
                    print(f"Debug - Batch {batch_idx}:")
                    print(
                        f"  Predictions shape: {predictions.shape}, range: [{predictions.min():.4f}, {predictions.max():.4f}]")
                    print(
                        f"  Targets shape: {spot_expressions.shape}, range: [{spot_expressions.min():.4f}, {spot_expressions.max():.4f}]")
                    print(
                        f"  predictions.requires_grad: {predictions.requires_grad}")

                # Calculate loss in log space: targets are already log1p-transformed
                loss = criterion(predictions, spot_expressions)

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
                    model.parameters(), max_norm=10.0)
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

        # Calculate test loss
        from .utils import evaluate_model
        test_loss = evaluate_model(model, test_loader, device)
        test_losses.append(test_loss)

        # Learning rate scheduling
        if scheduler is not None:
            scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {epoch_loss:.6f}, Test Loss: {test_loss:.6f}")
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
            torch.save(model.state_dict(), "best_hest_graph_model.pt")
            print(
                f"  *** Saving best model (Test Loss: {best_test_loss:.6f}, Epoch: {best_epoch}) ***")
        else:
            # Test loss has no significant improvement
            early_stopping_counter += 1
            print(
                f"  Early stopping counter: {early_stopping_counter}/{patience}")

            if early_stopping_counter >= patience:
                print(f"\n*** Early stopping triggered! ***")
                print(
                    f"Test loss did not improve for {patience} epochs (min_delta={min_delta})")
                print(
                    f"Best test loss: {best_test_loss:.6f} (Epoch {best_epoch})")
                break

    print(f"\n=== Training completed ===")
    print(f"Best test loss: {best_test_loss:.6f} (Epoch {best_epoch})")
    print(f"Total epochs: {len(train_losses)}")

    return train_losses, test_losses


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


def setup_model(feature_dim, num_genes, device):
    """
    Setup model with proper parameter initialization
    """
    from .models import StaticGraphTransformerPredictor

    # Create model
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
        n_pos=128  # HIST2ST style positional encoding range
    )

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

    # Ensure all parameters require gradients
    for param in model.parameters():
        param.requires_grad_(True)

    # Set model to training mode
    model.train()

    return model.to(device)
