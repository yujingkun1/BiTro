import os
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
import matplotlib.pyplot as plt
import time
import numpy as np
import psutil
import gc


def load_spatial_pretrained_weights(
    model: nn.Module,
    spatial_checkpoint_path: str,
    device: str | torch.device = "cuda",
    freeze_backbone: bool = False,
) -> nn.Module:
    """
    Load compatible weights from a pretrained spatial model checkpoint
    into the bulk OptimizedTransformerPredictor to enable transfer learning.

    The spatial model is `spitial_model.models.StaticGraphTransformerPredictor`
    trained via `spitial_model/train_transfer_learning.py`.

    This function:
    - Loads the spatial checkpoint (state_dict)
    - Copies over matching keys with identical tensor shapes:
        * gnn.*
        * feature_projection.*
        * transformer.*
        * output_projection.*  (only when shape matches num_genes)
    - Skips spatial-specific parameters such as:
        * gene_queries, gene_readout
        * x_embed, y_embed
    - Optionally freezes backbone layers (GNN + feature_projection + transformer)
      while keeping output head trainable.
    """
    device = torch.device(device)

    if not spatial_checkpoint_path:
        print("Warning: spatial checkpoint path not provided; skipping transfer initialization")
        return model

    if not os.path.exists(spatial_checkpoint_path):
        print(f"Warning: spatial checkpoint not found: {spatial_checkpoint_path}")
        return model

    print("\n=== Initializing bulk weights from a pretrained spatial checkpoint ===")
    print(f"Spatial checkpoint: {spatial_checkpoint_path}")

    try:
        spatial_state = torch.load(spatial_checkpoint_path, map_location=device)
        # Support both raw state_dict and {"state_dict": ...} formats.
        if isinstance(spatial_state, dict) and "state_dict" in spatial_state:
            spatial_state = spatial_state["state_dict"]
        if not isinstance(spatial_state, dict):
            raise ValueError("Spatial checkpoint does not contain a valid state_dict")

        model_state = model.state_dict()

        loaded_keys: list[str] = []
        skipped_keys: list[str] = []
        mismatched_keys: list[tuple[str, torch.Size, torch.Size]] = []

        for key, value in spatial_state.items():
            # Skip spatial-specific parameters.
            if any(
                s in key
                for s in [
                    "gene_queries",
                    "gene_readout",
                    "x_embed",
                    "y_embed",
                ]
            ):
                skipped_keys.append(key)
                continue

            if key in model_state:
                if model_state[key].shape == value.shape:
                    model_state[key] = value
                    loaded_keys.append(key)
                else:
                    mismatched_keys.append((key, model_state[key].shape, value.shape))
            else:
                # Skip keys that do not exist in the bulk model.
                skipped_keys.append(key)

        model.load_state_dict(model_state, strict=False)

        print(f"✓ Loaded {len(loaded_keys)} compatible layers from the spatial checkpoint into the bulk model")
        if loaded_keys:
            print(f"  Examples: {', '.join(loaded_keys[:5])}")

        if mismatched_keys:
            print(f"⚠ Skipped {len(mismatched_keys)} layers due to shape mismatch (e.g., different gene counts in the output head):")
            for k, ms, ss in mismatched_keys[:5]:
                print(f"  {k}: bulk {ms} vs spatial {ss}")
            if len(mismatched_keys) > 5:
                print(f"  ... and {len(mismatched_keys) - 5} more")

        if skipped_keys:
            print(f"ℹ Skipped {len(skipped_keys)} spatial-only or bulk-missing parameters (e.g., gene_queries/x_embed)")

        if freeze_backbone:
            print("\n=== Freezing bulk backbone (GNN + feature_projection + transformer) ===")
            frozen_params = 0
            trainable_params = 0

            # Freeze GNN.
            if hasattr(model, "gnn"):
                for _, p in model.gnn.named_parameters():
                    p.requires_grad = False
                    frozen_params += p.numel()

            # Freeze feature projection.
            if hasattr(model, "feature_projection"):
                for _, p in model.feature_projection.named_parameters():
                    p.requires_grad = False
                    frozen_params += p.numel()

            # Freeze Transformer.
            if hasattr(model, "transformer"):
                for _, p in model.transformer.named_parameters():
                    p.requires_grad = False
                    frozen_params += p.numel()

            # Count remaining trainable parameters (e.g., output head).
            for name, p in model.named_parameters():
                if p.requires_grad:
                    trainable_params += p.numel()

            print(f"✓ Frozen backbone parameters: {frozen_params:,}")
            print(f"✓ Remaining trainable parameters: {trainable_params:,}")

    except Exception as e:  # pylint: disable=broad-except
        print(f"Error: failed to load weights from spatial checkpoint: {e}")
        print("Continuing training with randomly initialized bulk model weights.")

    return model


def train_optimized_model(model, train_loader, test_loader, optimizer, scheduler=None, 
                         num_epochs=50, device="cuda", patience=10, min_delta=1e-6,
                         log_every=10, debug=False, enable_profiling=False, cleanup_interval=1,
                         cluster_loss_weight: float = 0.0):
    model.to(device)
    criterion = nn.MSELoss()
    scaler = GradScaler('cuda')

    best_loss = float('inf')
    best_test_loss = float('inf')
    early_stopping_counter = 0
    best_epoch = 0

    train_losses = []
    test_losses = []

    print("=== Starting optimized training (multi-graph batching) ===")
    print(f"Graph batch size: {model.graph_batch_size}")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        num_batches = 0
        batch_skip_count = 0
        patient_skip_count = 0

        print(f"\n=== Epoch {epoch+1} training ===")

        for batch_idx, batch in enumerate(train_loader):
            batch_start_time = time.time() if enable_profiling else None
            data_loading_time = 0.0
            
            expressions = batch['expressions'].to(device, non_blocking=True)
            spot_graphs_list = batch['spot_graphs_list']
            cluster_labels_list = batch.get('cluster_labels_list', None)

            log_this_batch = (batch_idx % log_every == 0) or debug
            if log_this_batch:
                print(f"\nBatch {batch_idx}: processing {len(spot_graphs_list)} patients")

            optimizer.zero_grad()

            batch_predictions = []
            # accumulate cluster loss for this batch across patients
            batch_cluster_loss = 0.0

            for i in range(len(spot_graphs_list)):
                spot_graphs = batch['spot_graphs_list'][i]
                all_cell_features = batch['all_cell_features_list'][i]
                all_cell_positions = batch['all_cell_positions_list'][i]
                has_graphs = batch['has_graphs_list'][i]

                if log_this_batch:
                    print(f"  Patient {i+1}: cell_features={all_cell_features.shape}, positions={all_cell_positions.shape}, has_graphs={has_graphs}, n_graphs={len(spot_graphs) if spot_graphs else 0}")

                if all_cell_features.shape[0] == 0:
                    if log_this_batch:
                        print(f"    Warning: skipping patient {i+1} (no cell features)")
                    patient_skip_count += 1
                    continue

                all_cell_features = all_cell_features.to(device, non_blocking=True)
                all_cell_positions = all_cell_positions.to(device, non_blocking=True)

                if has_graphs and len(spot_graphs) > 0:
                    for graph in spot_graphs:
                        if hasattr(graph, 'x') and graph.x is not None:
                            graph.x = graph.x.to(device, non_blocking=True)
                        if hasattr(graph, 'edge_index') and graph.edge_index is not None:
                            graph.edge_index = graph.edge_index.to(device, non_blocking=True)

                forward_start_time = time.time() if enable_profiling else None

                with autocast('cuda'):
                    if has_graphs and len(spot_graphs) > 0:
                        total_cells = sum([graph.x.shape[0] for graph in spot_graphs if hasattr(graph, 'x') and graph.x is not None])
                        max_cells_threshold = 150000

                        if total_cells <= max_cells_threshold:
                            if log_this_batch:
                                print(f"    Graph mode: {len(spot_graphs)} graphs -> {total_cells} cells (graph-enhanced)")
                            cell_predictions_list = model(spot_graphs)
                        else:
                            if log_this_batch:
                                print(f"    Large graph sample: {len(spot_graphs)} graphs -> {total_cells} cells (chunked to fit memory)")
                            target_cells_per_batch = 10000
                            batch_size_adaptive = max(32, len(spot_graphs) * target_cells_per_batch // total_cells)
                            all_cell_predictions_list = []
                            for batch_start in range(0, len(spot_graphs), batch_size_adaptive):
                                batch_end = min(batch_start + batch_size_adaptive, len(spot_graphs))
                                batch_graphs = spot_graphs[batch_start:batch_end]
                                batch_cells = sum([g.x.shape[0] for g in batch_graphs if hasattr(g, 'x')])
                                if log_this_batch:
                                    print(f"      Chunk {batch_start//batch_size_adaptive + 1}: {len(batch_graphs)} graphs -> {batch_cells} cells")
                                current_batch_predictions = model(batch_graphs)
                                all_cell_predictions_list.extend(current_batch_predictions)
                                torch.cuda.empty_cache()
                                del current_batch_predictions
                            cell_predictions_list = all_cell_predictions_list
                    else:
                        if log_this_batch:
                            print(f"    No-graph mode: {all_cell_features.shape[0]} cells (raw DINO features)")
                        cell_predictions = model.forward_raw_features(all_cell_features, all_cell_positions)
                        cell_predictions_list = [cell_predictions]

                    if cell_predictions_list:
                        # Detect whether model returned gene-level per-graph outputs (1D tensor length == num_genes)
                        first_pred = cell_predictions_list[0]
                        if isinstance(first_pred, torch.Tensor) and first_pred.dim() == 1 and first_pred.shape[0] == expressions.shape[1]:
                            # gene-level outputs per graph: stack then sum across graphs to aggregate per-patient
                            stacked = torch.stack([pred for pred in cell_predictions_list if pred.numel() > 0], dim=0)  # [num_graphs, G]
                            if stacked.shape[0] > 0:
                                aggregated_prediction = stacked.sum(dim=0, keepdim=True)  # [1, G]
                                if log_this_batch:
                                    print(f"    Patient {i+1} gene-level aggregation: n_graphs={stacked.shape[0]}, aggregated_shape={aggregated_prediction.shape}")
                            else:
                                aggregated_prediction = torch.zeros(1, expressions.shape[1], device=device)
                                if log_this_batch:
                                    print(f"    Patient {i+1} aggregation: no valid graphs; using zero prediction")
                        else:
                            # node-level outputs: concatenate per-node predictions and sum across nodes
                            all_cell_predictions = torch.cat([pred for pred in cell_predictions_list if pred.shape[0] > 0], dim=0)
                            # compute cluster loss on node-level predictions if cluster labels are available
                            cluster_loss_sample = 0.0
                            if cluster_loss_weight > 0 and cluster_labels_list is not None:
                                try:
                                    cl = cluster_labels_list[i]
                                    if cl is not None:
                                        # ensure tensor on device and length matches cells
                                        if not isinstance(cl, torch.Tensor):
                                            cl = torch.tensor(cl, dtype=torch.long)
                                        cl = cl.to(device)
                                        if cl.numel() == all_cell_predictions.shape[0]:
                                            unique_labels = torch.unique(cl)
                                            for label in unique_labels:
                                                if label.item() == -1:
                                                    continue
                                                mask = (cl == label)
                                                cluster_preds = all_cell_predictions[mask]
                                                if cluster_preds.size(0) > 1:
                                                    centroid = cluster_preds.mean(dim=0)
                                                    distances = torch.norm(cluster_preds - centroid, dim=1)
                                                    cluster_loss_sample += distances.mean()
                                except Exception:
                                    cluster_loss_sample = 0.0
                            if all_cell_predictions.shape[0] > 0:
                                aggregated_prediction = all_cell_predictions.sum(dim=0, keepdim=True)
                                if log_this_batch:
                                    print(f"    Patient {i+1} aggregation: n_cells={all_cell_predictions.shape[0]}, aggregated_shape={aggregated_prediction.shape}, cluster_loss_sample={cluster_loss_sample:.6f}")
                            else:
                                aggregated_prediction = torch.zeros(1, expressions.shape[1], device=device)
                                if log_this_batch:
                                    print(f"    Patient {i+1} aggregation: no valid cells; using zero prediction")
                            # accumulate cluster loss for batch
                            batch_cluster_loss += cluster_loss_sample
                    else:
                        aggregated_prediction = torch.zeros(1, expressions.shape[1], device=device)
                        if log_this_batch:
                            print(f"    Patient {i+1} aggregation: no predictions returned; using zero prediction")

                batch_predictions.append(aggregated_prediction)

            if not batch_predictions:
                if log_this_batch:
                    print(f"    Warning: Batch {batch_idx}: all patients skipped; no valid predictions")
                batch_skip_count += 1
                continue

            if len(batch_predictions) != len(spot_graphs_list):
                if log_this_batch:
                    print(f"    Warning: Batch {batch_idx}: only {len(batch_predictions)}/{len(spot_graphs_list)} patients produced valid predictions")

            predictions = torch.cat(batch_predictions, dim=0)
            if log_this_batch:
                print(f"  Batch {batch_idx} merged predictions: shape={predictions.shape}")

            if predictions.shape[0] != expressions.shape[0]:
                if log_this_batch:
                    print(f"    Warning: prediction/target batch size mismatch: {predictions.shape[0]} vs {expressions.shape[0]}")
                expressions = expressions[:predictions.shape[0]]

            with autocast('cuda'):
                with autocast('cuda'):
                    pred_sum = predictions.sum().item()
                    if pred_sum <= 1e-10 or not torch.isfinite(predictions).all():
                        if log_this_batch:
                            print("    Error: invalid predictions (sum too small or non-finite); skipping batch")
                        batch_skip_count += 1
                        continue

                    epsilon = 1e-8
                    sum_pred = predictions.sum(dim=1, keepdim=True) + epsilon
                    normalized_pred = predictions / sum_pred
                    result = torch.clamp(normalized_pred * 1000000.0, min=0.0, max=1e6)

                    if torch.isnan(result).any() or torch.isinf(result).any():
                        if log_this_batch:
                            print("    Error: normalized result contains NaN/Inf; skipping batch")
                        batch_skip_count += 1
                        continue

                    loss = criterion(result, expressions)
                    # add cluster loss term (weighted) aggregated across patients in this batch
                    if cluster_loss_weight and batch_cluster_loss:
                        # batch_cluster_loss already summed across patients
                        loss = loss + cluster_loss_weight * batch_cluster_loss
                if log_this_batch:
                    print(f"  Loss: {loss.item():.6f}")
                if torch.isnan(loss) or torch.isinf(loss):
                    if log_this_batch:
                        print("    Error: loss is NaN/Inf; skipping batch")
                    batch_skip_count += 1
                    continue

                backward_start_time = time.time() if enable_profiling else None
                if log_this_batch:
                    print("  Backward pass...")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                scaler.step(optimizer)
                scaler.update()
                if log_this_batch:
                    print("  Backward pass done")
                if enable_profiling:
                    backward_time = time.time() - backward_start_time if backward_start_time else 0.0
                    forward_time = (backward_start_time - forward_start_time) if forward_start_time else 0.0
                    total_batch_time = time.time() - batch_start_time if batch_start_time else 0.0
                    if log_this_batch:
                        print(f"  Profiling: total={total_batch_time:.3f}s, data={data_loading_time:.3f}s, forward={forward_time:.3f}s, backward={backward_time:.3f}s")

                running_loss += loss.item()
                num_batches += 1

                # Important: only delete large tensors after monitoring/logging.
                del predictions, result, loss
                del batch_predictions
                del expressions, spot_graphs_list
                if cleanup_interval and cleanup_interval > 0:
                    if (batch_idx + 1) % cleanup_interval == 0 and torch.cuda.is_available():
                        torch.cuda.empty_cache()

        if num_batches == 0:
            print(f"Epoch {epoch+1}: all batches were skipped")
            print(f"  Skipped batches: {batch_skip_count}")
            print(f"  Skipped patients: {patient_skip_count}")
            continue

        epoch_loss = running_loss / num_batches
        train_losses.append(epoch_loss)

        print(f"\nEpoch {epoch+1} training summary:")
        print(f"  Total batches: {batch_idx + 1}")
        print(f"  Trained batches: {num_batches}")
        print(f"  Skipped batches: {batch_skip_count}")
        print(f"  Skipped patients: {patient_skip_count}")
        print(f"  Mean loss: {epoch_loss:.6f}")

        model.eval()
        test_loss = 0.0
        test_batches = 0
        with torch.no_grad():
            for batch in test_loader:
                expressions = batch['expressions'].to(device, non_blocking=True)
                spot_graphs_list = batch['spot_graphs_list']

                batch_predictions = []
                for i in range(len(spot_graphs_list)):
                    spot_graphs = spot_graphs_list[i]
                    all_cell_features = batch['all_cell_features_list'][i]
                    all_cell_positions = batch['all_cell_positions_list'][i]
                    has_graphs = batch['has_graphs_list'][i]

                    if all_cell_features.shape[0] == 0:
                        continue

                    all_cell_features = all_cell_features.to(device, non_blocking=True)
                    all_cell_positions = all_cell_positions.to(device, non_blocking=True)

                    if has_graphs and len(spot_graphs) > 0:
                        for graph in spot_graphs:
                            if hasattr(graph, 'x') and graph.x is not None:
                                graph.x = graph.x.to(device, non_blocking=True)
                            if hasattr(graph, 'edge_index') and graph.edge_index is not None:
                                graph.edge_index = graph.edge_index.to(device, non_blocking=True)

                    if has_graphs and len(spot_graphs) > 0:
                        total_cells = sum([graph.x.shape[0] for graph in spot_graphs if hasattr(graph, 'x')])
                        max_cells_threshold = 150000
                        if total_cells <= max_cells_threshold:
                            cell_predictions_list = model(spot_graphs)
                        else:
                            target_cells_per_batch = 10000
                            batch_size_adaptive = max(32, len(spot_graphs) * target_cells_per_batch // total_cells)
                            all_cell_predictions_list = []
                            for batch_start in range(0, len(spot_graphs), batch_size_adaptive):
                                batch_end = min(batch_start + batch_size_adaptive, len(spot_graphs))
                                batch_graphs = spot_graphs[batch_start:batch_end]
                                current_predictions = model(batch_graphs)
                                all_cell_predictions_list.extend(current_predictions)
                                torch.cuda.empty_cache()
                                del current_predictions
                            cell_predictions_list = all_cell_predictions_list
                    else:
                        cell_predictions = model.forward_raw_features(all_cell_features, all_cell_positions)
                        cell_predictions_list = [cell_predictions]

                    if cell_predictions_list:
                        first_pred = cell_predictions_list[0]
                        if isinstance(first_pred, torch.Tensor) and first_pred.dim() == 1 and first_pred.shape[0] == expressions.shape[1]:
                            stacked = torch.stack([pred for pred in cell_predictions_list if pred.numel() > 0], dim=0)
                            if stacked.shape[0] > 0:
                                aggregated_prediction = stacked.sum(dim=0, keepdim=True)
                            else:
                                aggregated_prediction = torch.zeros(1, expressions.shape[1], device=device)
                        else:
                            all_cell_predictions = torch.cat([pred for pred in cell_predictions_list if pred.shape[0] > 0], dim=0)
                            if all_cell_predictions.shape[0] > 0:
                                aggregated_prediction = all_cell_predictions.sum(dim=0, keepdim=True)
                            else:
                                aggregated_prediction = torch.zeros(1, expressions.shape[1], device=device)
                    else:
                        aggregated_prediction = torch.zeros(1, expressions.shape[1], device=device)

                    batch_predictions.append(aggregated_prediction)

                if batch_predictions:
                    predictions = torch.cat(batch_predictions, dim=0)
                    sum_pred = predictions.sum(dim=1, keepdim=True).clamp(min=1e-8)
                    normalized_pred = predictions / sum_pred
                    result = normalized_pred * 1000000.0
                    loss = criterion(result, expressions)
                    if torch.isfinite(loss):
                        test_loss += loss.item()
                        test_batches += 1

                del predictions, result, loss
                del batch_predictions
                del expressions, spot_graphs_list
                torch.cuda.empty_cache()

        test_loss = test_loss / max(test_batches, 1)
        test_losses.append(test_loss)

        if scheduler is not None:
            scheduler.step()

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {epoch_loss:.6f}, Test Loss: {test_loss:.6f}")

        if cleanup_interval and cleanup_interval > 0:
            torch.cuda.empty_cache()
            gc.collect()

        if test_loss < best_test_loss - min_delta:
            best_test_loss = test_loss
            best_epoch = epoch + 1
            early_stopping_counter = 0
            torch.save(model.state_dict(), "best_PRAD_lora_model_cluster_norm_attention.pt")
            print("  *** Saved best model ***")
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                print(f"Early stopping triggered. Best test loss: {best_test_loss:.6f} (epoch {best_epoch})")
                break

        if epoch_loss < best_loss:
            best_loss = epoch_loss

    print(f"\nTraining complete. Best test loss: {best_test_loss:.6f}")

    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss', color='blue')
    plt.plot(range(1, len(test_losses) + 1), test_losses, label='Test Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Optimized Bulk Static Training Loss (372 Genes, Multi-Graph Batch)')
    plt.legend()
    plt.grid(True)
    plt.savefig('bulk_BRCA_lora_loss_cluster_norm_attention.png')
    plt.close()

    return train_losses, test_losses
