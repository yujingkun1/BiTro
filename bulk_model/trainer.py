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
        print("⚠ 未提供 spatial 模型路径，跳过迁移学习初始化。")
        return model

    if not os.path.exists(spatial_checkpoint_path):
        print(f"⚠ Spatial checkpoint 未找到: {spatial_checkpoint_path}")
        return model

    print("\n=== 从 Spatial 预训练模型初始化 Bulk 模型权重 ===")
    print(f"Spatial checkpoint: {spatial_checkpoint_path}")

    try:
        spatial_state = torch.load(spatial_checkpoint_path, map_location=device)
        # 兼容直接保存 state_dict 或包含 'state_dict' 的情况
        if isinstance(spatial_state, dict) and "state_dict" in spatial_state:
            spatial_state = spatial_state["state_dict"]
        if not isinstance(spatial_state, dict):
            raise ValueError("Spatial checkpoint 不包含有效的 state_dict 字典")

        model_state = model.state_dict()

        loaded_keys: list[str] = []
        skipped_keys: list[str] = []
        mismatched_keys: list[tuple[str, torch.Size, torch.Size]] = []

        for key, value in spatial_state.items():
            # 跳过明显的空间特异参数
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
                # 其他在 bulk 模型中不存在的 key 也跳过
                skipped_keys.append(key)

        model.load_state_dict(model_state, strict=False)

        print(f"✓ 成功从 spatial 模型加载 {len(loaded_keys)} 层参数到 bulk 模型")
        if loaded_keys:
            print(f"  示例: {', '.join(loaded_keys[:5])}")

        if mismatched_keys:
            print(f"⚠ 因形状不匹配跳过 {len(mismatched_keys)} 层（例如输出 head 基因数不同）:")
            for k, ms, ss in mismatched_keys[:5]:
                print(f"  {k}: bulk {ms} vs spatial {ss}")
            if len(mismatched_keys) > 5:
                print(f"  ... 以及另外 {len(mismatched_keys) - 5} 层")

        if skipped_keys:
            print(f"ℹ 跳过 {len(skipped_keys)} 个 spatial 特有或 bulk 中不存在的参数（如 gene_queries/x_embed 等）")

        if freeze_backbone:
            print("\n=== 冻结 Bulk 模型 Backbone（GNN + feature_projection + transformer）===")
            frozen_params = 0
            trainable_params = 0

            # 冻结 GNN
            if hasattr(model, "gnn"):
                for _, p in model.gnn.named_parameters():
                    p.requires_grad = False
                    frozen_params += p.numel()

            # 冻结特征投影
            if hasattr(model, "feature_projection"):
                for _, p in model.feature_projection.named_parameters():
                    p.requires_grad = False
                    frozen_params += p.numel()

            # 冻结 transformer
            if hasattr(model, "transformer"):
                for _, p in model.transformer.named_parameters():
                    p.requires_grad = False
                    frozen_params += p.numel()

            # 统计仍然可训练的参数（例如 output_projection 的最后几层）
            for name, p in model.named_parameters():
                if p.requires_grad:
                    trainable_params += p.numel()

            print(f"✓ Backbone 冻结参数量: {frozen_params:,}")
            print(f"✓ 仍可训练参数量: {trainable_params:,}")

    except Exception as e:  # pylint: disable=broad-except
        print(f"❌ 从 spatial checkpoint 加载权重失败: {e}")
        print("将使用随机初始化的 bulk 模型继续训练。")

    return model


def train_optimized_model(model, train_loader, test_loader, optimizer, scheduler=None,
                         num_epochs=50, device="cuda", patience=10, min_delta=1e-6,
                         log_every=10, debug=False, enable_profiling=False, cleanup_interval=1):
    model.to(device)
    criterion = nn.MSELoss()
    scaler = GradScaler('cuda')

    best_loss = float('inf')
    best_test_loss = float('inf')
    early_stopping_counter = 0
    best_epoch = 0

    train_losses = []
    test_losses = []

    print("=== 开始优化训练（批量处理多图）===")
    print(f"图批量大小: {model.graph_batch_size}")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        num_batches = 0
        batch_skip_count = 0
        patient_skip_count = 0

        print(f"\n=== Epoch {epoch+1} 开始训练 ===")

        for batch_idx, batch in enumerate(train_loader):
            batch_start_time = time.time() if enable_profiling else None
            data_loading_time = 0.0
            
            expressions = batch['expressions'].to(device, non_blocking=True)
            spot_graphs_list = batch['spot_graphs_list']

            log_this_batch = (batch_idx % log_every == 0) or debug
            if log_this_batch:
                print(f"\nBatch {batch_idx}: 开始处理 {len(spot_graphs_list)} 个患者")

            optimizer.zero_grad()

            batch_predictions = []

            for i in range(len(spot_graphs_list)):
                spot_graphs = batch['spot_graphs_list'][i]
                all_cell_features = batch['all_cell_features_list'][i]
                all_cell_positions = batch['all_cell_positions_list'][i]
                has_graphs = batch['has_graphs_list'][i]

                if log_this_batch:
                    print(f"  患者 {i+1}: 细胞特征形状={all_cell_features.shape}, 位置形状={all_cell_positions.shape}, 有图={has_graphs}, 图数量={len(spot_graphs) if spot_graphs else 0}")

                if all_cell_features.shape[0] == 0:
                    if log_this_batch:
                        print(f"    ⚠️ 跳过患者 {i+1}：没有细胞特征数据")
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
                                print(f"    有图处理：{len(spot_graphs)}个图 → {total_cells}个细胞 (图增强)")
                            cell_predictions_list = model(spot_graphs)
                        else:
                            if log_this_batch:
                                print(f"    超大有图患者：{len(spot_graphs)}个图 → {total_cells}个细胞 (梯度累积分批)")
                            target_cells_per_batch = 10000
                            batch_size_adaptive = max(32, len(spot_graphs) * target_cells_per_batch // total_cells)
                            all_cell_predictions_list = []
                            for batch_start in range(0, len(spot_graphs), batch_size_adaptive):
                                batch_end = min(batch_start + batch_size_adaptive, len(spot_graphs))
                                batch_graphs = spot_graphs[batch_start:batch_end]
                                batch_cells = sum([g.x.shape[0] for g in batch_graphs if hasattr(g, 'x')])
                                if log_this_batch:
                                    print(f"      分批{batch_start//batch_size_adaptive + 1}: {len(batch_graphs)}个图 → {batch_cells}个细胞")
                                current_batch_predictions = model(batch_graphs)
                                all_cell_predictions_list.extend(current_batch_predictions)
                                torch.cuda.empty_cache()
                                del current_batch_predictions
                            cell_predictions_list = all_cell_predictions_list
                    else:
                        if log_this_batch:
                            print(f"    无图处理：{all_cell_features.shape[0]}个细胞 (原始DINO特征)")
                        cell_predictions = model.forward_raw_features(all_cell_features, all_cell_positions)
                        cell_predictions_list = [cell_predictions]

                    if cell_predictions_list:
                        all_cell_predictions = torch.cat([pred for pred in cell_predictions_list if pred.shape[0] > 0], dim=0)
                        if all_cell_predictions.shape[0] > 0:
                            aggregated_prediction = all_cell_predictions.sum(dim=0, keepdim=True)
                            if log_this_batch:
                                print(f"    患者 {i+1} 预测聚合：细胞数={all_cell_predictions.shape[0]}, 聚合结果形状={aggregated_prediction.shape}")
                        else:
                            aggregated_prediction = torch.zeros(1, expressions.shape[1], device=device)
                            if log_this_batch:
                                print(f"    患者 {i+1} 预测聚合：没有有效细胞，使用零预测")
                    else:
                        aggregated_prediction = torch.zeros(1, expressions.shape[1], device=device)
                        if log_this_batch:
                            print(f"    患者 {i+1} 预测聚合：没有预测结果，使用零预测")

                batch_predictions.append(aggregated_prediction)

            if not batch_predictions:
                if log_this_batch:
                    print(f"    ⚠️ Batch {batch_idx}: 所有患者都被跳过，没有有效预测")
                batch_skip_count += 1
                continue

            if len(batch_predictions) != len(spot_graphs_list):
                if log_this_batch:
                    print(f"    ⚠️ Batch {batch_idx}: {len(spot_graphs_list)}个患者中只有{len(batch_predictions)}个有效")

            predictions = torch.cat(batch_predictions, dim=0)
            if log_this_batch:
                print(f"  Batch {batch_idx} 合并预测：形状={predictions.shape}")

            if predictions.shape[0] != expressions.shape[0]:
                if log_this_batch:
                    print(f"    ⚠️ 预测和真实值数量不匹配: {predictions.shape[0]} vs {expressions.shape[0]}")
                expressions = expressions[:predictions.shape[0]]

            with autocast('cuda'):
                pred_sum = predictions.sum().item()
                if pred_sum <= 1e-10 or not torch.isfinite(predictions).all():
                    if log_this_batch:
                        print(f"    ❌ 警告：预测异常，跳过这个batch")
                    batch_skip_count += 1
                    continue

                epsilon = 1e-8
                sum_pred = predictions.sum(dim=1, keepdim=True) + epsilon
                normalized_pred = predictions / sum_pred
                result = torch.clamp(normalized_pred * 1000000.0, min=0.0, max=1e6)

                if torch.isnan(result).any() or torch.isinf(result).any():
                    if log_this_batch:
                        print(f"    ❌ 警告：归一化结果包含NaN或Inf！跳过")
                    batch_skip_count += 1
                    continue

                loss = criterion(result, expressions)
                if log_this_batch:
                    print(f"  计算损失：{loss.item():.6f}")
                if torch.isnan(loss) or torch.isinf(loss):
                    if log_this_batch:
                        print(f"    ❌ 警告：损失为NaN或Inf，跳过这个batch")
                    batch_skip_count += 1
                    continue

                backward_start_time = time.time() if enable_profiling else None
                if log_this_batch:
                    print(f"  开始反向传播...")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                scaler.step(optimizer)
                scaler.update()
                if log_this_batch:
                    print(f"  反向传播完成")
                if enable_profiling:
                    backward_time = time.time() - backward_start_time if backward_start_time else 0.0
                    forward_time = (backward_start_time - forward_start_time) if forward_start_time else 0.0
                    total_batch_time = time.time() - batch_start_time if batch_start_time else 0.0
                    if log_this_batch:
                        print(f"  性能统计: 总时间={total_batch_time:.3f}s, 数据加载={data_loading_time:.3f}s, 前向={forward_time:.3f}s, 反向={backward_time:.3f}s")

                running_loss += loss.item()
                num_batches += 1

                # 🔧 关键修复：监控完成后再清理大tensor
                del predictions, result, loss
                del batch_predictions
                del expressions, spot_graphs_list
                if cleanup_interval and cleanup_interval > 0:
                    if (batch_idx + 1) % cleanup_interval == 0 and torch.cuda.is_available():
                        torch.cuda.empty_cache()

        if num_batches == 0:
            print(f"Epoch {epoch+1}: 所有batch都被跳过")
            print(f"  跳过的batch数: {batch_skip_count}")
            print(f"  跳过的患者数: {patient_skip_count}")
            continue

        epoch_loss = running_loss / num_batches
        train_losses.append(epoch_loss)

        print(f"\nEpoch {epoch+1} 训练统计:")
        print(f"  总batch数: {batch_idx + 1}")
        print(f"  成功训练的batch数: {num_batches}")
        print(f"  跳过的batch数: {batch_skip_count}")
        print(f"  跳过的患者数: {patient_skip_count}")
        print(f"  平均损失: {epoch_loss:.6f}")

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
            torch.save(model.state_dict(), "best_BRCA_lora_model_transfer.pt")
            print(f"  *** 保存最佳模型 ***")
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                print(f"早停触发！最佳测试损失: {best_test_loss:.6f} (Epoch {best_epoch})")
                break

        if epoch_loss < best_loss:
            best_loss = epoch_loss

    print(f"\n训练完成! 最佳测试损失: {best_test_loss:.6f}")

    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss', color='blue')
    plt.plot(range(1, len(test_losses) + 1), test_losses, label='Test Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Optimized Bulk Static Training Loss (372 Genes, Multi-Graph Batch)')
    plt.legend()
    plt.grid(True)
    plt.savefig('bulk_BRCA_lora_loss_Transfer.png')
    plt.close()

    return train_losses, test_losses
