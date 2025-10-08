import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
import matplotlib.pyplot as plt


def train_optimized_model(model, train_loader, test_loader, optimizer, scheduler=None,
                         num_epochs=50, device="cuda", patience=10, min_delta=1e-6):
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
            expressions = batch['expressions'].to(device, non_blocking=True)
            spot_graphs_list = batch['spot_graphs_list']

            print(f"\nBatch {batch_idx}: 开始处理 {len(spot_graphs_list)} 个患者")

            optimizer.zero_grad()

            batch_predictions = []

            for i in range(len(spot_graphs_list)):
                spot_graphs = batch['spot_graphs_list'][i]
                all_cell_features = batch['all_cell_features_list'][i]
                all_cell_positions = batch['all_cell_positions_list'][i]
                has_graphs = batch['has_graphs_list'][i]

                print(f"  患者 {i+1}: 细胞特征形状={all_cell_features.shape}, 位置形状={all_cell_positions.shape}, 有图={has_graphs}, 图数量={len(spot_graphs) if spot_graphs else 0}")

                if all_cell_features.shape[0] == 0:
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

                with autocast('cuda'):
                    if has_graphs and len(spot_graphs) > 0:
                        total_cells = sum([graph.x.shape[0] for graph in spot_graphs if hasattr(graph, 'x') and graph.x is not None])
                        max_cells_threshold = 200000

                        if total_cells <= max_cells_threshold:
                            print(f"    有图处理：{len(spot_graphs)}个图 → {total_cells}个细胞 (图增强)")
                            cell_predictions_list = model(spot_graphs)
                        else:
                            print(f"    超大有图患者：{len(spot_graphs)}个图 → {total_cells}个细胞 (梯度累积分批)")
                            target_cells_per_batch = 10000
                            batch_size_adaptive = max(32, len(spot_graphs) * target_cells_per_batch // total_cells)
                            all_cell_predictions_list = []
                            for batch_start in range(0, len(spot_graphs), batch_size_adaptive):
                                batch_end = min(batch_start + batch_size_adaptive, len(spot_graphs))
                                batch_graphs = spot_graphs[batch_start:batch_end]
                                batch_cells = sum([g.x.shape[0] for g in batch_graphs if hasattr(g, 'x')])
                                print(f"      分批{batch_start//batch_size_adaptive + 1}: {len(batch_graphs)}个图 → {batch_cells}个细胞")
                                current_batch_predictions = model(batch_graphs)
                                all_cell_predictions_list.extend(current_batch_predictions)
                                torch.cuda.empty_cache()
                                del current_batch_predictions
                            cell_predictions_list = all_cell_predictions_list
                    else:
                        print(f"    无图处理：{all_cell_features.shape[0]}个细胞 (原始DINO特征)")
                        cell_predictions = model.forward_raw_features(all_cell_features, all_cell_positions)
                        cell_predictions_list = [cell_predictions]

                    if cell_predictions_list:
                        all_cell_predictions = torch.cat([pred for pred in cell_predictions_list if pred.shape[0] > 0], dim=0)
                        if all_cell_predictions.shape[0] > 0:
                            aggregated_prediction = all_cell_predictions.sum(dim=0, keepdim=True)
                            print(f"    患者 {i+1} 预测聚合：细胞数={all_cell_predictions.shape[0]}, 聚合结果形状={aggregated_prediction.shape}")
                        else:
                            aggregated_prediction = torch.zeros(1, expressions.shape[1], device=device)
                            print(f"    患者 {i+1} 预测聚合：没有有效细胞，使用零预测")
                    else:
                        aggregated_prediction = torch.zeros(1, expressions.shape[1], device=device)
                        print(f"    患者 {i+1} 预测聚合：没有预测结果，使用零预测")

                batch_predictions.append(aggregated_prediction)

            if not batch_predictions:
                print(f"    ⚠️ Batch {batch_idx}: 所有患者都被跳过，没有有效预测")
                batch_skip_count += 1
                continue

            if len(batch_predictions) != len(spot_graphs_list):
                print(f"    ⚠️ Batch {batch_idx}: {len(spot_graphs_list)}个患者中只有{len(batch_predictions)}个有效")

            predictions = torch.cat(batch_predictions, dim=0)
            print(f"  Batch {batch_idx} 合并预测：形状={predictions.shape}")

            if predictions.shape[0] != expressions.shape[0]:
                print(f"    ⚠️ 预测和真实值数量不匹配: {predictions.shape[0]} vs {expressions.shape[0]}")
                expressions = expressions[:predictions.shape[0]]

            with autocast('cuda'):
                pred_sum = predictions.sum().item()
                if pred_sum <= 1e-10 or not torch.isfinite(predictions).all():
                    print(f"    ❌ 警告：预测异常，跳过这个batch")
                    batch_skip_count += 1
                    continue

                epsilon = 1e-8
                sum_pred = predictions.sum(dim=1, keepdim=True) + epsilon
                normalized_pred = predictions / sum_pred
                result = torch.clamp(normalized_pred * 1000000.0, min=0.0, max=1e6)

                if torch.isnan(result).any() or torch.isinf(result).any():
                    print(f"    ❌ 警告：归一化结果包含NaN或Inf！跳过")
                    batch_skip_count += 1
                    continue

                loss = criterion(result, expressions)
                print(f"  计算损失：{loss.item():.6f}")
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"    ❌ 警告：损失为NaN或Inf，跳过这个batch")
                    batch_skip_count += 1
                    continue

                print(f"  开始反向传播...")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                scaler.step(optimizer)
                scaler.update()
                print(f"  反向传播完成")

                running_loss += loss.item()
                num_batches += 1

                if batch_idx % 5 == 0:
                    try:
                        gpu_util = torch.cuda.utilization(0) if torch.cuda.is_available() else 0
                    except (ModuleNotFoundError, RuntimeError):
                        gpu_util = "N/A"
                    gpu_mem_gb = torch.cuda.memory_allocated(device) / 1024**3 if torch.cuda.is_available() else 0
                    print(f"  Batch {batch_idx}: Loss={loss.item():.6f}, GPU利用率={gpu_util}%, GPU内存={gpu_mem_gb:.1f}GB")

                del predictions, result, loss
                del batch_predictions
                del expressions, spot_graphs_list
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
                        max_cells_threshold = 200000
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

        torch.cuda.empty_cache()
        import gc
        gc.collect()

        if test_loss < best_test_loss - min_delta:
            best_test_loss = test_loss
            best_epoch = epoch + 1
            early_stopping_counter = 0
            torch.save(model.state_dict(), "best_bulk_static_372_optimized_model.pt")
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
    plt.savefig('bulk_static_372_optimized_loss.png')
    plt.close()

    return train_losses, test_losses

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
import matplotlib.pyplot as plt


def train_optimized_model(model, train_loader, test_loader, optimizer, scheduler=None, 
                         num_epochs=50, device="cuda", patience=10, min_delta=1e-6):
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
            expressions = batch['expressions'].to(device, non_blocking=True)
            spot_graphs_list = batch['spot_graphs_list']

            print(f"\nBatch {batch_idx}: 开始处理 {len(spot_graphs_list)} 个患者")

            optimizer.zero_grad()

            batch_predictions = []

            for i in range(len(spot_graphs_list)):
                spot_graphs = spot_graphs_list[i]
                all_cell_features = batch['all_cell_features_list'][i]
                all_cell_positions = batch['all_cell_positions_list'][i]
                has_graphs = batch['has_graphs_list'][i]

                print(f"  患者 {i+1}: 细胞特征形状={all_cell_features.shape}, 位置形状={all_cell_positions.shape}, 有图={has_graphs}, 图数量={len(spot_graphs) if spot_graphs else 0}")

                if all_cell_features.shape[0] == 0:
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

                with autocast('cuda'):
                    if has_graphs and len(spot_graphs) > 0:
                        total_cells = sum([graph.x.shape[0] for graph in spot_graphs if hasattr(graph, 'x') and graph.x is not None])
                        max_cells_threshold = 200000

                        if total_cells <= max_cells_threshold:
                            print(f"    有图处理：{len(spot_graphs)}个图 → {total_cells}个细胞 (图增强)")
                            cell_predictions_list = model(spot_graphs)
                        else:
                            print(f"    超大有图患者：{len(spot_graphs)}个图 → {total_cells}个细胞 (梯度累积分批)")
                            target_cells_per_batch = 10000
                            batch_size_adaptive = max(32, len(spot_graphs) * target_cells_per_batch // total_cells)
                            all_cell_predictions_list = []
                            for batch_start in range(0, len(spot_graphs), batch_size_adaptive):
                                batch_end = min(batch_start + batch_size_adaptive, len(spot_graphs))
                                batch_graphs = spot_graphs[batch_start:batch_end]
                                batch_cells = sum([g.x.shape[0] for g in batch_graphs if hasattr(g, 'x')])
                                print(f"      分批{batch_start//batch_size_adaptive + 1}: {len(batch_graphs)}个图 → {batch_cells}个细胞")
                                current_batch_predictions = model(batch_graphs)
                                all_cell_predictions_list.extend(current_batch_predictions)
                                torch.cuda.empty_cache()
                                del current_batch_predictions
                            cell_predictions_list = all_cell_predictions_list
                    else:
                        print(f"    无图处理：{all_cell_features.shape[0]}个细胞 (原始DINO特征)")
                        cell_predictions = model.forward_raw_features(all_cell_features, all_cell_positions)
                        cell_predictions_list = [cell_predictions]

                    if cell_predictions_list:
                        all_cell_predictions = torch.cat([pred for pred in cell_predictions_list if pred.shape[0] > 0], dim=0)
                        if all_cell_predictions.shape[0] > 0:
                            aggregated_prediction = all_cell_predictions.sum(dim=0, keepdim=True)
                            print(f"    患者 {i+1} 预测聚合：细胞数={all_cell_predictions.shape[0]}, 聚合结果形状={aggregated_prediction.shape}")
                            agg_min = aggregated_prediction.min().item()
                            agg_max = aggregated_prediction.max().item()
                            agg_sum = aggregated_prediction.sum().item()
                            print(f"    聚合预测范围: [{agg_min:.6f}, {agg_max:.6f}], 总和: {agg_sum:.6f}")
                        else:
                            aggregated_prediction = torch.zeros(1, expressions.shape[1], device=device)
                            print(f"    患者 {i+1} 预测聚合：没有有效细胞，使用零预测")
                    else:
                        aggregated_prediction = torch.zeros(1, expressions.shape[1], device=device)
                        print(f"    患者 {i+1} 预测聚合：没有预测结果，使用零预测")

                batch_predictions.append(aggregated_prediction)

            if not batch_predictions:
                print(f"    ⚠️ Batch {batch_idx}: 所有患者都被跳过，没有有效预测")
                batch_skip_count += 1
                continue

            if len(batch_predictions) != len(spot_graphs_list):
                print(f"    ⚠️ Batch {batch_idx}: {len(spot_graphs_list)}个患者中只有{len(batch_predictions)}个有效")

            predictions = torch.cat(batch_predictions, dim=0)
            print(f"  Batch {batch_idx} 合并预测：形状={predictions.shape}")

            if predictions.shape[0] != expressions.shape[0]:
                print(f"    ⚠️ 预测和真实值数量不匹配: {predictions.shape[0]} vs {expressions.shape[0]}")
                expressions = expressions[:predictions.shape[0]]

            with autocast('cuda'):
                pred_min = predictions.min().item()
                pred_max = predictions.max().item()
                pred_sum = predictions.sum().item()
                pred_mean = predictions.mean().item()
                pred_std = predictions.std().item()

                expr_min = expressions.min().item()
                expr_max = expressions.max().item()
                expr_sum = expressions.sum().item()
                expr_mean = expressions.mean().item()
                expr_std = expressions.std().item()

                print(f"  原始预测值统计：min={pred_min:.6f}, max={pred_max:.6f}, sum={pred_sum:.6f}, mean={pred_mean:.6f}, std={pred_std:.6f}")
                print(f"  真实值统计：min={expr_min:.6f}, max={expr_max:.6f}, sum={expr_sum:.6f}, mean={expr_mean:.6f}, std={expr_std:.6f}")

                if pred_sum <= 1e-10:
                    print(f"    ❌ 警告：预测值接近全为0！总和={pred_sum:.10f}")
                    batch_skip_count += 1
                    continue

                if not torch.isfinite(predictions).all():
                    print(f"    ❌ 警告：预测值包含NaN或Inf！")
                    print(f"    NaN数量: {torch.isnan(predictions).sum().item()}")
                    print(f"    Inf数量: {torch.isinf(predictions).sum().item()}")
                    batch_skip_count += 1
                    continue

                epsilon = 1e-8
                sum_pred = predictions.sum(dim=1, keepdim=True) + epsilon
                print(f"  预测值行求和：min={sum_pred.min().item():.10f}, max={sum_pred.max().item():.10f}")
                normalized_pred = predictions / sum_pred
                print(f"  归一化后：min={normalized_pred.min().item():.10f}, max={normalized_pred.max().item():.10f}, sum={normalized_pred.sum().item():.10f}")
                result = normalized_pred * 1000000.0
                result = torch.clamp(result, min=0.0, max=1e6)
                if torch.isnan(result).any() or torch.isinf(result).any():
                    print(f"    ❌ 警告：归一化结果包含NaN或Inf！")
                    print(f"    原始预测值总和: {predictions.sum(dim=1)}")
                    print(f"    NaN数量: {torch.isnan(result).sum().item()}")
                    print(f"    Inf数量: {torch.isinf(result).sum().item()}")
                    batch_skip_count += 1
                    continue
                loss = criterion(result, expressions)
                print(f"  计算损失：{loss.item():.6f}")
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"    ❌ 警告：损失为NaN或Inf，跳过这个batch")
                    batch_skip_count += 1
                    continue
                print(f"  开始反向传播...")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                scaler.step(optimizer)
                scaler.update()
                print(f"  反向传播完成")

                running_loss += loss.item()
                num_batches += 1

                if batch_idx % 5 == 0:
                    try:
                        gpu_util = torch.cuda.utilization(0) if torch.cuda.is_available() else 0
                    except (ModuleNotFoundError, RuntimeError):
                        gpu_util = "N/A"
                    gpu_mem_gb = torch.cuda.memory_allocated(device) / 1024**3 if torch.cuda.is_available() else 0
                    print(f"  Batch {batch_idx}: Loss={loss.item():.6f}, GPU利用率={gpu_util}%, GPU内存={gpu_mem_gb:.1f}GB")

                del predictions, result, loss
                del batch_predictions
                del expressions, spot_graphs_list
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
                        max_cells_threshold = 200000
                        if total_cells <= max_cells_threshold:
                            cell_predictions_list = model(spot_graphs)
                        else:
                            print(f"    测试超大有图患者：{len(spot_graphs)}个图 → {total_cells}个细胞 (分批)")
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
        torch.cuda.empty_cache()
        import gc
        gc.collect()

        if test_loss < best_test_loss - min_delta:
            best_test_loss = test_loss
            best_epoch = epoch + 1
            early_stopping_counter = 0
            torch.save(model.state_dict(), "best_bulk_static_372_optimized_model.pt")
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
    plt.savefig('bulk_static_372_optimized_loss.png')
    plt.close()
    return train_losses, test_losses


