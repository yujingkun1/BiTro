#!/usr/bin/env python3
"""
评估训练好的 bulk 模型在测试集上的性能
计算 Pearson 相关系数和 JS 散度
"""

import os
import sys
import argparse
import math
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# 允许从项目根目录直接运行
_current_file_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_file_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from bulk_model.utils import load_gene_mapping
from bulk_model.dataset import BulkStaticGraphDataset372, collate_fn_bulk_372
from bulk_model.models import OptimizedTransformerPredictor


def parse_args():
    parser = argparse.ArgumentParser(description="评估 bulk 模型在测试集上的性能")
    parser.add_argument(
        "--model-path",
        type=str,
        default="/root/autodl-tmp/Cell2Gene/bulk_model/best_lora_model.pt",
        help="模型检查点路径",
    )
    parser.add_argument(
        "--graph-data-dir",
        type=str,
        default="/root/autodl-tmp/bulk_static_graphs_new_all_graph",
        help="图数据目录",
    )
    parser.add_argument(
        "--gene-list-file",
        type=str,
        default="/root/autodl-tmp/common_genes_misc_tenx_zen_897.txt",
        help="基因列表文件",
    )
    parser.add_argument(
        "--features-file",
        type=str,
        default="/root/autodl-tmp/features.tsv",
        help="特征文件",
    )
    parser.add_argument(
        "--tpm-csv-file",
        type=str,
        default="/root/autodl-tmp/tpm-TCGA-COAD-897-million.csv",
        help="TPM表达数据CSV文件",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "test"],
        help="评估的数据集分割",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="患者Batch Size",
    )
    parser.add_argument(
        "--graph-batch-size",
        type=int,
        default=128,
        help="图Batch Size",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="设备 (例如: cuda:0). 默认自动选择",
    )
    parser.add_argument(
        "--use-lora",
        action="store_true",
        default=True,
        help="使用LoRA",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=8,
        help="LoRA rank",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=16,
        help="LoRA alpha",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
        help="LoRA dropout",
    )
    parser.add_argument(
        "--lora-freeze-base",
        action="store_true",
        default=True,
        help="冻结LoRA基础权重",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="可选的CSV输出路径，保存每个样本的指标",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="可选的最大样本数限制（用于调试）",
    )
    return parser.parse_args()


def compute_pearson(pred: np.ndarray, target: np.ndarray, eps: float = 1e-12) -> float:
    """计算 Pearson 相关系数，防止零方差向量"""
    pred = pred.astype(np.float64)
    target = target.astype(np.float64)

    pred_std = pred.std()
    target_std = target.std()
    if pred_std < eps or target_std < eps:
        return float("nan")

    return float(np.corrcoef(pred, target)[0, 1])


def js_divergence(pred: np.ndarray, target: np.ndarray, eps: float = 1e-12) -> float:
    """计算两个非负向量之间的 Jensen-Shannon 散度"""
    p = pred / (pred.sum() + eps)
    q = target / (target.sum() + eps)
    m = 0.5 * (p + q)

    def _safe_kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * (np.log(a[mask] + eps) - np.log(b[mask] + eps))))

    return 0.5 * (_safe_kl(p, m) + _safe_kl(q, m))


def prepare_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def move_graphs_to_device(graphs: List, device: torch.device) -> None:
    """确保每个 PyG 图在目标设备上"""
    for graph in graphs or []:
        if hasattr(graph, "x") and graph.x is not None:
            graph.x = graph.x.to(device, non_blocking=True)
        if hasattr(graph, "edge_index") and graph.edge_index is not None:
            graph.edge_index = graph.edge_index.to(device, non_blocking=True)
        if hasattr(graph, "pos") and graph.pos is not None:
            graph.pos = graph.pos.to(device, non_blocking=True)


def aggregate_predictions(
    model: OptimizedTransformerPredictor,
    graphs: List,
    all_cell_features: torch.Tensor,
    all_cell_positions: torch.Tensor,
    has_graphs: bool,
    device: torch.device,
) -> torch.Tensor:
    """对单个样本运行模型并返回聚合预测"""
    if all_cell_features.shape[0] == 0:
        return torch.empty(0)

    all_cell_features = all_cell_features.to(device, non_blocking=True)
    all_cell_positions = all_cell_positions.to(device, non_blocking=True)

    if has_graphs and graphs:
        move_graphs_to_device(graphs, device)
        total_cells = sum(
            graph.x.shape[0]
            for graph in graphs
            if hasattr(graph, "x") and graph.x is not None
        )
        max_cells_threshold = 150000  # 与训练时一致

        if total_cells <= max_cells_threshold:
            cell_predictions_list = model(graphs)
        else:
            # 自适应分块处理非常大的样本
            target_cells_per_batch = 10000
            batch_size_adaptive = max(
                32, len(graphs) * target_cells_per_batch // max(total_cells, 1)
            )
            cell_predictions_list: List[torch.Tensor] = []
            for start in range(0, len(graphs), batch_size_adaptive):
                chunk = graphs[start : start + batch_size_adaptive]
                current_predictions = model(chunk)
                cell_predictions_list.extend(current_predictions)
                torch.cuda.empty_cache()
                del current_predictions
    else:
        cell_predictions = model.forward_raw_features(
            all_cell_features, all_cell_positions
        )
        cell_predictions_list = [cell_predictions]

    valid_preds = [
        pred for pred in cell_predictions_list if isinstance(pred, torch.Tensor) and pred.shape[0] > 0
    ]
    if not valid_preds:
        return torch.empty(0)

    concatenated = torch.cat(valid_preds, dim=0)
    return concatenated.sum(dim=0, keepdim=True)


def evaluate():
    args = parse_args()
    device = prepare_device(args.device)
    print(f"[评估] 使用设备: {device}")

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"模型检查点未找到: {args.model_path}")

    selected_genes, _ = load_gene_mapping(args.gene_list_file, args.features_file)
    if not selected_genes:
        raise RuntimeError("加载基因映射失败，请检查输入文件")

    dataset = BulkStaticGraphDataset372(
        graph_data_dir=args.graph_data_dir,
        split=args.split,
        selected_genes=selected_genes,
        max_samples=args.max_samples,
        tpm_csv_file=args.tpm_csv_file
    )

    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn_bulk_372,
        num_workers=0,
        pin_memory=False,
    )

    model = OptimizedTransformerPredictor(
        input_dim=dataset.feature_dim,
        gnn_hidden_dim=128,
        gnn_output_dim=128,
        embed_dim=256,
        num_genes=dataset.num_genes,
        num_layers=3,
        nhead=8,
        dropout=0.1,
        use_gnn=True,
        gnn_type='GAT',
        graph_batch_size=args.graph_batch_size,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_freeze_base=args.lora_freeze_base
    )
    
    print(f"[评估] 加载模型: {args.model_path}")
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    per_sample_metrics: List[Dict] = []

    print(f"[评估] 开始评估 {args.split} 集...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            expressions = batch["expressions"].to(device, non_blocking=True)
            spot_graphs_list = batch["spot_graphs_list"]
            all_cell_features_list = batch["all_cell_features_list"]
            all_cell_positions_list = batch["all_cell_positions_list"]
            has_graphs_list = batch["has_graphs_list"]
            slide_ids = batch["slide_ids"]
            patient_ids = batch["patient_ids"]

            batch_predictions: List[torch.Tensor] = []
            batch_targets: List[torch.Tensor] = []
            batch_slide_ids: List[str] = []
            batch_patient_ids: List[str] = []

            for idx in range(len(spot_graphs_list)):
                all_cell_features = all_cell_features_list[idx]
                all_cell_positions = all_cell_positions_list[idx]
                has_graphs = has_graphs_list[idx]

                if all_cell_features.shape[0] == 0:
                    continue

                graphs = spot_graphs_list[idx]
                aggregated_prediction = aggregate_predictions(
                    model,
                    graphs,
                    all_cell_features,
                    all_cell_positions,
                    has_graphs,
                    device,
                )

                if aggregated_prediction.shape[0] == 0:
                    continue

                batch_predictions.append(aggregated_prediction)
                batch_targets.append(expressions[idx].unsqueeze(0))
                batch_slide_ids.append(slide_ids[idx])
                batch_patient_ids.append(patient_ids[idx])

            if not batch_predictions:
                continue

            predictions = torch.cat(batch_predictions, dim=0)
            targets = torch.cat(batch_targets, dim=0)

            # 与训练时相同的归一化方式
            epsilon = 1e-8
            sum_pred = predictions.sum(dim=1, keepdim=True) + epsilon
            normalized_pred = predictions / sum_pred
            scaled_pred = torch.clamp(normalized_pred * 1000000.0, min=0.0, max=1e6)

            for sample_idx in range(scaled_pred.shape[0]):
                pred_np = scaled_pred[sample_idx].detach().cpu().numpy()
                target_np = targets[sample_idx].detach().cpu().numpy()

                pearson_val = compute_pearson(pred_np, target_np)
                js_val = js_divergence(pred_np, target_np)

                per_sample_metrics.append(
                    {
                        "slide_id": batch_slide_ids[sample_idx],
                        "patient_id": batch_patient_ids[sample_idx],
                        "pearson": pearson_val,
                        "js_divergence": js_val,
                    }
                )

            if (batch_idx + 1) % 10 == 0:
                print(f"  已处理 {batch_idx + 1} 个批次...")

    if not per_sample_metrics:
        print("[评估] 没有有效的评估样本，请检查数据集")
        return

    pearson_values = np.array(
        [m["pearson"] for m in per_sample_metrics if not math.isnan(m["pearson"])]
    )
    js_values = np.array(
        [m["js_divergence"] for m in per_sample_metrics if math.isfinite(m["js_divergence"])]
    )

    print("\n" + "="*60)
    print("=== 评估结果汇总 ===")
    print("="*60)
    print(f"评估样本数: {len(per_sample_metrics)}")
    if pearson_values.size > 0:
        print(
            f"Pearson 相关系数 - 均值: {pearson_values.mean():.4f}, "
            f"中位数: {np.median(pearson_values):.4f}, "
            f"标准差: {pearson_values.std():.4f}"
        )
        print(f"  Min: {pearson_values.min():.4f}, Max: {pearson_values.max():.4f}")
    else:
        print("Pearson 相关系数: 有效样本不足")

    if js_values.size > 0:
        print(
            f"JS 散度         - 均值: {js_values.mean():.4f}, "
            f"中位数: {np.median(js_values):.4f}, "
            f"标准差: {js_values.std():.4f}"
        )
        print(f"  Min: {js_values.min():.4f}, Max: {js_values.max():.4f}")
    else:
        print("JS 散度: 有效样本不足")
    print("="*60)

    if args.output_csv:
        df = pd.DataFrame(per_sample_metrics)
        df.to_csv(args.output_csv, index=False)
        print(f"\n[评估] 每个样本的指标已保存到: {args.output_csv}")


if __name__ == "__main__":
    evaluate()

