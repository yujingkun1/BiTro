#!/usr/bin/env python3
"""
评估训练好的 bulk 模型在测试集上的性能
计算 Pearson 相关系数 (Overall & Gene-wise) 和 JS 散度
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
    parser.add_argument("--model-path", type=str, default="/root/autodl-tmp/Cell2Gene/best_PRAD_lora_model.pt")
    parser.add_argument("--graph-data-dir", type=str, default="/root/autodl-tmp/bulk_PRAD_graphs_new_all_graph")
    parser.add_argument("--gene-list-file", type=str, default="/root/autodl-tmp/PRAD_intersection_genes.txt")
    parser.add_argument("--features-file", type=str, default="/root/autodl-tmp/features.tsv")
    parser.add_argument("--tpm-csv-file", type=str, default="/root/autodl-tmp/tpm-TCGA-PRAD-intersect-normalized.csv")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--graph-batch-size", type=int, default=128)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--use-lora", action="store_true", default=True)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-freeze-base", action="store_true", default=True)
    parser.add_argument("--output-csv", type=str, default=None, help="样本级指标输出")
    parser.add_argument("--output-gene-csv", type=str, default=None, help="基因级指标输出")
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def compute_pearson(vec1: np.ndarray, vec2: np.ndarray, eps: float = 1e-12) -> float:
    """计算 Pearson 相关系数"""
    if vec1.std() < eps or vec2.std() < eps:
        return float("nan")
    return float(np.corrcoef(vec1, vec2)[0, 1])


def js_divergence(pred: np.ndarray, target: np.ndarray, eps: float = 1e-12) -> float:
    """计算 Jensen-Shannon 散度"""
    p = pred / (pred.sum() + eps)
    q = target / (target.sum() + eps)
    m = 0.5 * (p + q)

    def _safe_kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * (np.log(a[mask] + eps) - np.log(b[mask] + eps))))

    return 0.5 * (_safe_kl(p, m) + _safe_kl(q, m))


def average_pairwise_pearson(vectors: np.ndarray) -> float:
    """计算样本间的平均相似度 (Inter-sample similarity)"""
    if vectors.shape[0] < 2:
        return float("nan")
    with np.errstate(invalid="ignore"):
        corr_matrix = np.corrcoef(vectors)
    upper_idx = np.triu_indices_from(corr_matrix, k=1)
    valid_values = corr_matrix[upper_idx]
    valid_values = valid_values[np.isfinite(valid_values)]
    return float(np.mean(valid_values)) if valid_values.size > 0 else float("nan")


def prepare_device(device_arg: str | None) -> torch.device:
    if device_arg: return torch.device(device_arg)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def aggregate_predictions(model, graphs, all_cell_features, all_cell_positions, has_graphs, device) -> torch.Tensor:
    """执行推理并聚合为 Bulk 表达量"""
    if all_cell_features.shape[0] == 0: return torch.empty(0)
    all_cell_features = all_cell_features.to(device)
    all_cell_positions = all_cell_positions.to(device)

    if has_graphs and graphs:
        # 确保图在设备上
        for g in graphs:
            if hasattr(g, 'x'): g.x = g.x.to(device)
            if hasattr(g, 'edge_index'): g.edge_index = g.edge_index.to(device)
        cell_preds = model(graphs)
    else:
        cell_preds = [model.forward_raw_features(all_cell_features, all_cell_positions)]

    valid_preds = [p for p in cell_preds if isinstance(p, torch.Tensor) and p.shape[0] > 0]
    if not valid_preds: return torch.empty(0)
    
    return torch.cat(valid_preds, dim=0).sum(dim=0, keepdim=True)


def evaluate():
    args = parse_args()
    device = prepare_device(args.device)
    print(f"[评估] 运行环境: {device}")

    # 1. 数据准备
    selected_genes, _ = load_gene_mapping(args.gene_list_file, args.features_file)
    dataset = BulkStaticGraphDataset372(
        graph_data_dir=args.graph_data_dir, split=args.split,
        selected_genes=selected_genes, max_samples=args.max_samples, tpm_csv_file=args.tpm_csv_file
    )
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_bulk_372)

    # 2. 模型加载
    model = OptimizedTransformerPredictor(
        input_dim=dataset.feature_dim, num_genes=dataset.num_genes,
        use_gnn=True, gnn_type='GAT', use_lora=args.use_lora,
        lora_r=args.lora_r, lora_alpha=args.lora_alpha
    )
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device).eval()

    # 3. 循环推理
    all_targets, all_preds = [], []
    sample_info = []

    print(f"[评估] 正在处理 {len(dataset)} 个样本...")
    with torch.no_grad():
        for batch in data_loader:
            targets = batch["expressions"].to(device)
            for i in range(len(batch["slide_ids"])):
                pred = aggregate_predictions(
                    model, batch["spot_graphs_list"][i], batch["all_cell_features_list"][i],
                    batch["all_cell_positions_list"][i], batch["has_graphs_list"][i], device
                )
                if pred.shape[0] == 0: continue
                
                # 归一化 (CPM/TPM style)
                pred_norm = (pred / (pred.sum() + 1e-8)) * 1e6
                
                all_targets.append(targets[i].cpu().numpy())
                all_preds.append(pred_norm.squeeze().cpu().numpy())
                sample_info.append({"slide_id": batch["slide_ids"][i], "patient_id": batch["patient_ids"][i]})

    # 4. 指标计算
    target_mtx = np.array(all_targets)  # (N_samples, N_genes)
    pred_mtx = np.array(all_preds)      # (N_samples, N_genes)

    # --- Overall Pearson (Per Sample) ---
    sample_pearsons = [compute_pearson(target_mtx[i], pred_mtx[i]) for i in range(len(target_mtx))]
    sample_jsds = [js_divergence(pred_mtx[i], target_mtx[i]) for i in range(len(target_mtx))]

    # --- Gene Pearson (Per Gene) ---
    # 计算每个基因在所有样本上的相关性
    gene_pearsons = []
    for g in range(target_mtx.shape[1]):
        r = compute_pearson(target_mtx[:, g], pred_mtx[:, g])
        gene_pearsons.append(r)
    gene_pearsons = np.array(gene_pearsons)

    # 5. 打印结果
    print("\n" + "="*60)
    print(f" 评估报告 - 样本数: {len(sample_info)} | 基因数: {len(selected_genes)}")
    print("="*60)
    
    print(f"Overall Pearson (样本内):  Mean: {np.nanmean(sample_pearsons):.4f} | Median: {np.nanmedian(sample_pearsons):.4f}")
    print(f"Gene Pearson    (基因跨样本): Mean: {np.nanmean(gene_pearsons):.4f} | Median: {np.nanmedian(gene_pearsons):.4f}")
    print(f"JS Divergence   (样本内):  Mean: {np.nanmean(sample_jsds):.4f}")
    
    print("-" * 60)
    # 样本间多样性检查 (衡量模型是否只是输出了一个“平均脸”)
    print(f"真实样本间平均 Pearson: {average_pairwise_pearson(target_mtx):.4f}")
    print(f"预测样本间平均 Pearson: {average_pairwise_pearson(pred_mtx):.4f}")
    print("="*60)

    # 6. 保存数据
    if args.output_csv:
        df_sample = pd.DataFrame(sample_info)
        df_sample['pearson'] = sample_pearsons
        df_sample['jsd'] = sample_jsds
        df_sample.to_csv(args.output_csv, index=False)
    
    if args.output_gene_csv:
        df_gene = pd.DataFrame({"gene_symbol": selected_genes, "gene_pearson": gene_pearsons})
        df_gene.to_csv(args.output_gene_csv, index=False)
        print(f"[输出] 基因级评估已保存至: {args.output_gene_csv}")

if __name__ == "__main__":
    evaluate()