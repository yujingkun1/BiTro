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
    # ========== 可在代码中直接修改的默认开关（在此处切换 True/False） ==========
    DEFAULT_USE_GENE_ATTENTION = False
    DEFAULT_APPLY_GENE_NORMALIZATION = False
    DEFAULT_ENABLE_CLUSTER_LOSS = False
    DEFAULT_CLUSTER_LOSS_WEIGHT = 0.0
    # =====================================================================
    parser.add_argument(
        "--model-path",
        type=str,
        default="/root/autodl-tmp/Cell2Gene/best_PRAD_lora_model.pt",
        help="模型检查点路径",
    )
    parser.add_argument(
        "--graph-data-dir",
        type=str,
        default="/root/autodl-tmp/bulk_PRAD_graphs_new_all_graph",
        help="图数据目录",
    )
    parser.add_argument(
        "--gene-list-file",
        type=str,
        default="/root/autodl-tmp/PRAD_intersection_genes.txt",
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
        default="/root/autodl-tmp/tpm-TCGA-PRAD-intersect-normalized.csv",
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
    parser.add_argument("--debug-logs", action="store_true", default=False,
                        help="启用调试日志，打印样本级预测统计信息")
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
    # Gene normalization control (default: enabled)
    group_norm = parser.add_mutually_exclusive_group()
    group_norm.add_argument("--apply-gene-normalization", dest="apply_gene_normalization", action="store_true",
                            help="启用基因 z-score 归一化 (默认)")
    group_norm.add_argument("--no-gene-normalization", dest="apply_gene_normalization", action="store_false",
                            help="禁用基因 z-score 归一化")
    # gene attention flags
    parser.add_argument("--use-gene-attention", dest="use_gene_attention", action="store_true",
                        help="启用 gene attention readout（默认为代码中设置）")
    parser.add_argument("--no-gene-attention", dest="use_gene_attention", action="store_false",
                        help="禁用 gene attention readout")
    # cluster loss flags (主要用于训练，但在这里保留为配置一致)
    parser.add_argument("--enable-cluster-loss", dest="enable_cluster_loss", action="store_true",
                        help="评估时是否启用 cluster loss（通常不启用，仅用于一致性）")
    parser.add_argument("--cluster-loss-weight", type=float, default=0.0,
                        help="聚类正则项权重（仅在 enable-cluster-loss=True 时生效）")
    # set defaults from code-level constants
    parser.set_defaults(use_gene_attention=DEFAULT_USE_GENE_ATTENTION)
    parser.set_defaults(apply_gene_normalization=DEFAULT_APPLY_GENE_NORMALIZATION)
    parser.set_defaults(enable_cluster_loss=DEFAULT_ENABLE_CLUSTER_LOSS)
    parser.set_defaults(cluster_loss_weight=DEFAULT_CLUSTER_LOSS_WEIGHT)
    parser.add_argument("--normalization-stats", type=str, default=None,
                        help="可选：归一化统计文件（JSON或npz）路径，包含 mean/std，用于非训练split时提供stats")
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


def average_pairwise_pearson(vectors: List[np.ndarray] | np.ndarray) -> float:
    """计算样本之间的平均 Pearson（仅对行之间做 pairwise）"""
    if isinstance(vectors, list):
        if not vectors:
            return float("nan")
        vectors = np.stack(vectors, axis=0)
    if vectors.shape[0] < 2:
        return float("nan")
    with np.errstate(invalid="ignore"):
        corr_matrix = np.corrcoef(vectors)
    upper_idx = np.triu_indices_from(corr_matrix, k=1)
    upper_values = corr_matrix[upper_idx]
    valid_values = upper_values[np.isfinite(upper_values)]
    if valid_values.size == 0:
        return float("nan")
    return float(np.mean(valid_values))


def average_pairwise_js(vectors: List[np.ndarray] | np.ndarray, eps: float = 1e-12) -> float:
    """计算样本之间的平均 JS 散度"""
    if isinstance(vectors, list):
        if not vectors:
            return float("nan")
        vectors = np.stack(vectors, axis=0)
    if vectors.shape[0] < 2:
        return float("nan")

    vectors = np.clip(vectors.astype(np.float64), a_min=0.0, a_max=None)
    sums = vectors.sum(axis=1, keepdims=True) + eps
    probs = vectors / sums

    total = 0.0
    count = 0
    for i in range(probs.shape[0]):
        for j in range(i + 1, probs.shape[0]):
            total += js_divergence(probs[i], probs[j], eps=eps)
            count += 1
    if count == 0:
        return float("nan")
    return float(total / count)


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

    # Load or compute normalization stats (for evaluation on non-train split)
    normalization_stats = None
    if args.apply_gene_normalization:
        import numpy as _np
        # If evaluating on non-train split and no stats provided, try to compute from training split
        if args.split != "train" and args.normalization_stats is None:
            print("[评估] 未提供 --normalization-stats，尝试从训练集计算 mean/std（这可能需要一些时间）...")
            try:
                train_dataset = BulkStaticGraphDataset372(
                    graph_data_dir=args.graph_data_dir,
                    split="train",
                    selected_genes=selected_genes,
                    max_samples=None,
                    tpm_csv_file=args.tpm_csv_file,
                    apply_gene_normalization=False,
                    normalization_stats=None,
                )
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=args.batch_size or 1,
                    shuffle=False,
                    collate_fn=collate_fn_bulk_372,
                    num_workers=0,
                    pin_memory=False,
                )

                # Welford 在线计算 mean/std，按基因维度（与 dataset.setup_gene_normalization 行为一致）
                count = 0
                mean = None
                M2 = None
                for b in train_loader:
                    exprs = b["expressions"]  # torch.Tensor, shape (B, num_genes)
                    arr = exprs.detach().cpu().numpy()
                    if mean is None:
                        mean = _np.zeros(arr.shape[1], dtype=_np.float64)
                        M2 = _np.zeros(arr.shape[1], dtype=_np.float64)
                    for row in arr:
                        count += 1
                        delta = row - mean
                        mean += delta / count
                        delta2 = row - mean
                        M2 += delta * delta2

                if count == 0:
                    raise RuntimeError("训练集为空，无法计算 normalization stats")
                variance = M2 / count
                std = _np.sqrt(variance)

                # 与 dataset.setup_gene_normalization 保持一致：对过小的 std 使用 1.0 防止除零
                eps = getattr(train_dataset, "_normalization_eps", 1e-6)
                std = _np.asarray(std, dtype=_np.float32)
                std[std < eps] = 1.0
                mean = _np.asarray(mean, dtype=_np.float32)

                normalization_stats = {"mean": mean.tolist(), "std": std.tolist()}
                print(f"[评估] 从训练集计算 normalization_stats 完成（样本数={count}）")
            except Exception as e:
                print(f"[评估] 无法从训练集计算 normalization_stats: {e}。将禁用基因归一化。")
                args.apply_gene_normalization = False
        elif args.normalization_stats is not None:
            # load JSON or npz
            import json
            stats_path = args.normalization_stats
            if stats_path.endswith(".json"):
                with open(stats_path, "r") as sf:
                    normalization_stats = json.load(sf)
            else:
                try:
                    arr = _np.load(stats_path, allow_pickle=True)
                    if isinstance(arr, dict) and "mean" in arr and "std" in arr:
                        normalization_stats = {"mean": _np.asarray(arr["mean"]).tolist(), "std": _np.asarray(arr["std"]).tolist()}
                    else:
                        normalization_stats = {"mean": _np.asarray(arr["mean"]).tolist(), "std": _np.asarray(arr["std"]).tolist()}
                except Exception as e:
                    raise ValueError(f"无法加载 normalization_stats 文件: {e}")

    dataset = BulkStaticGraphDataset372(
        graph_data_dir=args.graph_data_dir,
        split=args.split,
        selected_genes=selected_genes,
        max_samples=args.max_samples,
        tpm_csv_file=args.tpm_csv_file,
        apply_gene_normalization=args.apply_gene_normalization,
        normalization_stats=normalization_stats
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
        use_gene_attention=args.use_gene_attention,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_freeze_base=args.lora_freeze_base
    )
    
    print(f"[评估] 加载模型: {args.model_path}")
    state_dict = torch.load(args.model_path, map_location=device)
    # 尝试非严格加载以兼容可能缺失/多余的参数（例如 gene attention 开关差异）
    missing = None
    try:
        load_res = model.load_state_dict(state_dict, strict=False)
        # load_state_dict 返回一个 NamedTuple，有 missing_keys/unexpected_keys (PyTorch >=1.9)
        missing = getattr(load_res, "missing_keys", None)
        unexpected = getattr(load_res, "unexpected_keys", None)
        if missing:
            print(f"[评估] 加载模型时缺少参数（可能因为构造时关闭了某些功能）：{missing}")
        if unexpected:
            print(f"[评估] 加载模型时发现未使用的参数：{unexpected}")
    except TypeError:
        # 旧版返回 dict 或抛异常，降级为旧式不严格加载
        try:
            model.load_state_dict(state_dict)
        except RuntimeError as e:
            # 最后回退：尝试剥离 checkpoint 中多余的模块键（例如保存时包含 'model' 字段）
            if isinstance(state_dict, dict) and 'model' in state_dict and isinstance(state_dict['model'], dict):
                print("[评估] 检测到 checkpoint 包含 'model' 字段，尝试使用 state_dict['model']")
                model.load_state_dict(state_dict['model'], strict=False)
            else:
                raise
    model.to(device)
    model.eval()

    per_sample_metrics: List[Dict] = []
    all_target_vectors: List[np.ndarray] = []
    all_prediction_vectors: List[np.ndarray] = []
    all_target_vectors_raw: List[np.ndarray] = []

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

                # 获取原始 TPM（未经 z-score 归一化）以计算原始样本间相关性
                try:
                    patient_id_for_raw = batch_patient_ids[sample_idx]
                    raw_target_np = dataset.expressions_data.get(patient_id_for_raw, None)
                    if raw_target_np is None:
                        # 有些数据使用 slide->patient mapping；尝试使用 slide id
                        slide_for_raw = batch_slide_ids[sample_idx]
                        mapped_pid = dataset.slide_to_patient_mapping.get(slide_for_raw, None) if hasattr(dataset, 'slide_to_patient_mapping') else None
                        raw_target_np = dataset.expressions_data.get(mapped_pid, None) if mapped_pid is not None else None
                    if raw_target_np is not None:
                        raw_target_np = np.asarray(raw_target_np, dtype=np.float64)
                    else:
                        raw_target_np = None
                except Exception:
                    raw_target_np = None

                pearson_val = compute_pearson(pred_np, target_np)
                js_val = js_divergence(pred_np, target_np)

                # Optional debug logging for per-sample prediction stats
                if args.debug_logs and (len(per_sample_metrics) < 10):
                    raw_pred = predictions[sample_idx].detach().cpu().numpy()
                    ssum = raw_pred.sum()
                    top_idx = np.argsort(raw_pred)[-5:][::-1]
                    print(f"[debug] sample={batch_slide_ids[sample_idx]} sum={ssum:.3f} min={raw_pred.min():.6f} max={raw_pred.max():.6f} top5={top_idx.tolist()}")

                per_sample_metrics.append(
                    {
                        "slide_id": batch_slide_ids[sample_idx],
                        "patient_id": batch_patient_ids[sample_idx],
                        "pearson": pearson_val,
                        "js_divergence": js_val,
                    }
                )
                all_prediction_vectors.append(pred_np)
                all_target_vectors.append(target_np)
                if raw_target_np is not None:
                    all_target_vectors_raw.append(raw_target_np)

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
    # 计算两套平均 Pearson：一是基于原始 TPM（如果可用），一是基于脚本内部用于比较的 target vectors（可能已归一化）
    target_target_avg = float("nan")
    if all_target_vectors_raw:
        target_target_avg = average_pairwise_pearson(all_target_vectors_raw)
    else:
        target_target_avg = average_pairwise_pearson(all_target_vectors)
    pred_pred_avg = average_pairwise_pearson(all_prediction_vectors)
    target_target_js = average_pairwise_js(all_target_vectors)
    pred_pred_js = average_pairwise_js(all_prediction_vectors)

    if math.isfinite(target_target_avg):
        print(f"原始样本之间平均 Pearson: {target_target_avg:.4f}")
    else:
        print("原始样本之间平均 Pearson: 无法计算（样本不足或存在零方差）")
    if math.isfinite(pred_pred_avg):
        print(f"预测样本之间平均 Pearson: {pred_pred_avg:.4f}")
    else:
        print("预测样本之间平均 Pearson: 无法计算（样本不足或存在零方差）")

    if math.isfinite(target_target_js):
        print(f"原始样本之间平均 JS 散度: {target_target_js:.4f}")
    else:
        print("原始样本之间平均 JS 散度: 无法计算（样本不足或存在零方差）")
    if math.isfinite(pred_pred_js):
        print(f"预测样本之间平均 JS 散度: {pred_pred_js:.4f}")
    else:
        print("预测样本之间平均 JS 散度: 无法计算（样本不足或存在零方差）")
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

