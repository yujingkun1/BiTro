#!/usr/bin/env python3
"""
使用空转模型在bulk静态图上进行推理与评估（仅推理，不训练）。

流程概述：
- 读取bulk静态图数据（兼容patch图与“无图但有全细胞特征”的情况）
- 加载空转模型（spitial_model.models.StaticGraphTransformerPredictor）与权重
- 对每个切片/患者：逐patch推理并聚合为切片级表达预测
- 将预测与bulk表达（log1p）对齐后计算Pearson相关
- 输出每切片指标与总体指标，保存CSV/JSON/可选npy

注意：
- 基因顺序对齐：读取bulk图目录中的 bulk_graph_config.json 的 gene_names，
  同时读取提供的 gene_file（与空转训练一致的交集基因文件），将bulk表达重排到与空转模型一致的顺序。
"""

import os
import json
import argparse
from typing import List, Tuple, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.stats import pearsonr

# 本项目内导入
from spitial_model.models import StaticGraphTransformerPredictor
from spitial_model.utils import setup_device
from bulk_model.dataset import BulkStaticGraphDataset372, collate_fn_bulk_372


def read_gene_list(gene_file: str) -> List[str]:
    genes = []
    with open(gene_file, 'r') as f:
        for line in f:
            g = line.strip()
            if g and not g.startswith('Efficiently') and not g.startswith('Total') and not g.startswith('Detection') and not g.startswith('Samples'):
                genes.append(g)
    # 去重但保持次序
    seen = set()
    ordered = [g for g in genes if not (g in seen or seen.add(g))]
    return ordered


def load_bulk_config_gene_names(graph_dir: str) -> List[str]:
    cfg_path = os.path.join(graph_dir, 'bulk_graph_config.json')
    if not os.path.exists(cfg_path):
        # 兼容旧命名
        return []
    try:
        with open(cfg_path, 'r') as f:
            cfg = json.load(f)
        names = cfg.get('gene_names') or []
        return names
    except Exception:
        return []


def build_gene_reindex(bulk_gene_names: List[str], spatial_genes: List[str]) -> np.ndarray:
    """
    构造将bulk基因向量重排到spatial训练使用的gene顺序的索引数组。
    要求两边是同一组基因（通常为897交集基因），若有缺失则抛出警告并尝试交集。
    返回：shape [G'] 的整型索引，供 expression[reindex] 使用。
    """
    if not bulk_gene_names or not spatial_genes:
        # 无法对齐时，默认假设相同顺序
        return np.arange(len(spatial_genes), dtype=np.int64)

    name_to_idx = {g: i for i, g in enumerate(bulk_gene_names)}
    idxs = []
    missing = []
    for g in spatial_genes:
        i = name_to_idx.get(g)
        if i is None:
            missing.append(g)
        else:
            idxs.append(i)

    if missing:
        print(f"⚠️ 警告：bulk表达缺失 {len(missing)} 个基因，将忽略这些基因")
        # 仅保留存在于bulk的那部分基因
        present_spatial_genes = [g for g in spatial_genes if g in name_to_idx]
        # 根据保留下来的基因重建索引
        idxs = [name_to_idx[g] for g in present_spatial_genes]
        return np.asarray(idxs, dtype=np.int64)

    return np.asarray(idxs, dtype=np.int64)


def load_spatial_model(ckpt_path: str, input_dim: int, num_genes: int, device: torch.device) -> StaticGraphTransformerPredictor:
    model = StaticGraphTransformerPredictor(
        input_dim=input_dim,
        gnn_hidden_dim=128,
        gnn_output_dim=128,
        embed_dim=256,
        num_genes=num_genes,
        num_layers=2,
        nhead=8,
        dropout=0.3,
        use_gnn=True,
        gnn_type='GAT',
        n_pos=128,
        use_lora=True,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        lora_freeze_base=True,
    ).to(device)

    if ckpt_path and os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state, strict=False)
        print(f"✓ 已加载空转模型权重: {ckpt_path}")
    else:
        print(f"⚠️ 未找到模型权重: {ckpt_path}，将使用随机初始化权重")

    model.eval()
    return model


@torch.no_grad()
def predict_one_slide(model: StaticGraphTransformerPredictor,
                      graphs: List,                   # List[Data]
                      fallback_all_cells: Tuple[torch.Tensor, torch.Tensor] = None,  # (x, pos)
                      device: torch.device = torch.device('cpu')) -> np.ndarray:
    """
    对单个切片进行预测：
    - 若有patch级图：对每个图推理并对预测取平均
    - 若无图：使用所有细胞的(x, pos)构造单图（edge_index空）推理
    返回：切片级预测（num_genes,）numpy数组（与模型输出一致的log空间尺度）
    """
    batch_graphs = None
    if graphs and len(graphs) > 0:
        batch_graphs = [g.to(device) for g in graphs if g is not None]
    elif fallback_all_cells is not None:
        x, pos = fallback_all_cells
        from torch_geometric.data import Data
        g = Data(x=x.to(device), edge_index=torch.empty((2, 0), dtype=torch.long, device=device), pos=pos.to(device))
        batch_graphs = [g]
    else:
        return None

    preds = model(batch_graphs)  # [B, G]
    if preds is None or preds.numel() == 0:
        return None
    slide_pred = preds.mean(dim=0)  # [G]
    return slide_pred.detach().cpu().numpy()


def compute_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true is None or y_pred is None:
        return np.nan
    if y_true.ndim != 1:
        y_true = y_true.reshape(-1)
    if y_pred.ndim != 1:
        y_pred = y_pred.reshape(-1)
    if y_true.size != y_pred.size:
        m = min(y_true.size, y_pred.size)
        y_true, y_pred = y_true[:m], y_pred[:m]
    # 跳过零方差
    if np.var(y_true) < 1e-12 or np.var(y_pred) < 1e-12:
        return np.nan
    r, _ = pearsonr(y_true, y_pred)
    return float(r)


def run_inference(
    bulk_graph_dir: str,
    spatial_ckpt: str,
    gene_file: str,
    device_id: int = 0,
    split: str = 'test',   # 'train' | 'test' | 'both'
    batch_size: int = 1,
    num_workers: int = 2,
    save_dir: str = './infer_spatial_on_bulk_logs',
    save_predictions: bool = True,
):
    os.makedirs(save_dir, exist_ok=True)

    device = setup_device(device_id=device_id)

    # 读取基因顺序
    spatial_genes = read_gene_list(gene_file)
    bulk_gene_names = load_bulk_config_gene_names(bulk_graph_dir)
    reindex = build_gene_reindex(bulk_gene_names, spatial_genes)
    print(f"空转基因数: {len(spatial_genes)} | bulk基因数: {len(bulk_gene_names)} | 对齐后: {len(reindex)}")

    # 从bulk元数据中探测特征维度
    meta_path = os.path.join(bulk_graph_dir, f"bulk_{split}_metadata.json" if split in ('train','test') else f"bulk_test_metadata.json")
    feature_dim = 128
    try:
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        # 取任意一个条目的 cell_feature_dim
        if isinstance(meta, dict) and len(meta) > 0:
            sample_meta = next(iter(meta.values()))
            feature_dim = int(sample_meta.get('cell_feature_dim', 128))
    except Exception:
        pass

    # 加载模型
    model = load_spatial_model(spatial_ckpt, input_dim=feature_dim, num_genes=len(reindex), device=device)

    def infer_on_split(split_name: str):
        print(f"\n=== 在 {split_name} 集合上推理 ===")
        ds = BulkStaticGraphDataset372(graph_data_dir=bulk_graph_dir, split=split_name)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn_bulk_372)

        results = []
        all_preds = []
        all_gts = []

        for batch in dl:
            slide_ids = batch['slide_ids']
            patient_ids = batch['patient_ids']
            spot_graphs_list = batch['spot_graphs_list']  # List[List[Data]]
            exprs = batch['expressions']                  # [B, G_bulk]
            all_cell_features_list = batch['all_cell_features_list']
            all_cell_positions_list = batch['all_cell_positions_list']
            has_graphs_list = batch['has_graphs_list']

            for i in range(len(slide_ids)):
                graphs = spot_graphs_list[i]
                has_graphs = bool(has_graphs_list[i])

                fallback = None
                if (not graphs) or len(graphs) == 0:
                    x = all_cell_features_list[i]
                    p = all_cell_positions_list[i]
                    if isinstance(x, torch.Tensor) and isinstance(p, torch.Tensor) and x.size(0) == p.size(0) and x.size(0) > 0:
                        fallback = (x, p)

                pred = predict_one_slide(model, graphs, fallback_all_cells=fallback, device=device)
                if pred is None:
                    # 无法推理，跳过
                    continue

                # 对齐与变换 bulk 表达到log空间、顺序重排
                gt_bulk = exprs[i].detach().cpu().numpy()
                if gt_bulk.ndim != 1:
                    gt_bulk = gt_bulk.reshape(-1)
                # 先重排，再log1p
                if reindex is not None and reindex.size > 0 and reindex.max() < gt_bulk.size:
                    gt_bulk_aligned = gt_bulk[reindex]
                else:
                    gt_bulk_aligned = gt_bulk[:pred.shape[0]]
                gt_bulk_log = np.log1p(gt_bulk_aligned)

                # 若模型输出基因数与对齐后不一致，裁剪至共同长度
                Gc = min(pred.shape[0], gt_bulk_log.shape[0])
                corr = compute_pearson(gt_bulk_log[:Gc], pred[:Gc])

                results.append({
                    'slide_id': slide_ids[i],
                    'patient_id': patient_ids[i],
                    'has_graphs': has_graphs,
                    'num_graphs': len(graphs) if graphs else 0,
                    'pearson': corr
                })

                all_preds.append(pred[:Gc])
                all_gts.append(gt_bulk_log[:Gc])

        # 汇总
        summary = {
            'num_items': len(results),
            'mean_pearson': float(np.nanmean([r['pearson'] for r in results])) if results else np.nan,
            'median_pearson': float(np.nanmedian([r['pearson'] for r in results])) if results else np.nan,
        }
        if all_preds and all_gts:
            stacked_pred = np.stack(all_preds, axis=0)
            stacked_gt = np.stack(all_gts, axis=0)
            overall_corr, _ = pearsonr(stacked_gt.flatten(), stacked_pred.flatten())
            summary['overall_pearson_flat'] = float(overall_corr)
        else:
            summary['overall_pearson_flat'] = np.nan

        # 保存
        import pandas as pd
        df = pd.DataFrame(results)
        csv_path = os.path.join(save_dir, f"infer_{split_name}_per_slide.csv")
        df.to_csv(csv_path, index=False)
        with open(os.path.join(save_dir, f"infer_{split_name}_summary.json"), 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"保存结果: {csv_path}")
        print(f"统计: {summary}")

        if save_predictions and all_preds and all_gts:
            np.save(os.path.join(save_dir, f"infer_{split_name}_preds.npy"), np.stack(all_preds, axis=0))
            np.save(os.path.join(save_dir, f"infer_{split_name}_gts.npy"), np.stack(all_gts, axis=0))

    if split in ('train', 'test'):
        infer_on_split(split)
    elif split == 'both':
        infer_on_split('train')
        infer_on_split('test')
    else:
        raise ValueError("split 应为 'train' | 'test' | 'both'")


def parse_args():
    p = argparse.ArgumentParser(description='Use spatial model to infer on bulk graphs (no training).')
    p.add_argument('--graph_dir', type=str, required=True, help='bulk静态图输出目录（包含bulk_*_intra_patch_graphs.pkl等）')
    p.add_argument('--ckpt', type=str, required=False, default='./checkpoints/best_hest_graph_model_fold_0.pt', help='空转模型权重路径')
    p.add_argument('--gene_file', type=str, required=False, default='/data/yujk/hovernet2feature/HEST/tutorials/SA_process/common_genes_misc_tenx_zen_897.txt', help='空转训练使用的基因列表文件（与训练一致）')
    p.add_argument('--split', type=str, default='test', choices=['train', 'test', 'both'])
    p.add_argument('--device_id', type=int, default=0)
    p.add_argument('--batch_size', type=int, default=1)
    p.add_argument('--num_workers', type=int, default=2)
    p.add_argument('--save_dir', type=str, default='./infer_spatial_on_bulk_logs')
    p.add_argument('--no_save_preds', action='store_true', help='不保存预测矩阵')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_inference(
        bulk_graph_dir=args.graph_dir,
        spatial_ckpt=args.ckpt,
        gene_file=args.gene_file,
        device_id=args.device_id,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        save_dir=args.save_dir,
        save_predictions=(not args.no_save_preds),
    )


