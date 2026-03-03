#!/usr/bin/env python3
"""
Evaluate a trained bulk model on a dataset split.

This script reports per-sample metrics and summary statistics, including
Pearson correlation and Jensen-Shannon (JS) divergence.
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

# Allow running from the project root by adding it to sys.path.
_current_file_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_file_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from bulk_model.utils import load_gene_mapping
from bulk_model.dataset import BulkStaticGraphDataset372, collate_fn_bulk_372
from bulk_model.models import OptimizedTransformerPredictor


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate bulk model performance on a dataset split")
    # Defaults that can be edited in code (toggle True/False here).
    DEFAULT_USE_GENE_ATTENTION = False
    DEFAULT_APPLY_GENE_NORMALIZATION = False
    DEFAULT_ENABLE_CLUSTER_LOSS = False
    DEFAULT_CLUSTER_LOSS_WEIGHT = 0.0
    parser.add_argument(
        "--model-path",
        type=str,
        default="/root/autodl-tmp/BiTro/best_PRAD_lora_model.pt",
        help="Model checkpoint path",
    )
    parser.add_argument(
        "--graph-data-dir",
        type=str,
        default="/root/autodl-tmp/bulk_PRAD_graphs_new_all_graph",
        help="Graph data directory",
    )
    parser.add_argument(
        "--gene-list-file",
        type=str,
        default="/root/autodl-tmp/PRAD_intersection_genes.txt",
        help="Gene list file (one gene per line)",
    )
    parser.add_argument(
        "--features-file",
        type=str,
        default="/root/autodl-tmp/features.tsv",
        help="Feature mapping TSV file",
    )
    parser.add_argument(
        "--tpm-csv-file",
        type=str,
        default="/root/autodl-tmp/tpm-TCGA-PRAD-intersect-normalized.csv",
        help="TPM expression CSV file",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "test"],
        help="Dataset split to evaluate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Patient batch size",
    )
    parser.add_argument(
        "--graph-batch-size",
        type=int,
        default=128,
        help="Graph batch size",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (e.g., cuda:0). Default: auto",
    )
    parser.add_argument("--debug-logs", action="store_true", default=False,
                        help="Enable verbose debug logging (per-sample stats)")
    parser.add_argument(
        "--use-lora",
        action="store_true",
        default=True,
        help="Enable LoRA adapters",
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
        help="Freeze base weights when using LoRA",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Optional CSV output path to save per-sample metrics",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional limit on number of samples (for debugging)",
    )
    # Gene normalization control (default: enabled)
    group_norm = parser.add_mutually_exclusive_group()
    group_norm.add_argument("--apply-gene-normalization", dest="apply_gene_normalization", action="store_true",
                            help="Enable gene z-score normalization (default)")
    group_norm.add_argument("--no-gene-normalization", dest="apply_gene_normalization", action="store_false",
                            help="Disable gene z-score normalization")
    # gene attention flags
    parser.add_argument("--use-gene-attention", dest="use_gene_attention", action="store_true",
                        help="Enable gene attention readout (default comes from code-level config)")
    parser.add_argument("--no-gene-attention", dest="use_gene_attention", action="store_false",
                        help="Disable gene attention readout")
    # Cluster loss flags (mainly for training; kept for config compatibility).
    parser.add_argument("--enable-cluster-loss", dest="enable_cluster_loss", action="store_true",
                        help="Enable cluster loss during evaluation (usually off; for compatibility)")
    parser.add_argument("--cluster-loss-weight", type=float, default=0.0,
                        help="Cluster regularization weight (only effective when enable-cluster-loss=True)")
    # set defaults from code-level constants
    parser.set_defaults(use_gene_attention=DEFAULT_USE_GENE_ATTENTION)
    parser.set_defaults(apply_gene_normalization=DEFAULT_APPLY_GENE_NORMALIZATION)
    parser.set_defaults(enable_cluster_loss=DEFAULT_ENABLE_CLUSTER_LOSS)
    parser.set_defaults(cluster_loss_weight=DEFAULT_CLUSTER_LOSS_WEIGHT)
    parser.add_argument("--normalization-stats", type=str, default=None,
                        help="Optional normalization stats file (JSON/NPZ) with mean/std for non-train splits")
    return parser.parse_args()


def compute_pearson(pred: np.ndarray, target: np.ndarray, eps: float = 1e-12) -> float:
    """Compute Pearson correlation with zero-variance protection."""
    pred = pred.astype(np.float64)
    target = target.astype(np.float64)

    pred_std = pred.std()
    target_std = target.std()
    if pred_std < eps or target_std < eps:
        return float("nan")

    return float(np.corrcoef(pred, target)[0, 1])


def js_divergence(pred: np.ndarray, target: np.ndarray, eps: float = 1e-12) -> float:
    """Compute Jensen-Shannon divergence between two non-negative vectors."""
    p = pred / (pred.sum() + eps)
    q = target / (target.sum() + eps)
    m = 0.5 * (p + q)

    def _safe_kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * (np.log(a[mask] + eps) - np.log(b[mask] + eps))))

    return 0.5 * (_safe_kl(p, m) + _safe_kl(q, m))


def average_pairwise_pearson(vectors: List[np.ndarray] | np.ndarray) -> float:
    """Compute mean pairwise Pearson correlation across rows."""
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
    """Compute mean pairwise JS divergence across rows."""
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
    """Ensure each PyG graph is moved to the target device."""
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
    """Run model for a single sample and return an aggregated prediction."""
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
        max_cells_threshold = 150000  # Keep consistent with training.

        if total_cells <= max_cells_threshold:
            cell_predictions_list = model(graphs)
        else:
            # Adaptively chunk very large samples.
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
    print(f"[EVAL] Using device: {device}")

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {args.model_path}")

    selected_genes, _ = load_gene_mapping(args.gene_list_file, args.features_file)
    if not selected_genes:
        raise RuntimeError("Failed to load gene mapping; please check input files")

    # Load or compute normalization stats (for evaluation on non-train split)
    normalization_stats = None
    if args.apply_gene_normalization:
        import numpy as _np
        # If evaluating on non-train split and no stats provided, try to compute from training split
        if args.split != "train" and args.normalization_stats is None:
            print("[EVAL] --normalization-stats not provided; computing mean/std from the training split (may take some time)...")
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

                # Welford online mean/std per gene (aligned with dataset.setup_gene_normalization).
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
                    raise RuntimeError("Training split is empty; cannot compute normalization stats")
                variance = M2 / count
                std = _np.sqrt(variance)

                # Align with dataset.setup_gene_normalization: clamp tiny std to 1.0 to avoid div-by-zero.
                eps = getattr(train_dataset, "_normalization_eps", 1e-6)
                std = _np.asarray(std, dtype=_np.float32)
                std[std < eps] = 1.0
                mean = _np.asarray(mean, dtype=_np.float32)

                normalization_stats = {"mean": mean.tolist(), "std": std.tolist()}
                print(f"[EVAL] Computed normalization_stats from training split (n={count})")
            except Exception as e:
                print(f"[EVAL] Failed to compute normalization_stats from training split: {e}. Disabling gene normalization.")
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
                    raise ValueError(f"Unable to load normalization_stats file: {e}")

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
    
    print(f"[EVAL] Loading model: {args.model_path}")
    state_dict = torch.load(args.model_path, map_location=device)
    # Try non-strict loading to tolerate missing/unexpected keys (e.g., gene attention toggles).
    missing = None
    try:
        load_res = model.load_state_dict(state_dict, strict=False)
        # load_state_dict returns a NamedTuple with missing_keys/unexpected_keys (PyTorch >= 1.9).
        missing = getattr(load_res, "missing_keys", None)
        unexpected = getattr(load_res, "unexpected_keys", None)
        if missing:
            print(f"[EVAL] Missing keys while loading (may be due to disabled features): {missing}")
        if unexpected:
            print(f"[EVAL] Unexpected keys while loading: {unexpected}")
    except TypeError:
        # Older behavior fallback.
        try:
            model.load_state_dict(state_dict)
        except RuntimeError as e:
            # Last-resort fallback: some checkpoints nest parameters under a 'model' field.
            if isinstance(state_dict, dict) and 'model' in state_dict and isinstance(state_dict['model'], dict):
                print("[EVAL] Detected nested checkpoint field 'model'; trying state_dict['model']")
                model.load_state_dict(state_dict['model'], strict=False)
            else:
                raise
    model.to(device)
    model.eval()

    per_sample_metrics: List[Dict] = []
    all_target_vectors: List[np.ndarray] = []
    all_prediction_vectors: List[np.ndarray] = []
    all_target_vectors_raw: List[np.ndarray] = []

    print(f"[EVAL] Evaluating split: {args.split}")
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

            # Normalize predictions the same way as training.
            epsilon = 1e-8
            sum_pred = predictions.sum(dim=1, keepdim=True) + epsilon
            normalized_pred = predictions / sum_pred
            scaled_pred = torch.clamp(normalized_pred * 1000000.0, min=0.0, max=1e6)

            for sample_idx in range(scaled_pred.shape[0]):
                pred_np = scaled_pred[sample_idx].detach().cpu().numpy()
                target_np = targets[sample_idx].detach().cpu().numpy()

                # Try to retrieve raw TPM (non z-score) for target-target correlation.
                try:
                    patient_id_for_raw = batch_patient_ids[sample_idx]
                    raw_target_np = dataset.expressions_data.get(patient_id_for_raw, None)
                    if raw_target_np is None:
                        # Some datasets use slide->patient mapping; try slide_id.
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
                print(f"  Processed {batch_idx + 1} batches...")

    if not per_sample_metrics:
        print("[EVAL] No valid evaluation samples; please check the dataset")
        return

    pearson_values = np.array(
        [m["pearson"] for m in per_sample_metrics if not math.isnan(m["pearson"])]
    )
    js_values = np.array(
        [m["js_divergence"] for m in per_sample_metrics if math.isfinite(m["js_divergence"])]
    )

    print("\n" + "="*60)
    print("=== Evaluation summary ===")
    print("="*60)
    # Compute two mean Pearson scores: one based on raw TPM (if available), one based on
    # the vectors used for evaluation (may be normalized).
    target_target_avg = float("nan")
    if all_target_vectors_raw:
        target_target_avg = average_pairwise_pearson(all_target_vectors_raw)
    else:
        target_target_avg = average_pairwise_pearson(all_target_vectors)
    pred_pred_avg = average_pairwise_pearson(all_prediction_vectors)
    target_target_js = average_pairwise_js(all_target_vectors)
    pred_pred_js = average_pairwise_js(all_prediction_vectors)

    if math.isfinite(target_target_avg):
        print(f"Mean Pearson (target-target, raw if available): {target_target_avg:.4f}")
    else:
        print("Mean Pearson (target-target): unavailable (insufficient samples or zero variance)")
    if math.isfinite(pred_pred_avg):
        print(f"Mean Pearson (prediction-prediction): {pred_pred_avg:.4f}")
    else:
        print("Mean Pearson (prediction-prediction): unavailable (insufficient samples or zero variance)")

    if math.isfinite(target_target_js):
        print(f"Mean JS divergence (target-target): {target_target_js:.4f}")
    else:
        print("Mean JS divergence (target-target): unavailable (insufficient samples or zero variance)")
    if math.isfinite(pred_pred_js):
        print(f"Mean JS divergence (prediction-prediction): {pred_pred_js:.4f}")
    else:
        print("Mean JS divergence (prediction-prediction): unavailable (insufficient samples or zero variance)")
    print("="*60)
    print(f"Evaluation samples: {len(per_sample_metrics)}")
    if pearson_values.size > 0:
        print(
            f"Pearson - mean: {pearson_values.mean():.4f}, "
            f"median: {np.median(pearson_values):.4f}, "
            f"std: {pearson_values.std():.4f}"
        )
        print(f"  Min: {pearson_values.min():.4f}, Max: {pearson_values.max():.4f}")
    else:
        print("Pearson: insufficient valid samples")

    if js_values.size > 0:
        print(
            f"JS divergence - mean: {js_values.mean():.4f}, "
            f"median: {np.median(js_values):.4f}, "
            f"std: {js_values.std():.4f}"
        )
        print(f"  Min: {js_values.min():.4f}, Max: {js_values.max():.4f}")
    else:
        print("JS divergence: insufficient valid samples")
    print("="*60)

    if args.output_csv:
        df = pd.DataFrame(per_sample_metrics)
        df.to_csv(args.output_csv, index=False)
        print(f"\n[EVAL] Per-sample metrics saved to: {args.output_csv}")


if __name__ == "__main__":
    evaluate()

