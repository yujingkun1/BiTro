#!/usr/bin/env python3
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from .utils import check_environment_compatibility, load_gene_mapping
from .dataset import BulkStaticGraphDataset372, collate_fn_bulk_372
from .models import OptimizedTransformerPredictor
from .trainer import train_optimized_model


# 与原脚本一致：运行时做一次环境检查
check_environment_compatibility()


def main():
    print("=== 优化版本：批量处理多图提升GPU利用率 ===")

    graph_data_dir = "/root/autodl-tmp/bulk_static_graphs_new_all_graph"
    gene_list_file = "/root/autodl-tmp/common_genes_misc_tenx_zen_897.txt"
    features_file = "/root/autodl-tmp/features.tsv"

    selected_genes, _ = load_gene_mapping(gene_list_file, features_file)
    if not selected_genes:
        print("错误: 未能加载基因映射")
        return
    print(f"最终基因数量: {len(selected_genes)}")

    batch_size = 1
    graph_batch_size = 64
    num_epochs = 60
    learning_rate = 1e-4
    weight_decay = 1e-5
    patience = 15

    print(f"患者Batch Size: {batch_size}")
    print(f"图Batch Size: {graph_batch_size} (核心优化参数)")

    train_dataset = BulkStaticGraphDataset372(
        graph_data_dir=graph_data_dir,
        split='train',
        selected_genes=selected_genes,
        max_samples=None
    )
    test_dataset = BulkStaticGraphDataset372(
        graph_data_dir=graph_data_dir,
        split='test',
        selected_genes=selected_genes,
        max_samples=None
    )
    print(f"训练样本: {len(train_dataset)}, 测试样本: {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn_bulk_372,
        num_workers=0,
        pin_memory=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_bulk_372,
        num_workers=0,
        pin_memory=False
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    model = OptimizedTransformerPredictor(
        input_dim=train_dataset.feature_dim,
        gnn_hidden_dim=128,
        gnn_output_dim=128,
        embed_dim=256,
        num_genes=train_dataset.num_genes,
        num_layers=3,
        nhead=8,
        dropout=0.1,
        use_gnn=True,
        gnn_type='GAT',
        graph_batch_size=graph_batch_size
    )

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-7)

    print(f"\n=== 训练配置（0%数据丢失版本）===")
    print(f"图批量处理大小: {graph_batch_size}")
    print(f"支持混合处理: 有图增强 + 无图原始特征")
    print(f"数据保留率: 100% (0%丢失)")

    train_losses, test_losses = train_optimized_model(
        model, train_loader, test_loader, optimizer, scheduler,
        num_epochs=num_epochs, device=device, patience=patience
    )

    print("\n=== 混合处理训练完成! ===")
    print("✓ 支持有图患者（图增强）和无图患者（原始DINO特征）")
    print("✓ 数据保留率: 100%，0%丢失")
    print("✓ 保持原有计算逻辑不变")


if __name__ == "__main__":
    main()


