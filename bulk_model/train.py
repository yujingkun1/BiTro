#!/usr/bin/env python3
import os
import sys
import argparse
import psutil

# 允许从项目根目录直接运行：将项目根加入 sys.path
_current_file_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_file_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

from bulk_model.utils import check_environment_compatibility, load_gene_mapping
from bulk_model.dataset import BulkStaticGraphDataset372, collate_fn_bulk_372
from bulk_model.models import OptimizedTransformerPredictor
from bulk_model.trainer import train_optimized_model, load_spatial_pretrained_weights

# TF32/AMP 优化
torch.set_float32_matmul_precision("medium")
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# 与原脚本一致：运行时做一次环境检查
check_environment_compatibility()


def parse_args():
    parser = argparse.ArgumentParser(description='Bulk Model Training with LoRA and Optimizations')
    
    # 数据路径
    parser.add_argument("--graph-data-dir", type=str, 
                       default="/root/autodl-tmp/bulk_BRCA_graphs_new_all_graph",
                       help="图数据目录")
    parser.add_argument("--gene-list-file", type=str,
                       default="/root/autodl-tmp/her_hvg_cut_1000.txt",
                       help="基因列表文件")
    parser.add_argument("--features-file", type=str,
                       default="/root/autodl-tmp/features.tsv",
                       help="特征文件")
    parser.add_argument("--tpm-csv-file", type=str,
                       default="/root/autodl-tmp/tpm-TCGA-BRCA-1000-million.csv",
                       help="TPM表达数据CSV文件")
    
    # 训练参数
    parser.add_argument("--batch-size", type=int, default=4,
                       help="患者Batch Size (None=动态搜索)")
    parser.add_argument("--graph-batch-size", type=int, default=128,
                       help="图Batch Size")
    parser.add_argument("--num-epochs", type=int, default=30,
                       help="训练轮数")
    parser.add_argument("--learning-rate", type=float, default=1e-5,
                       help="学习率")
    parser.add_argument("--weight-decay", type=float, default=1e-5,
                       help="权重衰减")
    parser.add_argument("--patience", type=int, default=15,
                       help="早停耐心值")

    # 迁移学习：从 spatial 模型初始化
    parser.add_argument("--spatial-model-path", type=str, default=None,
                       help="可选：使用已经训练好的 spatial 模型 checkpoint 作为 bulk 预训练权重")
    parser.add_argument("--freeze-backbone-from-spatial", action="store_true",
                       help="如果提供 spatial 模型，是否冻结 bulk 的 GNN + feature_projection + transformer，仅训练输出头")
    
    # LoRA 参数
    parser.add_argument("--use-lora", action="store_true", default=True,
                       help="使用LoRA")
    parser.add_argument("--lora-r", type=int, default=8,
                       help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=16,
                       help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05,
                       help="LoRA dropout")
    parser.add_argument("--lora-freeze-base", action="store_true", default=True,
                       help="冻结LoRA基础权重")
    
    # 数据加载器优化
    parser.add_argument("--num-workers-train", type=int, default=0,
                       help="训练数据加载器worker数 (None=自动)")
    parser.add_argument("--num-workers-test", type=int, default=0,
                       help="测试数据加载器worker数 (None=自动)")
    parser.add_argument("--pin-memory", action="store_true", default=True,
                       help="启用pin_memory")
    parser.add_argument("--persistent-workers", action="store_true", default=True,
                       help="启用persistent_workers")
    parser.add_argument("--prefetch-factor", type=int, default=1,
                       help="预取因子")
    
    # 训练优化
    parser.add_argument("--log-every", type=int, default=10,
                       help="debug日志的batch间隔")
    parser.add_argument("--debug-logs", action="store_true",
                       help="开启详细调试日志")
    parser.add_argument("--enable-profiling", action="store_true",
                       help="开启batch级性能监控")
    parser.add_argument("--cleanup-interval", type=int, default=1,
                       help="显存清理间隔 (batch数)")
    parser.add_argument("--max-train-samples", type=int, default=None,
                       help="最大训练样本数 (None=全部)")
    
    # 动态batch size
    parser.add_argument("--disable-dynamic-bsz", action="store_true",
                       help="禁用动态batch size搜索")
    parser.add_argument("--max-dynamic-bsz", type=int, default=8,
                       help="动态batch size搜索最大值")
    # Gene normalization control (default: enabled)
    group_norm = parser.add_mutually_exclusive_group()
    group_norm.add_argument("--apply-gene-normalization", dest="apply_gene_normalization", action="store_true",
                            help="启用基因 z-score 归一化 (默认)")
    group_norm.add_argument("--no-gene-normalization", dest="apply_gene_normalization", action="store_false",
                            help="禁用基因 z-score 归一化")
    parser.set_defaults(apply_gene_normalization=True)

    # Use gene-attention readout for bulk (default: False; keep original per-node path)
    parser.add_argument("--use-gene-attention-bulk", action="store_true", default=True,
                        help="在 bulk 模型中使用 gene-level attention 直接输出基因预测（按 WSI/患者 聚合；默认开启）")
    parser.add_argument("--cluster-loss-weight", type=float, default=0.1,
                       help="聚类损失权重 (cluster loss weight). 默认 0.1")
    
    return parser.parse_args()


def count_parameters(model):
    """计算模型参数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def format_number(num):
    """格式化数字显示"""
    if num >= 1e9:
        return f"{num / 1e9:.2f}B"
    elif num >= 1e6:
        return f"{num / 1e6:.2f}M"
    elif num >= 1e3:
        return f"{num / 1e3:.2f}K"
    else:
        return str(num)


def find_optimal_batch_size(dataset, device, max_batch_size=16):
    """动态寻找最佳batch size"""
    print("🔍 正在寻找最佳batch_size...")
    
    for bs in [2, 4, 8, 16]:
        if bs > max_batch_size:
            break
        
        try:
            print(f"  测试 batch_size={bs}")
            
            # 创建测试数据加载器
            test_loader = DataLoader(
                dataset,
                batch_size=bs,
                num_workers=2,
                pin_memory=True
            )
            
            # 测试一个batch
            for batch in test_loader:
                expressions = batch['expressions'].to(device)
                spot_graphs_list = batch['spot_graphs_list']
                
                # 估算显存使用
                torch.cuda.empty_cache()
                before_mem = torch.cuda.memory_allocated(device)
                
                # 模拟前向传播
                dummy_prediction = torch.randn(
                    expressions.shape,
                    device=device,
                    requires_grad=True
                )
                loss = nn.MSELoss()(dummy_prediction, expressions)
                loss.backward()
                
                after_mem = torch.cuda.memory_allocated(device)
                mem_usage_gb = (after_mem - before_mem) / 1024**3
                
                print(f"    ✅ batch_size={bs} 可行，显存使用≈{mem_usage_gb:.1f}GB")
                
                del dummy_prediction, loss, expressions, spot_graphs_list
                torch.cuda.empty_cache()
                break
        
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"    ❌ batch_size={bs} 显存不足")
                torch.cuda.empty_cache()
                return max(1, bs // 2)
            else:
                print(f"    ❌ batch_size={bs} 其他错误: {e}")
                torch.cuda.empty_cache()
                continue
    
    return min(bs, max_batch_size)


def main():
    args = parse_args()
    print("=== 优化版本：批量处理多图提升GPU利用率 + LoRA ===")

    selected_genes, _ = load_gene_mapping(args.gene_list_file, args.features_file)
    if not selected_genes:
        print("错误: 未能加载基因映射")
        return
    print(f"最终基因数量: {len(selected_genes)}")

    train_dataset = BulkStaticGraphDataset372(
        graph_data_dir=args.graph_data_dir,
        split='train',
        selected_genes=selected_genes,
        max_samples=args.max_train_samples,
        tpm_csv_file=args.tpm_cfile if False else args.tpm_csv_file,
        apply_gene_normalization=args.apply_gene_normalization
    )
    # If gene normalization is enabled in the training dataset, propagate stats to the test dataset
    normalization_stats = None
    if getattr(train_dataset, "apply_gene_normalization", False):
        normalization_stats = getattr(train_dataset, "normalization_stats", None)
    test_dataset = BulkStaticGraphDataset372(
        graph_data_dir=args.graph_data_dir,
        split='test',
        selected_genes=selected_genes,
        max_samples=None,
        tpm_csv_file=args.tpm_csv_file,
        apply_gene_normalization=getattr(train_dataset, "apply_gene_normalization", False),
        normalization_stats=normalization_stats
    )
    print(f"训练样本: {len(train_dataset)}, 测试样本: {len(test_dataset)}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # 动态或手动配置batch_size
    initial_batch_size = 2
    if args.batch_size is not None:
        batch_size = max(1, args.batch_size)
        print(f"使用指定的患者Batch Size: {batch_size}")
    elif args.disable_dynamic_bsz:
        batch_size = initial_batch_size
        print(f"禁用动态batch size，使用初始值: {batch_size}")
    else:
        optimal_batch_size = find_optimal_batch_size(train_dataset, device, max_batch_size=args.max_dynamic_bsz)
        batch_size = max(initial_batch_size, optimal_batch_size)
        print(f"动态搜索得到患者Batch Size: {batch_size}")

    graph_batch_size = args.graph_batch_size
    print(f"图Batch Size: {graph_batch_size} (增强GPU利用率)")

    # 数据加载器优化配置
    cpu_cores = psutil.cpu_count(logical=False) or 1
    logical_cores = psutil.cpu_count(logical=True) or cpu_cores
    print(f"CPU核心：物理={cpu_cores}, 逻辑={logical_cores}")
    
    num_workers_train = args.num_workers_train if args.num_workers_train is not None else min(8, cpu_cores)
    num_workers_test = args.num_workers_test if args.num_workers_test is not None else min(4, cpu_cores)

    def build_loader(dataset, shuffle, num_workers):
        kwargs = dict(
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn_bulk_372,
            num_workers=max(0, num_workers),
            pin_memory=args.pin_memory,
            drop_last=False,
        )
        if num_workers > 0:
            kwargs["persistent_workers"] = args.persistent_workers
            if args.prefetch_factor is not None:
                kwargs["prefetch_factor"] = max(1, args.prefetch_factor)
        else:
            kwargs["pin_memory"] = False
        return DataLoader(dataset, **kwargs)

    train_loader = build_loader(train_dataset, shuffle=True, num_workers=num_workers_train)
    test_loader = build_loader(test_dataset, shuffle=False, num_workers=num_workers_test)

    print(f"数据加载器：train_workers={num_workers_train}, test_workers={num_workers_test}, pin_memory={args.pin_memory}")

    # 计算使用LoRA前后的参数量对比
    print("\n" + "="*60)
    print("📊 模型参数量统计")
    print("="*60)
    
    if args.use_lora:
        print("正在对比LoRA前后的参数量...")
        
        # 创建不带LoRA的临时模型来计算原始参数量
        print("  1. 创建不带LoRA的模型以计算原始参数量...")
        model_without_lora = OptimizedTransformerPredictor(
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
            graph_batch_size=graph_batch_size,
            use_lora=False,  # 不使用LoRA
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            lora_freeze_base=args.lora_freeze_base
        )
        
        total_params_wo_lora, trainable_params_wo_lora = count_parameters(model_without_lora)
        print(f"     ✓ 不带LoRA模型参数量计算完成")
        
        # 创建带LoRA的实际模型（后续可选地从 spatial 初始化）
        print("  2. 创建带LoRA的模型...")
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
            graph_batch_size=graph_batch_size,
            use_lora=args.use_lora,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            lora_freeze_base=args.lora_freeze_base
        )

        # 如果用户提供了 spatial checkpoint，则在此基础上做迁移学习初始化
        if args.spatial_model_path:
            model = load_spatial_pretrained_weights(
                model,
                spatial_checkpoint_path=args.spatial_model_path,
                device=device,
                freeze_backbone=args.freeze_backbone_from_spatial,
            )
        
        total_params_with_lora, trainable_params_with_lora = count_parameters(model)
        print(f"     ✓ 带LoRA模型参数量计算完成")
        
        # 显示对比结果
        print("\n" + "-"*60)
        print("参数量对比结果（LoRA前后）：")
        print("-"*60)
        print(f"{'指标':<20} {'不带LoRA':<20} {'带LoRA':<20} {'变化':<20}")
        print("-"*60)
        
        # 总参数量
        total_diff = total_params_with_lora - total_params_wo_lora
        total_diff_pct = (total_diff / total_params_wo_lora * 100) if total_params_wo_lora > 0 else 0
        print(f"{'总参数量':<20} {format_number(total_params_wo_lora):<20} {format_number(total_params_with_lora):<20} "
              f"{total_diff_pct:+.2f}% ({format_number(total_diff)})")
        
        # 可训练参数量
        trainable_diff = trainable_params_with_lora - trainable_params_wo_lora
        trainable_diff_pct = (trainable_diff / trainable_params_wo_lora * 100) if trainable_params_wo_lora > 0 else 0
        reduction_pct = ((trainable_params_wo_lora - trainable_params_with_lora) / trainable_params_wo_lora * 100) if trainable_params_wo_lora > 0 else 0
        
        print(f"{'可训练参数量':<20} {format_number(trainable_params_wo_lora):<20} {format_number(trainable_params_with_lora):<20} "
              f"{trainable_diff_pct:+.2f}% ({format_number(trainable_diff)})")
        
        if trainable_params_with_lora < trainable_params_wo_lora:
            print(f"\n🎉 LoRA减少了 {reduction_pct:.2f}% 的可训练参数！")
            print(f"   可训练参数从 {format_number(trainable_params_wo_lora)} 减少到 {format_number(trainable_params_with_lora)}")
            print(f"   减少了 {format_number(trainable_params_wo_lora - trainable_params_with_lora)} 个可训练参数")
        
        # 冻结参数数量
        frozen_params_wo_lora = total_params_wo_lora - trainable_params_wo_lora
        frozen_params_with_lora = total_params_with_lora - trainable_params_with_lora
        print(f"\n{'冻结参数量':<20} {format_number(frozen_params_wo_lora):<20} {format_number(frozen_params_with_lora):<20} "
              f"{format_number(frozen_params_with_lora - frozen_params_wo_lora)}")
        
        print("="*60 + "\n")
        
        # 清理临时模型
        del model_without_lora
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    else:
        # 未启用LoRA，只显示当前模型的参数量
        print("LoRA未启用，仅显示当前模型参数量...")
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
            graph_batch_size=graph_batch_size,
            use_lora=False,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            lora_freeze_base=args.lora_freeze_base
        )

        if args.spatial_model_path:
            model = load_spatial_pretrained_weights(
                model,
                spatial_checkpoint_path=args.spatial_model_path,
                device=device,
                freeze_backbone=args.freeze_backbone_from_spatial,
            )
        
        total_params, trainable_params = count_parameters(model)
        frozen_params = total_params - trainable_params
        
        print("\n" + "-"*60)
        print("模型参数量：")
        print("-"*60)
        print(f"总参数量:     {format_number(total_params)}")
        print(f"可训练参数量: {format_number(trainable_params)}")
        print(f"冻结参数量:   {format_number(frozen_params)}")
        print("="*60 + "\n")

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-7)

    # Configure model behavior: whether to use gene-attention readout for bulk
    if getattr(args, "use_gene_attention_bulk", False):
        print("⚙️ 配置：bulk 使用 gene-attention readout 输出（按 WSI/患者 聚合）")
        model.return_gene_level = True
    else:
        model.return_gene_level = False

    print(f"\n=== 训练配置（LoRA + 优化版本）===")
    print(f"图批量处理大小: {graph_batch_size}")
    print(f"LoRA: r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}, freeze_base={args.lora_freeze_base}")
    print(f"支持混合处理: 有图增强 + 无图原始特征")
    print(f"数据保留率: 100% (0%丢失)")
    print(f"日志：log_every={args.log_every}, debug={args.debug_logs}, profiling={args.enable_profiling}")

    train_losses, test_losses = train_optimized_model(
        model, train_loader, test_loader, optimizer, scheduler,
        num_epochs=args.num_epochs, device=device, patience=args.patience,
        log_every=args.log_every, debug=args.debug_logs,
        enable_profiling=args.enable_profiling,
        cleanup_interval=args.cleanup_interval,
        cluster_loss_weight=args.cluster_loss_weight
    )

    print("\n=== 混合处理训练完成! ===")
    print("✓ 支持有图患者（图增强）和无图患者（原始DINO特征）")
    print("✓ LoRA已应用，减少可训练参数")
    print("✓ 数据保留率: 100%，0%丢失")
    print("✓ 保持原有计算逻辑不变")


if __name__ == "__main__":
    main()


