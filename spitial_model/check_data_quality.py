#!/usr/bin/env python3
"""
数据质量检查脚本 - 检查PAAD基因表达分布
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
from scipy import stats
from collections import defaultdict

# 配置参数（与train.py保持一致）
hest_data_dir = "/data/yujk/hovernet2feature/HEST/hest_data"
hest_bench_paad_dir = "/data/yujk/hovernet2feature/HEST-Bench/PAAD"
gene_file = "/data/yujk/hovernet2feature/HEST-Bench/PAAD/var_50genes.txt"

def get_paad_samples():
    """从HEST-Bench/PAAD的splits文件中获取PAAD样本列表"""
    paad_samples = set()
    
    # 从splits目录中读取所有样本
    splits_dir = os.path.join(hest_bench_paad_dir, "splits")
    if os.path.exists(splits_dir):
        for filename in os.listdir(splits_dir):
            if filename.endswith('.csv'):
                csv_path = os.path.join(splits_dir, filename)
                try:
                    df = pd.read_csv(csv_path)
                    if 'sample_id' in df.columns:
                        paad_samples.update(df['sample_id'].tolist())
                except Exception as e:
                    print(f"警告: 读取 {csv_path} 失败: {e}")
    
    # 如果splits文件不存在，从adata目录获取
    if not paad_samples:
        adata_dir = os.path.join(hest_bench_paad_dir, "adata")
        if os.path.exists(adata_dir):
            for filename in os.listdir(adata_dir):
                if filename.endswith('.h5ad'):
                    sample_id = filename.replace('.h5ad', '')
                    paad_samples.add(sample_id)
    
    # 如果还是找不到，使用已知的PAAD样本（从utils.py中看到的）
    if not paad_samples:
        paad_samples = {'TENX116', 'TENX126', 'TENX140'}
        print("使用默认PAAD样本列表")
    
    return sorted(list(paad_samples))

def load_gene_list():
    """加载基因列表"""
    if not os.path.exists(gene_file):
        print(f"警告: 基因文件不存在: {gene_file}")
        return []
    
    genes = []
    with open(gene_file, 'r') as f:
        for line in f:
            gene = line.strip()
            if gene and not gene.startswith('Efficiently') and not gene.startswith('Total'):
                genes.append(gene)
    
    return genes

def check_sample_quality(sample_id, selected_genes):
    """检查单个样本的数据质量"""
    print(f"\n{'='*60}")
    print(f"检查样本: {sample_id}")
    print(f"{'='*60}")
    
    st_file = os.path.join(hest_data_dir, "st", f"{sample_id}.h5ad")
    if not os.path.exists(st_file):
        print(f"  ❌ ST文件不存在: {st_file}")
        return None
    
    try:
        adata = sc.read_h5ad(st_file)
        print(f"  ✓ 加载成功: {adata.n_obs} spots × {adata.n_vars} genes")
    except Exception as e:
        print(f"  ❌ 加载失败: {e}")
        return None
    
    # 检查基因覆盖
    adata_genes = set(adata.var_names)
    selected_genes_set = set(selected_genes)
    common_genes = selected_genes_set.intersection(adata_genes)
    missing_genes = selected_genes_set - adata_genes
    
    print(f"\n  基因覆盖情况:")
    print(f"    目标基因数: {len(selected_genes)}")
    print(f"    样本中存在的基因: {len(common_genes)}")
    print(f"    缺失的基因: {len(missing_genes)}")
    if missing_genes:
        print(f"    缺失基因列表: {list(missing_genes)[:10]}{'...' if len(missing_genes) > 10 else ''}")
    
    # 提取目标基因的表达数据
    if len(common_genes) == 0:
        print(f"  ❌ 没有共同基因，无法分析")
        return None
    
    # 获取共同基因的表达矩阵
    common_genes_list = [g for g in selected_genes if g in common_genes]
    gene_idx = [list(adata.var_names).index(g) for g in common_genes_list]
    expression = adata.X[:, gene_idx]
    
    # 转换为密集矩阵（如果是稀疏）
    if hasattr(expression, 'toarray'):
        expression = expression.toarray()
    expression = np.asarray(expression)
    
    # 应用log1p（与数据集中的处理一致）
    expression_log = np.log1p(expression)
    
    # 统计信息
    stats_dict = {
        'sample_id': sample_id,
        'n_spots': expression.shape[0],
        'n_genes': expression.shape[1],
        'n_common_genes': len(common_genes),
        'n_missing_genes': len(missing_genes),
    }
    
    # 原始表达统计（log1p之前）
    stats_dict['raw_mean'] = float(expression.mean())
    stats_dict['raw_std'] = float(expression.std())
    stats_dict['raw_min'] = float(expression.min())
    stats_dict['raw_max'] = float(expression.max())
    stats_dict['raw_median'] = float(np.median(expression))
    
    # 零值比例
    zero_ratio = (expression == 0).sum() / expression.size
    stats_dict['zero_ratio'] = float(zero_ratio)
    
    # Log1p后统计
    stats_dict['log_mean'] = float(expression_log.mean())
    stats_dict['log_std'] = float(expression_log.std())
    stats_dict['log_min'] = float(expression_log.min())
    stats_dict['log_max'] = float(expression_log.max())
    stats_dict['log_median'] = float(np.median(expression_log))
    
    print(f"\n  原始表达统计 (未log1p):")
    print(f"    均值: {stats_dict['raw_mean']:.4f}")
    print(f"    标准差: {stats_dict['raw_std']:.4f}")
    print(f"    范围: [{stats_dict['raw_min']:.4f}, {stats_dict['raw_max']:.4f}]")
    print(f"    中位数: {stats_dict['raw_median']:.4f}")
    print(f"    零值比例: {zero_ratio*100:.2f}%")
    
    print(f"\n  Log1p后统计:")
    print(f"    均值: {stats_dict['log_mean']:.4f}")
    print(f"    标准差: {stats_dict['log_std']:.4f}")
    print(f"    范围: [{stats_dict['log_min']:.4f}, {stats_dict['log_max']:.4f}]")
    print(f"    中位数: {stats_dict['log_median']:.4f}")
    
    # 每个基因的统计
    gene_stats = []
    for i, gene in enumerate(common_genes_list):
        gene_expr = expression_log[:, i]
        gene_stats.append({
            'gene': gene,
            'mean': float(gene_expr.mean()),
            'std': float(gene_expr.std()),
            'min': float(gene_expr.min()),
            'max': float(gene_expr.max()),
            'median': float(np.median(gene_expr)),
            'zero_ratio': float((expression[:, i] == 0).sum() / expression.shape[0]),
            'variance': float(gene_expr.var()),
        })
    
    stats_dict['gene_stats'] = gene_stats
    
    # 检查异常值
    # 使用IQR方法检测异常值
    q1 = np.percentile(expression_log, 25)
    q3 = np.percentile(expression_log, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = ((expression_log < lower_bound) | (expression_log > upper_bound)).sum()
    outlier_ratio = outliers / expression_log.size
    
    stats_dict['outlier_ratio'] = float(outlier_ratio)
    stats_dict['iqr_lower'] = float(lower_bound)
    stats_dict['iqr_upper'] = float(upper_bound)
    
    print(f"\n  异常值检测 (IQR方法):")
    print(f"    异常值比例: {outlier_ratio*100:.2f}%")
    print(f"    正常范围: [{lower_bound:.4f}, {upper_bound:.4f}]")
    
    # 检查基因表达方差
    gene_variances = [gs['variance'] for gs in gene_stats]
    low_variance_genes = [gs['gene'] for gs in gene_stats if gs['variance'] < 1e-6]
    
    stats_dict['low_variance_genes'] = low_variance_genes
    stats_dict['mean_gene_variance'] = float(np.mean(gene_variances))
    stats_dict['median_gene_variance'] = float(np.median(gene_variances))
    
    print(f"\n  基因方差统计:")
    print(f"    平均方差: {stats_dict['mean_gene_variance']:.6f}")
    print(f"    中位数方差: {stats_dict['median_gene_variance']:.6f}")
    print(f"    低方差基因数 (<1e-6): {len(low_variance_genes)}")
    if low_variance_genes:
        print(f"    低方差基因: {low_variance_genes[:5]}{'...' if len(low_variance_genes) > 5 else ''}")
    
    # 检查表达分布是否正态
    # 对每个基因进行Shapiro-Wilk测试（样本数限制）
    normal_genes = []
    non_normal_genes = []
    for gs in gene_stats:
        gene_expr = expression_log[:, common_genes_list.index(gs['gene'])]
        # 只对前5000个spot进行测试（Shapiro-Wilk有样本数限制）
        test_sample = gene_expr[:min(5000, len(gene_expr))]
        if len(test_sample) >= 3:
            try:
                stat, p_value = stats.shapiro(test_sample)
                if p_value > 0.05:
                    normal_genes.append(gs['gene'])
                else:
                    non_normal_genes.append(gs['gene'])
            except:
                pass
    
    stats_dict['normal_distributed_genes'] = len(normal_genes)
    stats_dict['non_normal_distributed_genes'] = len(non_normal_genes)
    
    print(f"\n  分布正态性 (Shapiro-Wilk测试):")
    print(f"    近似正态分布基因: {len(normal_genes)}")
    print(f"    非正态分布基因: {len(non_normal_genes)}")
    
    return stats_dict

def generate_summary_report(all_stats, output_dir):
    """生成汇总报告"""
    if not all_stats:
        print("没有可用的统计数据")
        return
    
    print(f"\n{'='*60}")
    print("汇总报告")
    print(f"{'='*60}")
    
    # 汇总统计
    n_samples = len(all_stats)
    total_spots = sum(s['n_spots'] for s in all_stats)
    avg_spots = total_spots / n_samples if n_samples > 0 else 0
    
    print(f"\n样本统计:")
    print(f"  样本数: {n_samples}")
    print(f"  总spots数: {total_spots}")
    print(f"  平均spots/样本: {avg_spots:.1f}")
    
    # 基因覆盖汇总
    all_common_genes = set()
    all_missing_genes = set()
    for s in all_stats:
        if 'gene_stats' in s:
            all_common_genes.update([gs['gene'] for gs in s['gene_stats']])
    
    print(f"\n基因覆盖汇总:")
    print(f"  至少在一个样本中存在的基因: {len(all_common_genes)}")
    
    # 表达统计汇总
    all_log_means = [s['log_mean'] for s in all_stats]
    all_log_stds = [s['log_std'] for s in all_stats]
    all_zero_ratios = [s['zero_ratio'] for s in all_stats]
    
    print(f"\n表达统计汇总 (log1p后):")
    print(f"  平均表达均值: {np.mean(all_log_means):.4f} ± {np.std(all_log_means):.4f}")
    print(f"  平均表达标准差: {np.mean(all_log_stds):.4f} ± {np.std(all_log_stds):.4f}")
    print(f"  平均零值比例: {np.mean(all_zero_ratios)*100:.2f}% ± {np.std(all_zero_ratios)*100:.2f}%")
    
    # 基因方差汇总
    all_gene_variances = []
    for s in all_stats:
        if 'gene_stats' in s:
            all_gene_variances.extend([gs['variance'] for gs in s['gene_stats']])
    
    if all_gene_variances:
        print(f"\n基因方差汇总:")
        print(f"  平均方差: {np.mean(all_gene_variances):.6f}")
        print(f"  中位数方差: {np.median(all_gene_variances):.6f}")
        print(f"  最小方差: {np.min(all_gene_variances):.6f}")
        print(f"  最大方差: {np.max(all_gene_variances):.6f}")
        low_var_count = sum(1 for v in all_gene_variances if v < 1e-6)
        print(f"  低方差基因数 (<1e-6): {low_var_count} ({low_var_count/len(all_gene_variances)*100:.1f}%)")
    
    # 保存详细报告
    os.makedirs(output_dir, exist_ok=True)
    report_file = os.path.join(output_dir, "data_quality_report.json")
    with open(report_file, 'w') as f:
        json.dump(all_stats, f, indent=2)
    print(f"\n详细报告已保存: {report_file}")
    
    # 生成可视化
    try:
        plot_expression_distribution(all_stats, output_dir)
    except Exception as e:
        print(f"生成可视化失败: {e}")

def plot_expression_distribution(all_stats, output_dir):
    """绘制表达分布图"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 每个样本的表达分布箱线图
    ax1 = axes[0, 0]
    sample_names = []
    log_means_list = []
    for s in all_stats:
        if 'gene_stats' in s:
            sample_names.append(s['sample_id'])
            log_means_list.append([gs['mean'] for gs in s['gene_stats']])
    
    if log_means_list:
        ax1.boxplot(log_means_list, labels=sample_names)
        ax1.set_title('Gene Expression Mean Distribution per Sample (log1p)')
        ax1.set_ylabel('Expression Value (log1p)')
        ax1.tick_params(axis='x', rotation=45)
    
    # 2. 基因方差分布
    ax2 = axes[0, 1]
    all_variances = []
    for s in all_stats:
        if 'gene_stats' in s:
            all_variances.extend([gs['variance'] for gs in s['gene_stats']])
    
    if all_variances:
        ax2.hist(all_variances, bins=50, edgecolor='black')
        ax2.set_title('Gene Variance Distribution')
        ax2.set_xlabel('Variance')
        ax2.set_ylabel('Number of Genes')
        ax2.set_yscale('log')
    
    # 3. 零值比例
    ax3 = axes[1, 0]
    zero_ratios = [s['zero_ratio'] for s in all_stats]
    sample_ids = [s['sample_id'] for s in all_stats]
    ax3.bar(range(len(zero_ratios)), [r*100 for r in zero_ratios])
    ax3.set_xticks(range(len(sample_ids)))
    ax3.set_xticklabels(sample_ids, rotation=45)
    ax3.set_title('Zero Expression Ratio per Sample')
    ax3.set_ylabel('Zero Ratio (%)')
    
    # 4. 表达范围
    ax4 = axes[1, 1]
    log_mins = [s['log_min'] for s in all_stats]
    log_maxs = [s['log_max'] for s in all_stats]
    x_pos = range(len(sample_ids))
    ax4.bar([x - 0.2 for x in x_pos], log_mins, width=0.4, label='Min', alpha=0.7)
    ax4.bar([x + 0.2 for x in x_pos], log_maxs, width=0.4, label='Max', alpha=0.7)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(sample_ids, rotation=45)
    ax4.set_title('Expression Range per Sample (log1p)')
    ax4.set_ylabel('Expression Value (log1p)')
    ax4.legend()
    
    plt.tight_layout()
    plot_file = os.path.join(output_dir, "data_quality_plots.png")
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"可视化图表已保存: {plot_file}")
    plt.close()

def main():
    """主函数"""
    print("="*60)
    print("PAAD数据质量检查")
    print("="*60)
    print(f"检查目标: HEST-Bench/PAAD中定义的样本在hest_data中的数据质量")
    
    # 加载基因列表
    selected_genes = load_gene_list()
    print(f"\n目标基因数: {len(selected_genes)}")
    if len(selected_genes) > 0:
        print(f"前10个基因: {selected_genes[:10]}")
    
    # 获取PAAD样本（从HEST-Bench/PAAD定义）
    paad_samples = get_paad_samples()
    print(f"\nPAAD样本数: {len(paad_samples)}")
    print(f"PAAD样本列表: {paad_samples}")
    
    # 检查这些样本在hest_data中是否存在
    print(f"\n检查样本在hest_data中的存在情况:")
    available_samples = []
    missing_samples = []
    for sample_id in paad_samples:
        st_file = os.path.join(hest_data_dir, "st", f"{sample_id}.h5ad")
        if os.path.exists(st_file):
            available_samples.append(sample_id)
            print(f"  ✓ {sample_id}: 存在")
        else:
            missing_samples.append(sample_id)
            print(f"  ✗ {sample_id}: 不存在 ({st_file})")
    
    if not available_samples:
        print("\n错误: 没有找到任何PAAD样本在hest_data中")
        return
    
    if missing_samples:
        print(f"\n警告: 以下PAAD样本在hest_data中不存在: {missing_samples}")
    
    # 只检查PAAD样本（在hest_data中存在的）
    all_stats = []
    for sample_id in available_samples:
        stats = check_sample_quality(sample_id, selected_genes)
        if stats:
            all_stats.append(stats)
    
    # 生成汇总报告
    if all_stats:
        output_dir = "/data/yujk/hovernet2feature/Cell2Gene/data_quality_reports"
        generate_summary_report(all_stats, output_dir)
    else:
        print("\n没有成功检查的样本")

if __name__ == "__main__":
    main()

