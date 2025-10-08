import os
import sys
import json
import pickle
import warnings
import psutil
import gc
import torch

warnings.filterwarnings("ignore", category=FutureWarning, module="torch.cuda.amp")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.loss")


def check_environment_compatibility():
    print("=== 环境兼容性检查 ===")
    python_version = sys.version_info
    required_python = (3, 12, 9)
    if python_version[:3] >= required_python:
        print(f"✅ Python版本: {python_version.major}.{python_version.minor}.{python_version.micro} (要求: {'.'.join(map(str, required_python))})")
    else:
        print(f"⚠️ Python版本过低: {python_version.major}.{python_version.minor}.{python_version.micro} (要求: {'.'.join(map(str, required_python))})")

    try:
        torch_version = torch.__version__
        print(f"✅ PyTorch版本: {torch_version}")
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
            print(f"✅ CUDA版本: {cuda_version}")
            print(f"✅ GPU设备: {gpu_count}个 - {gpu_name}")
            if torch.backends.cudnn.is_available():
                print(f"✅ cuDNN版本: {torch.backends.cudnn.version()}")
            else:
                print("⚠️ cuDNN不可用")
        else:
            print("⚠️ CUDA不可用，将使用CPU模式")
    except Exception as e:
        print(f"❌ PyTorch环境检查失败: {e}")

    dependencies = {
        'numpy': '2.2.4',
        'pandas': '2.2.3',
        'scikit-learn': '1.6.1',
        'matplotlib': '3.10.1',
        'psutil': '7.0.0',
    }
    for package, expected_version in dependencies.items():
        try:
            module = __import__(package)
            actual_version = getattr(module, '__version__', 'Unknown')
            print(f"✅ {package}: {actual_version} (期望: {expected_version})")
        except ImportError:
            print(f"❌ {package}: 未安装")

    print("=== 环境检查完成 ===\n")


def get_memory_usage():
    cpu_memory = psutil.virtual_memory().percent
    gpu_memory = 0
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated()
        max_allocated = torch.cuda.max_memory_allocated()
        if max_allocated > 0:
            gpu_memory = allocated / max_allocated * 100
        else:
            total_memory = torch.cuda.get_device_properties(0).total_memory
            gpu_memory = allocated / total_memory * 100
    return cpu_memory, gpu_memory


def safe_memory_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

import os
import gc
import psutil
import torch


def get_memory_usage():
    """获取当前内存和GPU内存使用情况（与原脚本一致）"""
    cpu_memory = psutil.virtual_memory().percent
    gpu_memory = 0
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated()
        max_allocated = torch.cuda.max_memory_allocated()
        if max_allocated > 0:
            gpu_memory = allocated / max_allocated * 100
        else:
            total_memory = torch.cuda.get_device_properties(0).total_memory
            gpu_memory = allocated / total_memory * 100
    return cpu_memory, gpu_memory


def safe_memory_cleanup():
    """安全的内存清理（与原脚本一致）"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_gene_mapping(gene_list_file, features_file):
    """加载基因映射：从基因名称到ENS ID（从原脚本迁移）"""
    print("=== 加载基因映射 ===")
    target_genes = set()
    with open(gene_list_file, 'r') as f:
        for line in f:
            gene = line.strip()
            if gene and not gene.startswith('Efficiently') and not gene.startswith('Total') and not gene.startswith('Detection') and not gene.startswith('Samples'):
                target_genes.add(gene)
    print(f"目标基因数量: {len(target_genes)}")

    gene_name_to_ens = {}
    if os.path.exists(features_file):
        with open(features_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    ens_id = parts[0]
                    gene_name = parts[1]
                    gene_name_to_ens[gene_name] = ens_id

    selected_ens_genes = []
    for gene_name in target_genes:
        if gene_name in gene_name_to_ens:
            selected_ens_genes.append(gene_name_to_ens[gene_name])

    print(f"成功映射基因数量: {len(selected_ens_genes)}")
    return selected_ens_genes, gene_name_to_ens


