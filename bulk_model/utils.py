"""
Utility helpers for the bulk model training pipeline.

This module provides:
- Environment and dependency checks
- CPU/GPU memory usage helpers
- Gene list / mapping utilities

Note:
    Some functions are duplicated for backward compatibility with earlier scripts.
    The last definition in the file takes precedence in Python.
"""

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
    """Print a lightweight environment compatibility report."""
    print("=== Environment compatibility check ===")
    python_version = sys.version_info
    required_python = (3, 12, 9)
    if python_version[:3] >= required_python:
        print(f"✓ Python: {python_version.major}.{python_version.minor}.{python_version.micro} (required: {'.'.join(map(str, required_python))})")
    else:
        print(f"Warning: Python version too low: {python_version.major}.{python_version.minor}.{python_version.micro} (required: {'.'.join(map(str, required_python))})")

    try:
        torch_version = torch.__version__
        print(f"✓ PyTorch: {torch_version}")
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
            print(f"✓ CUDA: {cuda_version}")
            print(f"✓ GPU: {gpu_count} device(s) - {gpu_name}")
            if torch.backends.cudnn.is_available():
                print(f"✓ cuDNN: {torch.backends.cudnn.version()}")
            else:
                print("Warning: cuDNN is not available")
        else:
            print("Warning: CUDA is not available; using CPU mode")
    except Exception as e:
        print(f"Error: PyTorch environment check failed: {e}")

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
            print(f"✓ {package}: {actual_version} (expected: {expected_version})")
        except ImportError:
            print(f"Error: {package} is not installed")

    print("=== Environment check complete ===\n")


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
    """Get current CPU and GPU memory usage (percentage).

    Returns:
        A tuple ``(cpu_percent, gpu_percent)`` where:
        - ``cpu_percent`` is the system RAM usage percentage.
        - ``gpu_percent`` is the estimated GPU memory usage percentage.
    """
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
    """Run garbage collection and clear CUDA cache (if available)."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_gene_mapping(gene_list_file, features_file):
    """Load a gene mapping from gene symbol to Ensembl ID.

    Args:
        gene_list_file: Path to a text file containing target gene symbols
            (one per line).
        features_file: Path to a TSV mapping file. The first two columns are
            expected to be ``ens_id`` and ``gene_symbol``.

    Returns:
        A tuple ``(selected_ens_genes, gene_name_to_ens)`` where:
        - ``selected_ens_genes`` is a list of Ensembl IDs corresponding to the
          target symbols found in ``features_file``.
        - ``gene_name_to_ens`` is a dict mapping gene symbol -> Ensembl ID.
    """
    print("=== Loading gene mapping ===")
    target_genes = set()
    with open(gene_list_file, 'r') as f:
        for line in f:
            gene = line.strip()
            if gene and not gene.startswith('Efficiently') and not gene.startswith('Total') and not gene.startswith('Detection') and not gene.startswith('Samples'):
                target_genes.add(gene)
    print(f"Target genes: {len(target_genes)}")

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

    print(f"Mapped genes: {len(selected_ens_genes)}")
    return selected_ens_genes, gene_name_to_ens


