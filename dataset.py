#!/usr/bin/env python3
"""
Dataset Module for Cell2Gene HEST Spatial Gene Expression Prediction

author: Jingkun Yu
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import json
import pickle
import scanpy as sc
import warnings
warnings.filterwarnings("ignore")

try:
    from torch_geometric.data import Data
    GEOMETRIC_AVAILABLE = True
except ImportError:
    GEOMETRIC_AVAILABLE = False
    print("Warning: PyTorch Geometric not available")


class HESTSpatialDataset(Dataset):
    """
    HEST Spatial Transcriptomics Dataset for Cell2Gene prediction
    """
    
    def __init__(self, 
                 hest_data_dir,
                 graph_dir,
                 sample_ids,
                 feature_dim=128, 
                 mode='train', 
                 seed=42,
                 gene_file=None):
        
        self.feature_dim = feature_dim
        self.mode = mode
        self.graph_dir = graph_dir
        self.hest_data_dir = hest_data_dir
        self.sample_ids = sample_ids if isinstance(sample_ids, list) else [sample_ids]
        self.seed = seed
        self.gene_file = gene_file
        
        print(f"=== Initializing HEST Dataset (mode: {mode}) ===")
        print(f"Sample count: {len(self.sample_ids)}")
        print(f"Sample list: {self.sample_ids}")
        
        # 设置随机种子
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # 加载基因映射
        self.load_gene_mapping()
        
        # 加载HEST数据
        self.load_hest_data()
        
        # 加载图数据
        self.load_graph_data()
        
        print(f"=== Dataset initialization complete ===")
        print(f"Total spots: {len(self.all_spots_data)}")
        print(f"Graph files available: {self.graphs_available}")
        print(f"Genes count: {self.num_genes}")
    
    def load_gene_mapping(self):
        """Load intersection gene list"""
        print("=== Loading intersection gene list ===")
        
        # Load intersection gene list
        if self.gene_file is None:
            self.gene_file = "/data/yujk/hovernet2feature/HEST/tutorials/SA_process/common_genes_misc_tenx_zen_897.txt"
        
        # Read intersection gene list
        print(f"Loading intersection gene list from: {self.gene_file}")
        self.intersection_genes = set()
        with open(self.gene_file, 'r') as f:
            for line in f:
                gene = line.strip()
                if gene and not gene.startswith('Efficiently') and not gene.startswith('Total') and not gene.startswith('Detection') and not gene.startswith('Samples'):
                    self.intersection_genes.add(gene)
        
        print(f"Intersection genes count: {len(self.intersection_genes)}")
        
        # HEST data uses gene names directly, no ENS ID conversion needed
        self.selected_genes = list(self.intersection_genes)
        print(f"Final genes count: {len(self.selected_genes)}")
    
    def load_hest_data(self):
        """Load HEST data files directly"""
        print("=== Loading HEST data files ===")
        
        self.hest_data = {}
        self.all_spots_data = []
        
        for sample_id in self.sample_ids:
            try:
                print(f"Loading sample: {sample_id}")
                
                sample_info = {}
                
                # 1. Load AnnData file
                st_file = os.path.join(self.hest_data_dir, "st", f"{sample_id}.h5ad")
                if os.path.exists(st_file):
                    adata = sc.read_h5ad(st_file)
                    sample_info['adata'] = adata
                    print(f"  - AnnData: {adata.n_obs} spots × {adata.n_vars} genes")
                else:
                    print(f"  - Warning: AnnData file not found: {st_file}")
                    continue
                
                # 2. Load metadata
                metadata_file = os.path.join(self.hest_data_dir, "metadata", f"{sample_id}.json")
                if os.path.exists(metadata_file):
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    sample_info['metadata'] = metadata
                    print(f"  - Metadata: {metadata.get('tissue', 'unknown')} tissue")
                else:
                    sample_info['metadata'] = {}
                    print(f"  - Warning: Metadata file not found: {metadata_file}")
                
                # Store sample info
                self.hest_data[sample_id] = sample_info
                
                # Extract spots data
                adata = sample_info['adata']
                for spot_idx in range(adata.n_obs):
                    spot_id = adata.obs.index[spot_idx]
                    # Initialize with zero expression (will be updated later)
                    gene_expression = np.zeros(len(self.selected_genes))
                    
                    self.all_spots_data.append({
                        'sample_id': sample_id,
                        'spot_idx': spot_idx,
                        'spot_id': spot_id,
                        'gene_expression': gene_expression
                    })
                
            except Exception as e:
                print(f"  Error loading {sample_id}: {e}")
                continue
        
        print(f"Successfully loaded {len(self.hest_data)} samples")
        print(f"Total spots loaded: {len(self.all_spots_data)}")
        
        # Update gene expression data with intersection genes
        if self.hest_data:
            print(f"Using specified {len(self.selected_genes)} intersection genes")
            
            for spot_data in self.all_spots_data:
                sample_id = spot_data['sample_id']
                adata = self.hest_data[sample_id]['adata']
                spot_idx = spot_data['spot_idx']
                
                # Get expression data for intersection genes
                available_genes = set(adata.var.index).intersection(self.intersection_genes)
                if available_genes:
                    # Order genes according to specified order
                    ordered_genes = [g for g in self.selected_genes if g in available_genes]
                    gene_mask = adata.var.index.isin(ordered_genes)
                    gene_expression = adata.X[spot_idx, gene_mask]
                    if hasattr(gene_expression, 'toarray'):
                        gene_expression = gene_expression.toarray().flatten()
                    
                    # Ensure consistent dimensions
                    if len(gene_expression) != len(ordered_genes):
                        print(f"Warning: spot {spot_idx} gene expression dimension mismatch: {len(gene_expression)} vs {len(ordered_genes)}")
                        # Pad with zeros to ensure consistent dimensions
                        padded_expression = np.zeros(len(self.selected_genes))
                        padded_expression[:len(gene_expression)] = gene_expression
                        gene_expression = padded_expression
                    
                    # Apply log1p transformation to gene expression data
                    # log1p(x) = log(1+x) is more stable than log(x) for small values
                    gene_expression = np.log1p(gene_expression)
                    
                    spot_data['gene_expression'] = gene_expression
                    spot_data['available_genes'] = ordered_genes
                else:
                    print(f"Warning: spot {spot_idx} has no intersection genes")
                    # Apply log1p transformation even for zero expression
                    spot_data['gene_expression'] = np.log1p(np.zeros(len(self.selected_genes)))
                    spot_data['available_genes'] = self.selected_genes
            
            # Calculate final gene count and verify log transformation
            if self.all_spots_data:
                self.num_genes = len(self.selected_genes)
                self.common_genes = self.selected_genes
                print(f"Final gene count: {self.num_genes}")
                print(f"Expected gene count: {len(self.selected_genes)}")
                
                # Verify log transformation effect
                all_expressions = np.concatenate([spot['gene_expression'] for spot in self.all_spots_data])
                print(f"Gene expression after log1p transformation:")
                print(f"  Range: [{all_expressions.min():.4f}, {all_expressions.max():.4f}]")
                print(f"  Mean: {all_expressions.mean():.4f}, Std: {all_expressions.std():.4f}")
            else:
                self.num_genes = len(self.selected_genes)
                self.common_genes = self.selected_genes
        else:
            print("Error: Failed to load any sample data")
            self.num_genes = 0
            self.common_genes = []
    
    def load_graph_data(self):
        """Load graph data"""
        print("Loading graph data...")
        
        self.graphs_available = 0
        for spot_data in self.all_spots_data:
            sample_id = spot_data['sample_id']
            spot_id = spot_data['spot_id']
            
            # Construct graph file path
            graph_file = os.path.join(self.graph_dir, f"{sample_id}_{spot_id}_graph.pkl")
            
            if os.path.exists(graph_file):
                try:
                    with open(graph_file, 'rb') as f:
                        graph_data = pickle.load(f)
                    spot_data['graph'] = graph_data
                    self.graphs_available += 1
                except Exception as e:
                    print(f"Warning: Failed to load graph for {sample_id}_{spot_id}: {e}")
                    spot_data['graph'] = None
            else:
                spot_data['graph'] = None
        
        print(f"Graph data loaded: {self.graphs_available}/{len(self.all_spots_data)} spots")
    
    def __len__(self):
        return len(self.all_spots_data)
    
    def __getitem__(self, idx):
        spot_data = self.all_spots_data[idx]
        
        # Get gene expression (target)
        spot_expressions = torch.FloatTensor(spot_data['gene_expression'])
        
        # Get graph data
        graph_data = spot_data.get('graph', None)
        if graph_data is not None and GEOMETRIC_AVAILABLE:
            # Convert to PyTorch Geometric Data object
            spot_graph = Data(
                x=torch.FloatTensor(graph_data['node_features']),
                edge_index=torch.LongTensor(graph_data['edge_index']),
                pos=torch.FloatTensor(graph_data.get('positions', [])) if 'positions' in graph_data else None
            )
        else:
            # Create empty graph if no graph data available
            spot_graph = Data(
                x=torch.zeros(1, self.feature_dim),
                edge_index=torch.zeros(2, 0, dtype=torch.long),
                pos=torch.zeros(1, 2)
            )
        
        return {
            'spot_expressions': spot_expressions,
            'spot_graphs': spot_graph,
            'sample_id': spot_data['sample_id'],
            'spot_id': spot_data['spot_id']
        }


def collate_fn_hest_graph(batch):
    """
    Custom collate function for HEST graph data
    """
    spot_expressions = torch.stack([item['spot_expressions'] for item in batch])
    spot_graphs = [item['spot_graphs'] for item in batch]
    sample_ids = [item['sample_id'] for item in batch]
    spot_ids = [item['spot_id'] for item in batch]
    
    return {
        'spot_expressions': spot_expressions,
        'spot_graphs': spot_graphs,
        'sample_ids': sample_ids,
        'spot_ids': spot_ids
    }