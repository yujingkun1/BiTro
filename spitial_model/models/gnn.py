#!/usr/bin/env python3
"""
Graph Neural Network Models for Cell2Gene

author: Jingkun Yu
"""

import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")

try:
    import torch_geometric
    from torch_geometric.nn import GATConv, GCNConv
    GNN_AVAILABLE = True
    print(f"✓ PyTorch Geometric version: {torch_geometric.__version__}")
except ImportError as e:
    GNN_AVAILABLE = False
    print(f"警告: PyTorch Geometric不可用: {e}")


class StaticGraphGNN(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, gnn_type='GAT', dropout=0.3):
        super(StaticGraphGNN, self).__init__()
        self.num_layers = num_layers
        self.gnn_type = gnn_type
        self.dropout_rate = dropout
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        if gnn_type == 'GAT':
            self.convs.append(GATConv(input_dim, hidden_dim, heads=4, concat=True, dropout=dropout))
            current_dim = hidden_dim * 4
        else:
            self.convs.append(GCNConv(input_dim, hidden_dim))
            current_dim = hidden_dim
            
        self.norms.append(nn.LayerNorm(current_dim))
        
        for _ in range(num_layers - 2):
            if gnn_type == 'GAT':
                self.convs.append(GATConv(current_dim, hidden_dim, heads=4, concat=True, dropout=dropout))
                current_dim = hidden_dim * 4
            else:
                self.convs.append(GCNConv(current_dim, hidden_dim))
                current_dim = hidden_dim
            self.norms.append(nn.LayerNorm(current_dim))
        
        if num_layers > 1:
            if gnn_type == 'GAT':
                self.convs.append(GATConv(current_dim, output_dim, heads=1, concat=False, dropout=dropout))
            else:
                self.convs.append(GCNConv(current_dim, output_dim))
        
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        
        
    def forward(self, x, edge_index, batch=None):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = self.norms[i](x)
                x = self.activation(x)
                x = self.dropout(x)
            
        return x
