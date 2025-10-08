#!/usr/bin/env python3
"""
Transformer Model for Cell2Gene Spatial Gene Expression Prediction

author: Jingkun Yu
"""

import torch
import torch.nn as nn
from .gnn import StaticGraphGNN, GNN_AVAILABLE


class StaticGraphTransformerPredictor(nn.Module):
    """
    Combined GNN + Transformer model for spatial gene expression prediction
    """
    
    def __init__(self, 
                 input_dim=128,  
                 gnn_hidden_dim=128,
                 gnn_output_dim=128,  
                 embed_dim=256,
                 num_genes=18080,
                 num_layers=3,
                 nhead=8,
                 dropout=0.1,
                 use_gnn=True,
                 gnn_type='GAT',
                 n_pos=128):  
        
        super(StaticGraphTransformerPredictor, self).__init__()
        self.use_gnn = use_gnn and GNN_AVAILABLE
        self.embed_dim = embed_dim
        
        if self.use_gnn:
            self.gnn = StaticGraphGNN(
                input_dim=input_dim,
                hidden_dim=gnn_hidden_dim,
                output_dim=gnn_output_dim,
                num_layers=2,
                gnn_type=gnn_type
            )
            transformer_input_dim = gnn_output_dim
        else:
            transformer_input_dim = input_dim
            print("warning: not use GNN module")
        
        # projection
        self.feature_projection = nn.Linear(transformer_input_dim, embed_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # output: using an mlp 
        self.output_projection = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_genes),
            nn.Softplus()  # 添加 Softplus 激活确保输出非负且数值稳定
        )
        
        # positional encoding
        self.n_pos = n_pos
        self.x_embed = nn.Embedding(n_pos, embed_dim)
        self.y_embed = nn.Embedding(n_pos, embed_dim)
        self.embed_dim = embed_dim
    
    def generate_spatial_pos_encoding(self, positions, embed_dim):
        batch_size = positions.shape[0]
        
        # 将连续坐标归一化到[0, n_pos-1]范围
        # 使用min-max归一化避免坐标超出范围
        x_coords = positions[:, 0]
        y_coords = positions[:, 1]
        
        # 计算坐标的范围并归一化
        if x_coords.numel() > 0:
            x_min, x_max = x_coords.min(), x_coords.max()
            y_min, y_max = y_coords.min(), y_coords.max()
            
            # 防止除零错误
            x_range = x_max - x_min if x_max > x_min else 1.0
            y_range = y_max - y_min if y_max > y_min else 1.0
            
            # 归一化到[0, n_pos-1]
            x_normalized = ((x_coords - x_min) / x_range * (self.n_pos - 1)).long()
            y_normalized = ((y_coords - y_min) / y_range * (self.n_pos - 1)).long()
            
            # 确保索引在有效范围内
            x_indices = torch.clamp(x_normalized, 0, self.n_pos - 1)
            y_indices = torch.clamp(y_normalized, 0, self.n_pos - 1)
        else:
            # 如果没有坐标数据，使用零索引
            x_indices = torch.zeros(batch_size, dtype=torch.long, device=positions.device)
            y_indices = torch.zeros(batch_size, dtype=torch.long, device=positions.device)
        
        # 获取x和y的嵌入
        x_emb = self.x_embed(x_indices)  # [batch_size, embed_dim]
        y_emb = self.y_embed(y_indices)  # [batch_size, embed_dim]
        
        pos_enc = x_emb + y_emb  # [batch_size, embed_dim]
        
        return pos_enc
        
    def forward(self, batch_graphs, return_attention=False):
        device = next(self.parameters()).device
        batch_outputs = []
        
        for graph in batch_graphs:
            if graph is None or graph.x.shape[0] == 0:
                # 空图的情况，返回零向量
                zero_output = torch.zeros(1, self.output_projection[-2].out_features, device=device, requires_grad=True)
                batch_outputs.append(zero_output)
                continue
            
            # 将图数据移到正确设备
            graph = graph.to(device)
            
            # GNN处理
            if self.use_gnn and hasattr(graph, 'edge_index') and graph.edge_index.shape[1] > 0:
                node_features = self.gnn(graph.x, graph.edge_index)
            else:
                node_features = graph.x
            
            # 投影到Transformer维度
            node_features = self.feature_projection(node_features)  # [num_nodes, embed_dim]
            
            # HIST2ST风格的空间位置编码（基于细胞坐标）
            if hasattr(graph, 'pos') and graph.pos is not None:
                # 使用HIST2ST风格的位置编码（数值稳定）
                spatial_pos_enc = self.generate_spatial_pos_encoding(graph.pos, node_features.shape[1])
                node_features = node_features.unsqueeze(0) + spatial_pos_enc.unsqueeze(0)
            else:
                # 备用：创建默认位置（如果没有pos属性）
                num_nodes = node_features.shape[0]
                # 使用规律分布的位置而非随机位置
                default_pos = torch.stack([
                    torch.arange(num_nodes, device=node_features.device).float(),
                    torch.zeros(num_nodes, device=node_features.device)
                ], dim=1)
                spatial_pos_enc = self.generate_spatial_pos_encoding(default_pos, node_features.shape[1])
                node_features = node_features.unsqueeze(0) + spatial_pos_enc.unsqueeze(0)
            
            # Transformer处理
            transformer_output = self.transformer(node_features)  # [1, seq_len, embed_dim]
            
            # 先根据细胞的特征预测细胞表达，再对细胞的基因表达进行求和
            cell_gene_predictions = self.output_projection(transformer_output)  # [1, seq_len, num_genes]
            gene_prediction = cell_gene_predictions.sum(dim=1)  # [1, num_genes]
            batch_outputs.append(gene_prediction)
        
        # 合并批次输出
        if batch_outputs:
            return torch.cat(batch_outputs, dim=0)  # [batch_size, num_genes]
        else:
            # 全空的批次
            return torch.zeros(len(batch_graphs), self.output_projection[-2].out_features, device=device, requires_grad=True)