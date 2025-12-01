#!/usr/bin/env python3
"""
Transformer Model for Cell2Gene Spatial Gene Expression Prediction

author: Jingkun Yu
"""

import torch
import math
import torch.nn as nn
from .gnn import StaticGraphGNN, GNN_AVAILABLE
from .lora import apply_lora_to_linear_modules, LoRALinear, LoRAMultiheadSelfAttention
from torch_geometric.data import Batch


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
                 n_pos=128,
                 use_lora: bool = True,
                 lora_r: int = 8,
                 lora_alpha: int = 16,
                 lora_dropout: float = 0.05,
                 lora_freeze_base: bool = True):  
        
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
        
        # output (legacy): per-node to G, then sum. Kept for shape references but not used in forward
        self.output_projection = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_genes),
            nn.Softplus()
        )

        # Gene-specific attention readout: learn queries for each gene and a shared scalar head
        self.gene_queries = nn.Parameter(torch.empty(num_genes, embed_dim))
        nn.init.xavier_uniform_(self.gene_queries)
        self.gene_readout = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1),
            nn.Softplus()
        )
        
        # positional encoding
        self.n_pos = n_pos
        self.x_embed = nn.Embedding(n_pos, embed_dim)
        self.y_embed = nn.Embedding(n_pos, embed_dim)
        self.embed_dim = embed_dim

        # Apply LoRA to selected linear modules to reduce trainable parameters
        if use_lora:
            def match_fn(name: str, module: nn.Module) -> bool:
                # target: feature_projection; attention out_proj; output_projection Linear
                # FFN linear1/linear2 are handled explicitly below for robustness
                if name.endswith('feature_projection'):
                    return True
                if name.endswith('self_attn.out_proj'):
                    return True
                if name.startswith('output_projection') and isinstance(module, nn.Linear):
                    return True
                return False

            wrapped = apply_lora_to_linear_modules(
                self,
                match_fn=match_fn,
                r=lora_r,
                alpha=lora_alpha,
                dropout=lora_dropout,
                freeze_base=lora_freeze_base,
            )

            # Explicitly wrap Transformer FFN layers (linear1/linear2) to ensure coverage across PyTorch versions
            ffn_wrapped = 0
            if hasattr(self, 'transformer') and hasattr(self.transformer, 'layers'):
                for layer in getattr(self.transformer, 'layers', []):
                    for attr in ('linear1', 'linear2'):
                        lin = getattr(layer, attr, None)
                        if isinstance(lin, nn.Linear):
                            setattr(layer, attr, LoRALinear(
                                lin,
                                r=lora_r,
                                alpha=lora_alpha,
                                dropout=lora_dropout,
                                freeze_base=lora_freeze_base,
                            ))
                            ffn_wrapped += 1

            # Replace self-attention with LoRA-enabled Q and V projections while
            # reusing (possibly LoRA-wrapped) out_proj
            attn_lora_wrapped = 0
            if hasattr(self, 'transformer') and hasattr(self.transformer, 'layers'):
                for layer in getattr(self.transformer, 'layers', []):
                    attn = getattr(layer, 'self_attn', None)
                    if attn is not None:
                        setattr(layer, 'self_attn', LoRAMultiheadSelfAttention(
                            attn,
                            r=lora_r,
                            alpha=lora_alpha,
                            dropout=lora_dropout,
                            freeze_base=lora_freeze_base,
                        ))
                        attn_lora_wrapped += 1

            total_wrapped = wrapped + ffn_wrapped
            print(f"✓ LoRA applied to {total_wrapped} linear modules (FFN added: {ffn_wrapped}) (r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout})")
            print(f"✓ LoRA added to attention Q/V on {attn_lora_wrapped} layers")
    
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
        
    def forward(self, batch_graphs, return_attention=False, return_node_embeddings: bool = False):
        device = next(self.parameters()).device
        if not isinstance(batch_graphs, (list, tuple)) or len(batch_graphs) == 0:
            return torch.zeros(0, self.output_projection[-2].out_features, device=device)
        
        # 第一步：收集所有有效图并处理GNN部分（保持原样，逐图处理）
        valid_graphs_data = []
        valid_indices = []
        for idx, g in enumerate(batch_graphs):
            if g is None:
                continue
            gx = g.x if hasattr(g, 'x') else None
            if gx is None or gx.numel() == 0:
                continue
            edge_index = getattr(g, 'edge_index', None)
            if edge_index is not None:
                edge_index = edge_index.to(device)
            gx = gx.to(device)

            # GNN 编码（单图，保持原样）
            if self.use_gnn and edge_index is not None and edge_index.numel() > 0:
                node_features = self.gnn(gx, edge_index)
            else:
                node_features = gx

            # 线性投影到 Transformer 维度
            node_features = self.feature_projection(node_features)  # [S, E]

            # 空间位置编码（单图 min-max 归一化）
            pos = getattr(g, 'pos', None)
            if pos is None:
                S = node_features.size(0)
                pos = torch.stack([
                    torch.arange(S, device=device, dtype=torch.float32),
                    torch.zeros(S, device=device, dtype=torch.float32)
                ], dim=1)
            else:
                pos = pos.to(device)

            pos_enc = self.generate_spatial_pos_encoding(pos, node_features.size(1))

            # 保存序列和原始索引
            seq = node_features + pos_enc  # [S, E]
            valid_graphs_data.append({
                'seq': seq,
                'seq_len': seq.size(0),
                'orig_idx': idx
            })
            valid_indices.append(idx)
        
        if not valid_graphs_data:
            if return_node_embeddings:
                return (
                    torch.zeros(0, self.output_projection[-2].out_features, device=device),
                    [],
                    []
                )
            return torch.zeros(0, self.output_projection[-2].out_features, device=device)
        
        # 第二步：批量处理Transformer部分（这是关键优化）
        # 找到最大序列长度
        max_seq_len = max(item['seq_len'] for item in valid_graphs_data)
        batch_size = len(valid_graphs_data)
        embed_dim = valid_graphs_data[0]['seq'].size(1)
        
        # 创建padded batch
        padded_seqs = torch.zeros(batch_size, max_seq_len, embed_dim, device=device)
        
        for i, item in enumerate(valid_graphs_data):
            seq_len = item['seq_len']
            padded_seqs[i, :seq_len, :] = item['seq']
        
        # 批量传入Transformer（这是性能提升的关键！）
        # Note: We don't use src_key_padding_mask to avoid NestedTensor compatibility issues
        # The Transformer will process all positions, but we only use valid positions in the output
        transformer_output = self.transformer(padded_seqs)  # [B, max_seq_len, E]
        
        # 第三步：逐图处理后续的attention和readout（因为每个图的序列长度不同）
        predictions = []
        node_embeddings_list = []
        processed_indices = []
        
        for i, item in enumerate(valid_graphs_data):
            seq_len = item['seq_len']
            orig_idx = item['orig_idx']
            
            # 提取该图的Transformer输出（去掉padding部分）
            H = transformer_output[i, :seq_len, :]  # [S, E]

            # Gene-specific attention aggregation over nodes
            # Attention weights per gene over S nodes: softmax(Q H^T / sqrt(E))
            attn_logits = torch.matmul(self.gene_queries, H.transpose(0, 1)) / math.sqrt(self.embed_dim)  # [G, S]
            attn_weights = torch.softmax(attn_logits, dim=1)  # [G, S]
            # Pooled representation for each gene
            Z = torch.matmul(attn_weights, H)  # [G, E]
            # Shared scalar head per gene
            y = self.gene_readout(Z).squeeze(-1)  # [G]
            predictions.append(y)
            if return_node_embeddings:
                node_embeddings_list.append(H)
                processed_indices.append(orig_idx)

        if not predictions:
            if return_node_embeddings:
                return (
                    torch.zeros(0, self.output_projection[-2].out_features, device=device),
                    [],
                    []
                )
            return torch.zeros(0, self.output_projection[-2].out_features, device=device)

        pred_tensor = torch.stack(predictions, dim=0)  # [B, G]
        if return_node_embeddings:
            return pred_tensor, node_embeddings_list, processed_indices
        return pred_tensor