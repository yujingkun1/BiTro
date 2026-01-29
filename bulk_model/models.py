import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import math

try:
    import torch_geometric
    from torch_geometric.nn import GATConv, GCNConv
    GNN_AVAILABLE = True
    print(f"PyTorch Geometric version: {torch_geometric.__version__}")
except ImportError as e:
    GNN_AVAILABLE = False
    print(f"Warning: PyTorch Geometric not available: {e}")

# Import LoRA utilities
try:
    from .models.lora import apply_lora_to_linear_modules, LoRALinear, LoRAMultiheadSelfAttention
except ImportError:
    # Fallback for direct import
    import sys
    import os
    _models_dir = os.path.join(os.path.dirname(__file__), 'models')
    if _models_dir not in sys.path:
        sys.path.insert(0, _models_dir)
    from lora import apply_lora_to_linear_modules, LoRALinear, LoRAMultiheadSelfAttention


class StaticGraphGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, gnn_type='GAT'):
        super(StaticGraphGNN, self).__init__()
        self.num_layers = num_layers
        self.gnn_type = gnn_type

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        if gnn_type == 'GAT':
            self.convs.append(GATConv(input_dim, hidden_dim, heads=4, concat=True, dropout=0.1))
            current_dim = hidden_dim * 4
        else:
            self.convs.append(GCNConv(input_dim, hidden_dim))
            current_dim = hidden_dim

        self.norms.append(nn.LayerNorm(current_dim))

        for _ in range(num_layers - 2):
            if gnn_type == 'GAT':
                self.convs.append(GATConv(current_dim, hidden_dim, heads=4, concat=True, dropout=0.1))
                current_dim = hidden_dim * 4
            else:
                self.convs.append(GCNConv(current_dim, hidden_dim))
                current_dim = hidden_dim
            self.norms.append(nn.LayerNorm(current_dim))

        if num_layers > 1:
            if gnn_type == 'GAT':
                self.convs.append(GATConv(current_dim, output_dim, heads=1, concat=False, dropout=0.1))
            else:
                self.convs.append(GCNConv(current_dim, output_dim))

        self.dropout = nn.Dropout(0.1)
        self.activation = nn.ReLU()

    def forward(self, x, edge_index, batch=None):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = self.norms[i](x)
                x = self.activation(x)
                x = self.dropout(x)
        return x


class OptimizedTransformerPredictor(nn.Module):
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
                 graph_batch_size=32,
                 use_lora: bool = True,
                 lora_r: int = 8,
                 lora_alpha: int = 16,
                 lora_dropout: float = 0.05,
                 lora_freeze_base: bool = True):

        super(OptimizedTransformerPredictor, self).__init__()
        self.use_gnn = use_gnn and GNN_AVAILABLE
        self.embed_dim = embed_dim
        self.graph_batch_size = graph_batch_size

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

        self.feature_projection = nn.Linear(transformer_input_dim, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_projection = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_genes),
            nn.Softplus()
        )

        self.pos_encoding = nn.Parameter(torch.randn(20000, embed_dim) * 0.1)

        # Gene-specific attention readout (match spatial model)
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
        # Control whether forward returns gene-level predictions via attention
        self.return_gene_level = False

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

    def forward_single_graph(self, graph):
        if graph is None or graph.x.shape[0] == 0:
            device = next(self.parameters()).device
            return torch.zeros(1, self.output_projection[-1].out_features, device=device)

        if self.use_gnn and hasattr(graph, 'edge_index') and graph.edge_index.shape[1] > 0:
            node_features = self.gnn(graph.x, graph.edge_index)
        else:
            node_features = graph.x

        node_features = self.feature_projection(node_features)

        seq_len = node_features.shape[0]
        if seq_len <= self.pos_encoding.shape[0]:
            pos_enc = self.pos_encoding[:seq_len].unsqueeze(0)
            node_features = node_features.unsqueeze(0) + pos_enc
        else:
            pos_enc = self.pos_encoding[:seq_len % self.pos_encoding.shape[0]].unsqueeze(0)
            node_features = node_features.unsqueeze(0) + pos_enc

        transformer_output = self.transformer(node_features)
        cell_representations = transformer_output.squeeze(0)
        cell_predictions = self.output_projection(cell_representations)
        return cell_predictions

    def forward_batch_graphs(self, graph_list):
        if not graph_list:
            return []

        device = next(self.parameters()).device

        all_cell_features = []
        all_cell_positions = []
        cell_counts = []
        valid_graphs = []

        for graph in graph_list:
            if graph is None or not hasattr(graph, 'x') or graph.x.shape[0] == 0:
                cell_counts.append(0)
                valid_graphs.append(None)
                continue

            graph = graph.to(device)

            if self.use_gnn and hasattr(graph, 'edge_index') and graph.edge_index.shape[1] > 0:
                def gnn_forward(x, edge_index):
                    return self.gnn(x, edge_index)
                node_features = checkpoint(gnn_forward, graph.x, graph.edge_index, use_reentrant=False)
            else:
                node_features = graph.x

            all_cell_features.append(node_features)
            all_cell_positions.append(graph.pos)
            cell_counts.append(node_features.shape[0])
            valid_graphs.append(graph)

        if not all_cell_features:
            return [torch.zeros(1, self.output_projection[-1].out_features, device=device) for _ in graph_list]

        all_cells = torch.cat(all_cell_features, dim=0)
        all_positions = torch.cat(all_cell_positions, dim=0)
        total_cells = all_cells.shape[0]
        print(f"    批量处理：{len(graph_list)}个图 → {total_cells}个细胞的大序列")

        all_projected = self.feature_projection(all_cells)

        import math
        def generate_spatial_pos_encoding(positions, embed_dim):
            batch_size, coord_dim = positions.shape
            pos_enc = torch.zeros(batch_size, embed_dim, device=positions.device)
            div_term = torch.exp(torch.arange(0, embed_dim//2, 2, device=positions.device).float() * -(math.log(10000.0) / (embed_dim//2)))
            pos_enc[:, 0::4] = torch.sin(positions[:, 0:1] * div_term)
            pos_enc[:, 1::4] = torch.cos(positions[:, 0:1] * div_term)
            pos_enc[:, 2::4] = torch.sin(positions[:, 1:2] * div_term)
            pos_enc[:, 3::4] = torch.cos(positions[:, 1:2] * div_term)
            return pos_enc

        spatial_pos_enc = generate_spatial_pos_encoding(all_positions, all_projected.shape[1])
        pos_enc = spatial_pos_enc.unsqueeze(0)
        all_input = all_projected.unsqueeze(0) + pos_enc

        def transformer_forward(x):
            return self.transformer(x)

        def output_projection_forward(x):
            return self.output_projection(x)

        transformer_output = checkpoint(transformer_forward, all_input, use_reentrant=False)

        # By default we preserve existing behavior: return per-node predictions
        if not getattr(self, "return_gene_level", False):
            all_predictions = checkpoint(output_projection_forward, transformer_output.squeeze(0), use_reentrant=False)

            results = []
            start_idx = 0
            for count in cell_counts:
                if count == 0:
                    results.append(torch.zeros(1, self.output_projection[-1].out_features, device=device))
                else:
                    graph_predictions = all_predictions[start_idx:start_idx + count]
                    results.append(graph_predictions)
                    start_idx += count

            del all_cells, all_positions, all_projected, spatial_pos_enc
            del all_input, transformer_output, all_predictions

            return results
        else:
            # Alternative path: compute gene-level predictions using gene attention
            H_all = transformer_output.squeeze(0)  # [total_cells, E]
            results = []
            start_idx = 0
            for count in cell_counts:
                if count == 0:
                    results.append(torch.zeros(self.gene_queries.size(0), device=device))
                else:
                    H = H_all[start_idx:start_idx + count]  # [S, E]
                    # compute attention logits: [G, S]
                    attn_logits = torch.matmul(self.gene_queries, H.transpose(0, 1)) / math.sqrt(self.embed_dim)
                    attn_weights = torch.softmax(attn_logits, dim=1)  # [G, S]
                    Z = torch.matmul(attn_weights, H)  # [G, E]
                    y = self.gene_readout(Z).squeeze(-1)  # [G]
                    results.append(y)
                    start_idx += count

            del all_cells, all_positions, all_projected, spatial_pos_enc
            del all_input, transformer_output, H_all

            return results

    def forward_raw_features(self, all_cell_features, all_cell_positions):
        if all_cell_features.shape[0] == 0:
            device = next(self.parameters()).device
            return torch.zeros(1, self.output_projection[-1].out_features, device=device)

        device = next(self.parameters()).device
        all_cell_features = all_cell_features.to(device)
        all_cell_positions = all_cell_positions.to(device)

        projected_features = self.feature_projection(all_cell_features)

        import math
        def generate_spatial_pos_encoding(positions, embed_dim):
            batch_size, coord_dim = positions.shape
            pos_enc = torch.zeros(batch_size, embed_dim, device=positions.device)
            div_term = torch.exp(torch.arange(0, embed_dim//2, 2, device=positions.device).float() * -(math.log(10000.0) / (embed_dim//2)))
            pos_enc[:, 0::4] = torch.sin(positions[:, 0:1] * div_term)
            pos_enc[:, 1::4] = torch.cos(positions[:, 0:1] * div_term)
            pos_enc[:, 2::4] = torch.sin(positions[:, 1:2] * div_term)
            pos_enc[:, 3::4] = torch.cos(positions[:, 1:2] * div_term)
            return pos_enc

        spatial_pos_enc = generate_spatial_pos_encoding(all_cell_positions, projected_features.shape[1])
        input_with_pos = projected_features.unsqueeze(0) + spatial_pos_enc.unsqueeze(0)

        transformer_output = self.transformer(input_with_pos)
        all_predictions = self.output_projection(transformer_output.squeeze(0))
        return all_predictions

    def forward_hybrid_patient(self, spot_graphs, all_cell_features, all_cell_positions, has_graphs):
        if has_graphs and len(spot_graphs) > 0:
            return self.forward_batch_graphs(spot_graphs)
        else:
            return [self.forward_raw_features(all_cell_features, all_cell_positions)]

    def forward(self, batch_graphs, return_attention=False):
        return self.forward_batch_graphs(batch_graphs)


