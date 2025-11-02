#!/usr/bin/env python3
"""
LoRA utilities for injecting low-rank adapters into Linear layers.

author: Jingkun Yu
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable


class LoRALinear(nn.Module):
    """
    A drop-in Linear wrapper that adds a low-rank adapter:  y = W x + scale * (B @ A) x

    - Base weight/bias from the wrapped linear can be optionally frozen
    - Only the low-rank A (in_features x r) and B (r x out_features) are trainable
    """

    def __init__(self, base_linear: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.0, freeze_base: bool = True):
        super().__init__()
        assert isinstance(base_linear, nn.Linear)
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.has_bias = base_linear.bias is not None

        # Copy base layer parameters
        self.weight = nn.Parameter(base_linear.weight.detach().clone())
        if freeze_base:
            self.weight.requires_grad_(False)
        if self.has_bias:
            self.bias = nn.Parameter(base_linear.bias.detach().clone())
            if freeze_base:
                self.bias.requires_grad_(False)
        else:
            self.bias = None

        # LoRA parameters
        self.r = int(r)
        self.alpha = float(alpha)
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else None
        self.scaling = self.alpha / max(self.r, 1)

        if self.r > 0:
            # A: down-projection (in_features -> r), B: up-projection (r -> out_features)
            self.lora_A = nn.Parameter(torch.zeros(self.in_features, self.r))
            self.lora_B = nn.Parameter(torch.zeros(self.r, self.out_features))
            # Initialize following LoRA paper
            nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
            nn.init.zeros_(self.lora_B)
        else:
            # Degenerate case: no adapters
            self.lora_A = None
            self.lora_B = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = x.matmul(self.weight.t())
        if self.bias is not None:
            base_out = base_out + self.bias

        if self.r > 0 and self.lora_A is not None and self.lora_B is not None:
            x_in = self.dropout(x) if self.dropout is not None else x
            # (N, in) @ (in, r) -> (N, r) @ (r, out) -> (N, out)
            lora_out = x_in.matmul(self.lora_A).matmul(self.lora_B) * self.scaling
            return base_out + lora_out
        else:
            return base_out


class LoRAMultiheadSelfAttention(nn.Module):
    """
    A self-attention module that adds LoRA adapters to Q and V projections
    while reusing the original out_proj (which may itself be LoRA-wrapped).

    This module assumes batch_first=True inputs: [batch, seq, embed_dim].
    """

    def __init__(self, base_mha: nn.Module, r: int = 8, alpha: int = 16, dropout: float = 0.0, freeze_base: bool = True):
        super().__init__()
        assert hasattr(base_mha, 'embed_dim') and hasattr(base_mha, 'num_heads')
        self.embed_dim = int(base_mha.embed_dim)
        self.num_heads = int(base_mha.num_heads)
        self.head_dim = self.embed_dim // self.num_heads
        assert self.head_dim * self.num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        # Expose common attributes expected by torch.nn.Transformer
        self.batch_first = True
        self.dropout = float(base_mha.dropout if hasattr(base_mha, 'dropout') else dropout)
        self.attn_dropout = nn.Dropout(self.dropout)
        self.add_zero_attn = False
        self.bias_k = None
        self.bias_v = None
        # Compatibility placeholders for nn.MultiheadAttention interface
        self.in_proj_weight = None
        self.in_proj_bias = None
        self.q_proj_weight = None
        self.k_proj_weight = None
        self.v_proj_weight = None

        # Build separate linear layers for q, k, v from base_mha parameters
        use_combined = hasattr(base_mha, 'in_proj_weight') and base_mha.in_proj_weight is not None
        if use_combined:
            w_q = base_mha.in_proj_weight[:self.embed_dim, :].detach().clone()
            w_k = base_mha.in_proj_weight[self.embed_dim:2 * self.embed_dim, :].detach().clone()
            w_v = base_mha.in_proj_weight[2 * self.embed_dim:, :].detach().clone()
            b_q = None
            b_k = None
            b_v = None
            if getattr(base_mha, 'in_proj_bias', None) is not None:
                b_q = base_mha.in_proj_bias[:self.embed_dim].detach().clone()
                b_k = base_mha.in_proj_bias[self.embed_dim:2 * self.embed_dim].detach().clone()
                b_v = base_mha.in_proj_bias[2 * self.embed_dim:].detach().clone()
        else:
            # Separate weights (PyTorch variant)
            w_q = base_mha.q_proj_weight.detach().clone()
            w_k = base_mha.k_proj_weight.detach().clone()
            w_v = base_mha.v_proj_weight.detach().clone()
            # biases are stored differently; prefer None if unavailable
            b_q = getattr(base_mha, 'bias_q', None)
            b_q = b_q.detach().clone() if b_q is not None else None
            b_k = getattr(base_mha, 'bias_k', None)
            b_k = b_k.detach().clone() if b_k is not None else None
            b_v = getattr(base_mha, 'bias_v', None)
            b_v = b_v.detach().clone() if b_v is not None else None

        # Build base linears
        q_linear = nn.Linear(self.embed_dim, self.embed_dim, bias=b_q is not None)
        q_linear.weight = nn.Parameter(w_q)
        if b_q is not None:
            q_linear.bias = nn.Parameter(b_q)

        k_linear = nn.Linear(self.embed_dim, self.embed_dim, bias=b_k is not None)
        k_linear.weight = nn.Parameter(w_k)
        if b_k is not None:
            k_linear.bias = nn.Parameter(b_k)

        v_linear = nn.Linear(self.embed_dim, self.embed_dim, bias=b_v is not None)
        v_linear.weight = nn.Parameter(w_v)
        if b_v is not None:
            v_linear.bias = nn.Parameter(b_v)

        # Wrap Q and V with LoRA; keep K as base
        self.q_proj = LoRALinear(q_linear, r=r, alpha=alpha, dropout=dropout, freeze_base=freeze_base)
        self.k_proj = k_linear
        if freeze_base:
            self.k_proj.weight.requires_grad_(False)
            if self.k_proj.bias is not None:
                self.k_proj.bias.requires_grad_(False)
        self.v_proj = LoRALinear(v_linear, r=r, alpha=alpha, dropout=dropout, freeze_base=freeze_base)

        # Reuse the existing out_proj (may already be LoRA-wrapped)
        self.out_proj = base_mha.out_proj

    def _shape_projection(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, S, E] -> [B, num_heads, S, head_dim]
        B, S, _ = x.shape
        x = x.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        return x

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                need_weights: bool = False,
                attn_mask: torch.Tensor = None,
                key_padding_mask: torch.Tensor = None,
                average_attn_weights: bool = True,
                is_causal: bool = False):
        # Assume batch_first=True
        B, S, E = query.shape
        assert E == self.embed_dim, "Input embedding dim mismatch"

        # Projections with LoRA on Q and V
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Reshape for multi-head
        q = self._shape_projection(q)
        k = self._shape_projection(k)
        v = self._shape_projection(v)

        # Prefer fused scaled_dot_product_attention when masks allow and weights not requested
        can_use_sdp = (attn_mask is None) and (key_padding_mask is None) and (not need_weights)
        if can_use_sdp:
            context = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=is_causal,
            )
            attn_weights = None
        else:
            scale = (self.head_dim) ** -0.5
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, H, S, S]

            if attn_mask is not None:
                if attn_mask.dim() == 2:
                    attn_scores = attn_scores + attn_mask.view(1, 1, S, S)
                else:
                    attn_scores = attn_scores + attn_mask

            if is_causal:
                causal_mask = torch.ones(S, S, device=attn_scores.device, dtype=torch.bool).tril()
                attn_scores = attn_scores.masked_fill(~causal_mask.view(1, 1, S, S), float('-inf'))

            if key_padding_mask is not None:
                mask = key_padding_mask.view(B, 1, 1, S)
                attn_scores = attn_scores.masked_fill(mask, float('-inf'))

            attn_weights = torch.softmax(attn_scores, dim=-1)
            attn_weights = self.attn_dropout(attn_weights)
            context = torch.matmul(attn_weights, v)  # [B, H, S, D]

        # Merge heads back: [B, S, E]
        context = context.transpose(1, 2).contiguous().view(B, S, E)
        out = self.out_proj(context)

        if need_weights and attn_weights is not None:
            if average_attn_weights:
                # Average over heads -> [B, S, S]
                return out, attn_weights.mean(dim=1)
            else:
                # Return per-head weights -> [B, H, S, S]
                return out, attn_weights
        else:
            return out, None

def _set_module_by_name(root: nn.Module, name: str, new_module: nn.Module):
    """Replace a submodule on a root module by its dotted path name."""
    parts = name.split('.')
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_module)


def apply_lora_to_linear_modules(model: nn.Module,
                                 match_fn: Callable[[str, nn.Module], bool],
                                 r: int = 8,
                                 alpha: int = 16,
                                 dropout: float = 0.0,
                                 freeze_base: bool = True) -> int:
    """
    Wrap matched nn.Linear modules with LoRALinear.

    Returns the number of modules wrapped.
    """
    replaced = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and match_fn(name, module):
            lora_layer = LoRALinear(module, r=r, alpha=alpha, dropout=dropout, freeze_base=freeze_base)
            _set_module_by_name(model, name, lora_layer)
            replaced += 1
    return replaced


