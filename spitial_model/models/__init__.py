#!/usr/bin/env python3
"""
Model package initialization

author: Jingkun Yu
"""

from .gnn import StaticGraphGNN
from .transformer import StaticGraphTransformerPredictor

__all__ = ['StaticGraphGNN', 'StaticGraphTransformerPredictor']