#!/usr/bin/env python3
"""
Test script for Pearson correlation loss function
"""

import torch
from utils import pearson_correlation_loss

# Test case 1: Perfect correlation (should give low loss)
batch_size = 4
num_genes = 8
predictions = torch.randn(batch_size, num_genes, requires_grad=True)
targets = predictions.clone().detach().requires_grad_(False)

loss = pearson_correlation_loss(predictions, targets)
print(f"Test 1 - Perfect correlation: Loss = {loss.item():.6f} (should be close to 0)")

# Test case 2: Random predictions (should give higher loss)
predictions_random = torch.randn(batch_size, num_genes, requires_grad=True)
targets_random = torch.randn(batch_size, num_genes, requires_grad=False)

loss_random = pearson_correlation_loss(predictions_random, targets_random)
print(f"Test 2 - Random predictions: Loss = {loss_random.item():.6f} (should be higher)")

# Test case 3: Gradient computation (should not error)
try:
    loss_grad = pearson_correlation_loss(predictions, targets)
    loss_grad.backward()
    print(f"Test 3 - Gradient computation: ✓ Gradients computed successfully")
    print(f"         Predictions gradient norm: {predictions.grad.norm().item():.6f}")
except Exception as e:
    print(f"Test 3 - Gradient computation: ✗ Error: {e}")

# Test case 4: Batch with different scales
targets_scaled = targets * 10
loss_scaled = pearson_correlation_loss(predictions, targets_scaled)
print(f"Test 4 - Scaled targets: Loss = {loss_scaled.item():.6f} (should be similar to Test 1)")

print("\nAll tests completed!")
