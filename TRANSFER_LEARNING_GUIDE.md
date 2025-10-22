# Cell2Gene 迁移学习指南 (Transfer Learning Guide)

## 概述

本指南说明如何使用 **BulkModel** 的预训练权重来训练 **SpatialModel**，实现迁移学习 (Transfer Learning)。

---

## 核心功能

### 1. **三种迁移学习策略**

#### 策略 A: 完整微调 (Full Fine-tuning) - 推荐用于数据充足场景
```
Transfer Strategy: "full"
- 加载所有兼容的 BulkModel 权重
- 所有层都参与训练
- 学习率: 1e-6 (保守)
```

#### 策略 B: 冻结骨干网络 (Frozen Backbone) - 推荐用于数据较少场景
```
Transfer Strategy: "frozen_backbone"
- 加载所有兼容的 BulkModel 权重
- 冻结: GNN + Feature Projection 层
- 训练: Transformer + Output Projection 层
- 优点: 收敛快，参数更新少，防止过拟合
```

#### 策略 C: 冻结编码器 (Frozen Encoder) - 实验性
```
Transfer Strategy: "frozen_decoder"
- 加载所有兼容的 BulkModel 权重
- 冻结: GNN + Transformer 层
- 训练: Feature Projection + Output Projection 层
- 使用场景: 基因表达预测头需要特殊调整
```

---

## 使用方法

### 方法 1: 使用专门的迁移学习脚本 (推荐)

```bash
cd /data/yujk/hovernet2feature/Cell2Gene

# 基本使用 - 完整微调策略
python spitial_model/train_transfer_learning.py

# 使用冻结骨干网络策略
TRANSFER_STRATEGY=frozen_backbone python spitial_model/train_transfer_learning.py

# 使用 Leave-One-Out 交叉验证
CV_MODE=loo python spitial_model/train_transfer_learning.py

# 指定特定样本进行 LOO
CV_MODE=loo LOO_HELDOUT="sample1,sample2" python spitial_model/train_transfer_learning.py
```

### 方法 2: 在原有训练脚本中启用迁移学习

修改 `spitial_model/train.py` 中的参数:

```bash
cd /data/yujk/hovernet2feature/Cell2Gene

# 启用迁移学习
USE_TRANSFER_LEARNING=true python spitial_model/train.py

# 启用迁移学习 + 冻结骨干
USE_TRANSFER_LEARNING=true FREEZE_BACKBONE=true python spitial_model/train.py

# 禁用迁移学习 (从头开始训练)
USE_TRANSFER_LEARNING=false python spitial_model/train.py
```

---

## 环境变量配置

### 核心参数

| 变量名 | 说明 | 默认值 | 可选值 |
|--------|------|--------|--------|
| `TRANSFER_STRATEGY` | 迁移学习策略 | `full` | `full`, `frozen_backbone` |
| `USE_TRANSFER_LEARNING` | 是否启用迁移学习 | `true` | `true`, `false` |
| `FREEZE_BACKBONE` | 是否冻结骨干网络 | `false` | `true`, `false` |
| `CV_MODE` | 交叉验证模式 | `kfold` | `kfold`, `loo` |
| `LOO_HELDOUT` | Leave-One-Out 指定样本 | 无 | 样本名称,逗号分隔 |

---

## 权重加载机制

### 自动处理

脚本会自动:

1. ✓ 检查 BulkModel 权重文件是否存在
2. ✓ 加载兼容的权重层
3. ✓ 自动跳过因形状不匹配的层
4. ✓ 报告加载统计信息

### 权重加载示例输出

```
=== Loading Pretrained Weights from Bulk Model ===
Bulk model path: /data/yujk/hovernet2feature/best_bulk_static_372_optimized_model.pt
Loaded state dict with 125 keys

✓ Successfully loaded 78 weight layers from bulkmodel
⚠ Skipped 2 layers due to shape mismatch:
    output_projection.1.weight: model (512, 256) vs bulk (512, 372)
    output_projection.1.bias: model (512,) vs bulk (512,)
⚠ Skipped 45 layers not present in spatial model
```

---

## 模型架构对比

### BulkModel 架构
```
输入 (特征维度: 128)
  ↓
[GNN: GAT] → 输出维度: 128
  ↓
[Feature Projection] → embed_dim: 256
  ↓
[Transformer] → 3层, 8个头
  ↓
[Output Projection] → 372个基因
```

### SpatialModel 架构
```
输入 (特征维度: 128)
  ↓
[GNN: GAT] → 输出维度: 128 (← BulkModel 权重)
  ↓
[Feature Projection] → embed_dim: 256 (← BulkModel 权重)
  ↓
[Transformer] → 2层, 8个头 (← 部分 BulkModel 权重)
  ↓
[Output Projection] → 897个基因 (新随机初始化)
```

### 权重兼容性

| 层 | 兼容性 | 说明 |
|----|--------|------|
| GNN 模块 | ✓ 完全兼容 | 输入/输出维度相同 |
| Feature Projection | ✓ 完全兼容 | 维度完全匹配 (128→256) |
| Transformer | ✓ 部分兼容 | 层数可能不同 (3 vs 2) |
| Output Projection | ✗ 不兼容 | 输出基因数不同 (372 vs 897) |
| Positional Encoding | ✗ 不兼容 | 编码方式不同 |

---

## 性能建议

### 推荐的学习率配置

```
迁移学习 + 完整微调:      lr = 1e-6  (保守)
迁移学习 + 冻结骨干:      lr = 1e-5  (正常)
从头开始训练:             lr = 1e-4  (较高)
```

### 超参数建议

```yaml
迁移学习配置:
  batch_size: 16
  num_epochs: 70
  learning_rate: 1e-6
  weight_decay: 1e-5
  early_stopping_patience: 5
  early_stopping_min_delta: 1e-6
```

---

## 输出和结果

### 结果目录结构

```
Cell2Gene/
├── log_normalized_transfer_full/           # "full" 策略结果
│   ├── temp_fold_results.json
│   ├── final_10fold_results.json
│   ├── fold_0_metrics.json
│   ├── fold_0_predictions.npy
│   └── fold_0_training_curve.png
│
├── log_normalized_transfer_frozen_backbone/ # "frozen_backbone" 策略结果
│   ├── temp_fold_results.json
│   ├── final_10fold_results.json
│   └── ...
│
├── checkpoints_transfer_full/
│   ├── best_hest_graph_model_fold_0.pt
│   ├── best_hest_graph_model_fold_1.pt
│   └── ...
└── ...
```

### 关键输出指标

```
=== Final Summary ===
Average overall correlation: 0.756234 ± 0.045123
Average gene correlation: 0.654321 ± 0.032456
Average final test loss: 0.123456 ± 0.012345
```

---

## 常见问题 (FAQ)

### Q1: 我应该使用哪种迁移学习策略?

**A:** 
- 数据充足 (>100 样本): 使用 `frozen_backbone` 或 `full`
- 数据较少 (<50 样本): 优先使用 `frozen_backbone`
- 测试新想法: 使用 `full` 获得最大灵活性

### Q2: 权重未能完全加载怎么办?

**A:** 这是正常的! 因为:
- Output Projection 层的输出维度不同 (372 vs 897)
- Positional Encoding 方式不同
- 某些层在 SpatialModel 中不存在

脚本会自动跳过这些层，用随机初始化的权重代替。

### Q3: 如何比较迁移学习 vs 从头开始训练?

**A:** 运行两个实验:
```bash
# 迁移学习
TRANSFER_STRATEGY=frozen_backbone python spitial_model/train_transfer_learning.py

# 从头开始
USE_TRANSFER_LEARNING=false python spitial_model/train.py

# 对比 log_normalized_transfer_frozen_backbone/ 和 log_normalized/ 目录的结果
```

### Q4: 可以混合使用两个脚本吗?

**A:** 不推荐。使用专门脚本 `train_transfer_learning.py` 更清晰，或在原脚本中设置环境变量。

### Q5: 冻结骨干网络会有什么影响?

**A:**
- ✓ 优点: 训练速度快，显存占用少，防止过拟合
- ✓ 优点: 只需优化 2-3 个新层
- ✗ 缺点: 灵活性下降，可能无法完全适应 Spatial 数据
- 建议: 先用冻结策略快速验证，再用完整微调获得最优结果

---

## 故障排除

### 问题: "Error loading bulkmodel weights: ..."

**解决方案:**
1. 检查权重文件路径是否正确
2. 确认权重文件完整 (检查文件大小 > 100MB)
3. 尝试重新加载: `torch.load(path, map_location='cpu')`

### 问题: "No trainable parameters!"

**解决方案:**
1. 不要同时冻结所有层
2. 确认 `FREEZE_BACKBONE=true` 时不是在冻结所有参数

### 问题: 训练速度很慢

**解决方案:**
1. 使用 `frozen_backbone` 策略减少计算量
2. 增加 `batch_size` (如硬件允许)
3. 检查 GPU 利用率: `nvidia-smi`

---

## 技术细节

### 权重加载算法

```python
for key, value in bulk_state_dict.items():
    if key in model_state_dict:  # 键存在?
        if model_state_dict[key].shape == value.shape:  # 形状相同?
            model_state_dict[key] = value  # 加载权重
        else:
            mismatched_keys.append(key)  # 记录不匹配
    else:
        skipped_keys.append(key)  # 记录多余的键
```

### 冻结机制

```python
if freeze_backbone:
    # 冻结预训练部分
    for param in model.gnn.parameters():
        param.requires_grad = False
    for param in model.feature_projection.parameters():
        param.requires_grad = False
    
    # 保持下游层可训练
    for param in model.transformer.parameters():
        param.requires_grad = True
    for param in model.output_projection.parameters():
        param.requires_grad = True
```

---

## 相关文献

迁移学习在基因表达预测中的应用:

1. Devlin et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers
2. He et al. (2016). Deep Residual Learning for Image Recognition
3. Yosinski et al. (2014). How Transferable are Features in Deep Neural Networks?

---

## 更新日志

### v1.0 (当前版本)
- ✓ 支持三种迁移学习策略
- ✓ 自动权重加载和形状处理
- ✓ 冻结骨干网络支持
- ✓ 10-fold 和 LOO 交叉验证
- ✓ 详细的权重加载统计

---

## 联系支持

如有问题，请联系: Jingkun Yu


