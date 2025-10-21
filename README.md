# Cell2Gene: Spatial Gene Expression Prediction

A modular implementation for predicting spatial gene expression from histology images using Graph Neural Networks and Transformers.

## Project Structure



## Usage

### Spatial Model Training

The hest dataset is already preprocessed the data well, so we just need to normalize the image using vahadane method for preprocessing of feature extraction.
#### 1. Extract_spatial_features
```bash
cd /data/yujk/hovernet2feature/Cell2Gene
python utils/extract_spatial_features_dinov3.py
```
#### 2. Graph Construction
```bash
python utils/spatial_graph_construction.py
```
#### 3. Train the Model

```bash
cd /data/yujk/hovernet2feature/Cell2Gene
python train.py
```
### Bulk Model Training
The model trained on bulk RNA-seq data to prepare for transfer learning.
#### Data preprocessing
Using these commands to preprocess the bulk RNA-seq data.
```bash
python bulk_model/pre_processing/select_wsi.py
python bulk_model/pre_processing/extract_features.py
python bulk_model/pre_processing/patch_normalization.py
```
####  Segment cells Using CellViT
```bash
CellViT-inference --config-dir cellVit.yaml
```
####  Extract Cell Features
```bash
python utils/extract_spatial_features_dinov3.py
```
####  Construct Cell Graph of Bulk Data
```bash
python utils/bulk_graph_construction.py
```
#### 4. Train the Bulk Model
```bash
python train_bulk.py
```

### Transfer Learning

Please compelete the step1,2 of Spatial Model Training and all steps of Bulk Model Training before running the transfer learning code.
```bash
python train_transfer.py
```



### Model Architecture

1. **Graph Neural Network**: Processes cell-level features with spatial relationships
2. **Feature Projection**: Maps GNN outputs to transformer dimensions
3. **Spatial Positional Encoding**: HIST2ST-style position encoding
4. **Transformer Encoder**: Captures long-range spatial dependencies
5. **MLP Output**: Predicts gene expression with Softplus activation

### Configuration

Key parameters in `train.py`:
- `batch_size`: Training batch size (default: 16)
- `learning_rate`: AdamW learning rate (default: 3e-6)
- `num_epochs`: Maximum training epochs (default: 60)
- `patience`: Early stopping patience (default: 50)
- `embed_dim`: Transformer embedding dimension (default: 256)

## Requirements

- PyTorch
- PyTorch Geometric
- scanpy
- pandas
- numpy
- scipy
- scikit-learn
- matplotlib
- seaborn
- tqdm