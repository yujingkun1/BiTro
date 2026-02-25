## BiTro: Bidirectional Transfer Learning Enhances Bulk and Spatial Transcriptomics Prediction in Cancer Pathological Images

BiTro is a modular pipeline for **predicting spatial gene expression from histology images**.  
The project supports both **spatial transcriptomics (HEST)** and **bulk RNA‑seq** settings, with optional **Bidirectional Transfer Learning**.

---

**The code is currently in the process of being refined and improved. In the future, an ipynb file of detailed whole process will be added.**

---

## 1. Installation

```bash
# clone github repo
git clone https://github.com/yujingkun1/BiTro.git

cd BiTro
conda env create -f environment.yml
conda activate BiTro
```

**Attention**：This environment is based on CUDA 11.8 / PyTorch 2.6(cu124), for different CUDA version, you may need to manually download these packages.

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

pip install torch-geometric
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.5.0+cu121.html
```

---

### Other dependances

- For cell segmentation, please clone **HoverNet** repository and install environment.
- Pretrained **DINOv3** weights and environments used by `extract_spatial_features_dinov3.py`.

---

## 2. Spatial Model Pipeline (HEST)

The HEST dataset is already preprocessed at the expression level. We only need to normalize histology images (e.g. Vahadane) and extract features before graph construction and training.

Firstly, the HEST repository should be cloned to download spatial transcriptomic data.

### 2.1. Extract cell‑level spatial features

```bash
cd BiTro
python utils/extract_spatial_features_dinov3.py
# run this file for cluster label
python utils/cluster_kmeans_features.py 
```

### 2.2. Construct spatial graphs

```bash
python utils/spatial_graph_construction.py
```

### 2.3. Train the spatial model

```bash
python spitial_model/train.py
```
---

## 3. Bulk Model Pipeline

The bulk model is trained on WSI‑level graphs to predict bulk RNA‑seq expression and can later be used for transfer learning to spatial models.

### 3.1. Bulk data preprocessing

First running HoverNet to segment all cells from images.

```bash
python utils/extract_bulk_features_dinov3.py
# run this file for cluster label (change file address)
python utils/cluster_kmeans_features.py
```


### 3.2. Construct cell graphs for bulk data

```bash
python utils/bulk_graph_construction.py
```

### 3.5. Train the bulk model

```bash
python bulk_model/train.py
```

---

## 4. Transfer Learning (Optional)

> Note: You can also train purely from scratch without transfer learning by
> disabling the corresponding flags in `spitial_model/train.py`.

To use a pretrained bulk model as backbone for the spatial model:


`spitial_model/train.py` already supports transfer learning via:

- `use_transfer_learning`: whether to load a bulk checkpoint.
- `bulk_model_path`: path to the pretrained bulk model.
- `freeze_backbone` / `TRANSFER_STRATEGY`: control what parts of the model are frozen.

Example (using environment variables):

```bash

export OUTPUT_DIR=./log_normalized_BRCA_transfer
export USE_TRANSFER_LEARNING=true
export BULK_MODEL_PATH=/data/yujk/hovernet2feature/best_bulk_static_372_optimized_model.pt
export FREEZE_BACKBONE=false        # or true depending on your strategy

python spitial_model/train.py
```

You can also expose these as command‑line arguments if desired (see comments in `spitial_model/train.py`).

---


