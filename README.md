## BiTro: Bidirectional Transfer Learning Enhances Bulk and Spatial Transcriptomics Prediction in Cancer Pathological Images

BiTro is a modular pipeline for **predicting spatial gene expression from histology images**.  
The project supports both **spatial transcriptomics (HEST)** and **bulk RNA‑seq** settings, with optional **Bidirectional Transfer Learning**.

[arXiv](https://arxiv.org/abs/2603.14897)

**Demo data**  
Please download the demo data from:  
https://drive.google.com/drive/folders/1VUQjz7QaVmPJqz8-ZGSb9GakjPRLPbB5?usp=drive_link  
Unzip and place the folder under `BiTro/demo_data`.
The HEST sample id used in ST training are provided in HEST_sample.txt.

After installing the environment, you can directly run `Bulk_pipeline.ipynb` and `ST_pipeline.ipynb` to reproduce the full models. For better performance, increase the number of training epochs in the notebooks (the demo defaults are intentionally small).

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

**Key parameters (spatial)**  
- `--hest_data_dir`: HEST data root  
- `--graph_dir`: spatial graph directory  
- `--features_dir`: spatial feature directory  
- `--gene_file`: target gene list  
- `--cv_mode`: `loo` or `kfold`

---

## 3. Bulk Model Pipeline

The bulk model is trained on WSI‑level graphs to predict bulk RNA‑seq expression and can later be used for transfer learning to spatial models.

### 3.1. Bulk data preprocessing

Recommended order (matches the demo notebook):
1) Extract patches  
2) Run HoverNet segmentation  
3) Extract DINOv3 features  
4) Add cluster labels on the full feature set  
5) Split train/test  

```bash
# 1) Extract patches (WSI -> patch PNGs)
python utils/extract_patches.py

# 2) Run HoverNet (tile mode)
# e.g. hovernet-with-feature-extract/run_tile.sh or run_tile_all.sh

# 3) Extract features (requires HoverNet outputs + patches)
python utils/extract_bulk_features_dinov3.py

# 4) Add cluster labels to all parquet features
python utils/cluster_parquet_features.py

# 5) Split features into train/test
python utils/split_features.py
```


### 3.2. Construct cell graphs for bulk data

```bash
python utils/bulk_graph_construction.py
```

### 3.3. Train the bulk model

```bash
python bulk_model/train.py \
  --graph-data-dir ./demo_data/Graphs/Bulk \
  --gene-list-file ./demo_data/Gene/BRCA.txt \
  --features-file ./demo_data/bulk/features.tsv \
  --tpm-csv-file ./demo_data/bulk/tpm-TCGA-BRCA-1000-million.csv \
  --batch-size 2 \
  --graph-batch-size 128 \
  --num-epochs 60 \
  --learning-rate 1e-4 \
  --weight-decay 1e-5 \
  --use-lora \
  --lora-r 8 \
  --lora-alpha 16 \
  --lora-dropout 0.05 \
  --lora-freeze-base
```

---

## 4. Transfer Learning (Optional)

### 4.1. Bulk → Spatial (transfer to spatial model)

`spitial_model/train.py` supports transfer learning via:

- `use_transfer_learning`: whether to load a bulk checkpoint  
- `bulk_model_path`: path to the pretrained bulk model  
- `freeze_backbone`: whether to freeze backbone layers  

Example (using environment variables):

```bash
export OUTPUT_DIR=./log_normalized_BRCA_transfer
export USE_TRANSFER_LEARNING=true
export BULK_MODEL_PATH=/path/to/bulk_model.pt
export FREEZE_BACKBONE=false

python spitial_model/train.py
```

### 4.2. Spatial → Bulk (transfer to bulk model)

`bulk_model/train.py` supports initializing from a spatial checkpoint:

- `--spatial-model-path`: path to a spatial model checkpoint  
- `--freeze-backbone-from-spatial`: freeze backbone layers when initializing  

Example:

```bash
python bulk_model/train.py \
  --graph-data-dir ./demo_data/Graphs/Bulk \
  --gene-list-file ./demo_data/Gene/BRCA.txt \
  --features-file ./demo_data/bulk/features.tsv \
  --tpm-csv-file ./demo_data/bulk/tpm-TCGA-BRCA-1000-million.csv \
  --spatial-model-path /path/to/spatial_checkpoint.pt \
  --freeze-backbone-from-spatial
```

---
