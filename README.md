## Cell2Gene: Spatial Gene Expression Prediction from Histology

Cell2Gene is a modular pipeline for **predicting spatial gene expression from histology images** using **cell-level graphs**, **Graph Neural Networks (GNNs)** and **Transformers**.  
The project supports both **spatial transcriptomics (HEST)** and **bulk RNA‑seq** settings, with optional **transfer learning from bulk models to spatial models**.

---

## 1. Features

- **Spatial model (HEST)**
  - Cell‑level feature extraction from histology using DINOv3.
  - Spatial graph construction (intra‑patch and inter‑patch cell graphs).
  - GNN + Transformer architecture for spot‑level gene expression prediction.
  - Supports **k‑fold** and **leave‑one‑out (LOO)** cross‑validation.

- **Bulk model**
  - Preprocessing pipeline from WSI → patches → cell graphs.
  - Bulk‑level gene expression prediction using graph‑based models.

- **Transfer learning**
  - Load a pretrained bulk model as backbone for spatial models.
  - Flexible configuration via environment variables / command‑line.

- **Evaluation & visualization**
  - Per‑gene and per‑spot Pearson correlation metrics.
  - Training curves, gene‑wise correlation histograms.
  - Aggregated statistics across folds (JSON + plots).

---

## 2. Repository Structure (Key Components)

- **`spitial_model/`**  
  - `train.py`: main training script for the HEST spatial model (supports CV, optional transfer learning).  
  - `dataset.py`: `HESTSpatialDataset`, data loading & normalization.  
  - `models/`: GNN, Transformer, and LoRA‑related model definitions.  
  - `trainer.py`: training loop, optimizer/scheduler setup, early stopping.  
  - `utils.py`: fold sampling, evaluation, plotting, and IO utilities.

- **`bulk_model/`**  
  - `pre_processing/`: WSI selection, patch feature extraction, normalization.  
  - `train.py`: training script for the bulk model.  
  - `dataset.py`, `models.py`, `trainer.py`, `utils.py`: bulk‑side counterparts.

- **`utils/`**
  - `extract_spatial_features_dinov3.py`: extract DINOv3 features for spatial data.  
  - `spatial_graph_construction.py`: build cell graphs for HEST.  
  - `bulk_graph_construction.py`: build cell graphs for bulk RNA‑seq.

- **`infer_spatial_on_bulk/`**
  - `infer.py`: infer spatial‑like predictions on bulk data using trained models.

- **`log_*` / `log_normalized*` / `cscc_normalized/`**  
  Output directories for different experiments (datasets, learning rates, gene sets, etc.), including:
  - Fold‑wise metrics (`fold_x_metrics.txt`, `fold_x_epoch_metrics.json`)  
  - Per‑fold predictions/targets (`.npy`)  
  - Best Pearson summaries (`fold_best_pearsons.json`, `final_loo_best_pearsons.json`)  
  - Plots (training curves, gene correlation histograms, across‑fold Pearson plots)

---

## 3. Installation

Create a conda or virtualenv environment and install the required packages:

```bash
conda create -n cell2gene python=3.10
conda activate cell2gene

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # or CPU / your CUDA version

pip install \
  torch-geometric \
  scanpy \
  pandas \
  numpy \
  scipy \
  scikit-learn \
  matplotlib \
  seaborn \
  tqdm
```

You will also need:

- A working **CellViT** installation (for cell segmentation).
- Pretrained **DINOv3** weights used by `extract_spatial_features_dinov3.py`.

---

## 4. Spatial Model Pipeline (HEST)

The HEST dataset is already preprocessed at the expression level. We only need to normalize histology images (e.g. Vahadane) and extract features before graph construction and training.

### 4.1. Extract cell‑level spatial features

```bash
cd /data/yujk/hovernet2feature/Cell2Gene
python utils/extract_spatial_features_dinov3.py
```

This script:
- Loads preprocessed HEST images.
- Applies stain normalization (Vahadane).
- Extracts DINOv3 features for each cell / patch and stores them to disk.

### 4.2. Construct spatial graphs

```bash
python utils/spatial_graph_construction.py
```

This script:
- Builds intra‑patch and inter‑patch graphs between cells.
- Saves graph structures (edges, node features, metadata) for training.

### 4.3. Train the spatial model

```bash
cd /data/yujk/hovernet2feature/Cell2Gene
python spitial_model/train.py
```

Key notes:
- Cross‑validation mode is controlled via `CV_MODE` (`kfold` or `loo`).
- Results (metrics, curves, Pearson plots) are saved under a chosen `output_dir`
  (see configuration section below).

---

## 5. Bulk Model Pipeline

The bulk model is trained on WSI‑level graphs to predict bulk RNA‑seq expression and can later be used for transfer learning to spatial models.

### 5.1. Bulk data preprocessing

```bash
cd /data/yujk/hovernet2feature/Cell2Gene

python bulk_model/pre_processing/select_wsi.py
python bulk_model/pre_processing/extract_features.py
python bulk_model/pre_processing/patch_normalization.py
```

These scripts:
- Select valid WSIs and corresponding expression profiles.
- Extract patch‑level features for each WSI.
- Normalize patch features for stability.

### 5.2. Cell segmentation using CellViT

```bash
CellViT-inference --config-dir cellVit.yaml
```

This produces cell segmentation masks / instances that will be used for cell‑level graphs.

### 5.3. Extract cell features for bulk WSIs

```bash
python utils/extract_spatial_features_dinov3.py
```

Reuses the DINOv3 extractor to produce features for segmented cells in bulk WSIs.

### 5.4. Construct cell graphs for bulk data

```bash
python utils/bulk_graph_construction.py
```

This script:
- Constructs cell‑level graphs for bulk WSIs.
- Saves graph data and mappings (`bulk_*_all_cell_features.pkl`, `bulk_*_intra_patch_graphs.pkl`, etc.).

### 5.5. Train the bulk model

```bash
cd /data/yujk/hovernet2feature/Cell2Gene
python bulk_model/train.py
```

This will:
- Train the bulk graph model on bulk RNA‑seq targets.
- Save checkpoints under `bulk_BRCA_graphs_checkpoints/` or other configured dirs.

---

## 6. Transfer Learning (Optional)

> Note: You can also train purely from scratch without transfer learning by
> disabling the corresponding flags in `spitial_model/train.py`.

To use a pretrained bulk model as backbone for the spatial model:

1. **Complete spatial feature extraction & graph construction**
   - Sections **4.1** and **4.2** above.
2. **Train and save the bulk model**
   - Section **5.5** above.
3. **Set transfer‑learning configuration and run spatial training with TL**

`spitial_model/train.py` already supports transfer learning via:

- `use_transfer_learning`: whether to load a bulk checkpoint.
- `bulk_model_path`: path to the pretrained bulk model.
- `freeze_backbone` / `TRANSFER_STRATEGY`: control what parts of the model are frozen.

Example (using environment variables):

```bash
cd /data/yujk/hovernet2feature/Cell2Gene

export OUTPUT_DIR=./log_normalized_BRCA_transfer
export USE_TRANSFER_LEARNING=true
export BULK_MODEL_PATH=/data/yujk/hovernet2feature/best_bulk_static_372_optimized_model.pt
export FREEZE_BACKBONE=false        # or true depending on your strategy

python spitial_model/train.py
```

You can also expose these as command‑line arguments if desired (see comments in `spitial_model/train.py`).

---

## 7. Model Architecture (Spatial Model)

The spatial model (in `spitial_model/models/`) follows a **GNN + Transformer** design:

1. **Graph Neural Network (GNN)**  
   Processes cell‑level features with spatial edges (intra‑patch / inter‑patch).

2. **Feature Projection Layer**  
   Maps GNN outputs to the Transformer embedding dimension.

3. **Spatial Positional Encoding**  
   HIST2ST‑style position encoding that injects spatial coordinates.

4. **Transformer Encoder**  
   Several self‑attention layers capture long‑range spatial dependencies.

5. **MLP Output Head**  
   Predicts multi‑gene expression for each spot with Softplus activation to ensure non‑negativity.

The bulk model uses a related graph‑based architecture adapted to WSI‑level graphs.

---

## 8. Configuration & Environment Variables

Key hyperparameters and options for the spatial model are defined in `spitial_model/train.py`:

- **Training**
  - `batch_size`: training batch size (e.g. 128)
  - `learning_rate`: AdamW learning rate (e.g. 1e‑4)
  - `num_epochs`: max epochs (e.g. 60–70)
  - `weight_decay`: weight decay (e.g. 1e‑5)
  - `patience`, `min_delta`: early stopping parameters

- **Data & genes**
  - `hest_data_dir`, `graph_dir`, `features_dir`: paths to HEST data, graphs and features.
  - `gene_file`: subset of genes used for training (e.g. common intersection genes).
  - `use_gene_normalization`: whether to apply per‑gene variance normalization.

- **Cross‑validation**
  - `CV_MODE`: `"kfold"` or `"loo"`.
  - `CV_HELDOUT` / `LOO_HELDOUT`: optional list of samples to hold out in LOO.

- **Transfer learning**
  - `USE_TRANSFER_LEARNING`: `"true"` / `"false"`.
  - `BULK_MODEL_PATH`: path to bulk model checkpoint.
  - `FREEZE_BACKBONE` or `TRANSFER_STRATEGY`: control which modules are trainable.

You can either:
- Edit default values directly in `spitial_model/train.py`, or  
- Use **environment variables** / **command‑line arguments** to override them when launching experiments.

---

## 9. Outputs & Metrics

For each experiment (e.g. under `log_normalized_BRCA_1e-3/` or similar), the code saves:

- **Per‑fold results**
  - `fold_x_metrics.txt`: overall MSE, overall correlation, mean/median gene correlation, spot‑wise metrics.
  - `fold_x_epoch_metrics.json`: epoch‑wise loss and correlation curves.
  - `fold_x_predictions.npy`, `fold_x_targets.npy`: arrays for further analysis.
  - `fold_x_training_curves.png`: loss and Pearson curves.
  - `fold_x_gene_correlation_hist.png`: histogram of gene‑wise correlations.

- **Aggregated results**
  - `final_10fold_results.json` / `final_loo_results.json`: full per‑fold metadata & metrics.
  - `fold_best_pearsons.json` / `final_*_best_pearsons.json`: best overall / gene Pearson per fold and their statistics.
  - `best_overall_pearson_across_folds.png`, `best_gene_pearson_across_folds.png`: across‑fold visualization.

These files allow quick comparison across hyperparameters, gene sets, and model variants.

---

## 10. Requirements Summary

- **Core**
  - PyTorch
  - PyTorch Geometric
  - NumPy, SciPy, pandas
  - scikit‑learn

- **Single‑cell / transcriptomics**
  - scanpy

- **Visualization & utilities**
  - matplotlib
  - seaborn
  - tqdm

Make sure CUDA / GPU drivers are properly configured if you plan to train large models.

---

## 11. Citation

If you use this codebase or ideas in your work, please consider citing the corresponding thesis / paper (to be added once available).

---

## 12. Contact

For questions or collaborations, please contact the author:

- **Author**: Jingkun Yu  
- **Project root**: `/data/yujk/hovernet2feature/Cell2Gene`

Feel free to open issues or propose improvements based on your own datasets and experiments.