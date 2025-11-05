import numpy as np
import cv2
import json
import scipy.io as sio
import os
import torch
import torchvision.transforms as transforms
import pandas as pd
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from skimage import measure
from multiprocessing import Pool, cpu_count, set_start_method
import multiprocessing as mp
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
import time
import logging
import psutil
import warnings
warnings.filterwarnings("ignore")

# Set multiprocessing start method to 'spawn' for CUDA compatibility
if __name__ == "__main__":
    set_start_method('spawn', force=True)

# 设置HuggingFace镜像（用于中国网络环境）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
print(f"已设置HuggingFace镜像: {os.environ.get('HF_ENDPOINT')}")

# DINOv3模型导入
try:
    import transformers
    from transformers import AutoModel, AutoImageProcessor
    DINOV3_AVAILABLE = True
    print("✓ DINOv3可用")
except ImportError as e:
    DINOV3_AVAILABLE = False
    print(f"错误: 请安装transformers库: pip install transformers")
    print(f"详细错误: {e}")

# Configure logging
logging.basicConfig(filename='feature_extraction.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Verify PyTorch version and CUDA availability
print(f"PyTorch version: {torch.__version__}")
logging.info(f"PyTorch version: {torch.__version__}")
# Remove version constraint to support newer PyTorch
print(f"CUDA available: {torch.cuda.is_available()}")
logging.info(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    try:
        visible = torch.cuda.device_count()
        names = ", ".join([torch.cuda.get_device_name(i) for i in range(visible)])
        print(f"GPUs ({visible}): {names}")
        logging.info(f"GPUs ({visible}): {names}")
    except Exception:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        logging.info(f"GPU: {torch.cuda.get_device_name(0)}")

# Global worker and model cache for multi-GPU inference
_WORKER_RANK = None
_WORKER_DEVICE = None
_MODEL_CACHE = {}

def _init_worker():
    """Initializer for worker processes to assign a fixed CUDA device per worker."""
    global _WORKER_RANK, _WORKER_DEVICE
    try:
        ident = mp.current_process()._identity
        rank = (ident[0] - 1) if ident else 0
    except Exception:
        rank = 0
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if num_gpus > 0:
        dev_index = rank % num_gpus
        _WORKER_DEVICE = f"cuda:{dev_index}"
        try:
            torch.cuda.set_device(dev_index)
        except Exception:
            pass
    else:
        _WORKER_DEVICE = "cpu"
    _WORKER_RANK = rank

def load_image(image_path):
    """Load an image from the given path."""
    original_img = cv2.imread(image_path)
    if original_img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    return cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

def extract_features_for_image(args):
    """Extract raw features for a single image patch without PCA."""
    image_path, json_path, mat_path, model_name, processed_log, batch_size = args
    image_name = os.path.splitext(os.path.basename(image_path))[0]  # e.g., "TCGA-AA-3979-01A-01-TS1..._patch_001"
    wsi_name = os.path.basename(os.path.dirname(image_path))
    
    # Extract patch number from image_name
    patch_number = image_name.split("_patch_")[-1] if "_patch_" in image_name else "unknown"
    
    logging.info(f"Processing patch: {image_name}")
    try:
        # Load image and segmentation map
        original_img = load_image(image_path)
        mat_data = sio.loadmat(mat_path)
        inst_map = mat_data['inst_map']
        
        # Prepare model once per worker/device
        global _MODEL_CACHE, _WORKER_DEVICE
        device = _WORKER_DEVICE if _WORKER_DEVICE else ('cuda:0' if torch.cuda.is_available() else 'cpu')
        if device not in _MODEL_CACHE:
            model, original_feature_dim = get_feature_extractor(model_name)
            model.eval()
            if torch.cuda.is_available() and device.startswith('cuda'):
                model = model.to(device)
            _MODEL_CACHE[device] = (model, original_feature_dim)
        model, original_feature_dim = _MODEL_CACHE[device]
        preprocess = get_preprocess_transform()
        
        # Extract cell features
        cell_features, cell_ids, cell_locations, cell_sizes = [], [], [], []
        unique_ids = np.unique(inst_map)
        unique_ids = unique_ids[unique_ids > 0]
        
        # Batched inference for higher GPU utilization
        batch_inputs = []
        batch_meta_ids = []
        batch_meta_locs = []
        batch_meta_sizes = []

        def flush_batch():
            nonlocal batch_inputs, batch_meta_ids, batch_meta_locs, batch_meta_sizes
            if not batch_inputs:
                return
            batch_tensor = torch.stack(batch_inputs, dim=0)
            if torch.cuda.is_available() and device.startswith('cuda'):
                batch_tensor = batch_tensor.to(device, non_blocking=True)
            with torch.inference_mode():
                if torch.cuda.is_available() and device.startswith('cuda'):
                    with torch.amp.autocast('cuda'):
                        out = model(batch_tensor)
                else:
                    out = model(batch_tensor)
                if isinstance(out, tuple):
                    out = out[0]
                if len(out.shape) == 4:
                    out = out.mean(dim=[2, 3])
                elif len(out.shape) == 3:
                    out = out.mean(dim=1)
                out_np = out.detach().cpu().numpy()
            for i in range(out_np.shape[0]):
                features = out_np[i]
                if np.isnan(features).any() or np.isinf(features).any():
                    features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
                cell_features.append(features)
                cell_ids.append(int(batch_meta_ids[i]))
                cell_locations.append(batch_meta_locs[i])
                cell_sizes.append(batch_meta_sizes[i])
            batch_inputs = []
            batch_meta_ids = []
            batch_meta_locs = []
            batch_meta_sizes = []

        for cell_id in unique_ids:
            cell_mask = (inst_map == cell_id).astype(np.uint8)
            props = measure.regionprops(cell_mask)[0]
            y_min, x_min, y_max, x_max = props.bbox
            centroid_y, centroid_x = props.centroid

            cell_roi = original_img[y_min:y_max, x_min:x_max].copy()
            mask_roi = cell_mask[y_min:y_max, x_min:x_max]
            masked_cell = np.zeros_like(cell_roi)
            masked_cell[mask_roi == 1] = cell_roi[mask_roi == 1]

            if masked_cell.shape[0] < 10 or masked_cell.shape[1] < 10:
                continue

            cell_pil = Image.fromarray(masked_cell)
            input_tensor = preprocess(cell_pil)
            # Keep on CPU for now; move as a stacked batch
            if torch.cuda.is_available() and device.startswith('cuda'):
                input_tensor = input_tensor.pin_memory()
            batch_inputs.append(input_tensor)
            batch_meta_ids.append(int(cell_id))
            batch_meta_locs.append((centroid_x, centroid_y))
            batch_meta_sizes.append((props.area, props.perimeter))

            if len(batch_inputs) >= batch_size:
                flush_batch()

        # Flush remaining
        flush_batch()
        
        # Log processed file
        with open(processed_log, 'a') as f:
            f.write(f"{image_path}\n")
        logging.info(f"Completed patch: {image_name}, extracted {len(cell_features)} cells")
        
    except Exception as e:
        logging.error(f"Error processing {image_path}: {str(e)}")
        print(f"Error processing {image_name}: {str(e)}")
        raise
    
    return wsi_name, image_name, patch_number, cell_features, cell_ids, cell_locations, cell_sizes, original_feature_dim

def get_feature_extractor(model_name):
    """Return a DinoV3 feature extractor model and its feature dimension."""
    if not DINOV3_AVAILABLE:
        raise ImportError("DINOv3不可用，请安装transformers")
    
    # 设置 DINOv3 模型路径
    dinov3_model_path = "/data/hdd1/yujk/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
    dinov3_repo_dir = "/data/hdd1/yujk/dinov3"
    
    if not os.path.exists(dinov3_model_path):
        raise RuntimeError(f"DINOv3模型文件不存在: {dinov3_model_path}")
    
    if not os.path.exists(dinov3_repo_dir):
        raise RuntimeError(f"DINOv3仓库不存在: {dinov3_repo_dir}")
    
    print(f"使用DINOv3仓库: {dinov3_repo_dir}")
    print(f"使用本地DINOv3模型: {dinov3_model_path}")
    
    try:
        # 使用 torch.hub.load 加载 DINOv3 ViT-L/16 模型
        print("使用torch.hub加载DINOv3-ViT-L/16模型...")
        
        dino_model = torch.hub.load(
            dinov3_repo_dir, 
            'dinov3_vitl16',  # DINOv3 ViT-L/16 模型
            source='local',
            weights=dinov3_model_path,  # 使用本地权重
            trust_repo=True
        )
        
        print("✓ 成功使用torch.hub加载DINOv3模型")
        
        # DINOv3-L 的特征维度
        feature_dim = 1024
        
        return dino_model, feature_dim
        
    except Exception as e:
        print(f"torch.hub加载失败: {e}")
        
        # 备选方案：手动实现DINOv3加载
        print("尝试手动加载...")
        try:
            # 直接加载模型文件
            checkpoint = torch.load(dinov3_model_path, map_location='cpu')
            
            # 检查是否是完整的模型对象
            if hasattr(checkpoint, 'forward'):
                # 直接是模型对象
                dino_model = checkpoint
                print("✓ 直接加载模型对象")
            else:
                # 是状态字典，需要先建立模型架构
                print("检测到状态字典，正在建立模型架构...")
                
                # 尝试使用timm或手动建立模型架构
                try:
                    import timm
                    # 使用timm创建DINOv3类似的模型
                    dino_model = timm.create_model(
                        'vit_large_patch16_224',
                        pretrained=False,
                        num_classes=0,  # 只要特征提取
                        global_pool=''
                    )
                    
                    # 尝试加载权重
                    if isinstance(checkpoint, dict):
                        if 'model' in checkpoint:
                            state_dict = checkpoint['model']
                        elif 'state_dict' in checkpoint:
                            state_dict = checkpoint['state_dict']
                        else:
                            state_dict = checkpoint
                    else:
                        state_dict = checkpoint
                    
                    # 尝试加载状态字典
                    missing_keys, unexpected_keys = dino_model.load_state_dict(state_dict, strict=False)
                    print(f"✓ 使用timm加载模型，缺少: {len(missing_keys)}, 意外: {len(unexpected_keys)}")
                    
                except ImportError:
                    print("timm不可用，请安装: pip install timm")
                    raise RuntimeError("无法加载DINOv3模型")
            
            feature_dim = 1024
            return dino_model, feature_dim
            
        except Exception as e2:
            raise RuntimeError(f"所有DINOv3加载方式都失败:\ntorch.hub: {e}\n手动加载: {e2}")

def get_preprocess_transform():
    """Return the preprocessing transformation pipeline for DinoV3."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def process_wsi(wsi_name, wsi_path, json_folder, mat_folder, output_folder, model_name, pca_components, chunk_size, batch_size):
    """Process all patches for a single WSI and save as one Parquet file with PCA."""
    print(f"\n{'='*60}")
    print(f"Processing WSI: {wsi_name}")
    print(f"{'='*60}")
    
    logging.info(f"Processing WSI: {wsi_name}")
    
    # 你的目录结构为:  /.../output/<WSI>/{json, mat}
    # 因此这里拼接到每个 WSI 的 json/mat 子目录
    json_wsi_path = os.path.join(json_folder, wsi_name, 'json')
    mat_wsi_path = os.path.join(mat_folder, wsi_name, 'mat')
    
    if not os.path.isdir(json_wsi_path) or not os.path.isdir(mat_wsi_path):
        logging.warning(f"Missing json or mat folder for {wsi_name}, skipping.")
        print(f"Missing data folders for {wsi_name}, skipping.")
        return
    
    image_files = [f for f in os.listdir(wsi_path) if f.endswith('.png')]
    total_patches = len(image_files)
    print(f"Total patches: {total_patches}")
    
    processed_log = os.path.join(output_folder, "processed_patches.log")
    processed_patches = set()
    if os.path.exists(processed_log):
        with open(processed_log, 'r') as f:
            processed_patches = set(line.strip() for line in f)
    
    args_list = []
    for image_file in image_files:
        image_path = os.path.join(wsi_path, image_file)
        json_path = os.path.join(json_wsi_path, image_file.replace('.png', '.json'))
        mat_path = os.path.join(mat_wsi_path, image_file.replace('.png', '.mat'))
        
        if not os.path.exists(json_path) or not os.path.exists(mat_path):
            logging.warning(f"Missing json or mat for {image_file} in {wsi_name}, skipped.")
            continue
        
        if image_path not in processed_patches:
            args_list.append((image_path, json_path, mat_path, model_name, processed_log, batch_size))
    
    if not args_list:
        print(f"All patches already processed for {wsi_name}")
        logging.info(f"No patches to process for {wsi_name} or all already processed.")
        return
    
    print(f"Processing {len(args_list)} new patches...")
    
    # Process patches in chunks and aggregate data
    all_features = []  # Collect all features across patches for PCA
    cell_ids_all = []
    cell_locations_all = []
    cell_sizes_all = []
    image_names_all = []
    patch_numbers_all = []
    
    total_cells = 0
    for i in range(0, len(args_list), chunk_size):
        chunk_args = args_list[i:i + chunk_size]
        chunk_num = i//chunk_size + 1
        total_chunks = (len(args_list) + chunk_size - 1)//chunk_size
        print(f"\nProcessing chunk {chunk_num}/{total_chunks} ({len(chunk_args)} patches)...")
        
        try:
            # Use one worker per GPU to maximize GPU utilization; fallback to CPU workers if no CUDA
            gpu_workers = torch.cuda.device_count() if torch.cuda.is_available() else 0
            num_workers = gpu_workers if gpu_workers > 0 else min(cpu_count(), 4)
            with Pool(processes=num_workers, initializer=_init_worker) as pool:
                results = list(tqdm(pool.imap(extract_features_for_image, chunk_args), 
                                   total=len(chunk_args), desc="Extracting Features"))
            
            chunk_cells = 0
            for wsi_name_chunk, image_name, patch_number, cell_features, cell_ids, cell_locations, cell_sizes, _ in results:
                all_features.extend(cell_features)
                cell_ids_all.extend(cell_ids)
                cell_locations_all.extend(cell_locations)
                cell_sizes_all.extend(cell_sizes)
                image_names_all.extend([image_name] * len(cell_features))
                patch_numbers_all.extend([patch_number] * len(cell_features))
                chunk_cells += len(cell_features)
            
            total_cells += chunk_cells
            print(f"Chunk {chunk_num} completed: {chunk_cells} cells extracted")
        
        except Exception as e:
            logging.error(f"Error in chunk {chunk_num} of WSI {wsi_name}: {str(e)}")
            print(f"Error in chunk {chunk_num}: {str(e)}")
            continue
    
    print(f"\nTotal cells extracted: {total_cells:,}")
    
    # Apply PCA across all cells in the WSI
    if all_features:
        print(f"Applying PCA (target: {pca_components} components)...")
        n_samples = len(all_features)
        effective_components = min(pca_components, n_samples, 1024)  # 1024 is the original feature dim for DinoV3
        
        if effective_components < 1:
            reduced_features = all_features  # No PCA if too few samples
            explained_variance_ratio = 1.0
            print(f"Insufficient samples for PCA, using original features")
        else:
            pca = PCA(n_components=effective_components)
            reduced_features = pca.fit_transform(np.array(all_features))
            explained_variance_ratio = pca.explained_variance_ratio_.sum()
            
            print(f"PCA completed:")
            print(f"   • Original dimensions: 1024")
            print(f"   • Reduced dimensions: {effective_components}")
            print(f"   • Explained variance ratio: {explained_variance_ratio:.4f} ({explained_variance_ratio*100:.2f}%)")
            
            # Show top components contribution
            top_components = min(10, len(pca.explained_variance_ratio_))
            print(f"   • Top {top_components} components:")
            for i in range(top_components):
                print(f"     PC{i+1}: {pca.explained_variance_ratio_[i]:.4f} ({pca.explained_variance_ratio_[i]*100:.2f}%)")
        
        # Prepare data for DataFrame
        wsi_data = []
        for idx, (image_name, patch_number, feat, cell_id, loc, size) in enumerate(zip(image_names_all, patch_numbers_all, reduced_features, cell_ids_all, cell_locations_all, cell_sizes_all)):
            unique_id = f"{wsi_name}_{image_name}_cell_{cell_id}"
            wsi_data.append([unique_id, image_name, cell_id, loc[0], loc[1], size[0], size[1]] + feat.tolist())
        
        # Save all data for this WSI to a single Parquet file
        feature_cols = [f"feature_{i}" for i in range(effective_components)]
        columns = ['unique_id', 'image_name', 'cell_id', 'x', 'y', 'area', 'perimeter'] + feature_cols
        df = pd.DataFrame(wsi_data, columns=columns)
        output_path = os.path.join(output_folder, f"{wsi_name}_features.parquet")
        df.to_parquet(output_path, engine='pyarrow', compression='snappy', index=False)
        
        print(f"\nResults saved:")
        print(f"   • File: {output_path}")
        print(f"   • Cells: {len(wsi_data):,}")
        print(f"   • Features per cell: {effective_components}")
        print(f"   • File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
        
        logging.info(f"Features for {wsi_name} saved: {len(wsi_data):,} cells, {effective_components} features, variance explained: {explained_variance_ratio:.4f}")
    else:
        logging.info(f"No data to save for WSI {wsi_name}")
        print(f"No data to save for WSI {wsi_name}")

def batch_extract_features(image_folder, json_folder, mat_folder, output_folder, model_name='dinov3', pca_components=128, chunk_size=100, batch_size=64):
    """Process all WSIs and manage the extraction process."""
    print(f"\n{'='*80}")
    print(f"DinoV3 Feature Extraction Pipeline")
    print(f"{'='*80}")
    print(f"Image folder: {image_folder}")
    print(f"JSON folder: {json_folder}")
    print(f"MAT folder: {mat_folder}")
    print(f"Output folder: {output_folder}")
    print(f"Model: {model_name}")
    print(f"PCA components: {pca_components}")
    print(f"Chunk size: {chunk_size}")
    print(f"Batch size (cells): {batch_size}")
    
    os.makedirs(output_folder, exist_ok=True)
    processed_wsi_log = os.path.join(output_folder, "processed_wsi.log")
    
    processed_wsi = set()
    if os.path.exists(processed_wsi_log):
        with open(processed_wsi_log, 'r') as f:
            processed_wsi = set(line.strip() for line in f)
    
    wsi_names = [d for d in os.listdir(image_folder) if os.path.isdir(os.path.join(image_folder, d))]
    total_wsi = len(wsi_names)
    remaining_wsi = [name for name in wsi_names if name not in processed_wsi]
    
    print(f"\nStatus:")
    print(f"   • Total WSIs: {total_wsi}")
    print(f"   • Already processed: {len(processed_wsi)}")
    print(f"   • Remaining: {len(remaining_wsi)}")
    
    if len(processed_wsi) > 0:
        print(f"   • Processed WSIs: {sorted(list(processed_wsi))}")
    
    if len(remaining_wsi) == 0:
        print(f"All WSIs already processed!")
        return
    
    print(f"\nStarting processing...")
    
    for idx, wsi_name in enumerate(remaining_wsi):
        print(f"\nWSI Progress: {idx+1}/{len(remaining_wsi)}")
        
        wsi_path = os.path.join(image_folder, wsi_name)
        process_wsi(wsi_name, wsi_path, json_folder, mat_folder, output_folder, model_name, pca_components, chunk_size, batch_size)
        
        # Mark WSI as processed
        with open(processed_wsi_log, 'a') as f:
            f.write(f"{wsi_name}\n")
    
    print(f"\nAll processing completed!")

if __name__ == "__main__":
    # 统一使用绝对路径，且 json_folder / mat_folder 指向 output 根目录
    image_folder = "/data/hdd1/yujk/hovernet-with-feature-extract/output_patches"
    json_folder = "/data/hdd1/yujk/hovernet-with-feature-extract/output"
    mat_folder = "/data/hdd1/yujk/hovernet-with-feature-extract/output"
    output_folder = "/data/hdd1/yujk/extracted_features_dinov3"
    
    batch_extract_features(
        image_folder=image_folder,
        json_folder=json_folder,
        mat_folder=mat_folder,
        output_folder=output_folder,
        model_name='dinov3',  # Changed to dinov3
        pca_components=128,
        chunk_size=100,
        batch_size=128
    )