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
from multiprocessing import cpu_count, set_start_method, get_context
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
import time
import logging
import psutil
import warnings
warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# Global caches & resource controls
# --------------------------------------------------------------------------
MODEL_CACHE = {}
PREPROCESS_CACHE = None
WORKER_DEVICE_ID = None
ORIGINAL_FEATURE_DIM = 1024  # DinoV3 ViT-L/16 embedding size


def elevate_process_priority():
    """Attempt to raise the priority of the current process and pin all CPUs."""
    try:
        proc = psutil.Process(os.getpid())
        # Give access to all CPUs (max utilisation)
        if hasattr(proc, "cpu_affinity"):
            cpu_total = psutil.cpu_count() or os.cpu_count()
            if cpu_total:
                proc.cpu_affinity(list(range(cpu_total)))
        # Set highest priority possible on Linux (-20 nice)
        try:
            proc.nice(-20)
        except psutil.AccessDenied:
            # Fallback: try slightly lower priority if we cannot get -20
            proc.nice(-10)
    except Exception as exc:
        print(f"Warning: Unable to elevate process priority: {exc}")


def get_preprocess_cached():
    """Return (and cache) the preprocessing pipeline."""
    global PREPROCESS_CACHE
    if PREPROCESS_CACHE is None:
        PREPROCESS_CACHE = get_preprocess_transform()
    return PREPROCESS_CACHE


def get_model_for_device(model_name, device_id=None):
    """
    Load and cache the DinoV3 model per device.
    device_id: None for CPU, otherwise integer GPU index.
    """
    global MODEL_CACHE
    device_key = "cpu" if device_id is None else f"cuda:{device_id}"
    if device_key in MODEL_CACHE:
        return MODEL_CACHE[device_key]

    model, feature_dim = get_feature_extractor(model_name)
    model.eval()

    if device_id is not None and torch.cuda.is_available():
        torch.cuda.set_device(device_id)
        model = model.to(device_key)

    MODEL_CACHE[device_key] = (model, feature_dim)
    return MODEL_CACHE[device_key]


def worker_initializer(device_queue):
    """Initializer for worker processes to claim a specific GPU device."""
    global WORKER_DEVICE_ID
    elevate_process_priority()
    try:
        WORKER_DEVICE_ID = device_queue.get_nowait()
    except Exception:
        WORKER_DEVICE_ID = None

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
cuda_available = torch.cuda.is_available()
print(f"CUDA available: {cuda_available}")
logging.info(f"CUDA available: {cuda_available}")
if cuda_available:
    gpu_count = torch.cuda.device_count()
    print(f"Detected {gpu_count} CUDA device(s).")
    logging.info(f"Detected {gpu_count} CUDA device(s).")
    for gpu_idx in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(gpu_idx)
        print(f"GPU {gpu_idx}: {gpu_name}")
        logging.info(f"GPU {gpu_idx}: {gpu_name}")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

def load_image(image_path):
    """Load an image from the given path."""
    original_img = cv2.imread(image_path)
    if original_img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    return cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

def extract_features_for_image(args):
    """Extract raw features for a single image patch without PCA."""
    image_path, json_path, mat_path, model_name, processed_log = args
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
        
        # Load pre-trained model
        gpu_enabled = torch.cuda.is_available()
        device_id = WORKER_DEVICE_ID if gpu_enabled else None
        model, original_feature_dim = get_model_for_device(
            model_name, device_id if (gpu_enabled and device_id is not None) else None
        )
        preprocess = get_preprocess_cached()

        CUDA_AVAILABLE = gpu_enabled and device_id is not None
        if CUDA_AVAILABLE:
            device = torch.device(f"cuda:{device_id}")
        else:
            device = torch.device("cpu")
        
        # Extract cell features
        cell_features, cell_ids, cell_locations, cell_sizes = [], [], [], []
        unique_ids = np.unique(inst_map)
        unique_ids = unique_ids[unique_ids > 0]
        
        with torch.no_grad():
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
                input_tensor = preprocess(cell_pil).unsqueeze(0)
                if CUDA_AVAILABLE:
                    input_tensor = input_tensor.to(device, non_blocking=True)
                
                # Extract features using DinoV3
                if CUDA_AVAILABLE:
                    torch.cuda.set_device(device_id)
                    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                        batch_features = model(input_tensor)
                        # 如果返回的是tuple，取第一个元素
                        if isinstance(batch_features, tuple):
                            batch_features = batch_features[0]
                        # 如果是4D tensor (batch, channels, height, width)，取global average pooling
                        if len(batch_features.shape) == 4:
                            batch_features = batch_features.mean(dim=[2, 3])  # Global average pooling
                        elif len(batch_features.shape) == 3:
                            batch_features = batch_features.mean(dim=1)  # 平均token特征
                        features = batch_features.squeeze().flatten().detach().cpu().numpy()
                else:
                    batch_features = model(input_tensor)
                    if isinstance(batch_features, tuple):
                        batch_features = batch_features[0]
                    if len(batch_features.shape) == 4:
                        batch_features = batch_features.mean(dim=[2, 3])
                    elif len(batch_features.shape) == 3:
                        batch_features = batch_features.mean(dim=1)
                    features = batch_features.squeeze().flatten().cpu().numpy()
                
                # 检查并清理NaN/Inf值
                if np.isnan(features).any() or np.isinf(features).any():
                    features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
                
                cell_features.append(features)
                cell_ids.append(int(cell_id))
                cell_locations.append((centroid_x, centroid_y))
                cell_sizes.append((props.area, props.perimeter))
        
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
    dinov3_model_path = "/data/yujk/hovernet2feature/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
    dinov3_repo_dir = "/data/yujk/hovernet2feature/dinov3"
    
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


def check_wsi_feature_file_integrity(wsi_name, output_folder,
                                     min_file_size_mb=1.0,
                                     min_cells_threshold=10):
    """
    检查单个 WSI 的特征文件是否完整/可用。
    主要用于恢复运行时，过滤掉之前因为 OOM 或异常中断但仍被记录为“已完成”的样本。

    规则（任一不满足则视为不完整）：
      1. parquet 文件存在；
      2. 文件大小至少 min_file_size_mb；
      3. 能正常打开，并且包含基本列；
      4. 行数 > 0（如果你希望更严格，可以把 min_cells_threshold 调大）。
    """
    output_path = os.path.join(output_folder, f"{wsi_name}_features.parquet")

    if not os.path.exists(output_path):
        logging.warning(f"[INTEGRITY] {wsi_name}: parquet 文件不存在: {output_path}")
        return False

    file_size_mb = os.path.getsize(output_path) / 1024 / 1024
    if file_size_mb < min_file_size_mb:
        logging.warning(f"[INTEGRITY] {wsi_name}: 文件过小 ({file_size_mb:.2f} MB) -> 可疑")
        # 对于极少细胞的切片，可能仍然是合法结果，所以继续做进一步检查

    try:
        # 只读 very small sample，避免加载整个大表
        # max_rows 需要较新的 pyarrow，如果版本不支持，则会忽略该参数
        table = pq.read_table(output_path)
        num_rows = table.num_rows
        cols = set(table.column_names)
    except Exception as e:
        logging.error(f"[INTEGRITY] {wsi_name}: 读取 parquet 失败: {e}")
        return False

    required_cols = {"unique_id", "image_name", "cell_id", "x", "y", "area", "perimeter"}
    if not required_cols.issubset(cols):
        logging.warning(f"[INTEGRITY] {wsi_name}: 缺少必要列, 现有列: {sorted(list(cols))}")
        return False

    if num_rows <= 0:
        logging.warning(f"[INTEGRITY] {wsi_name}: 行数为 0，视为不完整")
        return False

    if num_rows < min_cells_threshold:
        logging.info(f"[INTEGRITY] {wsi_name}: 细胞数 ({num_rows}) 少于阈值 {min_cells_threshold}，"
                     f"如果这是预期情况可以调低阈值。")

    logging.info(f"[INTEGRITY] {wsi_name}: 通过完整性检查, 行数={num_rows}, 大小={file_size_mb:.2f} MB")
    return True

def process_wsi(wsi_name, wsi_path, json_folder, mat_folder, output_folder,
                model_name, pca_components, chunk_size, force_reprocess=False):
    """
    Process all patches for a single WSI and save as one Parquet file with PCA.

    返回值:
        success: bool
            True  表示该 WSI 已经“处理完”（包括数据缺失而被跳过的情况）；
            False 表示处理中出现异常（如 OOM），下次续跑时应重新尝试。
    """
    print(f"\n{'='*60}")
    print(f"Processing WSI: {wsi_name}")
    print(f"{'='*60}")
    
    logging.info(f"Processing WSI: {wsi_name}")
    
    json_wsi_path = os.path.join(json_folder, wsi_name, "json")
    mat_wsi_path = os.path.join(mat_folder, wsi_name, "mat")

    if not os.path.isdir(json_wsi_path) or not os.path.isdir(mat_wsi_path):
        logging.warning(
            "Missing patch artefacts for %s. Expected json_dir at %s and mat_dir at %s. Skipping.",
            wsi_name,
            json_wsi_path,
            mat_wsi_path,
        )
        print(f"Missing data folders for {wsi_name}, skipping.")
        # 数据本身缺失，不属于 OOM 类型错误，视为“已处理”，避免无限重试
        return True
    
    print(f"   • JSON dir: {json_wsi_path}")
    print(f"   • MAT dir: {mat_wsi_path}")
    
    image_files = [f for f in os.listdir(wsi_path) if f.endswith('.png')]
    total_patches = len(image_files)
    print(f"Total patches: {total_patches}")
    
    processed_log = os.path.join(output_folder, "processed_patches.log")
    processed_patches = set()
    if not force_reprocess and os.path.exists(processed_log):
        with open(processed_log, 'r') as f:
            processed_patches = set(line.strip() for line in f)
    elif force_reprocess:
        print(f"   • 强制重跑 {wsi_name} 的所有 patch（忽略 processed_patches.log）")
    
    args_list = []
    for image_file in image_files:
        image_path = os.path.join(wsi_path, image_file)
        json_path = os.path.join(json_wsi_path, image_file.replace('.png', '.json'))
        mat_path = os.path.join(mat_wsi_path, image_file.replace('.png', '.mat'))
        
        if not os.path.exists(json_path) or not os.path.exists(mat_path):
            logging.warning(f"Missing json or mat for {image_file} in {wsi_name}, skipped.")
            continue
        
        if image_path not in processed_patches:
            args_list.append((image_path, json_path, mat_path, model_name, processed_log))
    
    if not args_list:
        print(f"All patches already processed for {wsi_name}")
        logging.info(f"No patches to process for {wsi_name} or all already processed.")
        # 如果已经有对应 parquet 文件，则认为完成；否则视为不完整
        if check_wsi_feature_file_integrity(wsi_name, output_folder):
            return True
        else:
            print(f"No new patches, but特征文件可能不完整，将在下一轮重试 {wsi_name}")
            return False
    
    print(f"Processing {len(args_list)} new patches...")
    
    # Process patches in parallel and aggregate data
    all_features = []  # Collect all features across patches for PCA
    cell_ids_all = []
    cell_locations_all = []
    cell_sizes_all = []
    image_names_all = []
    patch_numbers_all = []
    
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    cpu_cap = cpu_count()
    if cpu_cap is None:
        cpu_cap = psutil.cpu_count(logical=True) or 1
    worker_count = min(max(1, len(args_list)), num_gpus if num_gpus > 0 else cpu_cap)
    print(f"   • Parallel workers: {worker_count} (GPUs: {num_gpus})")

    ctx = get_context("spawn")
    device_queue = ctx.Queue()
    if num_gpus > 0:
        for gpu_id in range(num_gpus):
            device_queue.put(gpu_id)
    else:
        for _ in range(worker_count):
            device_queue.put(None)

    total_cells = 0
    try:
        with ctx.Pool(
            worker_count,
            initializer=worker_initializer,
            initargs=(device_queue,),
        ) as pool:
            for (
                wsi_name_chunk,
                image_name,
                patch_number,
                cell_features,
                cell_ids,
                cell_locations,
                cell_sizes,
                _,
            ) in tqdm(
                pool.imap(
                    extract_features_for_image,
                    args_list,
                    chunksize=max(1, chunk_size // max(1, worker_count)),
                ),
                total=len(args_list),
                desc="Extracting Features",
            ):
                all_features.extend(cell_features)
                cell_ids_all.extend(cell_ids)
                cell_locations_all.extend(cell_locations)
                cell_sizes_all.extend(cell_sizes)
                image_names_all.extend([image_name] * len(cell_features))
                patch_numbers_all.extend([patch_number] * len(cell_features))
                total_cells += len(cell_features)
    except Exception as e:
        # 这里很可能包含 GPU OOM 等错误，视为“本次处理失败，下次需要重跑”
        logging.error(f"Error during WSI {wsi_name}: {str(e)}")
        print(f"Error during WSI processing: {str(e)}")
        return False

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
        try:
            df.to_parquet(output_path, engine='pyarrow', compression='snappy', index=False)
        except Exception as e:
            logging.error(f"Error saving parquet for {wsi_name}: {e}")
            print(f"Error saving parquet for {wsi_name}: {e}")
            return False
        
        print(f"\nResults saved:")
        print(f"   • File: {output_path}")
        print(f"   • Cells: {len(wsi_data):,}")
        print(f"   • Features per cell: {effective_components}")
        print(f"   • File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
        
        logging.info(f"Features for {wsi_name} saved: {len(wsi_data):,} cells, {effective_components} features, variance explained: {explained_variance_ratio:.4f}")
        return True
    else:
        logging.info(f"No data to save for WSI {wsi_name}")
        print(f"No data to save for WSI {wsi_name}")
        # 没有任何 cell，视为已成功处理（数据本身为空）
        return True

def batch_extract_features(image_folder, json_folder, mat_folder, output_folder, model_name='dinov3', pca_components=128, chunk_size=100):
    """Process all WSIs and manage the extraction process."""
    elevate_process_priority()
    logical_cpus = psutil.cpu_count(logical=True) or cpu_count() or 1
    torch.set_num_threads(logical_cpus)

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
    
    os.makedirs(output_folder, exist_ok=True)
    processed_wsi_log = os.path.join(output_folder, "processed_wsi.log")
    
    processed_wsi = set()
    if os.path.exists(processed_wsi_log):
        with open(processed_wsi_log, 'r') as f:
            processed_wsi = set(line.strip() for line in f)

    # ------------------------------------------------------------------
    # 在恢复运行前，对已经标记为“完成”的 WSI 做一次完整性检查
    #  - 如果 parquet 文件不存在或明显不完整，则从 processed_wsi 中移除，
    #    并加入到 force_reprocess_wsi 集合中，下一轮会强制重跑。
    # ------------------------------------------------------------------
    force_reprocess_wsi = set()
    if processed_wsi:
        print("\nChecking integrity of previously processed WSIs ...")
        still_ok = set()
        for name in sorted(processed_wsi):
            ok = check_wsi_feature_file_integrity(name, output_folder)
            if ok:
                still_ok.add(name)
            else:
                print(f"   • 检测到不完整或损坏: {name} -> 将重新处理")
                force_reprocess_wsi.add(name)
                # 这里可以选择删除旧文件，避免后续误用
                out_path = os.path.join(output_folder, f"{name}_features.parquet")
                if os.path.exists(out_path):
                    try:
                        os.remove(out_path)
                        print(f"     已删除旧特征文件: {out_path}")
                    except Exception as e:
                        print(f"     删除旧特征文件失败: {e}")
        processed_wsi = still_ok
    
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
        # 如果该 WSI 之前被判定为“结果不完整”，则强制重跑所有 patch
        success = process_wsi(
            wsi_name,
            wsi_path,
            json_folder,
            mat_folder,
            output_folder,
            model_name,
            pca_components,
            chunk_size,
            force_reprocess=(wsi_name in force_reprocess_wsi),
        )
        
        # 只有在明确 success 的情况下才标记为已处理，
        # 避免 OOM / 其他异常时错误地写入 processed_wsi.log。
        if success:
            with open(processed_wsi_log, 'a') as f:
                f.write(f"{wsi_name}\n")
        else:
            print(f"⚠️  WSI {wsi_name} 本轮处理失败（例如 OOM），不会写入 processed_wsi.log，下次可继续重试。")
    
    print(f"\nAll processing completed!")

if __name__ == "__main__":
    image_folder = "/mnt/elements2/PRAD"
    json_folder = "/mnt/elements1/PRAD"
    mat_folder = "/mnt/elements1/PRAD"
    output_folder = "/mnt/elements2/ouput_features/PRAD"
    
    batch_extract_features(
        image_folder=image_folder,
        json_folder=json_folder,
        mat_folder=mat_folder,
        output_folder=output_folder,
        model_name='dinov3',  # Changed to dinov3
        pca_components=128,
        chunk_size=100
    )