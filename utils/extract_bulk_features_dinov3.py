import numpy as np
import cv2
import json
import scipy.io as sio
import os
import re
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

# Configure an optional HuggingFace endpoint mirror (useful in some networks).
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
print(f"Using HuggingFace endpoint: {os.environ.get('HF_ENDPOINT')}")

# DINOv3 / transformers availability.
try:
    import transformers
    from transformers import AutoModel, AutoImageProcessor
    DINOV3_AVAILABLE = True
    print("✓ DINOv3 available")
except ImportError as e:
    DINOV3_AVAILABLE = False
    print("Error: please install transformers (pip install transformers)")
    print(f"Details: {e}")

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

def parse_patch_coords_from_filename(filename):
    """
    Parse patch coordinates from filename.
    Expected pattern: *_level0_{x1}-{y1}-{x2}-{y2}.png
    """
    match = re.search(r"_level0_(\d+)-(\d+)-(\d+)-(\d+)", filename)
    if not match:
        return None
    x1, y1, x2, y2 = map(int, match.groups())
    return x1, y1, x2, y2

def infer_tile_size(image_files):
    """Infer tile size from patch filenames; fallback to 512 if not found."""
    for name in image_files:
        coords = parse_patch_coords_from_filename(name)
        if coords:
            x1, y1, x2, y2 = coords
            return max(1, x2 - x1), max(1, y2 - y1)
    return 512, 512

def infer_tile_origin(patch_coords):
    """Infer grid origin from a list of patch coordinate tuples."""
    if not patch_coords:
        return 0, 0
    xs = [c[0] for c in patch_coords]
    ys = [c[1] for c in patch_coords]
    return min(xs), min(ys)

def build_cells_by_patch_from_wsi_json(wsi_json_path, tile_w, tile_h, patch_set, origin_x, origin_y):
    """Build a mapping from patch bounds to cells using a WSI-level HoverNet JSON."""
    with open(wsi_json_path, "r") as f:
        data = json.load(f)
    nuc = data.get("nuc", {})
    cells_by_patch = {}
    for cell_id_str, cell in nuc.items():
        centroid = cell.get("centroid")
        contour = cell.get("contour")
        if not centroid or not contour:
            continue
        cx, cy = centroid
        if cx < origin_x or cy < origin_y:
            continue
        x1 = int((cx - origin_x) // tile_w) * tile_w + origin_x
        y1 = int((cy - origin_y) // tile_h) * tile_h + origin_y
        key = (x1, y1, x1 + tile_w, y1 + tile_h)
        if key not in patch_set:
            continue
        try:
            cell_id = int(cell_id_str)
        except Exception:
            cell_id = cell_id_str
        cells_by_patch.setdefault(key, []).append({
            "id": cell_id,
            "centroid": (cx, cy),
            "contour": contour,
        })
    return cells_by_patch

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
                        # If the model returns a tuple, take the first element.
                        if isinstance(batch_features, tuple):
                            batch_features = batch_features[0]
                        # If features are 4D (B, C, H, W), apply global average pooling.
                        if len(batch_features.shape) == 4:
                            batch_features = batch_features.mean(dim=[2, 3])  # Global average pooling
                        elif len(batch_features.shape) == 3:
                            batch_features = batch_features.mean(dim=1)  # Mean token features.
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
                
                # Clean NaN/Inf values (defensive).
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

def extract_features_for_patch_from_cells(args):
    """Extract raw features for a patch using WSI-level JSON cells (no per-patch mat)."""
    image_path, cells, model_name, processed_log, patch_coords = args
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    wsi_name = os.path.basename(os.path.dirname(image_path))
    patch_number = image_name.split("_patch_")[-1] if "_patch_" in image_name else "unknown"

    logging.info(f"Processing patch (WSI-level JSON): {image_name}")
    try:
        original_img = load_image(image_path)
        h, w = original_img.shape[:2]
        x1, y1, x2, y2 = patch_coords

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

        cell_features, cell_ids, cell_locations, cell_sizes = [], [], [], []

        for cell in cells:
            contour = np.array(cell["contour"], dtype=np.int32)
            if contour.ndim != 2 or contour.shape[0] < 3:
                continue
            # Convert to patch-local coordinates and clip to image bounds
            contour_local = contour.copy()
            contour_local[:, 0] -= x1
            contour_local[:, 1] -= y1
            contour_local[:, 0] = np.clip(contour_local[:, 0], 0, w - 1)
            contour_local[:, 1] = np.clip(contour_local[:, 1], 0, h - 1)

            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [contour_local], 1)

            if mask.sum() == 0:
                continue

            ys, xs = np.where(mask > 0)
            y_min, y_max = ys.min(), ys.max() + 1
            x_min, x_max = xs.min(), xs.max() + 1

            cell_roi = original_img[y_min:y_max, x_min:x_max].copy()
            mask_roi = mask[y_min:y_max, x_min:x_max]
            masked_cell = np.zeros_like(cell_roi)
            masked_cell[mask_roi == 1] = cell_roi[mask_roi == 1]

            if masked_cell.shape[0] < 10 or masked_cell.shape[1] < 10:
                continue

            cell_pil = Image.fromarray(masked_cell)
            input_tensor = preprocess(cell_pil).unsqueeze(0)
            if CUDA_AVAILABLE:
                input_tensor = input_tensor.to(device, non_blocking=True)

            if CUDA_AVAILABLE:
                torch.cuda.set_device(device_id)
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    batch_features = model(input_tensor)
                    if isinstance(batch_features, tuple):
                        batch_features = batch_features[0]
                    if len(batch_features.shape) == 4:
                        batch_features = batch_features.mean(dim=[2, 3])
                    elif len(batch_features.shape) == 3:
                        batch_features = batch_features.mean(dim=1)
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

            if np.isnan(features).any() or np.isinf(features).any():
                features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)

            centroid_x = cell["centroid"][0] - x1
            centroid_y = cell["centroid"][1] - y1
            area = int(mask.sum())
            perimeter = float(cv2.arcLength(contour_local, True))

            cell_features.append(features)
            cell_ids.append(cell["id"])
            cell_locations.append((centroid_x, centroid_y))
            cell_sizes.append((area, perimeter))

        with open(processed_log, 'a') as f:
            f.write(f"{image_path}\n")
        logging.info(f"Completed patch (WSI-level JSON): {image_name}, extracted {len(cell_features)} cells")

    except Exception as e:
        logging.error(f"Error processing {image_path}: {str(e)}")
        print(f"Error processing {image_name}: {str(e)}")
        raise

    return wsi_name, image_name, patch_number, cell_features, cell_ids, cell_locations, cell_sizes, original_feature_dim

def get_feature_extractor(model_name):
    """Return a DinoV3 feature extractor model and its feature dimension."""
    if not DINOV3_AVAILABLE:
        raise ImportError("DINOv3 is not available; please install transformers")
    
    # DINOv3 local model/repo paths.
    dinov3_model_path = "/data/yujk/hovernet2feature/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
    dinov3_repo_dir = "/data/yujk/hovernet2feature/dinov3"
    
    if not os.path.exists(dinov3_model_path):
        raise RuntimeError(f"DINOv3 model file not found: {dinov3_model_path}")
    
    if not os.path.exists(dinov3_repo_dir):
        raise RuntimeError(f"DINOv3 repository not found: {dinov3_repo_dir}")
    
    print(f"Using DINOv3 repository: {dinov3_repo_dir}")
    print(f"Using local DINOv3 weights: {dinov3_model_path}")
    
    try:
        # Load DINOv3 ViT-L/16 via torch.hub.
        print("Loading DINOv3 ViT-L/16 via torch.hub...")
        
        dino_model = torch.hub.load(
            dinov3_repo_dir, 
            'dinov3_vitl16',  # DINOv3 ViT-L/16 model.
            source='local',
            weights=dinov3_model_path,  # Local weights.
            trust_repo=True
        )
        
        print("✓ Loaded DINOv3 successfully via torch.hub")
        
        # DINOv3-L feature dimension.
        feature_dim = 1024
        
        return dino_model, feature_dim
        
    except Exception as e:
        print(f"torch.hub loading failed: {e}")
        
        # Fallback: manual loading.
        print("Trying manual loading...")
        try:
            # Load the checkpoint file directly.
            checkpoint = torch.load(dinov3_model_path, map_location='cpu')
            
            # Check whether this is a full model object.
            if hasattr(checkpoint, 'forward'):
                # Already a model instance.
                dino_model = checkpoint
                print("✓ Loaded model object directly")
            else:
                # State dict: need to build a compatible architecture first.
                print("Detected a state_dict; building model architecture...")
                
                # Try building a compatible ViT via timm.
                try:
                    import timm
                    # Create a DINOv3-like ViT backbone (feature extractor only).
                    dino_model = timm.create_model(
                        'vit_large_patch16_224',
                        pretrained=False,
                        num_classes=0,  # Feature extraction only.
                        global_pool=''
                    )
                    
                    # Load weights.
                    if isinstance(checkpoint, dict):
                        if 'model' in checkpoint:
                            state_dict = checkpoint['model']
                        elif 'state_dict' in checkpoint:
                            state_dict = checkpoint['state_dict']
                        else:
                            state_dict = checkpoint
                    else:
                        state_dict = checkpoint
                    
                    # Load state_dict non-strictly to tolerate key mismatches.
                    missing_keys, unexpected_keys = dino_model.load_state_dict(state_dict, strict=False)
                    print(f"✓ Loaded via timm (missing={len(missing_keys)}, unexpected={len(unexpected_keys)})")
                    
                except ImportError:
                    print("timm is not available. Install it with: pip install timm")
                    raise RuntimeError("Unable to load the DINOv3 model")
            
            feature_dim = 1024
            return dino_model, feature_dim
            
        except Exception as e2:
            raise RuntimeError(
                f"All DINOv3 loading methods failed:\n"
                f"torch.hub: {e}\n"
                f"manual loading: {e2}"
            )

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
    Check whether a WSI feature Parquet file looks complete/usable.

    This is mainly used for resuming interrupted runs: it filters out WSIs that
    were previously marked as "processed" but ended up with incomplete outputs
    due to OOM or other failures.

    A file is considered incomplete if any of the following is true:
    1) The Parquet file does not exist
    2) The file size is smaller than ``min_file_size_mb`` (heuristic)
    3) The file cannot be read or required columns are missing
    4) The table has 0 rows

    Args:
        wsi_name: WSI identifier (used as filename prefix).
        output_folder: Directory containing ``{wsi_name}_features.parquet``.
        min_file_size_mb: Minimum expected file size in MB (heuristic).
        min_cells_threshold: If row count is smaller than this threshold, a log
            message is emitted (but the file may still be valid).

    Returns:
        True if the file passes basic integrity checks, False otherwise.
    """
    output_path = os.path.join(output_folder, f"{wsi_name}_features.parquet")

    if not os.path.exists(output_path):
        logging.warning(f"[INTEGRITY] {wsi_name}: parquet file not found: {output_path}")
        return False

    file_size_mb = os.path.getsize(output_path) / 1024 / 1024
    if file_size_mb < min_file_size_mb:
        logging.warning(f"[INTEGRITY] {wsi_name}: file too small ({file_size_mb:.2f} MB) -> suspicious")
        # Extremely low-cell WSIs may still be valid, so continue checking.

    try:
        # Read-only sanity check (avoid loading an overly large table into memory).
        table = pq.read_table(output_path)
        num_rows = table.num_rows
        cols = set(table.column_names)
    except Exception as e:
        logging.error(f"[INTEGRITY] {wsi_name}: failed to read parquet: {e}")
        return False

    required_cols = {"unique_id", "image_name", "cell_id", "x", "y", "area", "perimeter"}
    if not required_cols.issubset(cols):
        logging.warning(f"[INTEGRITY] {wsi_name}: missing required columns, found: {sorted(list(cols))}")
        return False

    if num_rows <= 0:
        logging.warning(f"[INTEGRITY] {wsi_name}: 0 rows, considered incomplete")
        return False

    if num_rows < min_cells_threshold:
        logging.info(
            f"[INTEGRITY] {wsi_name}: row count ({num_rows}) < threshold {min_cells_threshold}. "
            f"If this is expected, lower min_cells_threshold."
        )

    logging.info(f"[INTEGRITY] {wsi_name}: integrity OK, rows={num_rows}, size={file_size_mb:.2f} MB")
    return True

def process_wsi(wsi_name, wsi_path, json_folder, mat_folder, output_folder,
                model_name, pca_components, chunk_size, force_reprocess=False):
    """
    Process all patches for a single WSI and save as one Parquet file with PCA.

    Returns:
        success: bool
            True means the WSI is considered "done" for this run (including the
            case where required inputs are missing and the WSI is skipped).
            False means the run failed part-way (e.g., OOM) and should be retried.
    """
    print(f"\n{'='*60}")
    print(f"Processing WSI: {wsi_name}")
    print(f"{'='*60}")
    
    logging.info(f"Processing WSI: {wsi_name}")
    
    json_wsi_path = os.path.join(json_folder, wsi_name, "json")
    mat_wsi_path = os.path.join(mat_folder, wsi_name, "mat")
    wsi_json_path = os.path.join(json_folder, "json", f"{wsi_name}.json")

    use_wsi_level_json = False
    if os.path.isdir(json_wsi_path) and os.path.isdir(mat_wsi_path):
        print(f"   • JSON dir: {json_wsi_path}")
        print(f"   • MAT dir: {mat_wsi_path}")
    elif os.path.isfile(wsi_json_path):
        use_wsi_level_json = True
        print(f"   • JSON file: {wsi_json_path}")
        print(f"   • MAT dir: not used (WSI-level JSON mode)")
    else:
        logging.warning(
            "Missing patch artefacts for %s. Expected json_dir at %s and mat_dir at %s, "
            "or WSI-level JSON at %s. Skipping.",
            wsi_name,
            json_wsi_path,
            mat_wsi_path,
            wsi_json_path,
        )
        print(f"Missing data folders for {wsi_name}, skipping.")
        # Missing inputs are treated as "done" to avoid infinite retries.
        return True
    
    image_files = [f for f in os.listdir(wsi_path) if f.endswith('.png')]
    total_patches = len(image_files)
    print(f"Total patches: {total_patches}")
    
    processed_log = os.path.join(output_folder, "processed_patches.log")
    processed_patches = set()
    if not force_reprocess and os.path.exists(processed_log):
        with open(processed_log, 'r') as f:
            processed_patches = set(line.strip() for line in f)
    elif force_reprocess:
        print(f"   • Forcing reprocess of all patches for {wsi_name} (ignoring processed_patches.log)")
    
    args_list = []
    if use_wsi_level_json:
        patch_coords_list = []
        patch_coords_map = {}
        for image_file in image_files:
            coords = parse_patch_coords_from_filename(image_file)
            if coords:
                patch_coords_list.append(coords)
                patch_coords_map[image_file] = coords
        tile_w, tile_h = infer_tile_size(image_files)
        origin_x, origin_y = infer_tile_origin(patch_coords_list)
        patch_set = set(patch_coords_list)
        print(f"   • Tile size inferred: {tile_w}x{tile_h}")
        print(f"   • Tile origin inferred: ({origin_x}, {origin_y})")
        print(f"   • Loading WSI-level JSON for {wsi_name}...")
        cells_by_patch = build_cells_by_patch_from_wsi_json(
            wsi_json_path, tile_w, tile_h, patch_set, origin_x, origin_y
        )

        for image_file in image_files:
            image_path = os.path.join(wsi_path, image_file)
            if image_path in processed_patches:
                continue
            coords = patch_coords_map.get(image_file)
            if not coords:
                logging.warning(f"Unable to parse patch coords for {image_file} in {wsi_name}, skipped.")
                continue
            cells = cells_by_patch.get(coords, [])
            if not cells:
                continue
            args_list.append((image_path, cells, model_name, processed_log, coords))
    else:
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
        # If a parquet output exists and passes integrity checks, consider it done;
        # otherwise treat it as incomplete.
        if check_wsi_feature_file_integrity(wsi_name, output_folder):
            return True
        else:
            print(f"No new patches, but the feature file may be incomplete; will retry in the next run: {wsi_name}")
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
            worker_fn = extract_features_for_patch_from_cells if use_wsi_level_json else extract_features_for_image
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
                    worker_fn,
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
        # This may include GPU OOM or similar issues: treat as a failed run that should be retried.
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
        # No cells: consider the WSI successfully processed (data is empty).
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
    # Before resuming, validate previously "processed" WSIs:
    # - If the parquet file is missing or obviously incomplete, remove it from
    #   processed_wsi and add it to force_reprocess_wsi for a full re-run.
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
                print(f"   • Detected incomplete/corrupted output: {name} -> will reprocess")
                force_reprocess_wsi.add(name)
                # Optionally delete old outputs to avoid accidental reuse.
                out_path = os.path.join(output_folder, f"{name}_features.parquet")
                if os.path.exists(out_path):
                    try:
                        os.remove(out_path)
                        print(f"     Deleted old feature file: {out_path}")
                    except Exception as e:
                        print(f"     Failed to delete old feature file: {e}")
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
        # If this WSI was previously marked as incomplete, force reprocess all patches.
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
        
        # Only mark as processed on explicit success to avoid writing false positives
        # on OOM or other failures.
        if success:
            with open(processed_wsi_log, 'a') as f:
                f.write(f"{wsi_name}\n")
        else:
            print(f"Warning: WSI {wsi_name} failed in this run (e.g., OOM); not writing processed_wsi.log. It will be retried next time.")
    
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
