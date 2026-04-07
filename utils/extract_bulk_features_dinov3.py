import os
import re
import json
import cv2
import timm
import torch
import psutil
import logging
import warnings
import traceback
import numpy as np
import pandas as pd
import scipy.io as sio
import pyarrow.parquet as pq
import torchvision.transforms as transforms

from PIL import Image
from tqdm import tqdm
from skimage import measure
from multiprocessing import cpu_count, set_start_method, get_context

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# Environment / thread control (important for multiprocessing stability)
# --------------------------------------------------------------------------
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

try:
    cv2.setNumThreads(0)
except Exception:
    pass

try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass

# --------------------------------------------------------------------------
# Global caches & resource controls
# --------------------------------------------------------------------------
MODEL_CACHE = {}
PREPROCESS_CACHE = None
WORKER_DEVICE_ID = None

UNI_MODEL_PATH = "/data/hdd2/shanggk/BiTro/UNI.bin"
ORIGINAL_FEATURE_DIM = 1024  # UNI ViT-L/16 embedding size


def elevate_process_priority():
    """Attempt to raise the priority of the current process and pin all CPUs."""
    try:
        proc = psutil.Process(os.getpid())
        if hasattr(proc, "cpu_affinity"):
            cpu_total = psutil.cpu_count() or os.cpu_count()
            if cpu_total:
                proc.cpu_affinity(list(range(cpu_total)))
        try:
            proc.nice(-20)
        except psutil.AccessDenied:
            try:
                proc.nice(-10)
            except Exception:
                pass
    except Exception as exc:
        print(f"Warning: Unable to elevate process priority: {exc}")


def get_preprocess_cached():
    """Return and cache the preprocessing pipeline."""
    global PREPROCESS_CACHE
    if PREPROCESS_CACHE is None:
        PREPROCESS_CACHE = get_preprocess_transform()
    return PREPROCESS_CACHE


def get_model_for_device(model_name, device_id=None):
    """
    Load and cache the UNI model per device.
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
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    try:
        cv2.setNumThreads(0)
    except Exception:
        pass

    try:
        WORKER_DEVICE_ID = device_queue.get_nowait()
    except Exception:
        WORKER_DEVICE_ID = None


# Configure logging
logging.basicConfig(
    filename="feature_extraction.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Verify PyTorch version and CUDA availability
print(f"PyTorch version: {torch.__version__}")
logging.info(f"PyTorch version: {torch.__version__}")

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


def get_feature_extractor(model_name):
    """Return a UNI feature extractor model and its feature dimension."""
    if model_name.lower() != "uni":
        print(f"Warning: model_name={model_name}, but this script loads UNI.")

    if not os.path.exists(UNI_MODEL_PATH):
        raise RuntimeError(f"UNI model file not found: {UNI_MODEL_PATH}")

    print(f"Using local UNI weights: {UNI_MODEL_PATH}")
    print("Loading UNI ViT-L/16 via timm...")

    model = timm.create_model(
        "vit_large_patch16_224",
        img_size=224,
        patch_size=16,
        init_values=1e-5,
        num_classes=0,
        dynamic_img_size=False,   # fixed 224 input
    )

    checkpoint = torch.load(UNI_MODEL_PATH, map_location="cpu")

    if isinstance(checkpoint, dict):
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    if isinstance(state_dict, dict):
        state_dict = {
            k.replace("module.", "", 1) if k.startswith("module.") else k: v
            for k, v in state_dict.items()
        }

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    feature_dim = 1024
    print(f"Loaded UNI successfully. Output feature dim = {feature_dim}")

    return model, feature_dim


def get_preprocess_transform():
    """Return the preprocessing transformation pipeline for UNI."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])


def run_uni_feature_extraction_batch(model, input_tensor, cuda_available=False, device_id=None):
    """
    Run UNI on a batch and return a 2D numpy feature array [B, D].
    input_tensor: [B, 3, 224, 224]
    """
    if input_tensor.ndim != 4:
        raise ValueError(f"Unexpected input tensor shape: {tuple(input_tensor.shape)}")

    with torch.inference_mode():
        if cuda_available:
            torch.cuda.set_device(device_id)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                batch_features = model(input_tensor)
        else:
            batch_features = model(input_tensor)

        if isinstance(batch_features, tuple):
            batch_features = batch_features[0]

        if batch_features.ndim == 4:
            batch_features = batch_features.mean(dim=[2, 3])
        elif batch_features.ndim == 3:
            batch_features = batch_features.mean(dim=1)

        batch_features = batch_features.detach().float().cpu().numpy()

        if np.isnan(batch_features).any() or np.isinf(batch_features).any():
            batch_features = np.nan_to_num(batch_features, nan=0.0, posinf=1.0, neginf=-1.0)

        return batch_features


def extract_features_for_image(args):
    """Extract raw UNI features for a single image patch without PCA, using batched inference."""
    image_path, json_path, mat_path, model_name, processed_log, infer_batch_size = args
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    wsi_name = os.path.basename(os.path.dirname(image_path))
    patch_number = image_name.split("_patch_")[-1] if "_patch_" in image_name else "unknown"

    logging.info(f"Processing patch: {image_name}")

    try:
        original_img = load_image(image_path)
        mat_data = sio.loadmat(mat_path)

        if "inst_map" not in mat_data:
            raise KeyError(f"'inst_map' not found in {mat_path}")

        inst_map = mat_data["inst_map"]

        gpu_enabled = torch.cuda.is_available()
        device_id = WORKER_DEVICE_ID if gpu_enabled else None

        model, original_feature_dim = get_model_for_device(
            model_name,
            device_id if (gpu_enabled and device_id is not None) else None,
        )

        preprocess = get_preprocess_cached()
        CUDA_AVAILABLE = gpu_enabled and device_id is not None
        device = torch.device(f"cuda:{device_id}") if CUDA_AVAILABLE else torch.device("cpu")

        cell_tensors = []
        cell_meta = []

        unique_ids = np.unique(inst_map)
        unique_ids = unique_ids[unique_ids > 0]

        for cell_id in unique_ids:
            try:
                cell_mask = (inst_map == cell_id).astype(np.uint8)

                if cell_mask.sum() == 0:
                    continue

                props_list = measure.regionprops(cell_mask)
                if not props_list:
                    continue

                props = max(props_list, key=lambda p: p.area)

                y_min, x_min, y_max, x_max = props.bbox
                centroid_y, centroid_x = props.centroid

                if y_max <= y_min or x_max <= x_min:
                    logging.warning(f"[CELL SKIP] Invalid bbox in {image_name}, cell_id={cell_id}, bbox={props.bbox}")
                    continue

                cell_roi = original_img[y_min:y_max, x_min:x_max].copy()
                mask_roi = cell_mask[y_min:y_max, x_min:x_max]

                if cell_roi.size == 0 or mask_roi.size == 0:
                    logging.warning(f"[CELL SKIP] Empty ROI in {image_name}, cell_id={cell_id}")
                    continue

                masked_cell = np.zeros_like(cell_roi)
                masked_cell[mask_roi == 1] = cell_roi[mask_roi == 1]

                if masked_cell.shape[0] < 10 or masked_cell.shape[1] < 10:
                    continue

                cell_pil = Image.fromarray(masked_cell).convert("RGB")
                input_tensor = preprocess(cell_pil)  # [3, 224, 224]

                if input_tensor.ndim != 3:
                    raise ValueError(
                        f"Unexpected tensor shape in {image_name}, cell_id={cell_id}: {tuple(input_tensor.shape)}"
                    )

                cell_tensors.append(input_tensor)
                cell_meta.append((
                    int(cell_id),
                    (float(centroid_x), float(centroid_y)),
                    (float(props.area), float(props.perimeter))
                ))

            except Exception as e:
                logging.error(f"[CELL ERROR] image={image_name}, cell_id={cell_id}, error={repr(e)}")
                logging.error(traceback.format_exc())
                print(f"[CELL ERROR] image={image_name}, cell_id={cell_id}, error={repr(e)}")
                continue

        cell_features, cell_ids, cell_locations, cell_sizes = [], [], [], []

        for start in range(0, len(cell_tensors), infer_batch_size):
            batch_tensors = cell_tensors[start:start + infer_batch_size]
            batch_meta = cell_meta[start:start + infer_batch_size]

            if len(batch_tensors) == 0:
                continue

            batch_tensor = torch.stack(batch_tensors, dim=0)

            if CUDA_AVAILABLE:
                batch_tensor = batch_tensor.pin_memory().to(device, non_blocking=True)

            batch_features = run_uni_feature_extraction_batch(
                model=model,
                input_tensor=batch_tensor,
                cuda_available=CUDA_AVAILABLE,
                device_id=device_id,
            )

            for feat, (cid, loc, size) in zip(batch_features, batch_meta):
                cell_features.append(feat)
                cell_ids.append(cid)
                cell_locations.append(loc)
                cell_sizes.append(size)

        with open(processed_log, "a") as f:
            f.write(f"{image_path}\n")

        logging.info(f"Completed patch: {image_name}, extracted {len(cell_features)} cells")

    except Exception as e:
        logging.error(f"Error processing {image_path}: {repr(e)}")
        logging.error(traceback.format_exc())
        print(f"Error processing {image_name}: {repr(e)}")
        print(traceback.format_exc())
        raise

    return wsi_name, image_name, patch_number, cell_features, cell_ids, cell_locations, cell_sizes, original_feature_dim


def extract_features_for_patch_from_cells(args):
    """Extract raw UNI features for a patch using WSI-level JSON cells, using batched inference."""
    image_path, cells, model_name, processed_log, patch_coords, infer_batch_size = args
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    wsi_name = os.path.basename(os.path.dirname(image_path))
    patch_number = image_name.split("_patch_")[-1] if "_patch_" in image_name else "unknown"

    logging.info(f"Processing patch, WSI-level JSON mode: {image_name}")

    try:
        original_img = load_image(image_path)
        h, w = original_img.shape[:2]
        x1, y1, x2, y2 = patch_coords

        gpu_enabled = torch.cuda.is_available()
        device_id = WORKER_DEVICE_ID if gpu_enabled else None

        model, original_feature_dim = get_model_for_device(
            model_name,
            device_id if (gpu_enabled and device_id is not None) else None,
        )

        preprocess = get_preprocess_cached()
        CUDA_AVAILABLE = gpu_enabled and device_id is not None
        device = torch.device(f"cuda:{device_id}") if CUDA_AVAILABLE else torch.device("cpu")

        cell_tensors = []
        cell_meta = []

        for cell in cells:
            try:
                contour = np.array(cell["contour"], dtype=np.int32)

                if contour.ndim != 2 or contour.shape[0] < 3:
                    continue

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

                if y_max <= y_min or x_max <= x_min:
                    continue

                cell_roi = original_img[y_min:y_max, x_min:x_max].copy()
                mask_roi = mask[y_min:y_max, x_min:x_max]

                if cell_roi.size == 0 or mask_roi.size == 0:
                    continue

                masked_cell = np.zeros_like(cell_roi)
                masked_cell[mask_roi == 1] = cell_roi[mask_roi == 1]

                if masked_cell.shape[0] < 10 or masked_cell.shape[1] < 10:
                    continue

                cell_pil = Image.fromarray(masked_cell).convert("RGB")
                input_tensor = preprocess(cell_pil)  # [3, 224, 224]

                if input_tensor.ndim != 3:
                    raise ValueError(
                        f"Unexpected tensor shape in {image_name}, cell_id={cell['id']}: {tuple(input_tensor.shape)}"
                    )

                centroid_x = float(cell["centroid"][0] - x1)
                centroid_y = float(cell["centroid"][1] - y1)
                area = float(mask.sum())
                perimeter = float(cv2.arcLength(contour_local, True))

                cell_tensors.append(input_tensor)
                cell_meta.append((
                    cell["id"],
                    (centroid_x, centroid_y),
                    (area, perimeter),
                ))

            except Exception as e:
                logging.error(
                    f"[CELL ERROR][WSI JSON] image={image_name}, cell_id={cell.get('id', 'unknown')}, error={repr(e)}"
                )
                logging.error(traceback.format_exc())
                print(f"[CELL ERROR][WSI JSON] image={image_name}, cell_id={cell.get('id', 'unknown')}, error={repr(e)}")
                continue

        cell_features, cell_ids, cell_locations, cell_sizes = [], [], [], []

        for start in range(0, len(cell_tensors), infer_batch_size):
            batch_tensors = cell_tensors[start:start + infer_batch_size]
            batch_meta = cell_meta[start:start + infer_batch_size]

            if len(batch_tensors) == 0:
                continue

            batch_tensor = torch.stack(batch_tensors, dim=0)

            if CUDA_AVAILABLE:
                batch_tensor = batch_tensor.pin_memory().to(device, non_blocking=True)

            batch_features = run_uni_feature_extraction_batch(
                model=model,
                input_tensor=batch_tensor,
                cuda_available=CUDA_AVAILABLE,
                device_id=device_id,
            )

            for feat, (cid, loc, size) in zip(batch_features, batch_meta):
                cell_features.append(feat)
                cell_ids.append(cid)
                cell_locations.append(loc)
                cell_sizes.append(size)

        with open(processed_log, "a") as f:
            f.write(f"{image_path}\n")

        logging.info(f"Completed patch, WSI-level JSON mode: {image_name}, extracted {len(cell_features)} cells")

    except Exception as e:
        logging.error(f"Error processing {image_path}: {repr(e)}")
        logging.error(traceback.format_exc())
        print(f"Error processing {image_name}: {repr(e)}")
        print(traceback.format_exc())
        raise

    return wsi_name, image_name, patch_number, cell_features, cell_ids, cell_locations, cell_sizes, original_feature_dim


def check_wsi_feature_file_integrity(wsi_name, output_folder, min_file_size_mb=1.0, min_cells_threshold=10):
    """Check whether a WSI feature Parquet file looks complete and usable."""
    output_path = os.path.join(output_folder, f"{wsi_name}_features.parquet")

    if not os.path.exists(output_path):
        logging.warning(f"[INTEGRITY] {wsi_name}: parquet file not found: {output_path}")
        return False

    file_size_mb = os.path.getsize(output_path) / 1024 / 1024

    if file_size_mb < min_file_size_mb:
        logging.warning(f"[INTEGRITY] {wsi_name}: file too small ({file_size_mb:.2f} MB)")

    try:
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
        logging.info(f"[INTEGRITY] {wsi_name}: row count ({num_rows}) < threshold {min_cells_threshold}")

    logging.info(f"[INTEGRITY] {wsi_name}: integrity OK, rows={num_rows}, size={file_size_mb:.2f} MB")
    return True


def process_wsi(
    wsi_name,
    wsi_path,
    json_folder,
    mat_folder,
    output_folder,
    model_name,
    pca_components=None,
    chunk_size=100,
    force_reprocess=False,
    infer_batch_size=64,
):
    """
    Process all patches for a single WSI and save one Parquet file.

    PCA has been removed.
    The saved feature columns are raw UNI features, usually 1024 dimensions.
    """
    print(f"\n{'=' * 60}")
    print(f"Processing WSI: {wsi_name}")
    print(f"{'=' * 60}")

    logging.info(f"Processing WSI: {wsi_name}")

    json_wsi_path = os.path.join(json_folder, wsi_name, "json")
    mat_wsi_path = os.path.join(mat_folder, wsi_name, "mat")
    wsi_json_path = os.path.join(json_folder, "json", f"{wsi_name}.json")

    use_wsi_level_json = False

    if os.path.isdir(json_wsi_path) and os.path.isdir(mat_wsi_path):
        print(f"   JSON dir: {json_wsi_path}")
        print(f"   MAT dir: {mat_wsi_path}")
    elif os.path.isfile(wsi_json_path):
        use_wsi_level_json = True
        print(f"   JSON file: {wsi_json_path}")
        print("   MAT dir: not used, WSI-level JSON mode")
    else:
        logging.warning(
            "Missing patch artefacts for %s. Expected json_dir at %s and mat_dir at %s, or WSI-level JSON at %s. Skipping.",
            wsi_name,
            json_wsi_path,
            mat_wsi_path,
            wsi_json_path,
        )
        print(f"Missing data folders for {wsi_name}, skipping.")
        return True

    image_files = [f for f in os.listdir(wsi_path) if f.endswith(".png")]
    total_patches = len(image_files)
    print(f"Total patches: {total_patches}")

    processed_log = os.path.join(output_folder, "processed_patches.log")
    processed_patches = set()

    if not force_reprocess and os.path.exists(processed_log):
        with open(processed_log, "r") as f:
            processed_patches = set(line.strip() for line in f)
    elif force_reprocess:
        print(f"   Forcing reprocess of all patches for {wsi_name}")

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

        print(f"   Tile size inferred: {tile_w}x{tile_h}")
        print(f"   Tile origin inferred: ({origin_x}, {origin_y})")
        print(f"   Loading WSI-level JSON for {wsi_name}...")

        cells_by_patch = build_cells_by_patch_from_wsi_json(
            wsi_json_path,
            tile_w,
            tile_h,
            patch_set,
            origin_x,
            origin_y,
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

            args_list.append((image_path, cells, model_name, processed_log, coords, infer_batch_size))
    else:
        for image_file in image_files:
            image_path = os.path.join(wsi_path, image_file)
            json_path = os.path.join(json_wsi_path, image_file.replace(".png", ".json"))
            mat_path = os.path.join(mat_wsi_path, image_file.replace(".png", ".mat"))

            if not os.path.exists(json_path) or not os.path.exists(mat_path):
                logging.warning(f"Missing json or mat for {image_file} in {wsi_name}, skipped.")
                continue

            if image_path not in processed_patches:
                args_list.append((image_path, json_path, mat_path, model_name, processed_log, infer_batch_size))

    if not args_list:
        print(f"All patches already processed for {wsi_name}")
        logging.info(f"No patches to process for {wsi_name} or all already processed.")

        if check_wsi_feature_file_integrity(wsi_name, output_folder):
            return True

        print(f"No new patches, but feature file may be incomplete; will retry next run: {wsi_name}")
        return False

    print(f"Processing {len(args_list)} new patches...")
    print(f"Inference batch size: {infer_batch_size}")

    all_features = []
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
    print(f"   Parallel workers: {worker_count} (GPUs: {num_gpus})")

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
                desc="Extracting UNI Features",
            ):
                if len(cell_features) == 0:
                    continue

                all_features.extend(cell_features)
                cell_ids_all.extend(cell_ids)
                cell_locations_all.extend(cell_locations)
                cell_sizes_all.extend(cell_sizes)
                image_names_all.extend([image_name] * len(cell_features))
                patch_numbers_all.extend([patch_number] * len(cell_features))
                total_cells += len(cell_features)

    except Exception as e:
        logging.error(f"Error during WSI {wsi_name}: {repr(e)}")
        logging.error(traceback.format_exc())
        print(f"Error during WSI processing: {repr(e)}")
        print(traceback.format_exc())
        return False

    print(f"\nTotal cells extracted: {total_cells:,}")

    if all_features:
        print("Keeping raw UNI features without PCA...")

        raw_features = np.asarray(all_features)

        if raw_features.ndim != 2:
            raw_features = np.vstack([np.asarray(feat).reshape(1, -1) for feat in all_features])

        feature_dim = raw_features.shape[1]

        print("UNI features kept:")
        print(f"   Original UNI feature dimensions: {feature_dim}")
        print(f"   Expected UNI dimensions: {ORIGINAL_FEATURE_DIM}")

        wsi_data = []

        for image_name, patch_number, feat, cell_id, loc, size in zip(
            image_names_all,
            patch_numbers_all,
            raw_features,
            cell_ids_all,
            cell_locations_all,
            cell_sizes_all,
        ):
            unique_id = f"{wsi_name}_{image_name}_cell_{cell_id}"
            wsi_data.append(
                [unique_id, image_name, cell_id, loc[0], loc[1], size[0], size[1]] + feat.tolist()
            )

        feature_cols = [f"feature_{i}" for i in range(feature_dim)]
        columns = ["unique_id", "image_name", "cell_id", "x", "y", "area", "perimeter"] + feature_cols

        df = pd.DataFrame(wsi_data, columns=columns)
        output_path = os.path.join(output_folder, f"{wsi_name}_features.parquet")

        try:
            df.to_parquet(output_path, engine="pyarrow", compression="snappy", index=False)
        except Exception as e:
            logging.error(f"Error saving parquet for {wsi_name}: {repr(e)}")
            logging.error(traceback.format_exc())
            print(f"Error saving parquet for {wsi_name}: {repr(e)}")
            return False

        print("\nResults saved:")
        print(f"   File: {output_path}")
        print(f"   Cells: {len(wsi_data):,}")
        print(f"   Features per cell: {feature_dim}")
        print(f"   File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")

        logging.info(
            f"UNI features for {wsi_name} saved: {len(wsi_data):,} cells, {feature_dim} features"
        )

        return True

    logging.info(f"No data to save for WSI {wsi_name}")
    print(f"No data to save for WSI {wsi_name}")
    return True


def batch_extract_features(
    image_folder,
    json_folder,
    mat_folder,
    output_folder,
    model_name="uni",
    pca_components=None,
    chunk_size=100,
    infer_batch_size=64,
):
    """Process all WSIs and manage the extraction process."""
    elevate_process_priority()

    logical_cpus = psutil.cpu_count(logical=True) or cpu_count() or 1

    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    print(f"\n{'=' * 80}")
    print("UNI Feature Extraction Pipeline")
    print(f"{'=' * 80}")
    print(f"Image folder: {image_folder}")
    print(f"JSON folder: {json_folder}")
    print(f"MAT folder: {mat_folder}")
    print(f"Output folder: {output_folder}")
    print(f"Model: {model_name}")
    print("PCA: disabled, raw UNI features will be saved")
    print(f"Chunk size: {chunk_size}")
    print(f"Infer batch size: {infer_batch_size}")
    print(f"UNI model path: {UNI_MODEL_PATH}")
    print(f"Logical CPUs detected: {logical_cpus}")

    os.makedirs(output_folder, exist_ok=True)

    processed_wsi_log = os.path.join(output_folder, "processed_wsi.log")

    processed_wsi = set()
    if os.path.exists(processed_wsi_log):
        with open(processed_wsi_log, "r") as f:
            processed_wsi = set(line.strip() for line in f)

    force_reprocess_wsi = set()

    if processed_wsi:
        print("\nChecking integrity of previously processed WSIs ...")
        still_ok = set()

        for name in sorted(processed_wsi):
            ok = check_wsi_feature_file_integrity(name, output_folder)

            if ok:
                still_ok.add(name)
            else:
                print(f"   Detected incomplete/corrupted output: {name} -> will reprocess")
                force_reprocess_wsi.add(name)

                out_path = os.path.join(output_folder, f"{name}_features.parquet")

                if os.path.exists(out_path):
                    try:
                        os.remove(out_path)
                        print(f"   Deleted old feature file: {out_path}")
                    except Exception as e:
                        print(f"   Failed to delete old feature file: {e}")

        processed_wsi = still_ok

    wsi_names = [
        d for d in os.listdir(image_folder)
        if os.path.isdir(os.path.join(image_folder, d))
    ]

    total_wsi = len(wsi_names)
    remaining_wsi = [name for name in wsi_names if name not in processed_wsi]

    print("\nStatus:")
    print(f"   Total WSIs: {total_wsi}")
    print(f"   Already processed: {len(processed_wsi)}")
    print(f"   Remaining: {len(remaining_wsi)}")

    if len(processed_wsi) > 0:
        print(f"   Processed WSIs: {sorted(list(processed_wsi))}")

    if len(remaining_wsi) == 0:
        print("All WSIs already processed!")
        return

    print("\nStarting processing...")

    for idx, wsi_name in enumerate(remaining_wsi):
        print(f"\nWSI Progress: {idx + 1}/{len(remaining_wsi)}")

        wsi_path = os.path.join(image_folder, wsi_name)

        success = process_wsi(
            wsi_name=wsi_name,
            wsi_path=wsi_path,
            json_folder=json_folder,
            mat_folder=mat_folder,
            output_folder=output_folder,
            model_name=model_name,
            pca_components=None,
            chunk_size=chunk_size,
            force_reprocess=(wsi_name in force_reprocess_wsi),
            infer_batch_size=infer_batch_size,
        )

        if success:
            with open(processed_wsi_log, "a") as f:
                f.write(f"{wsi_name}\n")
        else:
            print(
                f"Warning: WSI {wsi_name} failed in this run; "
                "not writing processed_wsi.log. It will be retried next time."
            )

    print("\nAll processing completed!")


if __name__ == "__main__":
    set_start_method("spawn", force=True)

    image_folder = "/data/hdd2/yujk/TCGA_patches/PRAD"
    json_folder = "/data/hdd2/yujk/hovernet_output/PRAD"
    mat_folder = "/data/hdd2/yujk/hovernet_output/PRAD"
    output_folder = "/data/hdd2/shanggk/BiTro/demo_data/Feature/Bulk/PRAD"

    batch_extract_features(
        image_folder=image_folder,
        json_folder=json_folder,
        mat_folder=mat_folder,
        output_folder=output_folder,
        model_name="uni",
        pca_components=None,
        chunk_size=100,
        infer_batch_size=64,   # 先试 64，不够再调 96 / 128
    )