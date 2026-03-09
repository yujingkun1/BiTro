#!/usr/bin/env python3
"""
HEST cell-level deep feature extractor for BiTro.

This script extracts per-cell deep embeddings using DINOv3 and applies PCA to
produce compact feature vectors suitable for downstream graph construction and
training.
"""

import gc
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import json
import h5py
import shapely.wkb as wkb
import warnings
from tqdm import tqdm
import pickle
from sklearn.decomposition import PCA
import cv2
import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
# Allow loading very large TIFFs (avoid DecompressionBomb warnings).
Image.MAX_IMAGE_PIXELS = None
warnings.filterwarnings("ignore")

# Optional dependency for Vahadane stain normalization.
try:
    import staintools
    STAINTOOLS_AVAILABLE = True
    print("✓ staintools available (Vahadane normalization can be enabled)")
except Exception as _e:
    STAINTOOLS_AVAILABLE = False
    print("Warning: staintools not available; stain normalization will be skipped. Install: pip install staintools")

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


class HESTCellFeatureExtractor:
    """Extract per-cell DINOv3 features for HEST samples."""

    def __init__(self,
                 hest_data_dir,
                 output_dir,
                 dinov3_model_path=None,
                 bulk_pca_model_path=None,
                 cell_patch_size=48,
                 dinov3_feature_dim=1024,
                 final_dino_dim=128,
                 device='cuda',
                 dino_batch_size=256,
                 cell_batch_size=50000,
                 num_workers=8,
                 assign_spot=False,
                 stain_normalize=False,
                 stain_method='vahadane',
                 stain_target_image=None,
                 use_normalized_wsi=False,
                 normalized_wsi_subdir="wsis_normalized",
                 raw_wsi_subdir="wsis"):

        if not DINOV3_AVAILABLE:
            raise ImportError("DINOv3 is not available; please install transformers")

        self.hest_data_dir = hest_data_dir
        self.output_dir = output_dir
        self.dinov3_model_path = dinov3_model_path
        self.bulk_pca_model_path = bulk_pca_model_path
        self.cell_patch_size = cell_patch_size
        self.dinov3_feature_dim = dinov3_feature_dim
        self.final_dino_dim = final_dino_dim
        self.final_feature_dim = final_dino_dim
        self.device = device
        self.dino_batch_size = dino_batch_size
        self.cell_batch_size = cell_batch_size
        self.num_workers = num_workers
        self.assign_spot = assign_spot
        self.stain_normalize = bool(stain_normalize)
        self.stain_method = str(stain_method).lower(
        ) if stain_method else 'vahadane'
        self.stain_target_image = stain_target_image
        self.stain_normalizer = None  # Initialized per-sample.
        self.use_normalized_wsi = bool(use_normalized_wsi)
        self.normalized_wsi_subdir = normalized_wsi_subdir
        self.raw_wsi_subdir = raw_wsi_subdir

        # If using pre-normalized WSIs, disable stain normalization to avoid
        # double-normalizing.
        if self.use_normalized_wsi and self.stain_normalize:
            print("Detected pre-normalized WSIs; disabling stain_normalize to avoid double normalization")
            self.stain_normalize = False

        # Ensure output directory exists.
        os.makedirs(output_dir, exist_ok=True)

        # Initialize DINOv3 model.
        self.init_dinov3_model()

        print("Extractor configuration:")
        print(f"  - DINOv3 batch size: {self.dino_batch_size}")
        print(f"  - Cell batch size: {self.cell_batch_size}")
        print(f"  - Workers: {self.num_workers}")
        print(f"  - Device: {self.device}")
        print(f"  - Final feature dim: {self.final_feature_dim} (DINOv3 + PCA)")
        print("  - Backbone: DINOv3-L")

    def init_dinov3_model(self):
        """Initialize the DINOv3 model (torch.hub with local weights)."""
        print("Initializing DINOv3 model...")

        if not self.dinov3_model_path or not os.path.exists(self.dinov3_model_path):
            raise RuntimeError(f"DINOv3 model file not found: {self.dinov3_model_path}")

        print(f"Using local DINOv3 weights: {self.dinov3_model_path}")

        try:
            # DINOv3 repository path (local).
            dinov3_repo_dir = "/data/yujk/hovernet2feature/dinov3"

            if not os.path.exists(dinov3_repo_dir):
                raise RuntimeError(f"DINOv3 repository not found: {dinov3_repo_dir}")

            print(f"Using DINOv3 repository: {dinov3_repo_dir}")

            # Load DINOv3 ViT-L/16 via torch.hub.
            print("Loading DINOv3 ViT-L/16 via torch.hub...")

            # Direct torch.hub loading.
            self.dino_model = torch.hub.load(
                dinov3_repo_dir,
                'dinov3_vitl16',  # DINOv3 ViT-L/16 model.
                source='local',
                weights=self.dinov3_model_path,  # Local weights.
                trust_repo=True
            )

            print("✓ Loaded DINOv3 successfully via torch.hub")

            # Set feature dimension.
            self.dinov3_feature_dim = 1024  # DINOv3-L feature dimension.

            # Set image preprocessing (ImageNet-style).
            from torchvision import transforms

            self.dino_processor_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

            print("✓ Image preprocessor configured")

        except Exception as e:
            print(f"torch.hub loading failed: {e}")

            # Fallback: manual loading.
            print("Trying manual loading...")
            try:
                # Load checkpoint file directly.
                checkpoint = torch.load(
                    self.dinov3_model_path, map_location='cpu')

                # Check whether this is already a model object.
                if hasattr(checkpoint, 'forward'):
                    # It's a model instance.
                    self.dino_model = checkpoint
                    print("✓ Loaded model object directly")
                else:
                    # State dict: build a compatible model architecture first.
                    print("Detected a state_dict; building model architecture...")

                    # Try building a compatible ViT via timm.
                    try:
                        import timm
                        # Create a DINOv3-like ViT backbone (feature extractor only).
                        self.dino_model = timm.create_model(
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
                        missing_keys, unexpected_keys = self.dino_model.load_state_dict(
                            state_dict, strict=False)
                        print(
                            f"✓ Loaded via timm (missing={len(missing_keys)}, unexpected={len(unexpected_keys)})")

                    except ImportError:
                        print("timm is not available. Install: pip install timm")
                        raise RuntimeError("Unable to load the DINOv3 model")

                # Set image preprocessing.
                from torchvision import transforms
                self.dino_processor_transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                ])

                self.dinov3_feature_dim = 1024

            except Exception as e2:
                raise RuntimeError(
                    f"All DINOv3 loading methods failed:\n"
                    f"torch.hub: {e}\n"
                    f"manual loading: {e2}"
                )

        # Finalize model attributes.
        self.dino_model.to(self.device)

        # Do not force FP16; let autocast manage precision.
        # if torch.cuda.is_available():
        #     self.dino_model = self.dino_model.half()  # (disabled) do not force FP16

        self.dino_model.eval()

        print(f"✓ DINOv3 model ready. Feature dim: {self.dinov3_feature_dim}")

    def _init_stain_normalizer_for_sample(self, sample_id, wsi, level):
        """Initialize (or skip) the stain normalizer for a sample.

        Preference order:
        1) Use a user-provided target image (if provided).
        2) Otherwise, fit a target region cropped from the middle of the WSI.
        """
        if not self.stain_normalize:
            return
        if self.stain_method != 'vahadane':
            print(f"Warning: stain method not implemented: {self.stain_method}; skipping normalization")
            return
        if not STAINTOOLS_AVAILABLE:
            print("Warning: staintools not available; skipping Vahadane normalization")
            return

        try:
            normalizer = staintools.StainNormalizer(method='vahadane')

            if self.stain_target_image and os.path.exists(self.stain_target_image):
                target_img = staintools.read_image(self.stain_target_image)
                target_img = staintools.LuminosityStandardizer.standardize(
                    target_img)
                normalizer.fit(target_img)
                self.stain_normalizer = normalizer
                print(f"✓ Initialized Vahadane normalization using provided target image: {self.stain_target_image}")
                return

            # Fit target image from a center crop of the WSI.
            try:
                wsi_width, wsi_height = wsi.level_dimensions[level]
                # Use a center crop with side length min(2048, half of the shorter side).
                side = min(2048, max(256, min(wsi_width, wsi_height) // 2))
                x0 = max(0, (wsi_width - side) // 2)
                y0 = max(0, (wsi_height - side) // 2)
                region = wsi.read_region((x0, y0), level, (side, side))
                region_rgb = np.array(region.convert('RGB'))
                region_rgb = staintools.LuminosityStandardizer.standardize(
                    region_rgb)
                normalizer.fit(region_rgb)
                self.stain_normalizer = normalizer
                print(
                    f"✓ Initialized sample {sample_id} with Vahadane normalization (fitted from the central WSI region {side}x{side} region)")
            except Exception as e:
                print(f"⚠️  Could not fit target stain from the WSI automatically; skipping normalization: {e}")
                self.stain_normalizer = None
        except Exception as e:
            print(f"⚠️  Failed to initialize Vahadane normalization; skipping: {e}")
            self.stain_normalizer = None

    def _normalize_patch(self, patch: np.ndarray) -> np.ndarray:
        """Apply stain normalization to a single RGB patch (fallback to original on failure)."""
        if not self.stain_normalize or self.stain_normalizer is None:
            return patch
        try:
            if patch is None or patch.size == 0:
                return patch
            if np.max(patch) < 5:
                return patch
            if patch.dtype != np.uint8:
                patch = patch.astype(np.uint8)
            patch_std = staintools.LuminosityStandardizer.standardize(patch)
            norm = self.stain_normalizer.transform(patch_std)
            norm = np.clip(norm, 0, 255).astype(np.uint8)
            return norm
        except Exception as e:
            return patch

    def monitor_resources(self):
        """Return a short CPU/RAM/GPU utilization string."""
        try:
            import psutil
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()

            resource_info = f"CPU: {cpu_percent:.1f}%, RAM: {memory.percent:.1f}%"

            if torch.cuda.is_available():
                gpu_memory_used = torch.cuda.memory_allocated() / 1024**3
                gpu_memory_total = torch.cuda.get_device_properties(
                    0).total_memory / 1024**3
                gpu_percent = (gpu_memory_used / gpu_memory_total) * 100
                resource_info += f", GPU: {gpu_percent:.1f}% ({gpu_memory_used:.2f}/{gpu_memory_total:.1f}GB)"

            return resource_info
        except ImportError:
            return "Resource monitoring requires psutil: pip install psutil"

    def load_sample_data(self, sample_id):
        """Load one sample's WSI path and segmentation table."""
        print(f"Loading sample data: {sample_id}")

        sample_data = {}

        candidates = []
        if self.use_normalized_wsi:
            candidates.extend([
                os.path.join(self.hest_data_dir, self.normalized_wsi_subdir,
                             f"normalized_{sample_id}_pyramid.tif"),
                os.path.join(
                    self.hest_data_dir, self.normalized_wsi_subdir, f"{sample_id}_pyramid.tif"),
                os.path.join(self.hest_data_dir,
                             self.normalized_wsi_subdir, f"{sample_id}.tif"),
                os.path.join(
                    self.hest_data_dir, self.normalized_wsi_subdir, f"normalized_{sample_id}.tif"),
            ])
        else:
            candidates.append(os.path.join(
                self.hest_data_dir, self.raw_wsi_subdir, f"{sample_id}.tif"))

        wsi_path = None
        for cand in candidates:
            if os.path.exists(cand):
                wsi_path = cand
                break
        if not wsi_path:
            raise FileNotFoundError(
                f"WSI file not found (use_normalized_wsi={self.use_normalized_wsi}); checked: {candidates}")

        cellvit_path = os.path.join(
            self.hest_data_dir, "cellvit_seg", f"{sample_id}_cellvit_seg.parquet")
        if not os.path.exists(cellvit_path):
            raise FileNotFoundError(f"Cell segmentation file does not exist: {cellvit_path}")

        cellvit_df = pd.read_parquet(cellvit_path)

        sample_data = {
            'wsi_path': wsi_path,
            'cellvit_df': cellvit_df,
            'sample_id': sample_id
        }

        print(f"  WSI: {wsi_path}")
        if self.use_normalized_wsi and self.normalized_wsi_subdir in wsi_path:
            print("  Using pre-normalized WSI (skipping a second stain normalization pass)")
        print(f"  Number of cells: {len(cellvit_df)}")

        return sample_data

    def extract_cell_patch(self, wsi_image, cell_geometry, patch_size=None):
        """Extract one cell-centered patch from a WSI image array."""
        if patch_size is None:
            patch_size = self.cell_patch_size

        try:
            if isinstance(cell_geometry, bytes):
                geom = wkb.loads(cell_geometry)
            else:
                geom = cell_geometry

            bounds = geom.bounds  # (minx, miny, maxx, maxy)

            center_x = (bounds[0] + bounds[2]) / 2
            center_y = (bounds[1] + bounds[3]) / 2

            half_size = patch_size // 2
            x1 = max(0, int(round(center_x - half_size)))
            y1 = max(0, int(round(center_y - half_size)))
            x2 = min(wsi_image.shape[1], int(round(center_x + half_size)))
            y2 = min(wsi_image.shape[0], int(round(center_y + half_size)))

            cell_patch = wsi_image[y1:y2, x1:x2]

            if cell_patch.shape[0] < patch_size or cell_patch.shape[1] < patch_size:
                padded_patch = np.zeros(
                    (patch_size, patch_size, 3), dtype=cell_patch.dtype)
                h, w = cell_patch.shape[:2]
                padded_patch[:h, :w] = cell_patch
                cell_patch = padded_patch

            elif cell_patch.shape[0] > patch_size or cell_patch.shape[1] > patch_size:
                cell_patch = cell_patch[:patch_size, :patch_size]

            cell_patch = self._normalize_patch(cell_patch)
            return cell_patch

        except Exception as e:
            print(f"Failed to extract cell patches: {e}")
            return np.zeros((patch_size, patch_size, 3), dtype=np.uint8)

    def extract_dino_features(self, cell_patches):
        """Extract DINOv3 features for cell patches (batched)."""
        if len(cell_patches) == 0:
            return np.zeros((0, self.dinov3_feature_dim))

        print(f"Starting extraction of {len(cell_patches)} patches with DINOv3 features...")
        print(f"Using batch size: {self.dino_batch_size} (optimized for much better GPU utilization)")

        features = []
        batch_size = self.dino_batch_size
        total_batches = (len(cell_patches) + batch_size - 1) // batch_size

        with torch.no_grad():
            for i in tqdm(range(0, len(cell_patches), batch_size), desc="DINOv3 feature extraction", total=total_batches):
                batch_patches = cell_patches[i:i+batch_size]

                try:
                    if hasattr(self, 'dino_processor_transform'):
                        processed_tensors = []
                        for patch in batch_patches:
                            if patch.max() <= 1.0:
                                patch = (patch * 255).astype(np.uint8)

                            from PIL import Image
                            pil_image = Image.fromarray(patch)

                            tensor = self.dino_processor_transform(pil_image)
                            processed_tensors.append(tensor)

                        batch_tensor = torch.stack(processed_tensors)

                    else:
                        processed_images = self._parallel_preprocess_images(
                            batch_patches)
                        inputs = self.dino_processor(
                            images=processed_images, return_tensors="pt")
                        batch_tensor = inputs['pixel_values']

                    batch_tensor = batch_tensor.to(
                        self.device, non_blocking=True)
                    # if torch.cuda.is_available():
                    #     batch_tensor = batch_tensor.to(self.device, non_blocking=True, dtype=torch.float16)
                    # else:
                    #     batch_tensor = batch_tensor.to(self.device, non_blocking=True)

                    if torch.cuda.is_available():
                        with torch.amp.autocast('cuda'):
                            if hasattr(self, 'dino_processor_transform'):
                                batch_features = self.dino_model(batch_tensor)
                                if isinstance(batch_features, tuple):
                                    batch_features = batch_features[0]
                                if len(batch_features.shape) == 4:
                                    batch_features = batch_features.mean(
                                        dim=[2, 3])  # Global average pooling
                                elif len(batch_features.shape) == 3:
                                    batch_features = batch_features.mean(
                                        dim=1)
                            else:
                                outputs = self.dino_model(
                                    pixel_values=batch_tensor)
                                batch_features = outputs.last_hidden_state.mean(
                                    dim=1)
                    else:
                        if hasattr(self, 'dino_processor_transform'):
                            batch_features = self.dino_model(batch_tensor)
                            if isinstance(batch_features, tuple):
                                batch_features = batch_features[0]
                            if len(batch_features.shape) == 4:
                                batch_features = batch_features.mean(dim=[
                                                                     2, 3])
                            elif len(batch_features.shape) == 3:
                                batch_features = batch_features.mean(dim=1)
                        else:
                            outputs = self.dino_model(
                                pixel_values=batch_tensor)
                            batch_features = outputs.last_hidden_state.mean(
                                dim=1)

                    batch_features_np = batch_features.cpu().numpy()

                    if np.isnan(batch_features_np).any() or np.isinf(batch_features_np).any():
                        print(f"  Warning: batch {i//batch_size} contains NaN/Inf values; cleaning them up")
                        batch_features_np = np.nan_to_num(
                            batch_features_np, nan=0.0, posinf=1.0, neginf=-1.0)

                    features.append(batch_features_np)

                    if i % (batch_size * 5) == 0:
                        resource_info = self.monitor_resources()
                        print(f"  Resource usage: {resource_info}")

                    if i % (batch_size * 20) == 0:
                        torch.cuda.empty_cache()

                except Exception as e:
                    print(f"Batch {i//batch_size} DINOv3 processing failed: {e}")
                    zero_features = np.zeros(
                        (len(batch_patches), self.dinov3_feature_dim))
                    features.append(zero_features)

        all_features = np.vstack(features) if features else np.zeros(
            (0, self.dinov3_feature_dim))

        print(f"DINOv3 feature extraction completed: {all_features.shape}")
        return all_features

    def _parallel_preprocess_images(self, batch_patches):
        """Preprocess patches in parallel (for HuggingFace processor)."""
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            processed_images = list(executor.map(
                self._preprocess_single_image, batch_patches))
        return processed_images

    def _preprocess_single_image(self, patch):
        """Preprocess a single patch."""
        if patch.max() <= 1.0:
            patch = (patch * 255).astype(np.uint8)
        return Image.fromarray(patch)

    def load_wsi_image(self, wsi_path):
        """Load a WSI image into an RGB numpy array (best-effort)."""
        try:
            try:
                import openslide
                wsi = openslide.OpenSlide(wsi_path)
                level = 0
                wsi_image = wsi.read_region(
                    (0, 0), level, wsi.level_dimensions[level])
                wsi_image = np.array(wsi_image.convert('RGB'))
                print(f"  Loaded WSI with openslide: {wsi_image.shape}")
                return wsi_image
            except ImportError:
                print("  openslide is unavailable; trying other methods...")

            import cv2
            wsi_image = cv2.imread(wsi_path)
            if wsi_image is not None:
                wsi_image = cv2.cvtColor(wsi_image, cv2.COLOR_BGR2RGB)
                print(f"  Loaded WSI with cv2: {wsi_image.shape}")
                return wsi_image

            # Fallback to PIL.
            from PIL import Image
            wsi_image = Image.open(wsi_path).convert('RGB')
            wsi_image = np.array(wsi_image)
            print(f"  Loaded WSI with PIL: {wsi_image.shape}")
            return wsi_image

        except Exception as e:
            print(f"  Failed to load WSI: {e}")
            return None

    class _SimpleWSI:
        """Minimal WSI wrapper for plain TIFFs (no pyramid/OpenSlide).

        Implements a small subset of the OpenSlide-like interface used in this
        script: dimensions, level_count, level_dimensions, level_downsamples,
        read_region, close.
        """

        def __init__(self, image_path: str):
            from PIL import Image
            # Avoid forcing full RGB decode here to reduce peak memory.
            self._img = Image.open(image_path)
            self._width, self._height = self._img.size
            self.level_count = 1
            self.level_dimensions = [(self._width, self._height)]
            self.level_downsamples = [1.0]
            self.dimensions = (self._width, self._height)

        def read_region(self, location, level, size):
            """OpenSlide-like read_region.

            Args:
                location: (x, y) coordinates in level-0 space.
                level: Pyramid level (only 0 is supported here).
                size: Output size (width, height).

            Out-of-bounds regions are filled with black pixels.
            """
            from PIL import Image

            if level != 0:
                # Only a single level is supported here; fall back to level 0.
                level = 0

            x, y = location
            w, h = size

            # Compute crop region and clamp to image bounds.
            left = max(0, int(x))
            top = max(0, int(y))
            right = min(self._width, int(x) + int(w))
            bottom = min(self._height, int(y) + int(h))

            # Create a black canvas for out-of-bounds padding.
            canvas = Image.new('RGB', (int(w), int(h)), (0, 0, 0))

            if right > left and bottom > top:
                crop = self._img.crop((left, top, right, bottom))
                # Paste crop at the relative offset.
                offset_x = max(0, left - int(x))
                offset_y = max(0, top - int(y))
                canvas.paste(crop, (offset_x, offset_y))

            return canvas

        def close(self):
            try:
                self._img.close()
            except Exception:
                pass

    def process_sample_with_independent_pca(self, sample_id):
        """Process one sample with per-sample PCA and save outputs."""
        print(f"\n=== Processing spatial sample: {sample_id} ===")
        print("Each sample trains its own PCA model with 128-dim DINOv3 features")

        sample_data = self.load_sample_data(sample_id)
        cellvit_df = sample_data['cellvit_df']
        wsi_path = sample_data['wsi_path']

        num_cells = len(cellvit_df)
        max_cells = num_cells
        cell_patches = []
        cell_positions = []  # (x, y) aligned with extracted patches.

        print(f"Preparing to extract features for all {num_cells} cells...")

        print("Trying to load the WSI image...")
        try:
            import openslide
            try:
                wsi = openslide.OpenSlide(wsi_path)
                print(f"  WSI dimensions: {wsi.dimensions}")
                print(f"  WSI level count: {wsi.level_count}")
                print(f"  WSI level dimensions: {wsi.level_dimensions}")
            except Exception as e:
                print(f"  OpenSlide could not load the file; falling back to regular image mode: {e}")
                wsi = self._SimpleWSI(wsi_path)
                print(f"  Simple WSI dimensions: {wsi.dimensions}")
                print(f"  Simple WSI level count: {wsi.level_count}")
                print(f"  Simple WSI level dimensions: {wsi.level_dimensions}")

            if self.use_normalized_wsi:
                level = 0 if getattr(wsi, 'level_count', 1) == 1 else min(
                    1, wsi.level_count - 1)
            else:
                if getattr(wsi, 'level_count', 1) > 1:
                    level = 1
                else:
                    level = 0
            print(f"  Using level {level}, dimensions: {wsi.level_dimensions[level]}")

            batch_size = self.cell_batch_size
            num_batches = (max_cells + batch_size - 1) // batch_size

            print(f"  Processing in {num_batches} batches, each with {batch_size} cells (significantly improving throughput)")

            self._init_stain_normalizer_for_sample(sample_id, wsi, level)

            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, max_cells)
                batch_cells = end_idx - start_idx

                print(
                    f"\n  Processing batch {batch_idx+1}/{num_batches}: cells {start_idx}-{end_idx-1} ({batch_cells} files)")

                print(f"\n  Using {self.num_workers} parallel worker processes to extract patches...")
                batch_patches, batch_positions = self._extract_patches_parallel(
                    cellvit_df.iloc[start_idx:end_idx], wsi, level, start_idx
                )

                cell_patches.extend(batch_patches)
                cell_positions.extend(batch_positions)

                del batch_patches
                gc.collect()

                print(
                    f"  Batch {batch_idx+1} completed; extracted {len(cell_patches)} patches")

                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.memory_allocated() / 1024**3
                    print(f"  Current GPU memory usage: {gpu_memory:.2f} GB")

            wsi.close()
            print(f"  Successfully extracted {len(cell_patches)} real cell patches from the WSI")

        except Exception as e:
            print(f"  WSI loading failed: {e}")
            print("  Could not extract real cell images from the WSI; terminating")
            raise RuntimeError(f"WSI image loading failed; cannot continue feature extraction: {e}") from e

        print("Extracting DINOv3 features...")
        dino_features = self.extract_dino_features(cell_patches)

        try:
            del cell_patches
        except Exception:
            pass
        gc.collect()

        print(f"For sample {sample_id}, training PCA dimensionality reduction independently...")
        final_features = self.apply_independent_pca(dino_features, sample_id)

        try:
            del dino_features
        except Exception:
            pass
        gc.collect()

        # Try to estimate the equivalent raw WSI level for normalized WSIs.
        normalized_equiv_level = None
        try:
            if self.use_normalized_wsi:
                import openslide
                raw_wsi_path = os.path.join(
                    self.hest_data_dir, self.raw_wsi_subdir, f"{sample_id}.tif")
                if os.path.exists(raw_wsi_path):
                    raw_wsi = openslide.OpenSlide(raw_wsi_path)
                    norm_w, norm_h = wsi.level_dimensions[level]
                    diffs = []
                    for li, (rw, rh) in enumerate(raw_wsi.level_dimensions):
                        diffs.append(abs(rw - norm_w) + abs(rh - norm_h))
                    normalized_equiv_level = int(
                        np.argmin(diffs)) if len(diffs) > 0 else None
                    raw_wsi.close()
        except Exception as _e:
            normalized_equiv_level = None

        metadata = {
            'sample_id': sample_id,
            'num_cells': num_cells,
            'feature_dim': self.final_feature_dim,
            'dino_dim': self.final_dino_dim,
            'patch_size': self.cell_patch_size,
            'wsi_level': int(level),
            'normalized_equiv_raw_level': normalized_equiv_level,
            'total_cells_processed': num_cells,
            'independent_pca': True,
            'pca_trained_on_sample': sample_id
        }

        positions = np.asarray(cell_positions, dtype=np.float32) if len(
            cell_positions) > 0 else np.zeros((num_cells, 2), dtype=np.float32)
        cell_index = np.arange(positions.shape[0], dtype=np.int64)
        spot_index = None
        if self.assign_spot:
            try:
                spot_index = self.assign_spot_indices(sample_id, positions)
            except Exception as e:
                print(f"[Warn] spot assignment failed for {sample_id}: {e}")

        output_file = self.save_features(
            sample_id, final_features, metadata, positions=positions, cell_index=cell_index, spot_index=spot_index)

        print(f"\n=== Performance statistics ===")
        total_cells = num_cells
        print(f"✓ Total processed cells: {total_cells:,}")
        print(f"✓ DINOv3 batch size: {self.dino_batch_size}")
        print(f"✓ Cell batch size: {self.cell_batch_size}")
        print(f"✓ Number of parallel workers: {self.num_workers}")
        print(f"✓ Final feature dimension: {self.final_feature_dim} (sample-specific PCA)")
        print(f"✓ Feature file: {output_file}")
        if torch.cuda.is_available():
            print(
                f"✓ Final GPU memory usage: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

        return {
            'sample_id': sample_id,
            'num_cells': num_cells,
            'final_feature_dim': self.final_feature_dim,
            'output_file': output_file
        }


    def _extract_patches_parallel(self, cellvit_batch, wsi, level, start_idx):
        """Extract cell patches in parallel using a thread pool."""
        batch_patches = []
        batch_positions = []

        # Use threads instead of processes because WSI objects are not serializable.
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            tasks = []
            for i, (_, row) in enumerate(cellvit_batch.iterrows()):
                task = executor.submit(
                    self._extract_single_patch_threaded,
                    row['geometry'],
                    wsi,
                    level,
                    self.cell_patch_size,
                    start_idx + i
                )
                tasks.append(task)

            for task in tqdm(tasks, desc=f"Batch-parallel patch extraction"):
                try:
                    patch, pos = task.result(timeout=10)  # 10s timeout
                    batch_patches.append(patch)
                    batch_positions.append(pos)
                except Exception as e:
                    print(f"  Parallel extraction failed; using the default patch: {e}")
                    batch_patches.append(
                        np.zeros((self.cell_patch_size, self.cell_patch_size, 3), dtype=np.uint8))
                    batch_positions.append((0.0, 0.0))

        return batch_patches, batch_positions

    def _extract_single_patch_threaded(self, geom_bytes, wsi, level, patch_size, cell_idx):
        """Thread-safe extraction of a single cell patch with black padding."""
        try:
            geom = wkb.loads(geom_bytes)
            centroid = geom.centroid
            center_x, center_y = centroid.x, centroid.y

            # Compute coordinates at the requested level (round for alignment).
            scale_factor = wsi.level_downsamples[level]
            half_size = patch_size // 2

            # WSI bounds at this level.
            wsi_width, wsi_height = wsi.level_dimensions[level]

            # Compute extraction region (rounded to integer pixels).
            x_start = int(round(center_x - half_size * scale_factor))
            y_start = int(round(center_y - half_size * scale_factor))
            x_end = x_start + int(round(patch_size * scale_factor))
            y_end = y_start + int(round(patch_size * scale_factor))

            # Black-padded patch.
            cell_patch = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)

            # Clamp to valid region inside the WSI.
            valid_x_start = max(0, x_start)
            valid_y_start = max(0, y_start)
            valid_x_end = min(wsi_width, x_end)
            valid_y_end = min(wsi_height, y_end)

            # If there is any valid region, extract it.
            if valid_x_start < valid_x_end and valid_y_start < valid_y_end:
                # Compute the extracted region size.
                region_width = int(
                    (valid_x_end - valid_x_start) / scale_factor)
                region_height = int(
                    (valid_y_end - valid_y_start) / scale_factor)

                if region_width > 0 and region_height > 0:
                    region = wsi.read_region(
                        (valid_x_start, valid_y_start),
                        level,
                        (region_width, region_height)
                    )

                    region_array = np.array(region.convert('RGB'))

                    # Compute placement offset within the output patch.
                    patch_x_start = max(
                        0, int(round((valid_x_start - x_start) / (scale_factor if scale_factor != 0 else 1))))
                    patch_y_start = max(
                        0, int(round((valid_y_start - y_start) / (scale_factor if scale_factor != 0 else 1))))
                    patch_x_end = min(
                        patch_size, patch_x_start + region_array.shape[1])
                    patch_y_end = min(
                        patch_size, patch_y_start + region_array.shape[0])

                    # Paste extracted region into the output patch.
                    if (patch_x_end > patch_x_start and patch_y_end > patch_y_start and
                            region_array.shape[0] > 0 and region_array.shape[1] > 0):

                        # Resize region_array to match the target slice.
                        target_height = patch_y_end - patch_y_start
                        target_width = patch_x_end - patch_x_start

                        if region_array.shape[:2] != (target_height, target_width):
                            region_array = cv2.resize(
                                region_array, (target_width, target_height))

                        cell_patch[patch_y_start:patch_y_end,
                                   patch_x_start:patch_x_end] = region_array

            # Optional stain normalization.
            cell_patch = self._normalize_patch(cell_patch)
            return cell_patch, (float(center_x), float(center_y))

        except Exception as e:
            if cell_idx < 5:  # Only show first few errors.
                print(f"    Cell {cell_idx} patch extraction failed: {e}")
            # Fallback to a black patch.
            return np.zeros((patch_size, patch_size, 3), dtype=np.uint8), (0.0, 0.0)

    def assign_spot_indices(self, sample_id, positions: np.ndarray):
        """Assign spot_index using nearest-neighbor matching with a radius constraint.

        The matching is based on HEST AnnData ``obsm['spatial']`` and is aligned
        with the logic in ``scripts/augment_features_with_positions.py``.

        Returns:
            A 1D int64 array of length N (cells). Unassigned cells are -1.
        """
        import scanpy as sc
        st_file = os.path.join(self.hest_data_dir, "st", f"{sample_id}.h5ad")
        meta_file = os.path.join(
            self.hest_data_dir, "metadata", f"{sample_id}.json")
        if not os.path.exists(st_file):
            raise FileNotFoundError(f"h5ad not found: {st_file}")

        adata = sc.read_h5ad(st_file)
        spatial = adata.obsm['spatial'] if 'spatial' in adata.obsm_keys(
        ) else None
        if spatial is None:
            raise RuntimeError(
                f"AnnData missing obsm['spatial'] for {sample_id}")

        metadata = {}
        if os.path.exists(meta_file):
            try:
                with open(meta_file, 'r') as f:
                    metadata = json.load(f)
            except Exception:
                metadata = {}

        pixel_size_um = metadata.get('pixel_size_um_estimated', 0.5)
        spot_diameter_px = metadata.get('spot_diameter', None)
        if spot_diameter_px is None:
            spot_radius_um = 25.0
        else:
            base_radius_um = (float(spot_diameter_px) /
                              2.0) * float(pixel_size_um)
            spot_radius_um = base_radius_um * 1.5

        n = positions.shape[0]
        M = spatial.shape[0]
        spot_index = np.full((n,), -1, dtype=np.int64)
        batch_size = 200000
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            pcs = positions[start:end]
            dists = np.linalg.norm(
                spatial[None, :, :] - pcs[:, None, :], axis=2)
            nearest = np.argmin(dists, axis=1)
            nearest_dist_px = dists[np.arange(end-start), nearest]
            nearest_dist_um = nearest_dist_px * float(pixel_size_um)
            within = nearest_dist_um <= float(spot_radius_um)
            spot_index[start:end][within] = nearest[within].astype(np.int64)
        return spot_index

    def apply_independent_pca(self, dino_features, sample_id):
        """Train a per-sample PCA and reduce DINOv3 features."""
        print(f"For sample {sample_id}, training an independent PCA reducer...")

        print(f"  Checking data quality...")
        nan_count = np.isnan(dino_features).sum()
        inf_count = np.isinf(dino_features).sum()

        if nan_count > 0:
            print(f"  Warning: found {nan_count} NaN values; replacing them with 0")
            dino_features = np.nan_to_num(dino_features, nan=0.0)

        if inf_count > 0:
            print(f"  Warning: found {inf_count} Inf values; replacing them with finite values")
            dino_features = np.nan_to_num(
                dino_features, posinf=1.0, neginf=-1.0)

        try:
            col_std = dino_features.std(axis=0)
            zero_std_cols = int(np.sum(col_std < 1e-8))
            print(f"  Diagnostics: number of DINO feature columns with std equal to or near 0: {zero_std_cols}/{dino_features.shape[1]}")
            std_percentiles = np.percentile(col_std, [0, 1, 5, 25, 50, 75, 95, 99, 100])
            print(f"  Diagnostics: column std percentiles: {std_percentiles}")

            n_samples = min(2000, dino_features.shape[0])
            rng = np.random.default_rng(42)
            idx = rng.choice(dino_features.shape[0], size=n_samples, replace=False)
            sample_rows = np.round(dino_features[idx, :], decimals=6)
            unique_rows = np.unique(sample_rows, axis=0)
            unique_frac = unique_rows.shape[0] / float(n_samples)
            print(f"  Diagnostics: in {n_samples} random samples, unique-row ratio: {unique_frac:.3f}")

            total_var = np.var(dino_features)
            print(f"  Diagnostics: overall DINO feature variance: {total_var:.6e}")

            try:
                preview_path = os.path.join(self.output_dir, f"{sample_id}_dino_features_preview.npy")
                np.save(preview_path, dino_features[:10])
                print(f"  Diagnostics: saved the first 10 DINO features to: {preview_path}")
            except Exception as _e:
                print(f"  Diagnostics: could not save preview: {_e}")
        except Exception as _e:
            print(f"  Diagnostics failed: {_e}")

        if np.all(dino_features == 0):
            print(f"  Error: all features are 0; PCA cannot be run")
            raise ValueError(f"Sample {sample_id} has all-zero DINOv3 features; feature extraction may have failed")

        n_samples = dino_features.shape[0]
        n_features = dino_features.shape[1]
        max_components = min(n_samples, n_features)

        actual_dino_dim = min(self.final_dino_dim, max_components - 1)

        if actual_dino_dim < self.final_dino_dim:
            print(
                f"  Warning: PCA dimension adjusted from {self.final_dino_dim} to {actual_dino_dim} (limited by the sample count)")

        if actual_dino_dim <= 0:
            print(f"  Error: could not determine a valid PCA dimension")
            raise ValueError(
                f"Sample {sample_id} could not determine a valid PCA dimension, sample count={n_samples},, feature count={n_features}")

        from sklearn.decomposition import PCA
        try:
            sample_pca = PCA(n_components=actual_dino_dim, random_state=42)
            reduced_features = sample_pca.fit_transform(dino_features)
        except Exception as e:
            print(f"  PCA training failed: {e}")
            print(
                f"  Feature statistics: min={dino_features.min():.6f}, max={dino_features.max():.6f}, mean={dino_features.mean():.6f}, std={dino_features.std():.6f}")
            raise ValueError(f"Sample {sample_id} PCA training failed: {e}") from e

        # Save PCA model with atomic write.
        sample_pca_path = os.path.join(
            self.output_dir, f"{sample_id}_dino_pca_model.pkl")
        temp_pca_path = sample_pca_path + ".tmp"
        
        try:
            with open(temp_pca_path, 'wb') as f:
                pickle.dump(sample_pca, f)
            os.replace(temp_pca_path, sample_pca_path)
        except Exception as e:
            if os.path.exists(temp_pca_path):
                try:
                    os.remove(temp_pca_path)
                except:
                    pass
            raise RuntimeError(f"Failed to save the PCA model: {e}") from e

        explained_variance = sample_pca.explained_variance_ratio_.sum()
        explained_variance_each = sample_pca.explained_variance_ratio_

        print(f"Sample {sample_id} PCA training completed:")
        print(f"  - Input dimension: {dino_features.shape[1]}")
        print(f"  - Output dimension: {actual_dino_dim}")
        print(
            f"  - Total explained variance ratio: {explained_variance:.4f} ({explained_variance*100:.2f}%)")

        print(f"  - Explained variance ratio of the first 10 principal components:")
        for i in range(min(10, len(explained_variance_each))):
            print(
                f"    PC{i+1}: {explained_variance_each[i]:.4f} ({explained_variance_each[i]*100:.2f}%)")

        cumulative_variance = np.cumsum(explained_variance_each)
        print(f"  - Cumulative explained variance:")
        milestones = [10, 20, 50, 100, 128]
        for milestone in milestones:
            if milestone <= len(cumulative_variance):
                print(
                    f"    First {milestone} principal components: {cumulative_variance[milestone-1]:.4f} ({cumulative_variance[milestone-1]*100:.2f}%)")

        print(f"  - PCA model saved to: {sample_pca_path}")

        return reduced_features

    def save_features(self, sample_id, combined_features, metadata, positions=None, cell_index=None, spot_index=None):
        """Save extracted features in a format compatible with downstream scripts.

        Always writes ``features`` and ``metadata``. If provided, also writes
        ``positions``, ``cell_index`` and ``spot_index``.

        Uses an atomic write (temp file + rename) to avoid partial outputs.
        Uses ``np.savez`` (uncompressed) to reduce peak memory usage.
        """
        output_file = os.path.join(
            self.output_dir, f"{sample_id}_combined_features.npz")
        # Use an explicit temp name to avoid ".npz" auto-suffix surprises.
        temp_file = output_file + ".tmp.npz"

        # Build payload dict.
        save_dict = {
            'features': combined_features,
            'metadata': metadata
        }
        if positions is not None:
            save_dict['positions'] = positions
        if cell_index is not None:
            save_dict['cell_index'] = cell_index
        if spot_index is not None:
            save_dict['spot_index'] = spot_index

        # Write temp file first (atomic replace).
        try:
            np.savez(temp_file, **save_dict)
            os.replace(temp_file, output_file)
        except Exception as e:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass
            raise RuntimeError(f"Failed to save the feature file: {e}") from e

        msg_extra = []
        if 'positions' in save_dict:
            msg_extra.append(f"positions={save_dict['positions'].shape}")
        if 'spot_index' in save_dict:
            msg_extra.append(f"spot_index={save_dict['spot_index'].shape}")
        print(
            f"Features saved: {output_file}{' | ' + ' '.join(msg_extra) if msg_extra else ''}")
        return output_file


def get_all_hest_samples(hest_data_dir):
    """Return available HEST sample IDs discovered on disk."""
    import glob
    import os

    # Find all cellvit segmentation parquet files.
    cellvit_pattern = os.path.join(
        hest_data_dir, "cellvit_seg", "*_cellvit_seg.parquet")
    cellvit_files = glob.glob(cellvit_pattern)

    # Extract sample IDs and keep those with an accessible WSI.
    sample_ids = []
    for file_path in cellvit_files:
        filename = os.path.basename(file_path)
        sample_id = filename.replace('_cellvit_seg.parquet', '')

        # Check whether a corresponding WSI exists (normalized preferred, then raw).
        normalized_candidates = [
            # Prefer pyramid variants.
            os.path.join(hest_data_dir, "wsis_normalized",
                         f"normalized_{sample_id}_pyramid.tif"),
            os.path.join(hest_data_dir, "wsis_normalized",
                         f"{sample_id}_pyramid.tif"),
            # Non-pyramid fallback.
            os.path.join(hest_data_dir, "wsis_normalized", f"{sample_id}.tif"),
            os.path.join(hest_data_dir, "wsis_normalized",
                         f"normalized_{sample_id}.tif"),
        ]
        raw_candidate = os.path.join(hest_data_dir, "wsis", f"{sample_id}.tif")
        if any(os.path.exists(p) for p in normalized_candidates) or os.path.exists(raw_candidate):
            sample_ids.append(sample_id)

    sample_ids.sort()
    return sample_ids


def get_progress_file(output_dir):
    """Return the progress file path."""
    return os.path.join(output_dir, "extraction_progress.json")


def load_progress(output_dir):
    """Load progress state from disk."""
    progress_file = get_progress_file(output_dir)
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r') as f:
                progress = json.load(f)
            print(f"✓ Loaded progress file: {progress_file}")
            return progress
        except Exception as e:
            print(f"⚠️  Failed to load progress file: {e}")
            return {}
    return {}


def save_progress(output_dir, progress):
    """Save progress state to disk."""
    progress_file = get_progress_file(output_dir)
    try:
        with open(progress_file, 'w') as f:
            json.dump(progress, f, indent=2)
        print(f"✓ Saved progress to: {progress_file}")
    except Exception as e:
        print(f"⚠️  Failed to save progress: {e}")


def verify_sample_file_integrity(output_dir, sample_id, expected_num_cells=None):
    """Validate the integrity of a saved sample output file.

    Args:
        output_dir: Output directory.
        sample_id: Sample ID.
        expected_num_cells: Optional expected cell count for validation.

    Returns:
        A tuple (is_complete, num_cells_in_file, error_msg).
    """
    output_file = os.path.join(
        output_dir, f"{sample_id}_combined_features.npz")
    pca_file = os.path.join(output_dir, f"{sample_id}_dino_pca_model.pkl")
    
    if not os.path.exists(output_file):
        return False, 0, "Feature file not found"
    if not os.path.exists(pca_file):
        return False, 0, "PCA model file not found"
    
    temp_file = output_file + ".tmp.npz"
    if os.path.exists(temp_file):
        return False, 0, "Temporary file exists; previous save may be incomplete"
    
    try:
        with np.load(output_file) as data:
            if 'features' not in data:
                return False, 0, "Missing required field: 'features'"
            if 'metadata' not in data:
                return False, 0, "Missing required field: 'metadata'"
            
            features = data['features']
            if features.size == 0:
                return False, 0, "Empty feature array"
            
            num_cells_in_file = features.shape[0]
            
            if len(features.shape) != 2:
                return False, num_cells_in_file, f"Invalid feature array shape: {features.shape}"
            
            try:
                metadata = data['metadata']
                # metadata may be stored as a numpy object; convert to dict when possible.
                if hasattr(metadata, 'item'):
                    metadata = metadata.item()
                elif isinstance(metadata, np.ndarray):
                    if metadata.ndim == 0:
                        metadata = metadata.item()
                    elif metadata.ndim == 1 and len(metadata) == 1:
                        metadata = metadata[0]
                
                if isinstance(metadata, dict):
                    metadata_num_cells = metadata.get('num_cells', None)
                    if metadata_num_cells is not None and metadata_num_cells != num_cells_in_file:
                        return False, num_cells_in_file, f"metadata num_cells ({metadata_num_cells}) != feature rows ({num_cells_in_file})"
            except Exception as e:
                pass
            
            if expected_num_cells is not None:
                if num_cells_in_file != expected_num_cells:
                    return False, num_cells_in_file, f"Cell count mismatch: file={num_cells_in_file}, expected={expected_num_cells}"
            
            if 'positions' in data:
                positions = data['positions']
                if positions.shape[0] != num_cells_in_file:
                    return False, num_cells_in_file, f"positions rows ({positions.shape[0]}) != feature rows ({num_cells_in_file})"
            
            if 'cell_index' in data:
                cell_index = data['cell_index']
                if len(cell_index) != num_cells_in_file:
                    return False, num_cells_in_file, f"cell_index length ({len(cell_index)}) != feature rows ({num_cells_in_file})"
            
            if 'spot_index' in data:
                spot_index = data['spot_index']
                if len(spot_index) != num_cells_in_file:
                    return False, num_cells_in_file, f"spot_index length ({len(spot_index)}) != feature rows ({num_cells_in_file})"
            
            if np.isnan(features).any() or np.isinf(features).any():
                return False, num_cells_in_file, "Features contain NaN/Inf values"
            
        return True, num_cells_in_file, "File looks complete"
        
    except Exception as e:
        return False, 0, f"Corrupted/unreadable file: {str(e)}"


def get_expected_num_cells(hest_data_dir, sample_id):
    """Get the expected cell count for a sample (from the cellvit_seg parquet).

    Args:
        hest_data_dir: HEST data directory.
        sample_id: Sample ID.

    Returns:
        Expected number of cells, or None if unavailable.
    """
    try:
        cellvit_path = os.path.join(
            hest_data_dir, "cellvit_seg", f"{sample_id}_cellvit_seg.parquet")
        if os.path.exists(cellvit_path):
            cellvit_df = pd.read_parquet(cellvit_path)
            return len(cellvit_df)
    except Exception as e:
        print(f"Warning: unable to get expected cell count for sample {sample_id}: {e}")
    return None


def is_sample_completed(output_dir, sample_id, expected_num_cells=None):
    """Check whether a sample is completed (including integrity validation).

    Args:
        output_dir: Output directory.
        sample_id: Sample ID.
        expected_num_cells: Optional expected cell count for validation.

    Returns:
        True if the output file looks complete, otherwise False.
    """
    is_complete, num_cells, error_msg = verify_sample_file_integrity(
        output_dir, sample_id, expected_num_cells)
    
    if not is_complete:
        print(f"⚠️  Sample {sample_id} has incomplete files: {error_msg}")
        if num_cells > 0:
            print(f"   Current files contain {num_cells} cells")
    
    return is_complete


def remove_incomplete_files(output_dir, sample_id):
    """Delete incomplete output artifacts (feature NPZ and PCA model file).

    Args:
        output_dir: Output directory.
        sample_id: Sample ID.
    """
    output_file = os.path.join(
        output_dir, f"{sample_id}_combined_features.npz")
    pca_file = os.path.join(output_dir, f"{sample_id}_dino_pca_model.pkl")
    temp_file = output_file + ".tmp.npz"
    
    removed_files = []
    for file_path in [output_file, pca_file, temp_file]:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                removed_files.append(os.path.basename(file_path))
            except Exception as e:
                print(f"⚠️  Failed to delete file {file_path}: {e}")
    
    if removed_files:
        print(f"🗑️  Deleted incomplete files: {', '.join(removed_files)}")


def main_independent_pca_extraction(target_samples=None):
    """Run independent-PCA feature extraction with resume support.

    Args:
        target_samples: Optional list of sample IDs to process. If None, the
            script runs in resume mode and processes all unfinished samples.
            Completed samples are skipped in both modes.
    """

    # Configuration.
    hest_data_dir = "/data/yujk/hovernet2feature/HEST/hest_data"
    output_dir = "/data/yujk/hovernet2feature/hest_dinov3_other_cancer"

    # Ensure output directory exists.
    os.makedirs(output_dir, exist_ok=True)

    # Discover available samples.
    all_samples = get_all_hest_samples(hest_data_dir)
    print(f"\nFound {len(all_samples)} available samples: {all_samples}")

    # If a target sample list is provided, validate/filter it.
    if target_samples is not None:
        if not isinstance(target_samples, (list, tuple)):
            raise ValueError("target_samples must be a list or tuple")
        # Deduplicate.
        target_samples = list(set(target_samples))
        # Validate existence.
        invalid_samples = [s for s in target_samples if s not in all_samples]
        if invalid_samples:
            print(f"⚠️  Warning: the following samples do not exist and will be ignored: {invalid_samples}")
        # Keep valid samples only.
        target_samples = [s for s in target_samples if s in all_samples]
        if not target_samples:
            print("❌ Error: no valid samples were found in the requested sample list")
            return
        print(f"\n📋 Explicit sample selection mode: processing {len(target_samples)} requested samples")
        print(f"   Requested samples: {sorted(target_samples)}")
        # Use the specified list as the candidate set.
        candidate_samples = target_samples
    else:
        print(f"\n🔄 Automatic resume mode: processing all unfinished samples")
        candidate_samples = all_samples

    # Load progress state.
    progress = load_progress(output_dir)
    completed_samples = set(progress.get('completed_samples', []))
    failed_samples = set(progress.get('failed_samples', []))

    # Validate outputs on disk (integrity check).
    print(f"\n=== Verifying the integrity of existing files ===")
    file_completed_samples = set()
    incomplete_samples = []
    
    for sample_id in candidate_samples:
        # Expected cell count from segmentation file (best-effort).
        expected_num_cells = get_expected_num_cells(hest_data_dir, sample_id)
        
        # Validate file integrity.
        if is_sample_completed(output_dir, sample_id, expected_num_cells):
            file_completed_samples.add(sample_id)
        else:
            # Track incomplete-but-existing outputs.
            output_file = os.path.join(
                output_dir, f"{sample_id}_combined_features.npz")
            if os.path.exists(output_file):
                incomplete_samples.append(sample_id)
    
    # Auto-fix incomplete outputs by deleting and reprocessing.
    if incomplete_samples:
        print(f"\n⚠️  Found {len(incomplete_samples)} incomplete files; they will be deleted and reprocessed automatically:")
        for sample_id in incomplete_samples:
            print(f"  - {sample_id}")
            remove_incomplete_files(output_dir, sample_id)

    # Merge progress sources.
    all_completed = completed_samples.union(file_completed_samples)

    # Determine remaining samples.
    remaining_samples = [s for s in candidate_samples if s not in all_completed]

    print(f"\n=== Processing status ===")
    if target_samples is not None:
        print(f"Requested sample count: {len(target_samples)}")
    print(f"Candidate sample count: {len(candidate_samples)}")
    print(f"Completed samples: {len(all_completed)} - {sorted(list(all_completed))}")
    print(f"Failed samples: {len(failed_samples)} - {sorted(list(failed_samples))}")
    print(f"Remaining samples: {len(remaining_samples)} - {sorted(remaining_samples)}")

    if not remaining_samples:
        if target_samples is not None:
            print("✅ All requested samples have already been processed!")
        else:
            print("✅ All samples have already been processed!")
        return

    # Optional prompt to resume (auto mode only).
    if target_samples is None and all_completed:
        print(f"\nDetected {len(all_completed)} completed samples")
        try:
            resume = input("Resume from the checkpoint and process the remaining samples? (y/n, default y): ").strip().lower()
            if resume in ['n', 'no']:
                print("User chose to restart processing from scratch")
                # Reset progress.
                remaining_samples = candidate_samples
                progress = {'completed_samples': [], 'failed_samples': []}
                save_progress(output_dir, progress)
        except KeyboardInterrupt:
            print("\nOperation cancelled by the user")
            return

    test_samples = remaining_samples  # Process remaining samples only.

    print("=== HEST spatial data independent PCA feature extraction (DINOv3, resumable) ===")
    print(f"Data directory: {hest_data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Feature setup: DINOv3 only, with independent PCA to 128 dims for each sample")
    print(f"Using level-1 WSI resolution (~0.5 μm/pixel)")
    print(f"48x48 pixel patches covering a physical area of 24x24 μm")

    # Create the feature extractor (auto-tuned parameters).
    try:
        # Auto-tune concurrency (cap threads).
        num_workers = min(mp.cpu_count(), 16)

        # GPU memory-based batch sizing.
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(
                0).total_memory / 1024**3
            print(f"Detected GPU memory: {gpu_memory_gb:.1f} GB")

            # Adjust batch sizes based on GPU memory.
            if gpu_memory_gb >= 24:  # 24GB+
                dino_batch_size = 512
                cell_batch_size = 100000
            elif gpu_memory_gb >= 16:  # 16-24GB
                dino_batch_size = 384
                cell_batch_size = 80000
            elif gpu_memory_gb >= 12:  # 12-16GB
                dino_batch_size = 256
                cell_batch_size = 60000
            elif gpu_memory_gb >= 8:   # 8-12GB
                dino_batch_size = 128
                cell_batch_size = 40000
            else:  # <8GB
                dino_batch_size = 64
                cell_batch_size = 20000
        else:
            print("No GPU detected; using CPU mode")
            dino_batch_size = 32
            cell_batch_size = 10000

        print(f"Automatically optimized configuration:")
        print(f"  - CPU cores: {mp.cpu_count()},, worker count: {num_workers}")
        print(f"  - DINOv3 batch size: {dino_batch_size}")
        print(f"  - Cell batch size: {cell_batch_size:,}")

        # DINOv3 weights path.
        dinov3_model_path = "/data/yujk/hovernet2feature/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"

        extractor = HESTCellFeatureExtractor(
            hest_data_dir=hest_data_dir,
            output_dir=output_dir,
            dinov3_model_path=dinov3_model_path,
            bulk_pca_model_path=None,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            dinov3_feature_dim=1024,
            dino_batch_size=dino_batch_size,
            cell_batch_size=cell_batch_size,
            num_workers=num_workers,
            assign_spot=True
        )

        # Process all remaining samples.
        print(f"\n=== Starting processing for {len(test_samples)} spatial samples ===")
        print(
            f"Preparing samples: {', '.join(test_samples[:5])}{'...' if len(test_samples) > 5 else ''}")

        sample_results = []

        # Performance tracking.
        import time
        start_time = time.time()

        # Process each sample (independent PCA + resume support).
        for sample_idx, sample_id in enumerate(test_samples):
            print(f"\nSample processing progress: {sample_idx+1}/{len(test_samples)}")
            print(f"{'='*50}")
            print(f"Processing spatial sample: {sample_id}")
            print(f"{'='*50}")

            # Skip if already completed (with integrity validation).
            expected_num_cells = get_expected_num_cells(hest_data_dir, sample_id)
            if is_sample_completed(output_dir, sample_id, expected_num_cells):
                print(f"✅ Sample {sample_id} is already complete and its files are intact; skipping")
                # Add to results for summary stats.
                try:
                    output_file = os.path.join(
                        output_dir, f"{sample_id}_combined_features.npz")
                    data = np.load(output_file)
                    num_cells = data['features'].shape[0]
                    final_feature_dim = data['features'].shape[1]
                    sample_results.append({
                        'sample_id': sample_id,
                        'num_cells': num_cells,
                        'final_feature_dim': final_feature_dim,
                        'output_file': output_file
                    })
                except Exception as e:
                    print(f"⚠️  Failed to read completed sample {sample_id} metadata: {e}")
                    print(f"   Corrupted files will be deleted and the sample will be reprocessed")
                    remove_incomplete_files(output_dir, sample_id)
                else:
                    continue

            # Failure handling: record and continue with next sample.
            try:
                sample_start_time = time.time()
                result = extractor.process_sample_with_independent_pca(
                    sample_id)
                sample_end_time = time.time()
                sample_time = sample_end_time - sample_start_time

                sample_results.append(result)

                # Update progress: add to completed list.
                current_progress = load_progress(output_dir)
                if 'completed_samples' not in current_progress:
                    current_progress['completed_samples'] = []
                if sample_id not in current_progress['completed_samples']:
                    current_progress['completed_samples'].append(sample_id)

                # Remove from failed list if present.
                if 'failed_samples' in current_progress and sample_id in current_progress['failed_samples']:
                    current_progress['failed_samples'].remove(sample_id)

                save_progress(output_dir, current_progress)

                print(f"\nSample {sample_id} completed (elapsed time: {sample_time:.1f} s):")
                print(f"  - Processed cells: {result['num_cells']:,}")
                print(f"  - Final feature dimension: {result['final_feature_dim']}")
                print(f"  - Feature file: {result['output_file']}")
                print(f"  - Processing speed: {result['num_cells']/sample_time:.0f} cells/s")

                # Live resource snapshot.
                resource_info = extractor.monitor_resources()
                print(f"  - Current resource usage: {resource_info}")

            except Exception as e:
                print(f"\n❌ Sample {sample_id} failed: {e}")
                print(f"   Skipping this sample and continuing with the next one...")
                import traceback
                print(f"   Error details: {traceback.format_exc()[:300]}...")

                # Update progress: add to failed list.
                current_progress = load_progress(output_dir)
                if 'failed_samples' not in current_progress:
                    current_progress['failed_samples'] = []
                if sample_id not in current_progress['failed_samples']:
                    current_progress['failed_samples'].append(sample_id)
                save_progress(output_dir, current_progress)

                continue  # Skip failed sample and continue.

        # Final progress update.
        final_progress = load_progress(output_dir)
        if 'completed_samples' not in final_progress:
            final_progress['completed_samples'] = []

        # Ensure all successful samples are listed as completed.
        for result in sample_results:
            if result['sample_id'] not in final_progress['completed_samples']:
                final_progress['completed_samples'].append(result['sample_id'])

        save_progress(output_dir, final_progress)

        # Abort if nothing succeeded.
        if not sample_results:
            print("\n❌ All samples failed; cannot continue")
            return

        if len(sample_results) < len(test_samples):
            failed_samples = set(test_samples) - \
                set([r['sample_id'] for r in sample_results])
            print(f"\n⚠️  The following samples failed: {failed_samples}")

        # Aggregate timing and throughput stats.
        end_time = time.time()
        total_time = end_time - start_time
        total_cells = sum(result['num_cells'] for result in sample_results)

        # Batch performance report.
        print(f"\n{'='*80}")
        print("=== Independent PCA processing for spatial data completed - performance report ===")
        print(f"{'='*80}")

        print(f"🏆 Batch processing statistics:")
        print(f"  - Successfully processed samples: {len(sample_results)}")
        print(f"  - Skipped/failed samples: {len(test_samples) - len(sample_results)}")
        print(f"  - Successfully processed samples: {[r['sample_id'] for r in sample_results]}")

        print(f"⚡ Performance statistics:")
        print(f"  - Total processing time: {total_time:.1f} s ({total_time/60:.1f} min)")
        print(f"  - Total processed cells: {total_cells:,}")
        print(f"  - Average processing speed: {total_cells/total_time:.0f} cells/s")
        print(f"  - Average time per sample: {total_time/len(sample_results):.1f} s")
        print(f"  - Average cells per sample: {total_cells//len(sample_results):,}")

        print(f"📊 Save statistics:")
        print(f"  - Successfully saved samples: {len(sample_results)}/{len(test_samples)}")
        print(f"  - Save success rate: {len(sample_results)/len(test_samples)*100:.1f}%")

        print(f"✅ Independent PCA configuration:")
        print(f"  - Processed samples: {len(sample_results)}/{len(test_samples)}")
        print(f"  - Feature dimension per cell: {extractor.final_feature_dim} (sample-specific PCA)")
        print(f"  - WSI resolution: level 1 (~0.5 μm/pixel)")
        print(
            f"  - Patch size: {extractor.cell_patch_size}×{extractor.cell_patch_size} pixels (24x24 μm)")
        print(f"  - Image source: real WSI cell images only")
        print(f"  - Output directory: {output_dir}")
        print(
            f"  - Performance tuning: DINOv3 batch size {extractor.dino_batch_size},, cell batch size {extractor.cell_batch_size:,},, multithreading with {extractor.num_workers}")

        print(f"\n📄 Sample details:")
        for result in sample_results:
            print(
                f"  - {result['sample_id']}: {result['num_cells']:,} cells, {result['final_feature_dim']}-dim features")

        # Final resource usage snapshot.
        final_resource_info = extractor.monitor_resources()
        print(f"\n  - Final resource usage: {final_resource_info}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

        # Ensure locals exist to avoid UnboundLocalError in the exception path.
        if 'sample_results' not in locals():
            sample_results = []
        if 'test_samples' not in locals():
            test_samples = []
        if 'extractor' not in locals():
            print("❗ Model initialization failed; cannot continue")
            return

        print(f"\n❗ Processing interrupted; current status:")
        print(f"  - Processed samples: {len(sample_results)}/{len(test_samples)}")
        if len(sample_results) > 0:
            print(
                f"  - Feature dimension per cell: {sample_results[0].get('final_feature_dim', 'N/A')}")
        print(f"  - WSI resolution: level 1 (~0.5 μm/pixel)")
        print(f"  - Patch size: 48×48 pixels (24x24 μm)")
        print(f"  - Image source: real WSI cell images only")
        print(f"  - Output directory: {output_dir}")

        if len(sample_results) > 0:
            print(f"\n📄 Completed sample details:")
            for result in sample_results:
                print(
                    f"  - {result['sample_id']}: {result['num_cells']:,} cells, {result['final_feature_dim']}-dim features")


if __name__ == "__main__":
    # Run independent-PCA extraction (DINOv3 features only).
    print("HEST cell feature extractor - DINOv3 features only (independent PCA)")
    print("Feature setup: DINOv3 768 dims -> PCA reduced to 128 dims")
    print("No morphology features; PCA is trained independently for each sample")
    print()
    
    # ============================================================
    # Usage:
    # 1) Resume mode (process all unfinished samples):
    #    main_independent_pca_extraction()
    #    or main_independent_pca_extraction(target_samples=None)
    #
    # 2) Targeted mode (process only selected samples):
    #    main_independent_pca_extraction(target_samples=['sample1', 'sample2'])
    #    Note: completed samples are skipped in both modes.
    # ============================================================
    
    # Option 1: resume mode (default).
    target_samples = None
    
    # Option 2: specify sample IDs (edit as needed).
    target_samples = [
        "INT1","INT10","INT11","INT12","INT13","INT14","INT15","INT16","INT17","INT18",
        "INT19","INT2","INT20","INT21","INT22","INT23","INT24","INT3","INT4","INT5",
        "INT6","INT7","INT8","INT9",
        "TENX111","TENX147","TENX148","TENX149",
        "NCBI642","NCBI643",
        "NCBI783","NCBI785","TENX95","TENX99",
        "TENX118","TENX141",
        "NCBI681","NCBI682","NCBI683","NCBI684",
        "TENX116","TENX126","TENX140",
        "MEND139","MEND140","MEND141","MEND142","MEND143","MEND144","MEND145","MEND146",
        "MEND147","MEND148","MEND149","MEND150","MEND151","MEND152","MEND153","MEND154",
        "MEND156","MEND157","MEND158","MEND159","MEND160","MEND161","MEND162",
        "ZEN36","ZEN40","ZEN48","ZEN49",
        "TENX115","TENX117"
    ]  # Replace with real sample IDs.
    
    try:
        main_independent_pca_extraction(target_samples=target_samples)
    except KeyboardInterrupt:
        print("\nOperation cancelled by the user")
    except Exception as e:
        print(f"Program execution error: {e}")
        import traceback
        traceback.print_exc()
