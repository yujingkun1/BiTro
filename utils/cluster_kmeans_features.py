#!/usr/bin/env python3
"""
KMeans clustering for BiTro feature NPZ files.

This script clusters cell-level feature embeddings and writes the cluster
assignments back into the original NPZ files (in-place).

Supported NPZ layouts:
1) Per-spot layout: one NPZ per sample containing a ``per_spot`` dict:
   ``per_spot[spot_idx] -> {'x': (num_cells, D), 'pos': (num_cells, 2), ...}``
   The script concatenates all cells in the sample, runs a single global KMeans,
   and writes labels back to each spot entry as ``'cluster'``.
2) Flat layout: ``features (N, D)`` and ``positions (N, 2)``, optionally with
   ``spot_ptr (M+1)`` or ``spot_index (N)`` for grouping.
   The script runs KMeans on ``features`` and writes ``cluster_labels (N,)``.

Notes:
- Writes are atomic: a temporary file is written and then replaces the original.
  When ``BACKUP=True``, a ``.bak`` copy is created once before the first write.
- Loading uses ``allow_pickle=True`` to preserve dict-like metadata entries.
"""

import os
import sys
import shutil
import tempfile
from typing import Dict, Tuple, List

import numpy as np

# ====== User-editable configuration (no CLI required) ======
# Directory containing ``{sample_id}_combined_features.npz`` files.
FEATURES_DIR: str = "/data/yujk/hovernet2feature/hest_dinov3_other_cancer"
# Sample IDs to process. Set to ``None`` to process all matching files.
SAMPLE_IDS: List[str] | None = None
# Number of KMeans clusters.
N_CLUSTERS: int = 7
# Whether to create a ``.bak`` backup before overwriting.
BACKUP: bool = True

# If True, skip files that already contain compatible clustering outputs.
SKIP_ALREADY_CLUSTERED: bool = True


def run_kmeans(features: np.ndarray, n_clusters: int, random_state: int = 42) -> np.ndarray:
    """Run KMeans (or MiniBatchKMeans fallback) on the given features.

    Args:
        features: Feature matrix of shape (N, D).
        n_clusters: Number of clusters.
        random_state: Random seed for reproducibility.

    Returns:
        Cluster labels of shape (N,), dtype int32.

    Raises:
        RuntimeError: If scikit-learn is unavailable or clustering fails.
    """
    try:
        # Prefer standard KMeans.
        from sklearn.cluster import KMeans
        # Use a fixed n_init for compatibility across scikit-learn versions.
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
        labels = kmeans.fit_predict(features)
        return labels.astype(np.int32)
    except Exception:
        # Fallback to MiniBatchKMeans (if available).
        try:
            from sklearn.cluster import MiniBatchKMeans
            mbk = MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state, batch_size=4096)
            labels = mbk.fit_predict(features)
            return labels.astype(np.int32)
        except Exception as e:
            raise RuntimeError(
                f"Unable to run clustering; please install scikit-learn. Original error: {e}"
            )


def process_npz_per_spot(npz_path: str, n_clusters: int, backup: bool = False) -> None:
    """Process a per-spot NPZ and write per-cell labels into each spot entry."""
    npz = np.load(npz_path, allow_pickle=True)
    keys = list(npz.keys())
    per_spot = npz['per_spot'].item()
    if not isinstance(per_spot, dict):
        raise ValueError("'per_spot' exists but is not a dict")

    # Collect and concatenate features across spots.
    feature_slices: List[Tuple[int, int, int]] = []  # (spot_idx, start, end)
    feature_list: List[np.ndarray] = []
    cursor = 0
    # For reproducibility, iterate spot_idx in ascending order.
    for spot_idx in sorted(per_spot.keys(), key=lambda x: int(x)):
        entry = per_spot[spot_idx]
        x = np.asarray(entry.get('x'), dtype=np.float32)
        if x.ndim != 2:
            raise ValueError(f"spot {spot_idx}: 'x' must be 2D, got shape {x.shape}")
        n_cells = x.shape[0]
        if n_cells == 0:
            feature_slices.append((int(spot_idx), cursor, cursor))
            continue
        feature_list.append(x)
        start, end = cursor, cursor + n_cells
        feature_slices.append((int(spot_idx), start, end))
        cursor = end

    if cursor == 0:
        print(f"[WARN] {os.path.basename(npz_path)}: no cell features under per_spot; skipping.")
        return

    features_all = np.concatenate(feature_list, axis=0)
    labels_all = run_kmeans(features_all, n_clusters=n_clusters)
    if labels_all.shape[0] != features_all.shape[0]:
        raise RuntimeError("KMeans output label count does not match input sample count")

    # Slice labels back into each spot entry.
    for spot_idx, start, end in feature_slices:
        if end <= start:
            per_spot[spot_idx]['cluster'] = np.zeros((0,), dtype=np.int32)
        else:
            per_spot[spot_idx]['cluster'] = labels_all[start:end].astype(np.int32)

    # Rebuild the saved dict (keep other keys unchanged).
    save_dict: Dict[str, object] = {}
    for k in keys:
        if k == 'per_spot':
            continue
        save_dict[k] = npz[k]
    save_dict['per_spot'] = per_spot

    _atomic_write_npz(npz_path, save_dict, backup=backup)


def process_npz_flat(npz_path: str, n_clusters: int, backup: bool = False) -> None:
    """Process a flat NPZ and write ``cluster_labels`` alongside existing arrays."""
    npz = np.load(npz_path, allow_pickle=True)
    keys = list(npz.keys())
    if 'features' not in keys:
        raise ValueError("Flat layout is missing the 'features' key")
    feats = np.asarray(npz['features'], dtype=np.float32)
    if feats.ndim != 2 or feats.shape[0] == 0:
        print(f"[WARN] {os.path.basename(npz_path)}: empty/invalid features array; skipping. Shape: {feats.shape}")
        return

    labels = run_kmeans(feats, n_clusters=n_clusters)
    if labels.shape[0] != feats.shape[0]:
        raise RuntimeError("KMeans output label count does not match input sample count")

    # Rebuild the saved dict (keep other keys unchanged).
    save_dict: Dict[str, object] = {}
    for k in keys:
        save_dict[k] = npz[k]
    save_dict['cluster_labels'] = labels.astype(np.int32)

    _atomic_write_npz(npz_path, save_dict, backup=backup)


def _atomic_write_npz(target_path: str, arrays: Dict[str, object], backup: bool = False) -> None:
    """Atomically write an NPZ file (optionally with a one-time backup)."""
    dir_name = os.path.dirname(target_path)
    base_name = os.path.basename(target_path)
    fd, tmp_path = tempfile.mkstemp(prefix=f".{base_name}.", suffix=".tmp.npz", dir=dir_name)
    os.close(fd)
    try:
        # Write to a temporary file first.
        np.savez(tmp_path, **arrays)
        # Optional one-time backup.
        if backup:
            backup_path = target_path + ".bak"
            if not os.path.exists(backup_path):
                shutil.copy2(target_path, backup_path)
        # Atomic replace.
        os.replace(tmp_path, target_path)
        print(f"[OK] Wrote back: {target_path}")
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def process_one_file(npz_path: str, n_clusters: int, backup: bool) -> None:
    """Detect NPZ layout and apply the appropriate clustering writer."""
    try:
        npz = np.load(npz_path, allow_pickle=True)
        keys = set(npz.keys())
    except Exception as e:
        print(f"[ERR] Failed to read: {npz_path}: {e}")
        return

    print(f"Processing: {npz_path}")

    # Resume-friendly: if clustering outputs already exist and match shapes, skip.
    if SKIP_ALREADY_CLUSTERED and 'per_spot' in keys:
        try:
            per_spot = npz['per_spot'].item()
            already = True
            for spot_idx in per_spot.keys():
                entry = per_spot[spot_idx]
                x = np.asarray(entry.get('x'), dtype=np.float32)
                cl = entry.get('cluster', None)
                if cl is None or getattr(cl, 'shape', None) is None or cl.shape[0] != x.shape[0]:
                    already = False
                    break
            if already:
                print(f"[SKIP] Clustering output exists and matches shapes: {npz_path}")
                return
        except Exception:
            # If validation fails, proceed with normal processing.
            pass

    if SKIP_ALREADY_CLUSTERED and 'features' in keys and 'positions' in keys:
        try:
            feats = np.asarray(npz['features'])
            if 'cluster_labels' in keys:
                cls = np.asarray(npz['cluster_labels'])
                if cls.shape[0] == feats.shape[0]:
                    print(f"[SKIP] Clustering output exists and matches shapes: {npz_path}")
                    return
        except Exception:
            pass

    if 'per_spot' in keys:
        process_npz_per_spot(npz_path, n_clusters=n_clusters, backup=backup)
    elif 'features' in keys and 'positions' in keys:
        process_npz_flat(npz_path, n_clusters=n_clusters, backup=backup)
    else:
        print(f"[WARN] Unrecognized NPZ layout (missing 'per_spot' or 'features'+'positions'): {npz_path}; skipping.")


def discover_npz_in_dir(features_dir: str, sample_ids: List[str]) -> List[str]:
    """Discover NPZ files in a directory (optionally filtered by sample IDs)."""
    paths: List[str] = []
    if sample_ids:
        for sid in sample_ids:
            candidate = os.path.join(features_dir, f"{sid}_combined_features.npz")
            if os.path.exists(candidate):
                paths.append(candidate)
            else:
                print(f"[WARN] Feature file not found for sample {sid}: {candidate}")
        return paths

    # If sample_ids is None, scan the directory.
    for name in os.listdir(features_dir):
        if not name.endswith("_combined_features.npz"):
            continue
        paths.append(os.path.join(features_dir, name))
    paths.sort()
    return paths


def main() -> None:
    features_dir = FEATURES_DIR
    sample_ids = SAMPLE_IDS
    n_clusters = N_CLUSTERS
    backup = BACKUP

    if not features_dir or not os.path.isdir(features_dir):
        print(f"[ERR] Please set a valid FEATURES_DIR at the top of the script. Current: {features_dir}")
        sys.exit(1)

    npz_paths = discover_npz_in_dir(features_dir, sample_ids)
    if not npz_paths:
        print("[WARN] No NPZ files found to process")
        sys.exit(0)

    print(f"Files to process: {len(npz_paths)} (n_clusters={n_clusters})")
    for path in npz_paths:
        process_one_file(path, n_clusters=n_clusters, backup=backup)


if __name__ == "__main__":
    main()


