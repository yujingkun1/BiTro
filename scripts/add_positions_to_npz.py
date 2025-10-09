#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

try:
    import shapely.wkb as wkb
except Exception:
    wkb = None


def extract_positions_from_cellvit(parquet_path):
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f'cellvit parquet not found: {parquet_path}')
    df = pd.read_parquet(parquet_path)
    xs, ys = [], []
    if 'geometry' in df.columns and wkb is not None:
        for geom_bytes in df['geometry']:
            try:
                geom = wkb.loads(geom_bytes)
                c = geom.centroid
                xs.append(float(c.x))
                ys.append(float(c.y))
            except Exception:
                xs.append(np.nan)
                ys.append(np.nan)
    else:
        # Fallback to grid-like default if geometry missing; keeps index order
        n = len(df)
        xs = [float(i % 100) for i in range(n)]
        ys = [float(i // 100) for i in range(n)]
    pos = np.stack([np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)], axis=1)
    return pos


def add_positions(npz_in_path, positions, out_path=None):
    data = np.load(npz_in_path, allow_pickle=True)
    items = {k: data[k] for k in data.files}
    if 'positions' in items:
        return False  # already has positions
    features = items.get('features', None)
    if features is None:
        raise ValueError('NPZ missing features array')
    n = features.shape[0]
    if positions.shape[0] != n:
        raise ValueError(f'positions length {positions.shape[0]} != features length {n}')
    items['positions'] = positions.astype(np.float32)
    target_path = out_path if out_path else npz_in_path
    np.savez_compressed(target_path, **items)
    return True


def main():
    parser = argparse.ArgumentParser(description='Add positions into NPZ feature files')
    parser.add_argument('--features-dir', required=True, help='Directory with {sample}_combined_features.npz')
    parser.add_argument('--hest-data-dir', required=True, help='HEST data dir (for cellvit_seg)')
    parser.add_argument('--samples', nargs='*', default=None, help='Sample IDs to process (default: all)')
    parser.add_argument('--out-dir', default=None, help='Output dir (default: in-place overwrite)')
    args = parser.parse_args()

    if args.samples:
        sample_ids = args.samples
    else:
        sample_ids = []
        for fn in os.listdir(args.features_dir):
            if fn.endswith('_combined_features.npz'):
                sample_ids.append(fn.replace('_combined_features.npz', ''))
        sample_ids.sort()

    os.makedirs(args.out_dir, exist_ok=True) if args.out_dir else None

    for sid in sample_ids:
        npz_path = os.path.join(args.features_dir, f'{sid}_combined_features.npz')
        if not os.path.exists(npz_path):
            print(f'[skip] {sid}: npz not found')
            continue
        try:
            data = np.load(npz_path, allow_pickle=True)
            if 'positions' in data.files:
                print(f'[ok] {sid}: positions already present')
                continue
        except Exception as e:
            print(f'[error] {sid}: cannot open npz: {e}')
            continue

        parquet_path = os.path.join(args.hest_data_dir, 'cellvit_seg', f'{sid}_cellvit_seg.parquet')
        try:
            pos = extract_positions_from_cellvit(parquet_path)
            out_path = os.path.join(args.out_dir, f'{sid}_combined_features.npz') if args.out_dir else None
            modified = add_positions(npz_path, pos, out_path=out_path)
            if modified:
                print(f'[write] {sid}: positions added -> {out_path or npz_path}')
        except Exception as e:
            print(f'[error] {sid}: {e}')


if __name__ == '__main__':
    main()
