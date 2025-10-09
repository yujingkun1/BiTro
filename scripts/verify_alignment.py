#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd


def load_npz(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    keys = set(data.keys())
    features = data['features'] if 'features' in keys else None
    positions = data['positions'] if 'positions' in keys else None
    spot_ptr = data['spot_ptr'] if 'spot_ptr' in keys else None
    spot_index = data['spot_index'] if 'spot_index' in keys else None
    per_spot = None
    if 'per_spot' in keys:
        try:
            per_spot = data['per_spot'].item()
        except Exception:
            per_spot = None
    return features, positions, spot_ptr, spot_index, per_spot


def quick_report(sample_id, features, positions, per_spot):
    report = {
        'sample_id': sample_id,
        'mode': 'per_spot' if per_spot is not None else 'flat',
        'features_shape': None,
        'positions_shape': None,
        'per_spot_count': None,
        'aligned': False,
        'message': ''
    }
    if per_spot is not None:
        report['per_spot_count'] = len(per_spot)
        # Check each entry has matching lengths
        ok = True
        for si, v in per_spot.items():
            if not isinstance(v, dict) or 'x' not in v or 'pos' not in v:
                ok = False
                break
            x = np.asarray(v['x'])
            p = np.asarray(v['pos'])
            if x.shape[0] != p.shape[0] or p.ndim != 2 or p.shape[1] != 2:
                ok = False
                break
        report['aligned'] = ok
        report['message'] = 'per_spot entries aligned' if ok else 'per_spot entries misaligned'
        return report

    if features is not None:
        report['features_shape'] = tuple(features.shape)
    if positions is not None:
        report['positions_shape'] = tuple(positions.shape)
    if features is None or positions is None:
        report['message'] = 'missing features or positions'
        return report

    n_feat = features.shape[0]
    n_pos = positions.shape[0]
    if n_feat == n_pos and positions.ndim == 2 and positions.shape[1] == 2:
        report['aligned'] = True
        report['message'] = 'features and positions lengths match'
    else:
        report['aligned'] = False
        report['message'] = f'length mismatch: features={n_feat}, positions={n_pos}'
    return report


def main():
    parser = argparse.ArgumentParser(description='Verify alignment between features and positions in NPZ files')
    parser.add_argument('--features-dir', required=True, help='Directory containing {sample}_combined_features.npz')
    parser.add_argument('--samples', nargs='*', default=None, help='Specific sample IDs to check (default: all in dir)')
    args = parser.parse_args()

    features_dir = args.features_dir
    if args.samples:
        sample_ids = args.samples
    else:
        sample_ids = []
        for fn in os.listdir(features_dir):
            if fn.endswith('_combined_features.npz'):
                sample_ids.append(fn.replace('_combined_features.npz', ''))
        sample_ids.sort()

    summaries = []
    for sid in sample_ids:
        npz_path = os.path.join(features_dir, f'{sid}_combined_features.npz')
        if not os.path.exists(npz_path):
            print(f'[skip] {sid}: file not found: {npz_path}')
            continue
        try:
            feats, poss, spot_ptr, spot_index, per_spot = load_npz(npz_path)
            rpt = quick_report(sid, feats, poss, per_spot)
            print(f"{sid}: mode={rpt['mode']}, aligned={rpt['aligned']}, msg={rpt['message']}, "
                  f"features={rpt['features_shape']}, positions={rpt['positions_shape']}, per_spot={rpt['per_spot_count']}")
            summaries.append(rpt)
        except Exception as e:
            print(f'[error] {sid}: {e}')

    # Final summary
    ok = sum(1 for r in summaries if r['aligned'])
    total = len(summaries)
    print(f'Aligned: {ok}/{total}')


if __name__ == '__main__':
    main()
