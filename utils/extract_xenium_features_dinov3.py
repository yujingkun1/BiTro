#!/usr/bin/env python3
"""
Xenium single-cell feature extraction (reusing the DINOv3 extractor).

This script reuses the DINOv3 extraction logic from
``extract_spatial_features_dinov3.py`` and only replaces the sample data loading
method to adapt to Xenium exports (``cells.csv`` and ``cell_boundaries.csv``).
"""
import os
import pandas as pd
import numpy as np
from shapely.geometry import Polygon
import shapely.wkb as wkb

# Reuse HESTCellFeatureExtractor from the spatial extractor.
# Prefer local import for script execution; fall back to package import.
try:
    from extract_spatial_features_dinov3 import HESTCellFeatureExtractor
except Exception:
    from Cell2Gene.utils.extract_spatial_features_dinov3 import HESTCellFeatureExtractor


class XeniumCellFeatureExtractor(HESTCellFeatureExtractor):
    """
    Xenium adapter for HESTCellFeatureExtractor.

    This subclass only overrides ``load_sample_data`` to read Xenium exported
    files. It assumes Xenium files are located under a fixed layout like:
    - Image: ``<xenium_root_dir>/Xenium_FFPE_Human_Breast_Cancer_Rep1_he_image.tif``
    - cells.csv: ``<xenium_root_dir>/outs/cells.csv``
    - cell_boundaries.csv: ``<xenium_root_dir>/outs/cell_boundaries.csv``
    - cell_feature_matrix.h5: ``<xenium_root_dir>/outs/cell_feature_matrix.h5``

    The returned dict format matches what the parent extractor expects:
    it includes ``'wsi_path'`` and ``'cellvit_df'`` (with a ``'geometry'`` column
    containing WKB bytes).
    """

    def __init__(self, xenium_root_dir, output_dir, *args, **kwargs):
        # xenium_root_dir example: /data/yujk/hovernet2feature/xenium
        super().__init__(hest_data_dir=xenium_root_dir, output_dir=output_dir, *args, **kwargs)
        self.xenium_root_dir = xenium_root_dir

    def load_sample_data(self, sample_id=None):
        """
        Build sample_data for Xenium.

        Args:
            sample_id: Ignored. Xenium uses fixed paths under ``xenium_root_dir``.
        """
        xenium_outs = os.path.join(self.xenium_root_dir, "outs")
        cells_csv = os.path.join(xenium_outs, "cells.csv")
        boundaries_csv = os.path.join(xenium_outs, "cell_boundaries.csv")

        wsi_path = os.path.join(self.xenium_root_dir, "Xenium_FFPE_Human_Breast_Cancer_Rep1_he_image.tif")
        if not os.path.exists(wsi_path):
            raise FileNotFoundError(f"Xenium image not found: {wsi_path}")

        if not os.path.exists(cells_csv):
            raise FileNotFoundError(f"cells.csv not found: {cells_csv}")
        if not os.path.exists(boundaries_csv):
            raise FileNotFoundError(f"cell_boundaries.csv not found: {boundaries_csv}")

        # Cell centers (Xenium cells.csv).
        cells_df = pd.read_csv(cells_csv)
        # Expected columns: cell_id, x_centroid, y_centroid.
        # If your Xenium export uses different names, adjust here.
        if 'cell_id' not in cells_df.columns or 'x_centroid' not in cells_df.columns or 'y_centroid' not in cells_df.columns:
            raise RuntimeError("cells.csv is missing expected columns: 'cell_id','x_centroid','y_centroid'")

        # Cell boundaries -> Polygon per cell_id.
        bnd_df = pd.read_csv(boundaries_csv)
        if 'cell_id' not in bnd_df.columns or 'vertex_x' not in bnd_df.columns or 'vertex_y' not in bnd_df.columns:
            raise RuntimeError("cell_boundaries.csv is missing expected columns: 'cell_id','vertex_x','vertex_y'")

        polygons = {}
        # Group by cell_id and build polygons (keep row order).
        for cid, group in bnd_df.groupby('cell_id'):
            coords = list(zip(group['vertex_x'].astype(float).tolist(), group['vertex_y'].astype(float).tolist()))
            try:
                poly = Polygon(coords)
                if not poly.is_valid:
                    poly = poly.buffer(0)
                polygons[int(cid)] = poly
            except Exception:
                # If polygon construction fails, use a tiny square as fallback.
                cx = cells_df.loc[cells_df['cell_id'] == cid, 'x_centroid'].values
                cy = cells_df.loc[cells_df['cell_id'] == cid, 'y_centroid'].values
                if len(cx) and len(cy):
                    x0, y0 = float(cx[0]), float(cy[0])
                    poly = Polygon([(x0-1, y0-1), (x0+1, y0-1), (x0+1, y0+1), (x0-1, y0+1)])
                else:
                    poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
                polygons[int(cid)] = poly

        # Build a DataFrame compatible with the parent extractor:
        # it must contain a 'geometry' column with WKB bytes per row.
        geom_list = []
        ids = []
        for _, row in cells_df.iterrows():
            cid = int(row['cell_id'])
            ids.append(cid)
            poly = polygons.get(cid, None)
            if poly is None:
                # Fallback: a tiny square around the centroid.
                cx = float(row['x_centroid'])
                cy = float(row['y_centroid'])
                poly = Polygon([(cx-1, cy-1), (cx+1, cy-1), (cx+1, cy+1), (cx-1, cy+1)])
            geom_list.append(poly.wkb)

        cellvit_df = pd.DataFrame({
            'cell_id': ids,
            'geometry': geom_list,
            # Keep centroids for optional downstream use.
            'x_centroid': cells_df['x_centroid'].astype(float).values,
            'y_centroid': cells_df['y_centroid'].astype(float).values
        })

        sample_data = {
            'wsi_path': wsi_path,
            'cellvit_df': cellvit_df,
            'sample_id': 'XENIUM_SAMPLE'
        }

        print(f"✓ Loaded Xenium data: image={wsi_path}, cells={len(cellvit_df)}")
        return sample_data


def main():
    xenium_root = "/data/yujk/hovernet2feature/xenium"
    output_dir = "/data/yujk/hovernet2feature/xenium_xenium_dinov3_features"
    os.makedirs(output_dir, exist_ok=True)

    # Adjust the DINOv3 model path to your local setup.
    dinov3_model_path = "/data/yujk/hovernet2feature/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"

    # Use conservative worker/batch settings to avoid OpenSlide/IO timeouts.
    extractor = XeniumCellFeatureExtractor(
        xenium_root_dir=xenium_root,
        output_dir=output_dir,
        dinov3_model_path=dinov3_model_path,
        device='cuda' if __import__('torch').cuda.is_available() else 'cpu',
        assign_spot=False,  # Xenium is single-cell; spot assignment can be disabled.
        num_workers=1,
        dino_batch_size=64,
        cell_batch_size=1000
    )

    # Process a single Xenium sample (load_sample_data ignores sample_id).
    result = extractor.process_sample_with_independent_pca(sample_id='XENIUM_SAMPLE')
    print("Done. Result:", result)


if __name__ == "__main__":
    main()


