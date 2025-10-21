#!/usr/bin/env python3
import os
import argparse
from typing import Optional, Tuple, List

import numpy as np

try:
    import openslide  # type: ignore
except Exception:
    openslide = None  # will fallback to PIL for simple reads

try:
    import tifffile  # type: ignore
except Exception:
    tifffile = None

from PIL import Image
# Disable PIL decompression bomb checks since we only read metadata (size)
Image.MAX_IMAGE_PIXELS = None  # type: ignore


def _read_image_size(path: str) -> Tuple[int, int]:
    """Return (width, height) for an image path using safe metadata reads.

    Prefer tifffile for TIFFs to avoid PIL safety checks; fallback to PIL size.
    """
    # Try tifffile first for TIFFs
    if tifffile is not None:
        try:
            with tifffile.TiffFile(path) as tf:  # type: ignore[attr-defined]
                page = tf.pages[0]
                w = int(page.imagewidth)
                h = int(page.imagelength)
                return (w, h)
        except Exception:
            pass

    # Fallback to PIL (MAX_IMAGE_PIXELS disabled above)
    with Image.open(path) as im:
        return im.size


def _get_raw_wsi_levels(raw_path: str) -> Tuple[List[Tuple[int, int]], Optional[List[float]]]:
    """Return raw WSI level dimensions and downsamples.

    If OpenSlide is available, use it. Otherwise, fallback to PIL as single-level.
    """
    if openslide is not None:
        try:
            slide = openslide.OpenSlide(raw_path)
            dims = list(slide.level_dimensions)
            downsamples = list(slide.level_downsamples)
            slide.close()
            return dims, downsamples
        except Exception:
            pass

    # Fallback: treat as single-level image
    w, h = _read_image_size(raw_path)
    return [(w, h)], None


def _closest_level(norm_wh: Tuple[int, int], raw_levels: List[Tuple[int, int]]) -> int:
    nw, nh = norm_wh
    best_i = 0
    best_score = float('inf')
    for i, (rw, rh) in enumerate(raw_levels):
        # relative size mismatch score
        s = abs(nw - rw) / max(rw, 1) + abs(nh - rh) / max(rh, 1)
        if s < best_score:
            best_score = s
            best_i = i
    return best_i


def analyze(sample: Optional[str], normalized_path: Optional[str], raw_path: Optional[str], base_dir: str) -> None:
    if not normalized_path and not sample:
        raise SystemExit("Provide --sample or --normalized")

    if sample and not normalized_path:
        normalized_path = os.path.join(
            base_dir, "HEST", "hest_data", "wsis_normalized", f"normalized_{sample}.tif")
    if sample and not raw_path:
        raw_path = os.path.join(
            base_dir, "HEST", "hest_data", "wsis", f"{sample}.tif")

    assert normalized_path is not None
    if not os.path.exists(normalized_path):
        raise SystemExit(f"Normalized WSI not found: {normalized_path}")

    norm_w, norm_h = _read_image_size(normalized_path)

    raw_levels: List[Tuple[int, int]] = []
    downsamples: Optional[List[float]] = None
    raw_w0 = raw_h0 = None

    if raw_path and os.path.exists(raw_path):
        raw_levels, downsamples = _get_raw_wsi_levels(raw_path)
        raw_w0, raw_h0 = raw_levels[0]
    else:
        print(
            "Warning: raw WSI not provided or not found. Will only report normalized size.")

    print("=== Normalized vs Raw WSI Level Analysis ===")
    if sample:
        print(f"Sample: {sample}")
    print(f"Normalized path: {normalized_path}")
    print(f"Normalized size (level 0 of normalized): {norm_w} x {norm_h}")

    if raw_path:
        print(
            f"Raw path: {raw_path} {'(not found)' if not os.path.exists(raw_path) else ''}")

    if raw_levels:
        print(f"Raw level count: {len(raw_levels)}")
        for i, (rw, rh) in enumerate(raw_levels):
            ds = f" downsample={downsamples[i]:.4f}" if downsamples is not None else ""
            print(f"  - level {i}: {rw} x {rh}{ds}")

        k = _closest_level((norm_w, norm_h), raw_levels)
        print(f"\nClosest raw level to normalized dimensions: level {k}")

        # Direct normalized->raw level-0 scale factors
        if raw_w0 and raw_h0:
            s0x = raw_w0 / norm_w
            s0y = raw_h0 / norm_h
            print(
                f"Estimated scale normalized_pixel -> raw_level0_pixel: sx={s0x:.6f}, sy={s0y:.6f}")
            if downsamples is not None:
                print(
                    f"Raw level {k} theoretical downsample vs level0: {downsamples[k]:.6f}")
    else:
        print("Raw WSI info unavailable. Cannot determine closest level.")

    print("\nMapping formulas:")
    print("- If a point is given in normalized pixels (xn, yn), convert to raw level-0 as:")
    print("    x0 = xn * (raw_w0 / norm_w),  y0 = yn * (raw_h0 / norm_h)")
    print("- If a point is given in raw level-L pixels (xL, yL) and you know level L downsample dL (level0->L):")
    print("    x0 = xL * dL,  y0 = yL * dL")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check which raw WSI level best matches a normalized TIFF and compute scaling.")
    parser.add_argument("--sample", type=str, default=None,
                        help="Sample ID, e.g., MISC33. Builds default paths under base_dir.")
    parser.add_argument("--normalized", type=str, default=None,
                        help="Explicit path to normalized TIFF.")
    parser.add_argument("--raw", type=str, default=None,
                        help="Explicit path to raw WSI (SVS/TIFF).")
    parser.add_argument("--base_dir", type=str, default="/data/yujk/hovernet2feature",
                        help="Project base dir for default paths.")
    args = parser.parse_args()

    analyze(args.sample, args.normalized, args.raw, args.base_dir)


if __name__ == "__main__":
    main()
