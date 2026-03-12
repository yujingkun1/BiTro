#!/usr/bin/env python3
"""
Add cluster_label to parquet feature files (per-file MiniBatchKMeans).
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans


def append_cluster_labels_to_parquet(feature_dir: Path, n_clusters: int, random_state: int) -> None:
    feature_dir = Path(feature_dir)
    files = sorted(feature_dir.glob("*.parquet"))
    print(f"[INFO] Found {len(files)} parquet files in {feature_dir}")

    for fp in files:
        df = pd.read_parquet(fp)
        feature_cols = sorted(
            [c for c in df.columns if c.startswith("feature_")],
            key=lambda x: int(x.split("_")[1]),
        )
        if len(feature_cols) == 0:
            print(f"[SKIP] no feature_* columns: {fp.name}")
            continue

        if "cluster_label" in df.columns and len(df["cluster_label"]) == len(df):
            print(f"[SKIP] already has cluster_label: {fp.name}")
            continue

        x = df[feature_cols].to_numpy(dtype=np.float32, copy=False)
        if x.shape[0] == 0:
            print(f"[SKIP] empty parquet: {fp.name}")
            continue

        k = min(n_clusters, x.shape[0])
        if k <= 1:
            labels = np.zeros((x.shape[0],), dtype=np.int32)
        else:
            km = MiniBatchKMeans(n_clusters=k, random_state=random_state, batch_size=4096)
            labels = km.fit_predict(x).astype(np.int32)

        df["cluster_label"] = labels
        df.to_parquet(fp, engine="pyarrow", compression="snappy", index=False)
        print(f"[OK] wrote cluster_label: {fp.name}")


def main() -> None:
    source_dir = Path(os.environ.get("CLUSTER_SOURCE_DIR", "."))
    n_clusters = int(os.environ.get("CLUSTER_N_CLUSTERS", "7"))
    random_state = int(os.environ.get("CLUSTER_RANDOM_STATE", "42"))

    if not source_dir.exists():
        print(f"[ERROR] Source directory does not exist: {source_dir}")
        return

    print(f"[INFO] Clustering parquet files in: {source_dir}")
    print(f"[INFO] n_clusters={n_clusters}, random_state={random_state}")
    append_cluster_labels_to_parquet(source_dir, n_clusters, random_state)


if __name__ == "__main__":
    main()
