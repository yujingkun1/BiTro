#!/usr/bin/env python3
"""
KMeans 聚类脚本：对 Cell2Gene HEST 的特征文件进行聚类，并将结果写回原 NPZ 文件。

支持两种特征组织格式：
1) per_spot: 单个样本一个 npz，其中包含 key 'per_spot' -> dict[int spot_idx -> { 'x': ndarray[num_cells,D], 'pos': ndarray[num_cells,2] }]
   - 本脚本会将该样本内所有细胞特征拼接做一次全局 KMeans 聚类，然后将每个 spot 的类别切片写回到该 spot 的 dict 中，新增字段 'cluster'。
2) 扁平格式：features(N,D), positions(N,2)，并配合 spot_ptr(M+1) 或 spot_index(N) 来分组至 spot（若存在）。
   - 本脚本会对 features 做 KMeans，并新增键 'cluster_labels'(N,) 写回同一 npz。

注意：
- 写回是“就地替换”（原文件名不变）。如指定 --backup，会在替换前保存同目录下的 .bak 备份。
- 读取时依赖 allow_pickle=True（原工程的数据加载已经这样做）。
"""

import os
import sys
import shutil
import tempfile
from typing import Dict, Tuple, List

import numpy as np

# ====== 用户可编辑配置（无需命令行） ======
# 指向存放 {sample_id}_combined_features.npz 的目录
FEATURES_DIR: str = "/data/yujk/hovernet2feature/hest_dinov3_other_cancer"
# 指定要处理的样本 ID 列表；留空或设为 None 则处理目录下全部匹配文件
SAMPLE_IDS: List[str] | None = None
# 聚类类别数
N_CLUSTERS: int = 7
# 是否在写回前生成 .bak 备份
BACKUP: bool = True

# 若为 True：当检测到已存在且尺寸匹配的聚类结果时，直接跳过该文件
SKIP_ALREADY_CLUSTERED: bool = True


def run_kmeans(features: np.ndarray, n_clusters: int, random_state: int = 42) -> np.ndarray:
    """对给定特征做 KMeans，返回每个样本的聚类标签。

    为兼容性考虑，这里使用 sklearn 的 KMeans（若可用）。如果 sklearn 不可用，则回退到一个简单的 MiniBatchKMeans 实现（如也不可用则报错）。
    """
    try:
        # 优先使用标准 KMeans
        from sklearn.cluster import KMeans
        # 为了兼容不同版本的 n_init 行为，统一指定固定次数
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
        labels = kmeans.fit_predict(features)
        return labels.astype(np.int32)
    except Exception:
        # 回退到 MiniBatchKMeans（如果可用）
        try:
            from sklearn.cluster import MiniBatchKMeans
            mbk = MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state, batch_size=4096)
            labels = mbk.fit_predict(features)
            return labels.astype(np.int32)
        except Exception as e:
            raise RuntimeError(f"无法执行聚类，请安装 scikit-learn。原始错误: {e}")


def process_npz_per_spot(npz_path: str, n_clusters: int, backup: bool = False) -> None:
    """处理包含 'per_spot' 的 npz：全样本拼接聚类，写回每个 spot 的 'cluster' 字段。"""
    npz = np.load(npz_path, allow_pickle=True)
    keys = list(npz.keys())
    per_spot = npz['per_spot'].item()
    if not isinstance(per_spot, dict):
        raise ValueError("'per_spot' 键存在但不是 dict 类型")

    # 收集并拼接所有 spot 的特征
    feature_slices: List[Tuple[int, int, int]] = []  # (spot_idx, start, end)
    feature_list: List[np.ndarray] = []
    cursor = 0
    # 为保证可重复，按 spot_idx 升序遍历
    for spot_idx in sorted(per_spot.keys(), key=lambda x: int(x)):
        entry = per_spot[spot_idx]
        x = np.asarray(entry.get('x'), dtype=np.float32)
        if x.ndim != 2:
            raise ValueError(f"spot {spot_idx} 的 'x' 维度不为2: 形状 {x.shape}")
        n_cells = x.shape[0]
        if n_cells == 0:
            feature_slices.append((int(spot_idx), cursor, cursor))
            continue
        feature_list.append(x)
        start, end = cursor, cursor + n_cells
        feature_slices.append((int(spot_idx), start, end))
        cursor = end

    if cursor == 0:
        print(f"[WARN] {os.path.basename(npz_path)}: per_spot 内无细胞特征，跳过。")
        return

    features_all = np.concatenate(feature_list, axis=0)
    labels_all = run_kmeans(features_all, n_clusters=n_clusters)
    if labels_all.shape[0] != features_all.shape[0]:
        raise RuntimeError("KMeans 输出标签长度与输入样本数不一致")

    # 将标签切片写回各个 spot
    for spot_idx, start, end in feature_slices:
        if end <= start:
            per_spot[spot_idx]['cluster'] = np.zeros((0,), dtype=np.int32)
        else:
            per_spot[spot_idx]['cluster'] = labels_all[start:end].astype(np.int32)

    # 重建保存内容（保持其他键不变）
    save_dict: Dict[str, object] = {}
    for k in keys:
        if k == 'per_spot':
            continue
        save_dict[k] = npz[k]
    save_dict['per_spot'] = per_spot

    _atomic_write_npz(npz_path, save_dict, backup=backup)


def process_npz_flat(npz_path: str, n_clusters: int, backup: bool = False) -> None:
    """处理扁平格式 npz：对 'features' 聚类并新增 'cluster_labels' 写回。"""
    npz = np.load(npz_path, allow_pickle=True)
    keys = list(npz.keys())
    if 'features' not in keys:
        raise ValueError("扁平格式缺少 'features' 键")
    feats = np.asarray(npz['features'], dtype=np.float32)
    if feats.ndim != 2 or feats.shape[0] == 0:
        print(f"[WARN] {os.path.basename(npz_path)}: features 为空或维度异常，跳过。形状: {feats.shape}")
        return

    labels = run_kmeans(feats, n_clusters=n_clusters)
    if labels.shape[0] != feats.shape[0]:
        raise RuntimeError("KMeans 输出标签长度与输入样本数不一致")

    # 重建保存内容（保持其他键不变）
    save_dict: Dict[str, object] = {}
    for k in keys:
        save_dict[k] = npz[k]
    save_dict['cluster_labels'] = labels.astype(np.int32)

    _atomic_write_npz(npz_path, save_dict, backup=backup)


def _atomic_write_npz(target_path: str, arrays: Dict[str, object], backup: bool = False) -> None:
    """原子写回 NPZ：写入临时文件后替换原文件；可选备份。"""
    dir_name = os.path.dirname(target_path)
    base_name = os.path.basename(target_path)
    fd, tmp_path = tempfile.mkstemp(prefix=f".{base_name}.", suffix=".tmp.npz", dir=dir_name)
    os.close(fd)
    try:
        # 保存到临时文件
        np.savez(tmp_path, **arrays)
        # 备份
        if backup:
            backup_path = target_path + ".bak"
            if not os.path.exists(backup_path):
                shutil.copy2(target_path, backup_path)
        # 原子替换
        os.replace(tmp_path, target_path)
        print(f"[OK] 写回: {target_path}")
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def process_one_file(npz_path: str, n_clusters: int, backup: bool) -> None:
    """按结构自动分支处理某个 npz 文件。"""
    try:
        npz = np.load(npz_path, allow_pickle=True)
        keys = set(npz.keys())
    except Exception as e:
        print(f"[ERR] 读取失败: {npz_path}: {e}")
        return

    print(f"处理: {npz_path}")

    # 断点续传：如已存在聚类结果且尺寸匹配，则跳过
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
                print(f"[SKIP] 已存在聚类结果且尺寸匹配: {npz_path}")
                return
        except Exception:
            # 校验失败则继续正常处理
            pass

    if SKIP_ALREADY_CLUSTERED and 'features' in keys and 'positions' in keys:
        try:
            feats = np.asarray(npz['features'])
            if 'cluster_labels' in keys:
                cls = np.asarray(npz['cluster_labels'])
                if cls.shape[0] == feats.shape[0]:
                    print(f"[SKIP] 已存在聚类结果且尺寸匹配: {npz_path}")
                    return
        except Exception:
            pass

    if 'per_spot' in keys:
        process_npz_per_spot(npz_path, n_clusters=n_clusters, backup=backup)
    elif 'features' in keys and 'positions' in keys:
        process_npz_flat(npz_path, n_clusters=n_clusters, backup=backup)
    else:
        print(f"[WARN] 未识别的结构（缺少 'per_spot' 或 'features'+'positions'）: {npz_path}，跳过。")


def discover_npz_in_dir(features_dir: str, sample_ids: List[str]) -> List[str]:
    """在目录中查找需处理的 npz 文件；如提供 sample_ids，则只挑选对应文件。"""
    paths: List[str] = []
    if sample_ids:
        for sid in sample_ids:
            candidate = os.path.join(features_dir, f"{sid}_combined_features.npz")
            if os.path.exists(candidate):
                paths.append(candidate)
            else:
                print(f"[WARN] 未找到样本 {sid} 的特征文件: {candidate}")
        return paths

    # 未指定 sample_ids 时，遍历目录
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
        print(f"[ERR] 请在脚本顶部设置有效的 FEATURES_DIR，目前为: {features_dir}")
        sys.exit(1)

    npz_paths = discover_npz_in_dir(features_dir, sample_ids)
    if not npz_paths:
        print("[WARN] 未发现需要处理的 npz 文件")
        sys.exit(0)

    print(f"待处理文件数: {len(npz_paths)} (n_clusters={n_clusters})")
    for path in npz_paths:
        process_one_file(path, n_clusters=n_clusters, backup=backup)


if __name__ == "__main__":
    main()


