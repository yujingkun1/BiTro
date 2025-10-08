import os
import json
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset


class BulkStaticGraphDataset372(Dataset):
    def __init__(self, graph_data_dir, split='train', selected_genes=None, max_samples=None, fold_config=None):
        super().__init__()
        self.graph_data_dir = graph_data_dir
        self.split = split
        self.selected_genes = selected_genes if selected_genes else []
        self.max_samples = max_samples
        self.fold_config = fold_config

        self.load_graph_data()

        if self.fold_config:
            self.apply_fold_filter()

        print(f"加载{split}集: {len(self.data_keys)}个数据项")

        if self.selected_genes:
            self.filter_genes()

    def load_graph_data(self):
        print(f"加载{self.split}集的静态图数据...")

        intra_graphs_file = os.path.join(self.graph_data_dir, f"bulk_{self.split}_intra_patch_graphs.pkl")
        inter_graphs_file = os.path.join(self.graph_data_dir, f"bulk_{self.split}_inter_patch_graphs.pkl")
        expressions_file = os.path.join(self.graph_data_dir, f"bulk_{self.split}_expressions.pkl")
        metadata_file = os.path.join(self.graph_data_dir, f"bulk_{self.split}_metadata.json")

        all_features_file = os.path.join(self.graph_data_dir, f"bulk_{self.split}_all_cell_features.pkl")
        all_positions_file = os.path.join(self.graph_data_dir, f"bulk_{self.split}_all_cell_positions.pkl")
        cluster_labels_file = os.path.join(self.graph_data_dir, f"bulk_{self.split}_cluster_labels.pkl")
        graph_status_file = os.path.join(self.graph_data_dir, f"bulk_{self.split}_graph_status.pkl")
        cell_mappings_file = os.path.join(self.graph_data_dir, f"bulk_{self.split}_cell_to_graph_mappings.pkl")
        slide_mappings_file = os.path.join(self.graph_data_dir, f"bulk_{self.split}_slide_to_patient_mapping.pkl")

        required_files = [intra_graphs_file, inter_graphs_file, expressions_file, metadata_file]
        for file_path in required_files:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Required file not found: {file_path}")

        with open(intra_graphs_file, 'rb') as f:
            self.intra_patch_graphs = pickle.load(f)
        with open(inter_graphs_file, 'rb') as f:
            self.inter_patch_graphs = pickle.load(f)

        print("加载完整细胞特征数据...")
        if os.path.exists(all_features_file):
            with open(all_features_file, 'rb') as f:
                self.all_cell_features = pickle.load(f)
            print(f"✅ 加载了所有细胞的DINO特征数据")
        else:
            print(f"⚠️ 未找到细胞特征文件: {all_features_file}")
            self.all_cell_features = {}

        if os.path.exists(all_positions_file):
            with open(all_positions_file, 'rb') as f:
                self.all_cell_positions = pickle.load(f)
            print(f"✅ 加载了所有细胞的空间坐标数据")
        else:
            print(f"⚠️ 未找到空间坐标文件: {all_positions_file}")
            self.all_cell_positions = {}

        if os.path.exists(cluster_labels_file):
            with open(cluster_labels_file, 'rb') as f:
                self.cluster_labels = pickle.load(f)
            print(f"✅ 加载了所有细胞的聚类标签数据")
        else:
            print(f"⚠️ 未找到聚类标签文件: {cluster_labels_file}")
            self.cluster_labels = {}

        if os.path.exists(graph_status_file):
            with open(graph_status_file, 'rb') as f:
                self.graph_status = pickle.load(f)
            print(f"✅ 加载了患者图状态数据")
        else:
            print(f"⚠️ 未找到图状态文件: {graph_status_file}")
            self.graph_status = {}

        if os.path.exists(cell_mappings_file):
            with open(cell_mappings_file, 'rb') as f:
                self.cell_to_graph_mappings = pickle.load(f)
            print(f"✅ 加载了细胞到图的映射数据")
        else:
            print(f"⚠️ 未找到细胞映射文件: {cell_mappings_file}")
            self.cell_to_graph_mappings = {}

        if os.path.exists(slide_mappings_file):
            with open(slide_mappings_file, 'rb') as f:
                self.slide_to_patient_mapping = pickle.load(f)
            print(f"✅ 加载了切片到患者的映射数据")
            self.slide_ids = list(self.intra_patch_graphs.keys())
            self.patient_ids = list(set(self.slide_to_patient_mapping.values()))
            print(f"  - 切片数: {len(self.slide_ids)}")
            print(f"  - 涉及患者数: {len(self.patient_ids)}")
        else:
            print(f"⚠️ 未找到切片映射文件，假设数据按患者组织")
            self.slide_to_patient_mapping = {}
            self.slide_ids = []
            self.patient_ids = list(self.intra_patch_graphs.keys())

        print("使用筛选后的897基因TPM数据...")
        tpm_csv_file = "/root/autodl-tmp/tpm-TCGA-COAD-897-million.csv"
        import pandas as pd
        tpm_df = pd.read_csv(tpm_csv_file, index_col=0)

        self.expressions_data = {}
        self.patient_id_mapping = {}
        for full_patient_id in tpm_df.columns:
            parts = full_patient_id.split('-')
            if len(parts) >= 4:
                truncated_id = '-'.join(parts[:4]) + '-01'
                self.expressions_data[truncated_id] = tpm_df[full_patient_id].values.astype(np.float32)
                self.patient_id_mapping[truncated_id] = full_patient_id
            else:
                self.expressions_data[full_patient_id] = tpm_df[full_patient_id].values.astype(np.float32)
                self.patient_id_mapping[full_patient_id] = full_patient_id

        print(f"✅ 加载了 {len(self.expressions_data)} 个患者的897基因表达数据")

        sample_patient = list(self.expressions_data.keys())[0]
        sample_sum = np.sum(self.expressions_data[sample_patient])
        print(f"验证 - 样本患者表达值总和: {sample_sum:.2f}")

        with open(metadata_file, 'r') as f:
            self.metadata = json.load(f)

        if self.slide_to_patient_mapping:
            if self.max_samples is not None:
                self.slide_ids = self.slide_ids[:self.max_samples]
            self.data_keys = self.slide_ids
            print(f"✅ 数据按切片组织: {len(self.slide_ids)} 个切片")
        else:
            self.patient_ids = list(self.expressions_data.keys())
            if self.max_samples is not None:
                self.patient_ids = self.patient_ids[:self.max_samples]
            self.data_keys = self.patient_ids
            print(f"✅ 数据按患者组织: {len(self.patient_ids)} 个患者")

        items_with_graphs = 0
        items_without_graphs = 0
        for data_key in self.data_keys:
            has_graphs = self.graph_status.get(data_key, True)
            if has_graphs:
                items_with_graphs += 1
            else:
                items_without_graphs += 1

        print(f"数据统计:")
        if self.slide_to_patient_mapping:
            print(f"  - 总切片数: {len(self.data_keys)}")
            print(f"  - 有图数据切片: {items_with_graphs}")
            print(f"  - 无图数据切片: {items_without_graphs} (仅使用原始DINO特征)")
        else:
            print(f"  - 总患者数: {len(self.data_keys)}")
            print(f"  - 有图数据患者: {items_with_graphs}")
            print(f"  - 无图数据患者: {items_without_graphs} (仅使用原始DINO特征)")

        self.feature_dim = self.metadata.get('feature_dim', 128) if isinstance(self.metadata, dict) else 128
        self.original_num_genes = len(list(self.expressions_data.values())[0]) if self.expressions_data else 18080

    def filter_genes(self):
        if not self.selected_genes:
            return
        target_gene_count = len(self.selected_genes)
        filtered_expressions = {}
        for patient_id, expression_data in self.expressions_data.items():
            if isinstance(expression_data, np.ndarray):
                filtered_expressions[patient_id] = expression_data[:target_gene_count]
            else:
                filtered_expressions[patient_id] = np.zeros(target_gene_count)
        self.expressions_data = filtered_expressions
        self.num_genes = target_gene_count
        print(f"基因过滤完成，最终基因数量: {self.num_genes}")
        if filtered_expressions:
            sample_patient = list(filtered_expressions.keys())[0]
            sample_data = filtered_expressions[sample_patient]
            sample_total = np.sum(sample_data)
            print(f"TPM数据验证：样本患者 {sample_patient} 表达值总和: {sample_total:.2f}")

    def __len__(self):
        return len(self.data_keys)

    def __getitem__(self, idx):
        data_key = self.data_keys[idx]
        if self.slide_to_patient_mapping:
            slide_id = data_key
            patient_id = self.slide_to_patient_mapping[slide_id]
        else:
            slide_id = data_key
            patient_id = data_key

        intra_graphs = self.intra_patch_graphs.get(data_key, {})
        expression = self.expressions_data.get(patient_id, np.zeros(getattr(self, 'num_genes', self.original_num_genes)))

        all_cell_features = self.all_cell_features.get(data_key, torch.empty((0, self.feature_dim)))
        all_cell_positions = self.all_cell_positions.get(data_key, torch.empty((0, 2)))
        cluster_labels = self.cluster_labels.get(data_key, torch.empty((0,)))
        has_graphs = self.graph_status.get(data_key, False)
        cell_mapping = self.cell_to_graph_mappings.get(data_key, None)

        spot_graphs = list(intra_graphs.values())

        if isinstance(expression, np.ndarray):
            expression = torch.tensor(expression, dtype=torch.float32)
        else:
            expression = torch.tensor(np.zeros(getattr(self, 'num_genes', self.original_num_genes)), dtype=torch.float32)

        if not isinstance(all_cell_features, torch.Tensor):
            all_cell_features = torch.empty((0, self.feature_dim))
        if not isinstance(all_cell_positions, torch.Tensor):
            all_cell_positions = torch.empty((0, 2))
        if not isinstance(cluster_labels, torch.Tensor):
            cluster_labels = torch.empty((0,))

        return {
            'slide_id': slide_id,
            'patient_id': patient_id,
            'spot_graphs': spot_graphs,
            'expression': expression,
            'all_cell_features': all_cell_features,
            'all_cell_positions': all_cell_positions,
            'cluster_labels': cluster_labels,
            'has_graphs': has_graphs,
            'cell_mapping': cell_mapping
        }


def collate_fn_bulk_372(batch):
    slide_ids = [item['slide_id'] for item in batch]
    patient_ids = [item['patient_id'] for item in batch]
    spot_graphs_list = [item['spot_graphs'] for item in batch]
    expressions = torch.stack([item['expression'] for item in batch])
    all_cell_features_list = [item['all_cell_features'] for item in batch]
    all_cell_positions_list = [item['all_cell_positions'] for item in batch]
    cluster_labels_list = [item['cluster_labels'] for item in batch]
    has_graphs_list = [item['has_graphs'] for item in batch]
    cell_mappings_list = [item['cell_mapping'] for item in batch]

    return {
        'slide_ids': slide_ids,
        'patient_ids': patient_ids,
        'spot_graphs_list': spot_graphs_list,
        'expressions': expressions,
        'all_cell_features_list': all_cell_features_list,
        'all_cell_positions_list': all_cell_positions_list,
        'cluster_labels_list': cluster_labels_list,
        'has_graphs_list': has_graphs_list,
        'cell_mappings_list': cell_mappings_list
    }

import os
import json
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset


class BulkStaticGraphDataset372(Dataset):
    def __init__(self, graph_data_dir, split='train', selected_genes=None, max_samples=None, fold_config=None):
        super().__init__()
        self.graph_data_dir = graph_data_dir
        self.split = split
        self.selected_genes = selected_genes if selected_genes else []
        self.max_samples = max_samples
        self.fold_config = fold_config
        self.load_graph_data()
        if self.fold_config:
            self.apply_fold_filter()
        print(f"加载{split}集: {len(self.data_keys)}个数据项")
        if self.selected_genes:
            self.filter_genes()

    def load_graph_data(self):
        print(f"加载{self.split}集的静态图数据...")
        intra_graphs_file = os.path.join(self.graph_data_dir, f"bulk_{self.split}_intra_patch_graphs.pkl")
        inter_graphs_file = os.path.join(self.graph_data_dir, f"bulk_{self.split}_inter_patch_graphs.pkl")
        expressions_file = os.path.join(self.graph_data_dir, f"bulk_{self.split}_expressions.pkl")
        metadata_file = os.path.join(self.graph_data_dir, f"bulk_{self.split}_metadata.json")
        all_features_file = os.path.join(self.graph_data_dir, f"bulk_{self.split}_all_cell_features.pkl")
        all_positions_file = os.path.join(self.graph_data_dir, f"bulk_{self.split}_all_cell_positions.pkl")
        cluster_labels_file = os.path.join(self.graph_data_dir, f"bulk_{self.split}_cluster_labels.pkl")
        graph_status_file = os.path.join(self.graph_data_dir, f"bulk_{self.split}_graph_status.pkl")
        cell_mappings_file = os.path.join(self.graph_data_dir, f"bulk_{self.split}_cell_to_graph_mappings.pkl")
        slide_mappings_file = os.path.join(self.graph_data_dir, f"bulk_{self.split}_slide_to_patient_mapping.pkl")

        required_files = [intra_graphs_file, inter_graphs_file, expressions_file, metadata_file]
        for file_path in required_files:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Required file not found: {file_path}")

        with open(intra_graphs_file, 'rb') as f:
            self.intra_patch_graphs = pickle.load(f)
        with open(inter_graphs_file, 'rb') as f:
            self.inter_patch_graphs = pickle.load(f)

        print("加载完整细胞特征数据...")
        if os.path.exists(all_features_file):
            with open(all_features_file, 'rb') as f:
                self.all_cell_features = pickle.load(f)
            print(f"✅ 加载了所有细胞的DINO特征数据")
        else:
            print(f"⚠️ 未找到细胞特征文件: {all_features_file}")
            self.all_cell_features = {}

        if os.path.exists(all_positions_file):
            with open(all_positions_file, 'rb') as f:
                self.all_cell_positions = pickle.load(f)
            print(f"✅ 加载了所有细胞的空间坐标数据")
        else:
            print(f"⚠️ 未找到空间坐标文件: {all_positions_file}")
            self.all_cell_positions = {}

        if os.path.exists(cluster_labels_file):
            with open(cluster_labels_file, 'rb') as f:
                self.cluster_labels = pickle.load(f)
            print(f"✅ 加载了所有细胞的聚类标签数据")
        else:
            print(f"⚠️ 未找到聚类标签文件: {cluster_labels_file}")
            self.cluster_labels = {}

        if os.path.exists(graph_status_file):
            with open(graph_status_file, 'rb') as f:
                self.graph_status = pickle.load(f)
            print(f"✅ 加载了患者图状态数据")
        else:
            print(f"⚠️ 未找到图状态文件: {graph_status_file}")
            self.graph_status = {}

        if os.path.exists(cell_mappings_file):
            with open(cell_mappings_file, 'rb') as f:
                self.cell_to_graph_mappings = pickle.load(f)
            print(f"✅ 加载了细胞到图的映射数据")
        else:
            print(f"⚠️ 未找到细胞映射文件: {cell_mappings_file}")
            self.cell_to_graph_mappings = {}

        if os.path.exists(slide_mappings_file):
            with open(slide_mappings_file, 'rb') as f:
                self.slide_to_patient_mapping = pickle.load(f)
            print(f"✅ 加载了切片到患者的映射数据")
            self.slide_ids = list(self.intra_patch_graphs.keys())
            self.patient_ids = list(set(self.slide_to_patient_mapping.values()))
            print(f"  - 切片数: {len(self.slide_ids)}")
            print(f"  - 涉及患者数: {len(self.patient_ids)}")
        else:
            print(f"⚠️ 未找到切片映射文件，假设数据按患者组织")
            self.slide_to_patient_mapping = {}
            self.slide_ids = []
            self.patient_ids = list(self.intra_patch_graphs.keys())

        print("使用筛选后的897基因TPM数据...")
        import pandas as pd
        tpm_csv_file = "/root/autodl-tmp/tpm-TCGA-COAD-897-million.csv"
        tpm_df = pd.read_csv(tpm_csv_file, index_col=0)
        self.expressions_data = {}
        self.patient_id_mapping = {}
        for full_patient_id in tpm_df.columns:
            parts = full_patient_id.split('-')
            if len(parts) >= 4:
                truncated_id = '-'.join(parts[:4]) + '-01'
                self.expressions_data[truncated_id] = tpm_df[full_patient_id].values.astype(np.float32)
                self.patient_id_mapping[truncated_id] = full_patient_id
            else:
                self.expressions_data[full_patient_id] = tpm_df[full_patient_id].values.astype(np.float32)
                self.patient_id_mapping[full_patient_id] = full_patient_id
        print(f"✅ 加载了 {len(self.expressions_data)} 个患者的897基因表达数据")
        sample_patient = list(self.expressions_data.keys())[0]
        sample_sum = np.sum(self.expressions_data[sample_patient])
        print(f"验证 - 样本患者表达值总和: {sample_sum:.2f}")

        with open(metadata_file, 'r') as f:
            self.metadata = json.load(f)

        if self.slide_to_patient_mapping:
            if self.max_samples is not None:
                self.slide_ids = self.slide_ids[:self.max_samples]
            self.data_keys = self.slide_ids
            print(f"✅ 数据按切片组织: {len(self.slide_ids)} 个切片")
        else:
            self.patient_ids = list(self.expressions_data.keys())
            if self.max_samples is not None:
                self.patient_ids = self.patient_ids[:self.max_samples]
            self.data_keys = self.patient_ids
            print(f"✅ 数据按患者组织: {len(self.patient_ids)} 个患者")

        items_with_graphs = 0
        items_without_graphs = 0
        for data_key in self.data_keys:
            has_graphs = self.graph_status.get(data_key, True)
            if has_graphs:
                items_with_graphs += 1
            else:
                items_without_graphs += 1
        print(f"数据统计:")
        if self.slide_to_patient_mapping:
            print(f"  - 总切片数: {len(self.data_keys)}")
            print(f"  - 有图数据切片: {items_with_graphs}")
            print(f"  - 无图数据切片: {items_without_graphs} (仅使用原始DINO特征)")
        else:
            print(f"  - 总患者数: {len(self.data_keys)}")
            print(f"  - 有图数据患者: {items_with_graphs}")
            print(f"  - 无图数据患者: {items_without_graphs} (仅使用原始DINO特征)")

        self.feature_dim = self.metadata.get('feature_dim', 128) if isinstance(self.metadata, dict) else 128
        self.original_num_genes = len(list(self.expressions_data.values())[0]) if self.expressions_data else 18080

    def filter_genes(self):
        if not self.selected_genes:
            return
        target_gene_count = len(self.selected_genes)
        filtered_expressions = {}
        for patient_id, expression_data in self.expressions_data.items():
            if isinstance(expression_data, np.ndarray):
                filtered_expressions[patient_id] = expression_data[:target_gene_count]
            else:
                filtered_expressions[patient_id] = np.zeros(target_gene_count)
        self.expressions_data = filtered_expressions
        self.num_genes = target_gene_count
        print(f"基因过滤完成，最终基因数量: {self.num_genes}")
        if filtered_expressions:
            sample_patient = list(filtered_expressions.keys())[0]
            sample_data = filtered_expressions[sample_patient]
            sample_total = np.sum(sample_data)
            print(f"TPM数据验证：样本患者 {sample_patient} 表达值总和: {sample_total:.2f}")

    def __len__(self):
        return len(self.data_keys)

    def __getitem__(self, idx):
        data_key = self.data_keys[idx]
        if self.slide_to_patient_mapping:
            slide_id = data_key
            patient_id = self.slide_to_patient_mapping[slide_id]
        else:
            slide_id = data_key
            patient_id = data_key
        intra_graphs = self.intra_patch_graphs.get(data_key, {})
        expression = self.expressions_data.get(patient_id, np.zeros(getattr(self, 'num_genes', self.original_num_genes)))
        all_cell_features = self.all_cell_features.get(data_key, torch.empty((0, self.feature_dim)))
        all_cell_positions = self.all_cell_positions.get(data_key, torch.empty((0, 2)))
        cluster_labels = self.cluster_labels.get(data_key, torch.empty((0,)))
        has_graphs = self.graph_status.get(data_key, False)
        cell_mapping = self.cell_to_graph_mappings.get(data_key, None)
        spot_graphs = list(intra_graphs.values())
        if isinstance(expression, np.ndarray):
            expression = torch.tensor(expression, dtype=torch.float32)
        else:
            expression = torch.tensor(np.zeros(getattr(self, 'num_genes', self.original_num_genes)), dtype=torch.float32)
        if not isinstance(all_cell_features, torch.Tensor):
            all_cell_features = torch.empty((0, self.feature_dim))
        if not isinstance(all_cell_positions, torch.Tensor):
            all_cell_positions = torch.empty((0, 2))
        if not isinstance(cluster_labels, torch.Tensor):
            cluster_labels = torch.empty((0,))
        return {
            'slide_id': slide_id,
            'patient_id': patient_id,
            'spot_graphs': spot_graphs,
            'expression': expression,
            'all_cell_features': all_cell_features,
            'all_cell_positions': all_cell_positions,
            'cluster_labels': cluster_labels,
            'has_graphs': has_graphs,
            'cell_mapping': cell_mapping
        }


def collate_fn_bulk_372(batch):
    slide_ids = [item['slide_id'] for item in batch]
    patient_ids = [item['patient_id'] for item in batch]
    spot_graphs_list = [item['spot_graphs'] for item in batch]
    expressions = torch.stack([item['expression'] for item in batch])
    all_cell_features_list = [item['all_cell_features'] for item in batch]
    all_cell_positions_list = [item['all_cell_positions'] for item in batch]
    cluster_labels_list = [item['cluster_labels'] for item in batch]
    has_graphs_list = [item['has_graphs'] for item in batch]
    cell_mappings_list = [item['cell_mapping'] for item in batch]
    return {
        'slide_ids': slide_ids,
        'patient_ids': patient_ids,
        'spot_graphs_list': spot_graphs_list,
        'expressions': expressions,
        'all_cell_features_list': all_cell_features_list,
        'all_cell_positions_list': all_cell_positions_list,
        'cluster_labels_list': cluster_labels_list,
        'has_graphs_list': has_graphs_list,
        'cell_mappings_list': cell_mappings_list
    }


