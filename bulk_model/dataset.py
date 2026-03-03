import os
import json
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset


def collate_fn_bulk_372(batch):
    slide_ids = [item['slide_id'] for item in batch]
    patient_ids = [item['patient_id'] for item in batch]
    spot_graphs_list = [item['spot_graphs'] for item in batch]
    
    # Ensure all expression tensors have the same length.
    expression_list = [item['expression'] for item in batch]
    if len(expression_list) > 0:
        # Target length is taken from the first tensor.
        target_size = expression_list[0].shape[0] if len(expression_list[0].shape) > 0 else expression_list[0].numel()
        # Pad/truncate to the same length when needed.
        normalized_expressions = []
        for expr in expression_list:
            if expr.shape[0] != target_size:
                if expr.shape[0] < target_size:
                    # Pad zeros.
                    padding = torch.zeros(target_size - expr.shape[0], dtype=expr.dtype, device=expr.device)
                    expr = torch.cat([expr, padding])
                else:
                    # Truncate.
                    expr = expr[:target_size]
            normalized_expressions.append(expr)
        expressions = torch.stack(normalized_expressions)
    else:
        expressions = torch.empty((0,))
    
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


class BulkStaticGraphDataset372(Dataset):
    def __init__(self, graph_data_dir, split='train', selected_genes=None, max_samples=None, fold_config=None, tpm_csv_file=None,
                 apply_gene_normalization: bool = True, normalization_stats: dict = None, normalization_eps: float = 1e-6):
        super().__init__()
        self.graph_data_dir = graph_data_dir
        self.split = split
        self.selected_genes = selected_genes if selected_genes else []
        self.max_samples = max_samples
        self.fold_config = fold_config
        self.tpm_csv_file = tpm_csv_file
        # Gene normalization settings
        self.apply_gene_normalization = apply_gene_normalization
        self._provided_normalization_stats = normalization_stats
        self._normalization_eps = normalization_eps
        self.gene_mean_np = None
        self.gene_std_np = None
        self.gene_mean_tensor = None
        self.gene_std_tensor = None
        self.load_graph_data()
        if self.fold_config:
            self.apply_fold_filter()
        print(f"Loaded split '{split}': {len(self.data_keys)} items")
        if self.selected_genes:
            self.filter_genes()
        # Prepare per-gene normalization stats (if enabled)
        try:
            self.setup_gene_normalization()
        except Exception as e:
            # For non-train splits without provided stats, raise a clearer error
            if self.apply_gene_normalization and self.split != 'train' and self._provided_normalization_stats is None:
                raise
            else:
                print(f"[normalization] setup skipped or failed: {e}")

    def load_graph_data(self):
        print(f"Loading static graph data for split '{self.split}'...")
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

        print("Loading full cell feature tables...")
        if os.path.exists(all_features_file):
            with open(all_features_file, 'rb') as f:
                self.all_cell_features = pickle.load(f)
            print("✓ Loaded DINO features for all cells")
        else:
            print(f"Warning: cell feature file not found: {all_features_file}")
            self.all_cell_features = {}

        if os.path.exists(all_positions_file):
            with open(all_positions_file, 'rb') as f:
                self.all_cell_positions = pickle.load(f)
            print("✓ Loaded spatial coordinates for all cells")
        else:
            print(f"Warning: cell position file not found: {all_positions_file}")
            self.all_cell_positions = {}

        if os.path.exists(cluster_labels_file):
            with open(cluster_labels_file, 'rb') as f:
                self.cluster_labels = pickle.load(f)
            print("✓ Loaded cluster labels for all cells")
        else:
            print(f"Warning: cluster label file not found: {cluster_labels_file}")
            self.cluster_labels = {}

        if os.path.exists(graph_status_file):
            with open(graph_status_file, 'rb') as f:
                self.graph_status = pickle.load(f)
            print("✓ Loaded per-patient graph status")
        else:
            print(f"Warning: graph status file not found: {graph_status_file}")
            self.graph_status = {}

        if os.path.exists(cell_mappings_file):
            with open(cell_mappings_file, 'rb') as f:
                self.cell_to_graph_mappings = pickle.load(f)
            print("✓ Loaded cell-to-graph mappings")
        else:
            print(f"Warning: cell mapping file not found: {cell_mappings_file}")
            self.cell_to_graph_mappings = {}

        if os.path.exists(slide_mappings_file):
            with open(slide_mappings_file, 'rb') as f:
                self.slide_to_patient_mapping = pickle.load(f)
            print("✓ Loaded slide-to-patient mapping")
            self.slide_ids = list(self.intra_patch_graphs.keys())
            self.patient_ids = list(set(self.slide_to_patient_mapping.values()))
            print(f"  - Slides: {len(self.slide_ids)}")
            print(f"  - Patients: {len(self.patient_ids)}")
        else:
            print("Warning: slide mapping file not found; assuming data is organized by patient")
            self.slide_to_patient_mapping = {}
            self.slide_ids = []
            self.patient_ids = list(self.intra_patch_graphs.keys())

        print("Loading TPM data (filtered gene set)...")
        if self.tpm_csv_file is None:
            raise ValueError("tpm_csv_file is required; please provide the TPM CSV path when creating the dataset")
        import pandas as pd
        tpm_df = pd.read_csv(self.tpm_csv_file, index_col=0)
        
        # Security: only load TPM for patients in the current split to avoid
        # data leakage across train/test.
        # First, determine which patient IDs belong to the current split.
        current_split_patient_ids = set()
        if self.slide_to_patient_mapping:
            # Slide-organized data: derive patient IDs from slide_to_patient_mapping.
            current_split_patient_ids = set(self.slide_to_patient_mapping.values())
        else:
            # Patient-organized data: use patient_ids directly.
            current_split_patient_ids = set(self.patient_ids) if hasattr(self, 'patient_ids') else set()
        
        print(f"[split] Using {len(current_split_patient_ids)} patient(s) for split '{self.split}' (leakage protection)")

        self.expressions_data = {}
        self.patient_id_mapping = {}
        
        # Robust patient ID matching: support multiple common formats.
        for full_patient_id in tpm_df.columns:
            matched_patient_id = None
            
            # Try direct match first.
            if full_patient_id in current_split_patient_ids:
                matched_patient_id = full_patient_id
            else:
                # Try matching after format conversions.
                parts = full_patient_id.split('-')
                candidate_ids = [full_patient_id]  # original
                
                if len(parts) >= 4:
                    # TCGA-like: first 4 parts + '-01'
                    candidate_ids.append('-'.join(parts[:4]) + '-01')
                    # Without sample type suffix.
                    candidate_ids.append('-'.join(parts[:4]))
                
                if len(parts) >= 3:
                    # First 3 parts.
                    candidate_ids.append('-'.join(parts[:3]))
                
                # Check if any candidate ID exists in the expected patient set.
                for candidate_id in candidate_ids:
                    if candidate_id in current_split_patient_ids:
                        matched_patient_id = candidate_id
                        break
            
            # If matched, load expression values.
            if matched_patient_id:
                self.expressions_data[matched_patient_id] = tpm_df[full_patient_id].values.astype(np.float32)
                self.patient_id_mapping[matched_patient_id] = full_patient_id
        print(f"✓ Loaded expression data for {len(self.expressions_data)} patient(s) (split '{self.split}' only)")
        
        # Sanity check: ensure data was loaded.
        if len(self.expressions_data) == 0:
            print("Error: no patient expression data was loaded")
            print(f"  Expected patients in split '{self.split}': {len(current_split_patient_ids)}")
            if len(current_split_patient_ids) > 0:
                print(f"  Example expected patient IDs (first 5): {list(current_split_patient_ids)[:5]}")
            print(f"  TPM CSV columns: {len(tpm_df.columns)}")
            if len(tpm_df.columns) > 0:
                print(f"  Example TPM CSV column names (first 5): {list(tpm_df.columns)[:5]}")
                # Diagnostics: check format compatibility.
                sample_expected = list(current_split_patient_ids)[0] if current_split_patient_ids else None
                sample_csv = list(tpm_df.columns)[0] if len(tpm_df.columns) > 0 else None
                if sample_expected and sample_csv:
                    print("  Format diagnostics:")
                    print(f"    - Expected patient ID: {sample_expected}")
                    print(f"    - Example CSV column: {sample_csv}")
                    # Try truncating CSV column format to see if it matches.
                    csv_parts = sample_csv.split('-')
                    if len(csv_parts) >= 4:
                        csv_truncated = '-'.join(csv_parts[:4]) + '-01'
                        print(f"    - Truncated CSV column: {csv_truncated}")
                        print(f"    - Matches expected: {csv_truncated == sample_expected}")
            raise ValueError(
                "No patient expression data could be loaded. Please check:\n"
                f"1. TPM CSV path is correct: {self.tpm_csv_file}\n"
                "2. Patient ID format matches (slide_to_patient_mapping IDs vs TPM CSV column names)\n"
                f"3. Split '{self.split}' contains valid patient data"
            )
        
        sample_patient = list(self.expressions_data.keys())[0]
        sample_sum = np.sum(self.expressions_data[sample_patient])
        print(f"Sanity check - sample patient expression sum: {sample_sum:.2f}")

        with open(metadata_file, 'r') as f:
            self.metadata = json.load(f)

        if self.slide_to_patient_mapping:
            if self.max_samples is not None:
                self.slide_ids = self.slide_ids[:self.max_samples]
            self.data_keys = self.slide_ids
            print(f"✓ Data organized by slide: {len(self.slide_ids)} slides")
        else:
            self.patient_ids = list(self.expressions_data.keys())
            if self.max_samples is not None:
                self.patient_ids = self.patient_ids[:self.max_samples]
            self.data_keys = self.patient_ids
            print(f"✓ Data organized by patient: {len(self.patient_ids)} patients")

        items_with_graphs = 0
        items_without_graphs = 0
        for data_key in self.data_keys:
            has_graphs = self.graph_status.get(data_key, True)
            if has_graphs:
                items_with_graphs += 1
            else:
                items_without_graphs += 1
        print("Dataset stats:")
        if self.slide_to_patient_mapping:
            print(f"  - Total slides: {len(self.data_keys)}")
            print(f"  - Slides with graphs: {items_with_graphs}")
            print(f"  - Slides without graphs: {items_without_graphs} (raw DINO features only)")
        else:
            print(f"  - Total patients: {len(self.data_keys)}")
            print(f"  - Patients with graphs: {items_with_graphs}")
            print(f"  - Patients without graphs: {items_without_graphs} (raw DINO features only)")

        self.feature_dim = self.metadata.get('feature_dim', 128) if isinstance(self.metadata, dict) else 128
        self.original_num_genes = len(list(self.expressions_data.values())[0]) if self.expressions_data else 18080

    def apply_fold_filter(self):
        """Optionally filter data according to fold_config (if provided)."""
        if not self.fold_config:
            return
        # If fold_config is provided, cross-validation filtering can be implemented here.
        # Currently skipped because split-based partitioning is already applied.
        pass

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
        print(f"Gene filtering complete. Final gene count: {self.num_genes}")
        if filtered_expressions:
            sample_patient = list(filtered_expressions.keys())[0]
            sample_data = filtered_expressions[sample_patient]
            sample_total = np.sum(sample_data)
            print(f"TPM sanity check: patient {sample_patient} expression sum: {sample_total:.2f}")

    def setup_gene_normalization(self):
        """
        Compute or load per-gene mean/std for z-score normalization of target TPM values.
        """
        if not self.apply_gene_normalization:
            print("[normalization] Gene normalization disabled")
            return

        if self._provided_normalization_stats is not None:
            print("[normalization] Using provided gene normalization statistics")
            if not isinstance(self._provided_normalization_stats, dict):
                raise TypeError("normalization_stats must be a dict containing 'mean' and 'std'")
            mean = self._provided_normalization_stats.get('mean')
            std = self._provided_normalization_stats.get('std')
            if mean is None or std is None:
                raise ValueError("normalization_stats must include both 'mean' and 'std'")
            mean = np.asarray(mean, dtype=np.float32)
            std = np.asarray(std, dtype=np.float32)
        else:
            if self.split != 'train':
                raise ValueError("Normalization stats must be provided when split is not 'train'")
            if not self.expressions_data:
                raise RuntimeError("Cannot compute normalization stats: expressions_data is empty")
            print("[normalization] Computing gene-wise mean/std from training TPM data")
            stacked = np.stack([v for v in self.expressions_data.values()], axis=0)
            mean = stacked.mean(axis=0).astype(np.float32)
            std = stacked.std(axis=0).astype(np.float32)

        std[std < self._normalization_eps] = 1.0

        self.gene_mean_np = mean
        self.gene_std_np = std
        import torch
        self.gene_mean_tensor = torch.from_numpy(mean.copy())
        self.gene_std_tensor = torch.from_numpy(std.copy())
        self.normalization_stats = {'mean': mean.copy(), 'std': std.copy()}
        print("[normalization] Gene normalization ready")

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
        # Apply per-gene z-score normalization if configured
        if self.apply_gene_normalization and getattr(self, 'gene_mean_tensor', None) is not None and getattr(self, 'gene_std_tensor', None) is not None:
            gm = self.gene_mean_tensor
            gs = self.gene_std_tensor
            # Align lengths if needed (e.g., after gene filtering)
            if gm.shape[0] != expression.shape[0]:
                n = expression.shape[0]
                gm = gm[:n]
                gs = gs[:n]
            expression = (expression - gm) / gs
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

