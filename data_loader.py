# utils/data_loader.py
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
import chardet


def detect_encoding(file_path):
    """Detect the encoding of a file"""
    try:
        with open(file_path, 'rb') as f:
            raw_data = f.read()
            result = chardet.detect(raw_data)
            encoding = result['encoding']
            confidence = result['confidence']
            print(f"Detected encoding for {file_path}: {encoding} (confidence: {confidence:.2f})")
            return encoding
    except Exception as e:
        print(f"Error detecting encoding for {file_path}: {e}")
        return 'utf-8'


def read_csv_with_encoding(file_path):
    """Read CSV file with automatic encoding detection"""
    # Try different encodings in order of likelihood
    encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'big5', 'latin-1', 'cp1252']

    # First try to detect encoding
    detected_encoding = detect_encoding(file_path)
    if detected_encoding and detected_encoding not in encodings:
        encodings.insert(0, detected_encoding)
    elif detected_encoding:
        # Move detected encoding to front
        encodings.remove(detected_encoding)
        encodings.insert(0, detected_encoding)

    for encoding in encodings:
        try:
            print(f"Trying to read {file_path} with encoding: {encoding}")
            df = pd.read_csv(file_path, encoding=encoding)
            print(f"Successfully read {file_path} with encoding: {encoding}")
            print(f"Shape: {df.shape}, Columns: {list(df.columns)}")
            return df
        except UnicodeDecodeError as e:
            print(f"Failed with {encoding}: {e}")
            continue
        except Exception as e:
            print(f"Error with {encoding}: {e}")
            continue

    raise ValueError(f"Could not read {file_path} with any of the tried encodings: {encodings}")


class HeteroNetworkData:
    def __init__(self, config):
        self.config = config

        # Read CSV files with encoding detection
        print(f"Reading {config.data_dir + config.circ_disease_file}")
        self.circ_disease_df = read_csv_with_encoding(config.data_dir + config.circ_disease_file)

        print(f"Reading {config.data_dir + config.circ_mirna_file}")
        self.circ_mirna_df = read_csv_with_encoding(config.data_dir + config.circ_mirna_file)

        print(f"Reading {config.data_dir + config.mirna_disease_file}")
        self.mirna_disease_df = read_csv_with_encoding(config.data_dir + config.mirna_disease_file)

        # Print data info
        print("\n=== Data Info ===")
        print(f"CircRNA-Disease associations: {self.circ_disease_df.shape}")
        print(f"CircRNA-miRNA associations: {self.circ_mirna_df.shape}")
        print(f"miRNA-Disease associations: {self.mirna_disease_df.shape}")

        # Print column names to help debug
        print(f"CircRNA-Disease columns: {list(self.circ_disease_df.columns)}")
        print(f"CircRNA-miRNA columns: {list(self.circ_mirna_df.columns)}")
        print(f"miRNA-Disease columns: {list(self.mirna_disease_df.columns)}")

        # Process data and build mappings
        self.process_data()

    def process_data(self):
        # Get column names (handle potential encoding issues in column names)
        circ_disease_cols = list(self.circ_disease_df.columns)
        circ_mirna_cols = list(self.circ_mirna_df.columns)
        mirna_disease_cols = list(self.mirna_disease_df.columns)

        # Try to automatically detect the correct column names
        # Common patterns for column names
        circ_col_patterns = ['circRNA', 'circ', 'circular', 'circularRNA']
        disease_col_patterns = ['disease', 'Disease', 'DISEASE']
        mirna_col_patterns = ['miRNA', 'mirna', 'miR', 'microRNA']

        def find_column(df_cols, patterns):
            for pattern in patterns:
                for col in df_cols:
                    if pattern.lower() in col.lower():
                        return col
            # If no pattern matches, return the first column
            return df_cols[0] if df_cols else None

        # Find column names
        circ_col_cd = find_column(circ_disease_cols, circ_col_patterns)
        disease_col_cd = find_column(circ_disease_cols, disease_col_patterns)

        circ_col_cm = find_column(circ_mirna_cols, circ_col_patterns)
        mirna_col_cm = find_column(circ_mirna_cols, mirna_col_patterns)

        mirna_col_md = find_column(mirna_disease_cols, mirna_col_patterns)
        disease_col_md = find_column(mirna_disease_cols, disease_col_patterns)

        print(f"\nDetected column mappings:")
        print(f"CircRNA-Disease: circRNA='{circ_col_cd}', disease='{disease_col_cd}'")
        print(f"CircRNA-miRNA: circRNA='{circ_col_cm}', miRNA='{mirna_col_cm}'")
        print(f"miRNA-Disease: miRNA='{mirna_col_md}', disease='{disease_col_md}'")

        # Create entity mappings
        circrnas_cd = set(self.circ_disease_df[circ_col_cd].unique()) if circ_col_cd else set()
        circrnas_cm = set(self.circ_mirna_df[circ_col_cm].unique()) if circ_col_cm else set()

        diseases_cd = set(self.circ_disease_df[disease_col_cd].unique()) if disease_col_cd else set()
        diseases_md = set(self.mirna_disease_df[disease_col_md].unique()) if disease_col_md else set()

        mirnas_cm = set(self.circ_mirna_df[mirna_col_cm].unique()) if mirna_col_cm else set()
        mirnas_md = set(self.mirna_disease_df[mirna_col_md].unique()) if mirna_col_md else set()

        self.circrnas = sorted(list(circrnas_cd | circrnas_cm))
        self.diseases = sorted(list(diseases_cd | diseases_md))
        self.mirnas = sorted(list(mirnas_cm | mirnas_md))

        # Create id mappings
        self.circ2id = {circ: idx for idx, circ in enumerate(self.circrnas)}
        self.disease2id = {disease: idx for idx, disease in enumerate(self.diseases)}
        self.mirna2id = {mirna: idx for idx, mirna in enumerate(self.mirnas)}

        # Store column names for later use
        self.circ_col_cd = circ_col_cd
        self.disease_col_cd = disease_col_cd
        self.circ_col_cm = circ_col_cm
        self.mirna_col_cm = mirna_col_cm
        self.mirna_col_md = mirna_col_md
        self.disease_col_md = disease_col_md

        # Dimensions
        self.num_circrnas = len(self.circrnas)
        self.num_diseases = len(self.diseases)
        self.num_mirnas = len(self.mirnas)

        print(f"\nEntity counts:")
        print(f"CircRNAs: {self.num_circrnas}")
        print(f"Diseases: {self.num_diseases}")
        print(f"miRNAs: {self.num_mirnas}")

        # Build adjacency matrices
        self.build_adjacency_matrices()

    def build_adjacency_matrices(self):
        # CircRNA-Disease adjacency
        self.circ_disease_adj = np.zeros((self.num_circrnas, self.num_diseases))
        if self.circ_col_cd and self.disease_col_cd:
            for _, row in self.circ_disease_df.iterrows():
                circ_id = self.circ2id.get(row[self.circ_col_cd])
                disease_id = self.disease2id.get(row[self.disease_col_cd])
                if circ_id is not None and disease_id is not None:
                    self.circ_disease_adj[circ_id, disease_id] = 1

        # CircRNA-miRNA adjacency
        self.circ_mirna_adj = np.zeros((self.num_circrnas, self.num_mirnas))
        if self.circ_col_cm and self.mirna_col_cm:
            for _, row in self.circ_mirna_df.iterrows():
                circ_id = self.circ2id.get(row[self.circ_col_cm])
                mirna_id = self.mirna2id.get(row[self.mirna_col_cm])
                if circ_id is not None and mirna_id is not None:
                    self.circ_mirna_adj[circ_id, mirna_id] = 1

        # miRNA-Disease adjacency
        self.mirna_disease_adj = np.zeros((self.num_mirnas, self.num_diseases))
        if self.mirna_col_md and self.disease_col_md:
            for _, row in self.mirna_disease_df.iterrows():
                mirna_id = self.mirna2id.get(row[self.mirna_col_md])
                disease_id = self.disease2id.get(row[self.disease_col_md])
                if mirna_id is not None and disease_id is not None:
                    self.mirna_disease_adj[mirna_id, disease_id] = 1

        print(f"\nAdjacency matrix stats:")
        print(f"CircRNA-Disease: {self.circ_disease_adj.shape}, non-zero: {np.count_nonzero(self.circ_disease_adj)}")
        print(f"CircRNA-miRNA: {self.circ_mirna_adj.shape}, non-zero: {np.count_nonzero(self.circ_mirna_adj)}")
        print(f"miRNA-Disease: {self.mirna_disease_adj.shape}, non-zero: {np.count_nonzero(self.mirna_disease_adj)}")

    def generate_metapaths(self):
        try:
            # 元路径: CircRNA-miRNA-CircRNA (CMC)
            cmc = np.matmul(self.circ_mirna_adj, self.circ_mirna_adj.T)

            # 元路径: Disease-miRNA-Disease (DMD)
            dmd = np.matmul(self.mirna_disease_adj.T, self.mirna_disease_adj)

            # 元路径: CircRNA-miRNA-Disease (CMD)
            cmd = np.matmul(self.circ_mirna_adj, self.mirna_disease_adj)

            # 元路径: Disease-miRNA-CircRNA (DMC)
            dmc = np.matmul(self.mirna_disease_adj.T, self.circ_mirna_adj.T)

            # 元路径: CircRNA-Disease-miRNA-CircRNA (CDMC)
            cdmc = np.matmul(np.matmul(self.circ_disease_adj, self.mirna_disease_adj.T), self.circ_mirna_adj.T)

            # 元路径: Disease-CircRNA-miRNA-Disease (DCMD)
            dcmd = np.matmul(np.matmul(self.circ_disease_adj.T, self.circ_mirna_adj), self.mirna_disease_adj)

            # 打印形状以便调试
            print(f"\nMetapath shapes:")
            print(f"CMC: {cmc.shape}")
            print(f"DMD: {dmd.shape}")
            print(f"CMD: {cmd.shape}")
            print(f"DMC: {dmc.shape}")
            print(f"CDMC: {cdmc.shape}")
            print(f"DCMD: {dcmd.shape}")

            # 归一化矩阵
            metapaths = {
                'CMC': self._normalize_adj(cmc),
                'DMD': self._normalize_adj(dmd),
                'CMD': self._normalize_adj(cmd),
                'DMC': self._normalize_adj(dmc),
                'CDMC': self._normalize_adj(cdmc),
                'DCMD': self._normalize_adj(dcmd)
            }

            return metapaths
        except Exception as e:
            print(f"元路径生成错误: {e}")
            # 打印矩阵形状以便调试
            print(f"circ_mirna_adj形状: {self.circ_mirna_adj.shape}")
            print(f"mirna_disease_adj形状: {self.mirna_disease_adj.shape}")
            print(f"circ_disease_adj形状: {self.circ_disease_adj.shape}")
            raise

    def _normalize_adj(self, adj):
        """Symmetrically normalize adjacency matrix."""
        # 对于非方形矩阵，我们只能对行进行归一化
        if adj.shape[0] != adj.shape[1]:
            print(f"非方形矩阵: {adj.shape}, 仅执行行归一化")
            adj_coo = sp.coo_matrix(adj)
            rowsum = np.array(adj_coo.sum(1)).flatten()
            d_inv = np.zeros_like(rowsum)
            d_inv[rowsum > 0] = 1.0 / rowsum[rowsum > 0]
            d_mat_inv = sp.diags(d_inv)
            return d_mat_inv @ adj_coo

        # 对于方形矩阵，执行对称归一化
        adj_coo = sp.coo_matrix(adj)
        rowsum = np.array(adj_coo.sum(1)).flatten()
        d_inv_sqrt = np.zeros_like(rowsum)
        d_inv_sqrt[rowsum > 0] = 1.0 / np.sqrt(rowsum[rowsum > 0])
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

        try:
            return d_mat_inv_sqrt @ adj_coo @ d_mat_inv_sqrt
        except Exception as e:
            print(f"归一化错误: {e}")
            return adj_coo

    def get_train_val_test_data(self):
        # Get positive and negative samples
        pos_pairs, neg_pairs = self._get_pos_neg_pairs()

        # Combine positive and negative samples
        all_pairs = np.vstack([pos_pairs, neg_pairs])
        all_labels = np.hstack([np.ones(len(pos_pairs)), np.zeros(len(neg_pairs))])

        # Split data
        train_idx, test_idx = train_test_split(
            np.arange(len(all_pairs)),
            test_size=self.config.test_ratio,
            stratify=all_labels,
            random_state=self.config.seed
        )

        train_pairs, test_pairs = all_pairs[train_idx], all_pairs[test_idx]
        train_labels, test_labels = all_labels[train_idx], all_labels[test_idx]

        train_idx, val_idx = train_test_split(
            np.arange(len(train_pairs)),
            test_size=self.config.val_ratio / (1 - self.config.test_ratio),
            stratify=train_labels,
            random_state=self.config.seed
        )

        val_pairs = train_pairs[val_idx]
        val_labels = train_labels[val_idx]
        train_pairs = train_pairs[train_idx]
        train_labels = train_labels[train_idx]

        return (
            (train_pairs, train_labels),
            (val_pairs, val_labels),
            (test_pairs, test_labels)
        )

    def _get_pos_neg_pairs(self):
        """Get positive and negative circRNA-disease pairs"""
        pos_pairs = []
        for i in range(self.num_circrnas):
            for j in range(self.num_diseases):
                if self.circ_disease_adj[i, j] == 1:
                    pos_pairs.append([i, j])

        pos_pairs = np.array(pos_pairs)

        # Generate negative samples (equal number as positive samples)
        neg_pairs = []
        while len(neg_pairs) < len(pos_pairs):
            i = np.random.randint(0, self.num_circrnas)
            j = np.random.randint(0, self.num_diseases)
            if self.circ_disease_adj[i, j] == 0:
                neg_pairs.append([i, j])

        neg_pairs = np.array(neg_pairs)

        return pos_pairs, neg_pairs

    def check_data_dimensions(self):
        """Check and print data dimensions for debugging"""
        print(f"\n=== Data Dimensions ===")
        print(f"Number of circRNAs: {self.num_circrnas}")
        print(f"Number of diseases: {self.num_diseases}")
        print(f"Number of miRNAs: {self.num_mirnas}")
        print(f"circ_disease_adj shape: {self.circ_disease_adj.shape}")
        print(f"circ_mirna_adj shape: {self.circ_mirna_adj.shape}")
        print(f"mirna_disease_adj shape: {self.mirna_disease_adj.shape}")

        # Check for empty matrices
        print(f"circ_disease_adj non-zero elements: {np.count_nonzero(self.circ_disease_adj)}")
        print(f"circ_mirna_adj non-zero elements: {np.count_nonzero(self.circ_mirna_adj)}")
        print(f"mirna_disease_adj non-zero elements: {np.count_nonzero(self.mirna_disease_adj)}")


class CircDiseaseDataset(Dataset):
    def __init__(self, pairs, labels):
        self.pairs = pairs
        self.labels = labels

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return {
            'circ_idx': self.pairs[idx, 0],
            'disease_idx': self.pairs[idx, 1],
            'label': self.labels[idx]
        }


def create_dataloaders(data, config):
    train_data, val_data, test_data = data.get_train_val_test_data()

    train_dataset = CircDiseaseDataset(train_data[0], train_data[1])
    val_dataset = CircDiseaseDataset(val_data[0], val_data[1])
    test_dataset = CircDiseaseDataset(test_data[0], test_data[1])

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)

    return train_loader, val_loader, test_loader