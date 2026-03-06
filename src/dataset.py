"""
TAWSEEM PyTorch Dataset
Custom Dataset class with MinMaxScaler normalization.
Supports both flat (MLP/XGBoost) and 2D (CNN) profile-level data.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler


class DNAProfileDataset(Dataset):
    """
    PyTorch Dataset for DNA mixture profiles.
    Features are normalized to [0, 1] using MinMaxScaler.
    """
    
    def __init__(self, features, labels, scaler=None, fit_scaler=True):
        if scaler is None:
            self.scaler = MinMaxScaler()
        else:
            self.scaler = scaler
        
        if fit_scaler:
            self.features = self.scaler.fit_transform(features)
        else:
            self.features = self.scaler.transform(features)
        
        self.features = torch.FloatTensor(self.features)
        self.labels = torch.LongTensor(labels - 1)  # NOC 1-5 → 0-4
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
    
    @property
    def n_features(self):
        return self.features.shape[1]


class DNAProfileCNNDataset(Dataset):
    """
    PyTorch Dataset for 1D CNN — returns (n_markers, n_features_per_marker) tensors.
    Each marker's features are scaled independently.
    """
    
    def __init__(self, features_2d, labels, scaler=None, fit_scaler=True):
        """
        Args:
            features_2d: (N, n_markers, n_features_per_marker)
            labels: (N,) with NOC 1-5
        """
        N, M, F = features_2d.shape
        
        # Scale each feature across all samples and markers
        flat = features_2d.reshape(N * M, F)
        if scaler is None:
            self.scaler = MinMaxScaler()
        else:
            self.scaler = scaler
        
        if fit_scaler:
            flat_scaled = self.scaler.fit_transform(flat)
        else:
            flat_scaled = self.scaler.transform(flat)
        
        self.features = torch.FloatTensor(flat_scaled.reshape(N, M, F))
        self.labels = torch.LongTensor(labels - 1)
        self.n_markers = M
        self.n_features_per_marker = F
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # Return (n_features_per_marker, n_markers) for Conv1d — channels first
        return self.features[idx].T, self.labels[idx]


def prepare_datasets(df, train_ratio=0.7, random_seed=42):
    """Split DataFrame into train/test datasets at profile level (legacy)."""
    np.random.seed(random_seed)
    profile_col = 'Sample File'
    profiles = df.groupby(profile_col)['NOC'].first()
    
    train_profiles, test_profiles = [], []
    for noc in sorted(profiles.unique()):
        noc_profiles = profiles[profiles == noc].index.tolist()
        np.random.shuffle(noc_profiles)
        n_train = int(len(noc_profiles) * train_ratio)
        train_profiles.extend(noc_profiles[:n_train])
        test_profiles.extend(noc_profiles[n_train:])
    
    train_df = df[df[profile_col].isin(train_profiles)]
    test_df = df[df[profile_col].isin(test_profiles)]
    
    unique_train_profiles = train_df[profile_col].unique()
    profile_id_map = {name: idx for idx, name in enumerate(unique_train_profiles)}
    train_profile_ids = train_df[profile_col].map(profile_id_map).values.astype(np.int64)
    
    feature_cols = [c for c in df.columns if c not in ('NOC', 'Sample File')]
    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df['NOC'].values.astype(np.int64)
    X_test = test_df[feature_cols].values.astype(np.float32)
    y_test = test_df['NOC'].values.astype(np.int64)
    
    train_dataset = DNAProfileDataset(X_train, y_train, fit_scaler=True)
    test_dataset = DNAProfileDataset(X_test, y_test, scaler=train_dataset.scaler, fit_scaler=False)
    
    print(f"  Train: {len(train_profiles)} profiles, {len(train_dataset)} rows")
    print(f"  Test:  {len(test_profiles)} profiles, {len(test_dataset)} rows")
    return train_dataset, test_dataset, train_dataset.scaler, train_profile_ids


def _extract_marker_features(row, height_cols, ol_cols, missing_cols):
    """Extract rich features from a single marker row."""
    heights = row[height_cols].values.astype(float)
    ol_flags = row[ol_cols].values.astype(float)
    missing_flags = row[missing_cols].values.astype(float)
    
    n_alleles = int((missing_flags == 0).sum())
    n_missing = int(missing_flags.sum())
    n_ol = int(ol_flags.sum())
    
    valid_heights = heights[missing_flags == 0]
    
    if len(valid_heights) > 0:
        sorted_h = np.sort(valid_heights)[::-1]
        h1 = sorted_h[0]
        h2 = sorted_h[1] if len(sorted_h) > 1 else 0
        h3 = sorted_h[2] if len(sorted_h) > 2 else 0
        sum_h = np.sum(valid_heights)
        mean_h = np.mean(valid_heights)
        std_h = np.std(valid_heights) if len(valid_heights) > 1 else 0
        h_ratio = h1 / (h2 + 1e-6) if h2 > 0 else 0
        h_range = h1 - np.min(valid_heights)
    else:
        h1 = h2 = h3 = sum_h = mean_h = std_h = h_ratio = h_range = 0
    
    # 11 features per marker
    return [n_alleles, h1, h2, h3, sum_h, mean_h, std_h, h_ratio, h_range, n_ol, n_missing]


def prepare_profile_datasets(df, train_ratio=0.7, random_seed=42):
    """
    Convert per-row data to per-PROFILE data with rich feature engineering.
    
    Returns:
        train_dataset, test_dataset, scaler, train_profile_ids,
        (X_train_2d, X_test_2d, y_train, y_test, class_weights)
    """
    np.random.seed(random_seed)
    profile_col = 'Sample File'
    
    # Sort and deduplicate
    df_sorted = df.sort_values([profile_col, 'Marker']).reset_index(drop=True)
    before = len(df_sorted)
    df_sorted = df_sorted.drop_duplicates(subset=[profile_col, 'Marker'], keep='first').reset_index(drop=True)
    if len(df_sorted) < before:
        print(f"  Removed {before - len(df_sorted)} duplicate rows")
    
    # Keep profiles with most common marker count
    mpp = df_sorted.groupby(profile_col).size()
    n_markers = mpp.mode().iloc[0]
    valid = mpp[mpp == n_markers].index
    if len(valid) < len(mpp):
        print(f"  Removed {len(mpp) - len(valid)} profiles with inconsistent markers")
        df_sorted = df_sorted[df_sorted[profile_col].isin(valid)].reset_index(drop=True)
    
    # Column groups
    height_cols = [f'Height {i}' for i in range(1, 11) if f'Height {i}' in df.columns]
    ol_cols = [f'OL_ind_{i}' for i in range(1, 11) if f'OL_ind_{i}' in df.columns]
    missing_cols = [f'Missing_Allele_{i}' for i in range(1, 11) if f'Missing_Allele_{i}' in df.columns]
    
    N_PER_MARKER = 11
    N_AGGREGATE = 10
    
    flat_profiles = []
    matrix_profiles = []
    labels = []
    
    for sample_file, group in df_sorted.groupby(profile_col, sort=False):
        marker_feats = []
        for _, row in group.iterrows():
            marker_feats.append(_extract_marker_features(row, height_cols, ol_cols, missing_cols))
        
        marker_feats = np.array(marker_feats, dtype=np.float32)
        allele_counts = marker_feats[:, 0]
        max_heights = marker_feats[:, 1]
        ol_counts = marker_feats[:, 9]
        sum_heights = marker_feats[:, 4]
        
        # Profile aggregates
        aggregates = np.array([
            np.max(allele_counts),               # MAC
            np.mean(allele_counts),
            np.std(allele_counts),
            np.sum(allele_counts >= 3),           # markers with 3+ alleles
            np.sum(allele_counts >= 5),           # markers with 5+ alleles
            np.sum(ol_counts),                   # total OL
            np.mean(max_heights),
            np.std(max_heights),
            np.sum(allele_counts),               # total peaks
            np.sum(sum_heights),                 # total height signal
        ], dtype=np.float32)
        
        flat_profiles.append(np.concatenate([marker_feats.flatten(), aggregates]))
        matrix_profiles.append(marker_feats)
        labels.append(group['NOC'].iloc[0])
    
    X_flat = np.array(flat_profiles, dtype=np.float32)
    X_matrix = np.array(matrix_profiles, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)
    
    print(f"  Profile-level: {len(X_flat)} profiles")
    print(f"    Flat: {X_flat.shape[1]} features ({n_markers}×{N_PER_MARKER} + {N_AGGREGATE})")
    print(f"    CNN:  {X_matrix.shape} (profiles × markers × features)")
    
    # Stratified split
    train_idx, test_idx = [], []
    for noc in sorted(np.unique(y)):
        noc_idx = np.where(y == noc)[0].tolist()
        np.random.shuffle(noc_idx)
        n_train = int(len(noc_idx) * train_ratio)
        train_idx.extend(noc_idx[:n_train])
        test_idx.extend(noc_idx[n_train:])
    
    X_train_flat, y_train = X_flat[train_idx], y[train_idx]
    X_test_flat, y_test = X_flat[test_idx], y[test_idx]
    X_train_2d, X_test_2d = X_matrix[train_idx], X_matrix[test_idx]
    
    train_profile_ids = np.arange(len(train_idx))
    
    # Flat datasets
    train_dataset = DNAProfileDataset(X_train_flat, y_train, fit_scaler=True)
    test_dataset = DNAProfileDataset(X_test_flat, y_test, scaler=train_dataset.scaler, fit_scaler=False)
    
    # Class weights for imbalanced data
    from collections import Counter
    counts = Counter(y_train.tolist())
    total = len(y_train)
    n_classes = len(counts)
    class_weights = {c: total / (n_classes * n) for c, n in counts.items()}
    
    print(f"  Train: {len(train_idx)}, Test: {len(test_idx)}")
    print(f"  Class distribution: {dict(sorted(counts.items()))}")
    
    return train_dataset, test_dataset, train_dataset.scaler, train_profile_ids, \
           (X_train_2d, X_test_2d, y_train, y_test, class_weights)
