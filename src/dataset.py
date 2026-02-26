"""
TAWSEEM PyTorch Dataset
Custom Dataset class with MinMaxScaler normalization.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler


class DNAProfileDataset(Dataset):
    """
    PyTorch Dataset for DNA mixture profiles.
    
    Each sample is one row (sample-locus pair) with features and NOC label.
    Features are normalized to [0, 1] using MinMaxScaler.
    """
    
    def __init__(self, features, labels, scaler=None, fit_scaler=True):
        """
        Args:
            features: numpy array of shape (n_samples, n_features)
            labels: numpy array of shape (n_samples,) with NOC values (1-5)
            scaler: MinMaxScaler instance (reuse from training set)
            fit_scaler: if True, fit the scaler on this data
        """
        # Normalize features to [0, 1]
        if scaler is None:
            self.scaler = MinMaxScaler()
        else:
            self.scaler = scaler
        
        if fit_scaler:
            self.features = self.scaler.fit_transform(features)
        else:
            self.features = self.scaler.transform(features)
        
        self.features = torch.FloatTensor(self.features)
        # Labels: convert NOC (1-5) to class indices (0-4)
        self.labels = torch.LongTensor(labels - 1)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
    
    @property
    def n_features(self):
        return self.features.shape[1]


def prepare_datasets(df, train_ratio=0.7, random_seed=42):
    """
    Split DataFrame into train/test datasets at the PROFILE level.
    
    Important: Split by profiles (not by rows) to prevent data leakage.
    All rows from the same profile stay in the same split.
    
    Args:
        df: processed DataFrame with features + NOC column
        train_ratio: fraction of profiles for training
        random_seed: random seed for reproducibility
    
    Returns:
        train_dataset, test_dataset, scaler, train_profile_ids
    """
    np.random.seed(random_seed)
    
    # Use 'Sample File' to identify unique profiles
    profile_col = 'Sample File'
    profiles = df.groupby(profile_col)['NOC'].first()
    
    train_profiles = []
    test_profiles = []
    
    # Stratified split by NOC
    for noc in sorted(profiles.unique()):
        noc_profiles = profiles[profiles == noc].index.tolist()
        np.random.shuffle(noc_profiles)
        
        n_train = int(len(noc_profiles) * train_ratio)
        train_profiles.extend(noc_profiles[:n_train])
        test_profiles.extend(noc_profiles[n_train:])
    
    # Split data
    train_df = df[df[profile_col].isin(train_profiles)]
    test_df = df[df[profile_col].isin(test_profiles)]
    
    # Create numeric profile IDs for profile-level CV split
    unique_train_profiles = train_df[profile_col].unique()
    profile_id_map = {name: idx for idx, name in enumerate(unique_train_profiles)}
    train_profile_ids = train_df[profile_col].map(profile_id_map).values.astype(np.int64)
    
    # Separate features and labels (exclude 'Sample File' from features)
    feature_cols = [c for c in df.columns if c not in ('NOC', 'Sample File')]
    
    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df['NOC'].values.astype(np.int64)
    X_test = test_df[feature_cols].values.astype(np.float32)
    y_test = test_df['NOC'].values.astype(np.int64)
    
    # Create datasets (fit scaler on train, apply to test)
    train_dataset = DNAProfileDataset(X_train, y_train, fit_scaler=True)
    test_dataset = DNAProfileDataset(X_test, y_test, scaler=train_dataset.scaler, fit_scaler=False)
    
    print(f"  Train: {len(train_profiles)} profiles, {len(train_dataset)} rows")
    print(f"  Test:  {len(test_profiles)} profiles, {len(test_dataset)} rows")
    print(f"  Features: {train_dataset.n_features}")
    
    return train_dataset, test_dataset, train_dataset.scaler, train_profile_ids


def prepare_profile_datasets(df, train_ratio=0.7, random_seed=42):
    """
    Convert per-row data to per-PROFILE data with smart feature aggregation.
    
    Instead of raw concatenation (22 × 70 = 1540 features), compute 
    meaningful aggregate features per profile that capture the key signals
    for NOC prediction.
    
    Features per marker (22 markers × ~3-4 features = ~66-88):
      - n_alleles: count of non-missing allele positions (key NOC signal)
      - max_height: highest peak intensity
      - mean_height: average peak intensity  
      - std_height: peak height variation (mixture indicator)
      - n_OL: count of OL indicators
      - height_ratio: max/min height ratio (mixture balance indicator)
    
    Profile-level summary (~6-8 features):
      - max_alleles_across_markers: max allele count seen
      - mean_alleles: average allele count
      - total_OL: total OL count across all markers
      - mean_max_height: average of max heights
      - std_max_height: variation of max heights across markers
    
    Returns:
        train_dataset, test_dataset, scaler, train_profile_ids
    """
    np.random.seed(random_seed)
    
    profile_col = 'Sample File'
    
    # Sort for consistent ordering
    df_sorted = df.sort_values([profile_col, 'Marker']).reset_index(drop=True)
    
    # Check all profiles have same number of markers
    markers_per_profile = df_sorted.groupby(profile_col).size()
    n_markers = markers_per_profile.iloc[0]
    assert markers_per_profile.nunique() == 1, \
        f"Not all profiles have same number of markers: {markers_per_profile.value_counts().to_dict()}"
    
    # Identify column groups
    allele_cols = [f'Allele {i}' for i in range(1, 11) if f'Allele {i}' in df.columns]
    height_cols = [f'Height {i}' for i in range(1, 11) if f'Height {i}' in df.columns]
    size_cols = [f'Size {i}' for i in range(1, 11) if f'Size {i}' in df.columns]
    ol_cols = [f'OL_ind_{i}' for i in range(1, 11) if f'OL_ind_{i}' in df.columns]
    missing_cols = [f'Missing_Allele_{i}' for i in range(1, 11) if f'Missing_Allele_{i}' in df.columns]
    
    profiles_list = []
    labels_list = []
    
    for sample_file, group in df_sorted.groupby(profile_col, sort=False):
        profile_features = []
        marker_allele_counts = []
        marker_max_heights = []
        marker_ol_counts = []
        
        for _, row in group.iterrows():
            # Per-marker features
            alleles = row[allele_cols].values.astype(float)
            heights = row[height_cols].values.astype(float)
            ol_flags = row[ol_cols].values.astype(float)
            missing_flags = row[missing_cols].values.astype(float)
            
            # Count of real alleles (non-missing, non-zero)
            n_alleles = int((missing_flags == 0).sum())
            
            # Height statistics (only for present alleles)
            valid_heights = heights[missing_flags == 0]
            if len(valid_heights) > 0:
                max_h = np.max(valid_heights)
                mean_h = np.mean(valid_heights)
                std_h = np.std(valid_heights) if len(valid_heights) > 1 else 0
                min_h = np.min(valid_heights)
                h_ratio = max_h / (min_h + 1e-6)  # height ratio (mixture balance)
            else:
                max_h = mean_h = std_h = h_ratio = 0
            
            # OL count
            n_ol = int(ol_flags.sum())
            
            # Per-marker feature vector
            profile_features.extend([n_alleles, max_h, mean_h, std_h, h_ratio, n_ol])
            
            # Track for profile-level aggregation
            marker_allele_counts.append(n_alleles)
            marker_max_heights.append(max_h)
            marker_ol_counts.append(n_ol)
        
        # Profile-level aggregate features
        marker_allele_counts = np.array(marker_allele_counts)
        marker_max_heights = np.array(marker_max_heights)
        
        profile_features.extend([
            np.max(marker_allele_counts),       # max alleles seen at any marker
            np.mean(marker_allele_counts),      # mean allele count
            np.std(marker_allele_counts),       # variation in allele counts
            np.sum(marker_ol_counts),           # total OL across profile
            np.mean(marker_max_heights),        # average max height
            np.std(marker_max_heights),         # height variation across markers
            np.sum(marker_allele_counts >= 3),  # markers with 3+ alleles (mixture evidence)
            np.sum(marker_allele_counts >= 5),  # markers with 5+ alleles (complex mixture)
        ])
        
        profiles_list.append(profile_features)
        labels_list.append(group['NOC'].iloc[0])
    
    X = np.array(profiles_list, dtype=np.float32)
    y = np.array(labels_list, dtype=np.int64)
    
    n_per_marker = 6  # features per marker
    n_aggregate = 8   # profile-level features
    total_features = n_markers * n_per_marker + n_aggregate
    
    print(f"  Profile-level: {len(X)} profiles × {X.shape[1]} features")
    print(f"    ({n_markers} markers × {n_per_marker} per-marker + {n_aggregate} aggregate)")
    
    # Stratified split by NOC
    train_idx = []
    test_idx = []
    
    for noc in sorted(np.unique(y)):
        noc_idx = np.where(y == noc)[0].tolist()
        np.random.shuffle(noc_idx)
        n_train = int(len(noc_idx) * train_ratio)
        train_idx.extend(noc_idx[:n_train])
        test_idx.extend(noc_idx[n_train:])
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    # Profile IDs for CV splitting
    train_profile_ids = np.arange(len(train_idx))
    
    # Create datasets
    train_dataset = DNAProfileDataset(X_train, y_train, fit_scaler=True)
    test_dataset = DNAProfileDataset(X_test, y_test, scaler=train_dataset.scaler, fit_scaler=False)
    
    print(f"  Train: {len(train_idx)} profiles")
    print(f"  Test:  {len(test_idx)} profiles")
    print(f"  Features: {train_dataset.n_features}")
    
    return train_dataset, test_dataset, train_dataset.scaler, train_profile_ids

