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
    Convert per-row data to per-PROFILE data, then split into train/test.
    
    Each profile's 22 markers are concatenated into one feature vector.
    This gives the model visibility of ALL markers at once.
    
    Input:  22 rows  × 70 features per profile (long format)
    Output:  1 row × 1540 features per profile (wide format)
    
    Returns:
        train_dataset, test_dataset, scaler, train_profile_ids
    """
    np.random.seed(random_seed)
    
    profile_col = 'Sample File'
    
    # Features to use per marker (exclude categoricals that are position-implicit)
    per_marker_cols = [c for c in df.columns 
                       if c not in ('NOC', 'Sample File', 'Dye', 'Marker')]
    
    # Sort by Sample File and Marker for consistent ordering
    df_sorted = df.sort_values([profile_col, 'Marker']).reset_index(drop=True)
    
    # Check that all profiles have the same number of markers
    markers_per_profile = df_sorted.groupby(profile_col).size()
    n_markers = markers_per_profile.iloc[0]
    assert markers_per_profile.nunique() == 1, \
        f"Not all profiles have same number of markers: {markers_per_profile.value_counts().to_dict()}"
    
    # Get unique marker names in sorted order (for feature naming)
    marker_names = df_sorted.groupby(profile_col)['Marker'].apply(list).iloc[0]
    
    # Pivot: group by profile, concatenate all marker features
    profiles_list = []
    labels_list = []
    sample_files = []
    
    for sample_file, group in df_sorted.groupby(profile_col, sort=False):
        # Group is already sorted by Marker
        row_features = group[per_marker_cols].values.flatten()
        profiles_list.append(row_features)
        labels_list.append(group['NOC'].iloc[0])
        sample_files.append(sample_file)
    
    X = np.array(profiles_list, dtype=np.float32)
    y = np.array(labels_list, dtype=np.int64)
    sample_files = np.array(sample_files)
    
    n_features_per_marker = len(per_marker_cols)
    print(f"  Profile-level: {len(X)} profiles × {X.shape[1]} features")
    print(f"    ({n_markers} markers × {n_features_per_marker} features/marker)")
    
    # Stratified split by NOC at profile level
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
    
    # Create datasets (fit scaler on train, apply to test)
    train_dataset = DNAProfileDataset(X_train, y_train, fit_scaler=True)
    test_dataset = DNAProfileDataset(X_test, y_test, scaler=train_dataset.scaler, fit_scaler=False)
    
    print(f"  Train: {len(train_idx)} profiles")
    print(f"  Test:  {len(test_idx)} profiles")
    print(f"  Features: {train_dataset.n_features}")
    
    return train_dataset, test_dataset, train_dataset.scaler, train_profile_ids

