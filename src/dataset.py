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
