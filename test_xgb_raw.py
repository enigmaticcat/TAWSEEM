"""
TEMPORARY TEST: XGBoost on raw concatenated features (1540 features).
Run: python test_xgb_raw.py
Delete this file after testing.
"""
import os, sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.config import RANDOM_SEED, TRAIN_RATIO
from src.data_preprocessing import preprocess_scenario

try:
    from xgboost import XGBClassifier
except ImportError:
    print("pip install xgboost first!")
    sys.exit(1)


def raw_concat_profile(df):
    """Concatenate all markers into 1 vector per profile (raw features)."""
    profile_col = 'Sample File'
    per_marker_cols = [c for c in df.columns if c not in ('NOC', 'Sample File', 'Dye', 'Marker')]
    
    df_sorted = df.sort_values([profile_col, 'Marker']).reset_index(drop=True)
    df_sorted = df_sorted.drop_duplicates(subset=[profile_col, 'Marker'], keep='first').reset_index(drop=True)
    
    # Keep only profiles with most common marker count
    mpp = df_sorted.groupby(profile_col).size()
    n_markers = mpp.mode().iloc[0]
    valid = mpp[mpp == n_markers].index
    df_sorted = df_sorted[df_sorted[profile_col].isin(valid)].reset_index(drop=True)
    
    profiles, labels = [], []
    for sf, group in df_sorted.groupby(profile_col, sort=False):
        profiles.append(group[per_marker_cols].values.flatten())
        labels.append(group['NOC'].iloc[0])
    
    return np.array(profiles, dtype=np.float32), np.array(labels)


def main():
    print("=" * 60)
    print("TEST: XGBoost on raw concatenated features")
    print("=" * 60)
    
    # Preprocess
    df = preprocess_scenario('single')
    X, y = raw_concat_profile(df)
    print(f"\nData: {X.shape[0]} profiles × {X.shape[1]} features")
    
    # Stratified split
    np.random.seed(RANDOM_SEED)
    train_idx, test_idx = [], []
    for noc in sorted(np.unique(y)):
        idx = np.where(y == noc)[0].tolist()
        np.random.shuffle(idx)
        n_train = int(len(idx) * TRAIN_RATIO)
        train_idx.extend(idx[:n_train])
        test_idx.extend(idx[n_train:])
    
    X_train, y_train = X[train_idx], y[train_idx] - 1  # NOC 1-5 → 0-4
    X_test, y_test = X[test_idx], y[test_idx] - 1
    
    # Scale
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    # --- XGBoost on raw features ---
    print("\n--- XGBoost (raw 1540 features) ---")
    xgb = XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.3,  # lower colsample for high-dim
        min_child_weight=3, reg_alpha=0.5, reg_lambda=2.0,
        random_state=RANDOM_SEED, eval_metric='mlogloss', verbosity=0,
    )
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    cv = cross_val_score(xgb, X_train, y_train, cv=skf, scoring='accuracy')
    print(f"CV: {cv.mean():.4f} ± {cv.std():.4f}  {[f'{s:.4f}' for s in cv]}")
    
    xgb.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, xgb.predict(X_train))
    test_acc = accuracy_score(y_test, xgb.predict(X_test))
    print(f"Train: {train_acc:.4f}, Test: {test_acc:.4f}")
    print(classification_report(y_test, xgb.predict(X_test), 
          target_names=[f'{i+1}P' for i in range(5)], digits=4))
    
    # --- Also try RandomForest on raw features ---
    from sklearn.ensemble import RandomForestClassifier
    print("\n--- RandomForest (raw 1540 features) ---")
    rf = RandomForestClassifier(
        n_estimators=1000, max_depth=None, min_samples_split=3,
        max_features=0.1,  # low for high-dim
        random_state=RANDOM_SEED, n_jobs=-1, class_weight='balanced',
    )
    cv = cross_val_score(rf, X_train, y_train, cv=skf, scoring='accuracy')
    print(f"CV: {cv.mean():.4f} ± {cv.std():.4f}  {[f'{s:.4f}' for s in cv]}")
    
    rf.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, rf.predict(X_train))
    test_acc = accuracy_score(y_test, rf.predict(X_test))
    print(f"Train: {train_acc:.4f}, Test: {test_acc:.4f}")
    print(classification_report(y_test, rf.predict(X_test),
          target_names=[f'{i+1}P' for i in range(5)], digits=4))


if __name__ == "__main__":
    main()
