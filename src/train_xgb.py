import os
import sys
import argparse
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    DATA_PROCESSED_DIR, RESULTS_DIR,
    TRAIN_RATIO, RANDOM_SEED, NUM_CV_FOLDS,
)
from src.data_preprocessing import preprocess_scenario
from src.dataset import prepare_profile_datasets

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    print("XGBoost chưa được cài. Chạy: pip install xgboost")
    sys.exit(1)


# ============================================================
# Cấu hình XGBoost
# ============================================================
XGB_PARAMS = dict(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=RANDOM_SEED,
    eval_metric="mlogloss",
    verbosity=0,
)


def run_xgb_scenario(scenario_name: str, skip_preprocessing: bool = False):
    print(f"\n{'#'*60}")
    print(f"# TAWSEEM XGBoost — Scenario: {scenario_name.upper()}")
    print(f"{'#'*60}")

    processed_path = os.path.join(DATA_PROCESSED_DIR, f"{scenario_name}_processed.csv")

    if skip_preprocessing and os.path.exists(processed_path):
        print(f"\n[1/4] Loading preprocessed data: {processed_path}")
        df = pd.read_csv(processed_path)
    else:
        print(f"\n[1/4] Preprocessing scenario '{scenario_name}'...")
        df = preprocess_scenario(scenario_name)

    print(f"\n[2/4] Feature engineering (profile-level)...")
    train_dataset, test_dataset, scaler, _, cnn_data = prepare_profile_datasets(
        df, train_ratio=TRAIN_RATIO, random_seed=RANDOM_SEED
    )

    X_train = train_dataset.features.numpy()   
    y_train = train_dataset.labels.numpy()      
    X_test  = test_dataset.features.numpy()
    y_test  = test_dataset.labels.numpy()

    _, _, y_train_raw, _, class_weights = cnn_data

    # Compute per-sample weights for XGBoost
    # y_train is 0-4 (shifted), class_weights keys are 1-5
    sample_weight = np.array(
        [class_weights.get(int(y) + 1, 1.0) for y in y_train],
        dtype=np.float32
    )
    print(f"  Class weights: { {k: f'{v:.3f}' for k, v in sorted(class_weights.items())} }")

    print(f"  Train: {X_train.shape[0]} profiles × {X_train.shape[1]} features")
    print(f"  Test:  {X_test.shape[0]} profiles × {X_test.shape[1]} features")

    print(f"\n[3/4] {NUM_CV_FOLDS}-Fold Cross-Validation...")
    model = XGBClassifier(**XGB_PARAMS)
    skf = StratifiedKFold(n_splits=NUM_CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    cv_start = time.time()
    cv_scores = cross_val_score(
        model, X_train, y_train, cv=skf, scoring="accuracy",
        fit_params={"sample_weight": sample_weight},
    )
    cv_time = time.time() - cv_start

    print(f"  CV Accuracy : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"  Per fold    : {[f'{s:.4f}' for s in cv_scores]}")
    print(f"  CV time     : {cv_time:.1f}s")

    print(f"\n[4/4] Training final XGBoost on full train set...")
    train_start = time.time()
    model.fit(X_train, y_train, sample_weight=sample_weight)
    elapsed = time.time() - train_start

    train_preds = model.predict(X_train)
    test_preds  = model.predict(X_test)

    train_acc = accuracy_score(y_train, train_preds)
    test_acc  = accuracy_score(y_test,  test_preds)

    print(f"\n  Train Accuracy : {train_acc:.4f}")
    print(f"  Test  Accuracy : {test_acc:.4f}")
    print(f"  Train time     : {elapsed:.1f}s")

    class_names = [f"{i+1}-Person" for i in range(5)]
    print(f"\n  Classification Report (Test):")
    print(classification_report(y_test, test_preds, target_names=class_names, digits=4))

    importances = model.feature_importances_
    top_idx = np.argsort(importances)[::-1][:15]
    print(f"  Top 15 Feature Importances:")
    for rank, idx in enumerate(top_idx, 1):
        print(f"    {rank:2d}. Feature {idx:4d}: {importances[idx]:.4f}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    cm = confusion_matrix(y_test, test_preds)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title(f"XGBoost — {scenario_name} (Test Acc: {test_acc:.4f})")
    plt.tight_layout()
    cm_path = os.path.join(RESULTS_DIR, f"{scenario_name}_xgb_confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"\n  Confusion matrix saved: {cm_path}")

    return {
        "scenario": scenario_name,
        "cv_acc": cv_scores.mean(),
        "cv_std": cv_scores.std(),
        "train_acc": train_acc,
        "test_acc": test_acc,
        "elapsed_time": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(
        description="TAWSEEM — XGBoost SOTA Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--scenario", type=str, default="single",
        choices=["single", "three", "four", "all"],
        help="Scenario to run (default: single)",
    )
    parser.add_argument(
        "--skip-preprocessing", action="store_true",
        help="Bỏ qua tiền xử lý nếu CSV đã tồn tại trong data/processed/",
    )
    args = parser.parse_args()

    scenarios = ["single", "three", "four"] if args.scenario == "all" else [args.scenario]

    all_results = []
    for scenario in scenarios:
        result = run_xgb_scenario(scenario, skip_preprocessing=args.skip_preprocessing)
        all_results.append(result)

    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("TỔNG HỢP KẾT QUẢ XGBOOST")
        print(f"{'='*60}")
        print(f"  {'Scenario':<10} {'CV Acc':>10} {'Test Acc':>10} {'Time':>8}")
        print(f"  {'-'*42}")
        for r in all_results:
            print(
                f"  {r['scenario']:<10} "
                f"{r['cv_acc']:.4f}±{r['cv_std']:.4f}  "
                f"{r['test_acc']:.4f}    "
                f"{r['elapsed_time']:.1f}s"
            )

    print("\n XGBoost training hoàn thành!")


if __name__ == "__main__":
    main()
