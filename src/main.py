"""
TAWSEEM: Main Entry Point
Run the full pipeline: preprocess → train (with CV) → evaluate → plot.
"""

import os
import sys
import argparse
import torch
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import DATA_PROCESSED_DIR, RESULTS_DIR, TRAIN_RATIO, RANDOM_SEED
from src.data_preprocessing import preprocess_scenario
from src.dataset import prepare_datasets
from src.train import cross_validate, train_final_model
from src.evaluate import generate_all_plots, plot_accuracy_comparison


def run_scenario(scenario_name, skip_preprocessing=False, skip_cv=False):
    """Run full pipeline for one scenario."""
    
    print(f"\n{'#'*60}")
    print(f"# TAWSEEM — Scenario: {scenario_name}")
    print(f"{'#'*60}")
    
    # --- Step 1: Preprocessing ---
    processed_path = os.path.join(DATA_PROCESSED_DIR, f"{scenario_name}_processed.csv")
    
    if skip_preprocessing and os.path.exists(processed_path):
        print(f"\nLoading preprocessed data from {processed_path}")
        df = pd.read_csv(processed_path)
    else:
        df = preprocess_scenario(scenario_name)
    
    # --- Step 2: Prepare PyTorch datasets ---
    print(f"\nPreparing PyTorch datasets...")
    train_dataset, test_dataset, scaler, train_profile_ids = prepare_datasets(
        df, train_ratio=TRAIN_RATIO, random_seed=RANDOM_SEED
    )
    
    n_features = train_dataset.n_features
    
    # --- Step 3: Device setup ---
    device = torch.device('cuda' if torch.cuda.is_available() else 
                          'mps' if torch.backends.mps.is_available() else 
                          'cpu')
    print(f"Using device: {device}")
    
    # --- Step 4: Cross-validation ---
    if not skip_cv:
        cv_accuracies = cross_validate(
            train_dataset, n_features, device, scenario_name,
            profile_ids=train_profile_ids
        )
    
    # --- Step 5: Train final model ---
    model, train_metrics, test_metrics, elapsed = train_final_model(
        train_dataset, test_dataset, n_features, device, scenario_name
    )
    
    # --- Step 6: Generate plots ---
    generate_all_plots(train_metrics, test_metrics, scenario_name)
    
    return {
        'train_acc': train_metrics['accuracy'],
        'test_acc': test_metrics['accuracy'],
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'elapsed_time': elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description='TAWSEEM DNA Profiling Tool')
    parser.add_argument('--scenario', type=str, default='single',
                        choices=['single', 'three', 'four', 'all'],
                        help='Scenario to run (default: single)')
    parser.add_argument('--skip-preprocessing', action='store_true',
                        help='Skip preprocessing if processed CSV exists')
    parser.add_argument('--skip-cv', action='store_true',
                        help='Skip cross-validation (faster)')
    args = parser.parse_args()
    
    if args.scenario == 'all':
        scenarios = ['single', 'three', 'four']
    else:
        scenarios = [args.scenario]
    
    all_results = {}
    
    for scenario in scenarios:
        results = run_scenario(
            scenario, 
            skip_preprocessing=args.skip_preprocessing,
            skip_cv=args.skip_cv,
        )
        all_results[scenario] = results
    
    # If multiple scenarios, create comparison plot
    if len(all_results) > 1:
        print(f"\n{'='*50}")
        print("COMPARISON ACROSS SCENARIOS")
        print(f"{'='*50}")
        
        os.makedirs(RESULTS_DIR, exist_ok=True)
        plot_accuracy_comparison(
            all_results,
            os.path.join(RESULTS_DIR, 'accuracy_comparison.png')
        )
        
        for name, res in all_results.items():
            print(f"  {name}: Train Acc={res['train_acc']:.4f}, "
                  f"Test Acc={res['test_acc']:.4f}, "
                  f"Time={res['elapsed_time']:.1f}s")
    
    print("\n✅ TAWSEEM pipeline completed!")
    return all_results


if __name__ == "__main__":
    main()
