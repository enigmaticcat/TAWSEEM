# TAWSEEM: DNA Profiling with Deep Learning

Reimplementation of the paper **"TAWSEEM: A Deep-Learning-Based Tool for Estimating the Number of Unknown Contributors in DNA Profiling"** (Electronics 2022, 11, 548).

Uses a **Multilayer Perceptron (MLP)** with PyTorch to classify DNA mixture profiles into 1-5 contributors, achieving ~97% accuracy on the PROVEDIt dataset.

## Quick Start (Google Colab)

1. Upload this repo to Google Drive or clone from GitHub
2. Open `colab_notebook.ipynb` in Google Colab
3. Run all cells — dataset will be auto-downloaded, preprocessed, and trained with GPU

## Quick Start (Local)

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download PROVEDIt dataset
# Place filtered CSVs in: PROVEDIt_1-5-Person CSVs Filtered/

# Run single scenario
python3 src/main.py --scenario single

# Run all 3 scenarios
python3 src/main.py --scenario all

# Skip preprocessing if already done
python3 src/main.py --scenario single --skip-preprocessing
```

## Project Structure

```
src/
├── config.py               # Paths, markers, hyperparameters
├── data_preprocessing.py    # 10-step preprocessing pipeline
├── dataset.py               # PyTorch Dataset + normalization
├── model.py                 # MLP (15 hidden layers, 7 dropout)
├── train.py                 # Training + 5-fold cross-validation
├── evaluate.py              # Metrics + visualization plots
└── main.py                  # CLI entry point
```

## Dataset

**PROVEDIt** (Project Research Openness for Validation with Empirical Data)  
Download: https://lftdi.camden.rutgers.edu/provedit  
File needed: `PROVEDIt_1-5-Person CSVs Filtered.zip` (22MB)

## Three Scenarios

| Scenario | Multiplexes | Profiles | Loci | Target Accuracy |
|---|---|---|---|---|
| Single | GF29 | 780 | 22 | ~90% |
| Four | All 4 | 6,000 | 13 | **~97%** |
| Three | IDPlus28+29+GF29 | 5,700 | 15 | ~96% |

## Reference

Alotaibi, H.; Alsolami, F.; Abozinadah, E.; Mehmood, R. TAWSEEM: A Deep Learning-Based Tool for Estimating the Number of Unknown Contributors in DNA Profiling. *Electronics* 2022, 11, 548.
