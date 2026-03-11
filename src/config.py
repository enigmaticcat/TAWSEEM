"""
TAWSEEM Configuration
Constants, paths, and hyperparameters for the TAWSEEM DNA profiling tool.
"""

import os

# ============================================================
# PATHS
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_DIR = os.path.join(PROJECT_ROOT, "PROVEDIt_1-5-Person CSVs Filtered")
DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# Multiplex folder names
MULTIPLEX_FOLDERS = {
    "IDPlus28": "PROVEDIt_1-5-Person CSVs Filtered_3130_IDPlus28cycles",
    "PP16HS32": "PROVEDIt_1-5-Person CSVs Filtered_3130_PP16HS32cycles",
    "GF29":     "PROVEDIt_1-5-Person CSVs Filtered_3500_GF29cycles",
    "IDPlus29": "PROVEDIt_1-5-Person CSVs Filtered_3500_IDPlus29cycles",
}

# ============================================================
# MARKER CONFIGURATIONS (per scenario)
# ============================================================

# All markers per multiplex (before removal of AMEL/Yindel)
MARKERS_GF29 = [
    "AMEL", "CSF1PO", "D10S1248", "D12S391", "D13S317", "D16S539",
    "D18S51", "D19S433", "D1S1656", "D21S11", "D22S1045", "D2S1338",
    "D2S441", "D3S1358", "D5S818", "D7S820", "D8S1179", "DYS391",
    "FGA", "SE33", "TH01", "TPOX", "Yindel", "vWA",
]

MARKERS_IDPLUS = [
    "AMEL", "CSF1PO", "D13S317", "D16S539", "D18S51", "D19S433",
    "D21S11", "D2S1338", "D3S1358", "D5S818", "D7S820", "D8S1179",
    "FGA", "TH01", "TPOX", "vWA",
]

MARKERS_PP16HS = [
    "AMEL", "CSF1PO", "D13S317", "D16S539", "D18S51", "D21S11",
    "D3S1358", "D5S818", "D7S820", "D8S1179", "FGA", "Penta D",
    "Penta E", "TH01", "TPOX", "vWA",
]

# Markers to REMOVE (paper Section 4.2.4: only AMEL and Yindel)
MARKERS_TO_REMOVE = {"AMEL", "Yindel"}

# 14 common markers across all 4 multiplexes (including AMEL)
COMMON_MARKERS_4 = sorted(
    set(MARKERS_GF29) & set(MARKERS_IDPLUS) & set(MARKERS_PP16HS)
)
# 16 common markers across 3 multiplexes: IDPlus28, IDPlus29, GF29
COMMON_MARKERS_3 = sorted(
    set(MARKERS_GF29) & set(MARKERS_IDPLUS)
)

# ============================================================
# SCENARIO CONFIGURATIONS
# ============================================================

SCENARIOS = {
    "single": {
        "name": "Single Multiplex (GF29 25sec)",
        "multiplexes": ["GF29"],
        "injection_times": ["25 sec"],
        "use_common_markers": False,  # Use all GF29 markers
        "markers_to_keep": sorted(set(MARKERS_GF29) - MARKERS_TO_REMOVE),
        "target_profiles_per_class": 156,
        "total_profiles": 780,
    },
    "four": {
        "name": "Four Multiplexes (14 loci)",
        "multiplexes": ["IDPlus28", "PP16HS32", "GF29", "IDPlus29"],
        "injection_times": None,  # Use all injection times
        "use_common_markers": True,
        "markers_to_keep": sorted(set(COMMON_MARKERS_4) - MARKERS_TO_REMOVE),
        "target_profiles_per_class": 1200,
        "total_profiles": 6000,
    },
    "three": {
        "name": "Three Multiplexes (16 loci)",
        "multiplexes": ["IDPlus28", "GF29", "IDPlus29"],
        "injection_times": None,  # Use all injection times
        "use_common_markers": True,
        "markers_to_keep": sorted(set(COMMON_MARKERS_3) - MARKERS_TO_REMOVE),
        "target_profiles_per_class": 1140,
        "total_profiles": 5700,
    },
}

# ============================================================
# DYE ENCODING
# ============================================================
DYE_ENCODING = {"B": 0, "G": 1, "P": 2, "R": 3, "Y": 4}

# Multiplex encoding (for multi-kit scenarios)
MULTIPLEX_ENCODING = {"IDPlus28": 0, "IDPlus29": 1, "GF29": 2, "PP16HS32": 3}

# ============================================================
# PREPROCESSING PARAMETERS
# ============================================================
MAX_ALLELES = 10  # Keep Allele 1-10, drop 11-100 (paper Figure 19)

# ============================================================
# MODEL HYPERPARAMETERS (Table 5 in paper)
# ============================================================
NUM_HIDDEN_LAYERS = 15
HIDDEN_DIM = 64
NUM_DROPOUT_LAYERS = 7
DROPOUT_RATE = 0.2
NUM_CLASSES = 5  # NOC: 1, 2, 3, 4, 5
LEARNING_RATE = 0.001  # Default Adam LR
EPOCHS = 200
BATCH_SIZE = 30
EARLY_STOPPING_PATIENCE = 20  # dừng sớm nếu val accuracy không cải thiện sau N epoch

# ============================================================
# TRAINING PARAMETERS
# ============================================================
TRAIN_RATIO = 0.7   # 70% train, 30% test
NUM_CV_FOLDS = 5     # 5-fold cross-validation
RANDOM_SEED = 42
