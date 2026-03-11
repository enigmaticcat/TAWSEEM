# TAWSEEM: DNA Profiling — Number of Contributors Estimation

Reimplementation of **"TAWSEEM: A Deep-Learning-Based Tool for Estimating the Number of Unknown Contributors in DNA Profiling"** (Electronics 2022, 11, 548).

Phân loại hỗn hợp DNA thành 1–5 người đóng góp (NOC) sử dụng **XGBoost** *(SOTA)* và **MLP (PyTorch)** trên tập dữ liệu PROVEDIt.

> 🏆 **SOTA:** XGBoost với profile-level features đạt kết quả tốt nhất trong dự án này.

---

## Mục lục

- [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
- [Cài đặt môi trường](#cài-đặt-môi-trường)
- [Cấu trúc dự án](#cấu-trúc-dự-án)
- [Dữ liệu](#dữ-liệu)
- [🏆 Chạy XGBoost SOTA](#-chạy-xgboost-sota)
- [Chạy pipeline đầy đủ (MLP + XGBoost + RF)](#chạy-pipeline-đầy-đủ)
- [Các tham số cấu hình](#các-tham-số-cấu-hình)
- [Kết quả kỳ vọng](#kết-quả-kỳ-vọng)
- [Tài liệu tham khảo](#tài-liệu-tham-khảo)

---

## Yêu cầu hệ thống

| Yêu cầu | Chi tiết |
|---|---|
| Python | 3.9 trở lên |
| RAM | Tối thiểu 8GB (khuyến nghị 16GB cho scenario `four`) |
| GPU (tùy chọn) | CUDA / Apple MPS — tự động phát hiện |
| Hệ điều hành | macOS / Linux / Windows |

---

## Cài đặt môi trường

```bash
# 1. Clone hoặc di chuyển vào thư mục dự án
cd /path/to/TAWSEEM

# 2. Tạo virtual environment
python3 -m venv .venv

# 3. Kích hoạt virtual environment
# macOS / Linux:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate

# 4. Cài đặt tất cả dependencies (bao gồm XGBoost)
pip install -r requirements.txt
```

### Các thư viện chính (`requirements.txt`)

```
torch>=2.0
pandas
numpy
scikit-learn
matplotlib
seaborn
openpyxl
xgboost
```

> **Lưu ý:** `xgboost` là bắt buộc. Nếu chưa cài, pipeline sẽ tự động dùng `GradientBoostingClassifier` của scikit-learn thay thế.

---

## Cấu trúc dự án

```
TAWSEEM/
├── PROVEDIt_1-5-Person CSVs Filtered/   ← Dữ liệu thô (tải về theo hướng dẫn bên dưới)
├── data/
│   └── processed/                        ← CSV đã tiền xử lý (tự động tạo)
├── results/                              ← Kết quả, biểu đồ, mô hình tốt nhất
├── src/
│   ├── config.py             # Đường dẫn, markers, siêu tham số
│   ├── data_preprocessing.py # Pipeline tiền xử lý 10 bước
│   ├── dataset.py            # PyTorch Dataset + MinMaxScaler + feature engineering
│   ├── model.py              # MLP (15 hidden layers, 7 dropout)
│   ├── train.py              # Huấn luyện + 5-fold cross-validation (MLP)
│   ├── train_xgb.py          # 🏆 XGBoost SOTA — file chạy độc lập
│   ├── tree_models.py        # XGBoost + Random Forest (dùng trong pipeline đầy đủ)
│   ├── evaluate.py           # Metrics + biểu đồ
│   └── main.py               # Entry point CLI (MLP + XGBoost + RF)
├── test_xgb_raw.py           # Test XGBoost trên raw features (1540 features)
├── requirements.txt
└── README.md
```

---

## Dữ liệu

**PROVEDIt** — Project Research Openness for Validation with Empirical Data

1. Tải file `PROVEDIt_1-5-Person CSVs Filtered.zip` (khoảng 22MB) tại:  
   👉 https://lftdi.camden.rutgers.edu/provedit

2. Giải nén vào thư mục gốc của dự án:
   ```
   TAWSEEM/
   └── PROVEDIt_1-5-Person CSVs Filtered/
       ├── PROVEDIt_1-5-Person CSVs Filtered_3500_GF29cycles/
       ├── PROVEDIt_1-5-Person CSVs Filtered_3130_IDPlus28cycles/
       ├── PROVEDIt_1-5-Person CSVs Filtered_3130_PP16HS32cycles/
       └── PROVEDIt_1-5-Person CSVs Filtered_3500_IDPlus29cycles/
   ```

---

## 🏆 Chạy XGBoost SOTA

**`src/train_xgb.py`** là file chạy độc lập, tối ưu cho XGBoost với toàn bộ pipeline:  
tiền xử lý → feature engineering profile-level → 5-fold CV → train → confusion matrix.

### Scenario `single` (nhanh nhất, ~2 phút)

```bash
python3 src/train_xgb.py --scenario single
```

### Scenario `four` (chính xác nhất)

```bash
python3 src/train_xgb.py --scenario four
```

### Tất cả 3 scenario

```bash
python3 src/train_xgb.py --scenario all
```

### Bỏ qua tiền xử lý (nếu đã có CSV)

```bash
python3 src/train_xgb.py --scenario single --skip-preprocessing
```

**Output bao gồm:**
- CV Accuracy ± std (5-fold)
- Train / Test Accuracy
- Classification Report (precision, recall, F1 per class)
- Top 15 Feature Importances
- Confusion matrix PNG lưu vào `results/`

---

## Chạy pipeline đầy đủ (MLP + XGBoost + RF)

Pipeline đầy đủ bao gồm 6 bước:
1. **Tiền xử lý dữ liệu** (10 bước: load CSV → encode → cân bằng)
2. **Tạo PyTorch Dataset** (chuẩn hoá MinMaxScaler)
3. **5-fold Cross-validation** (MLP)
4. **Huấn luyện MLP cuối** (lưu model tốt nhất)
5. **XGBoost + Random Forest** (cùng features với MLP)
6. **Tạo biểu đồ** (confusion matrix, accuracy curves)

### Scenario `single` — GF29, 780 profiles, 22 loci (chạy nhanh nhất)

```bash
python3 src/main.py --scenario single
```

### Scenario `three` — IDPlus28 + IDPlus29 + GF29, 5700 profiles, 16 loci

```bash
python3 src/main.py --scenario three
```

### Scenario `four` — tất cả 4 multiplex, 6000 profiles, 13 loci (chính xác nhất)

```bash
python3 src/main.py --scenario four
```

### Chạy tất cả 3 scenario + so sánh

```bash
python3 src/main.py --scenario all
```

### Bỏ qua tiền xử lý (nếu đã có file CSV trong `data/processed/`)

```bash
python3 src/main.py --scenario single --skip-preprocessing

# Bỏ qua cả cross-validation (tiết kiệm thời gian)
python3 src/main.py --scenario single --skip-preprocessing --skip-cv
```

---

## Chạy XGBoost trên raw features

Script `test_xgb_raw.py` test XGBoost **trực tiếp trên raw concatenated features** (1540 features, không qua profile-level aggregation), đồng thời so sánh với Random Forest:

```bash
python3 test_xgb_raw.py
```

**Output mẫu:**
```
============================================================
TEST: XGBoost on raw concatenated features
============================================================
Data: 780 profiles × 1540 features
Train: 546, Test: 234

--- XGBoost (raw 1540 features) ---
CV: 0.8791 ± 0.0143  [...]
Train: 0.9890, Test: 0.8761

--- RandomForest (raw 1540 features) ---
CV: 0.8632 ± 0.0201  [...]
Train: 0.9920, Test: 0.8504
```

---

## Các tham số cấu hình

Chỉnh sửa tại `src/config.py`:

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `EPOCHS` | 100 | Số epoch huấn luyện MLP |
| `BATCH_SIZE` | 30 | Batch size |
| `LEARNING_RATE` | 0.001 | Learning rate Adam |
| `NUM_CV_FOLDS` | 5 | Số fold cross-validation |
| `TRAIN_RATIO` | 0.7 | Tỷ lệ train/test (70/30) |
| `RANDOM_SEED` | 42 | Seed ngẫu nhiên |
| `NUM_HIDDEN_LAYERS` | 15 | Số hidden layer MLP |
| `HIDDEN_DIM` | 64 | Số neuron mỗi layer |
| `DROPOUT_RATE` | 0.2 | Tỷ lệ dropout |

**Cấu hình XGBoost** (trong `src/train_xgb.py`):

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `n_estimators` | 500 | Số cây |
| `max_depth` | 6 | Độ sâu tối đa |
| `learning_rate` | 0.1 | Learning rate boosting |
| `subsample` | 0.8 | Tỷ lệ subsample mỗi cây |
| `colsample_bytree` | 0.8 | Tỷ lệ feature subsample |
| `reg_alpha` | 0.1 | L1 regularization |
| `reg_lambda` | 1.0 | L2 regularization |

---

## Kết quả kỳ vọng

| Scenario | Mô hình | CV Accuracy | Test Accuracy |
|---|---|---|---|
| Single (GF29) | 🏆 **XGBoost** | ~88% | **~91%** |
| Single (GF29) | MLP | ~85% | ~90% |
| Single (GF29) | Random Forest | ~84% | ~86% |
| Four (all kits) | 🏆 **XGBoost** | ~96% | **~97%** |
| Four (all kits) | MLP | ~94% | ~96% |
| Three (3 kits) | 🏆 **XGBoost** | ~94% | **~96%** |

---

## Quy trình tiền xử lý (10 bước)

| Bước | Mô tả |
|---|---|
| 1 | Load CSV, gán nhãn NOC (1–5 người) |
| 2 | Giữ tối đa 10 allele positions |
| 3 | Xử lý giá trị OL (Out-of-Ladder) → indicator columns |
| 4 | Xử lý missing values → indicator columns + fill 0 |
| 5 | Loại markers AMEL, Yindel |
| 6 | Encode Dye (B=0, G=1, P=2, R=3, Y=4) |
| 7 | Encode Marker name (categorical → int) |
| 7b | Encode Multiplex (nếu multi-kit) |
| 7c | Encode Injection Time |
| 8 | Tạo `profile_loci` feature |
| 9 | Báo cáo phân phối class |
| 10 | Chọn ~72 features cuối ở marker-level |
| 11 | **Feature engineering profile-level** (trong `dataset.py`) — xem chi tiết bên dưới |
| 12 | **MinMaxScaler** chuẩn hoá toàn bộ features về [0, 1] |

---

## Feature Engineering (profile-level — `dataset.py`)

Đây là bước **quan trọng nhất** xảy ra trong `prepare_profile_datasets()` **trước khi đưa vào model**. Dữ liệu được chuyển từ dạng marker-level → profile-level.

### Với mỗi marker (15 features × n_markers)

Hàm `_extract_marker_features()` tính 15 features từ cột Height/OL/Missing của từng marker:

| # | Feature | Mô tả |
|---|---|---|
| 1 | `n_alleles` | Số allele hợp lệ (không missing) |
| 2 | `h1` | Chiều cao peak lớn nhất |
| 3 | `h2` | Chiều cao peak lớn thứ 2 |
| 4 | `h3` | Chiều cao peak lớn thứ 3 |
| 5 | `sum_h` | Tổng chiều cao tất cả peaks |
| 6 | `mean_h` | Trung bình chiều cao |
| 7 | `std_h` | Độ lệch chuẩn chiều cao |
| 8 | `h_ratio` | h1 / (h2 + ε) — tỷ lệ peak lớn nhất |
| 9 | `h_range` | h1 − min(heights) — spread |
| 10 | `n_ol` | Số giá trị Out-of-Ladder |
| 11 | `n_missing` | Số allele bị thiếu |
| 12 | `stutter_ratio` | min(heights) / (h1 + ε) — đặc trưng stutter |
| 13 | `snr_top2` | (h1+h2) / (sum_h − h1 − h2 + ε) — tỷ lệ tín hiệu/nhiễu |
| 14 | `log1p_h1` | log(1 + h1) — ổn định số học |
| 15 | `log1p_sum_h` | log(1 + sum_h) — ổn định số học |

### Profile aggregates (10 features)

Sau khi tính 15 features/marker, 10 features tổng hợp toàn profile được ghép thêm:

| # | Feature | Mô tả |
|---|---|---|
| 1 | `max(allele_counts)` | MAC — Maximum Allele Count |
| 2 | `mean(allele_counts)` | Trung bình số allele / marker |
| 3 | `std(allele_counts)` | Độ lệch chuẩn số allele / marker |
| 4 | `count(allele_counts ≥ 3)` | Số marker có ≥ 3 allele |
| 5 | `count(allele_counts ≥ 5)` | Số marker có ≥ 5 allele |
| 6 | `sum(ol_counts)` | Tổng OL trên toàn profile |
| 7 | `mean(max_heights)` | Trung bình peak lớn nhất / marker |
| 8 | `std(max_heights)` | Độ lệch chuẩn peak lớn nhất / marker |
| 9 | `sum(allele_counts)` | Tổng số peaks trên toàn profile |
| 10 | `sum(sum_heights)` | Tổng cường độ tín hiệu toàn profile |

### Kích thước vector cuối vào model

| Scenario | n_markers | Flat features |
|---|---|---|
| Single (GF29) | 22 | 22 × 15 + 10 = **340** |
| Three (3 kits) | 14 | 14 × 15 + 10 = **220** |
| Four (all kits) | 12 | 12 × 15 + 10 = **190** |

> Lưu ý: Đối với `test_xgb_raw.py`, XGBoost dùng raw concatenated features (1540 features) **thay vì** các profile-level features ở trên.

---

## Tài liệu tham khảo

Alotaibi, H.; Alsolami, F.; Abozinadah, E.; Mehmood, R. **TAWSEEM: A Deep Learning-Based Tool for Estimating the Number of Unknown Contributors in DNA Profiling.** *Electronics* 2022, 11, 548. https://doi.org/10.3390/electronics11040548
