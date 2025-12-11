# ĐỒ ÁN CUỐI KỲ: DỰ ĐOÁN GIÁ TAXI

**Môn:** Python cho Khoa học Dữ liệu - K23

## 📋 Mục lục

- [Giới thiệu](#giới-thiệu)
- [Cấu trúc dự án](#cấu-trúc-dự-án)
- [Cài đặt](#cài-đặt)
- [Hướng dẫn sử dụng](#hướng-dẫn-sử-dụng)
- [Chi tiết kỹ thuật](#chi-tiết-kỹ-thuật)
- [Kết quả](#kết-quả)

## 🎯 Giới thiệu

Dự án xây dựng pipeline học máy hoàn chỉnh để **dự đoán giá cước taxi** dựa trên các yếu tố như:

- Khoảng cách di chuyển
- Thời gian di chuyển
- Số hành khách
- Điều kiện giao thông
- Thời tiết
- Thời điểm trong ngày

### Mô hình sử dụng:

1. **Random Forest Regressor** - Ensemble learning
2. **Extra Trees Regressor** - Extremely Randomized Trees
3. **XGBoost Regressor** - Gradient boosting

### Tối ưu hyperparameters:

- Sử dụng **Optuna** để tự động tìm kiếm hyperparameters tối ưu
- So sánh hiệu suất dựa trên RMSE, MAE, R²

## 📁 Cấu trúc dự án

```
Do_An_Python/
│
├── src/                              # Source code chính
│   ├── __init__.py
│   ├── preprocessing/                # Module tiền xử lý dữ liệu
│   │   ├── __init__.py
│   │   ├── data_loader.py            # Class DataLoader (pre-split processing)
│   │   └── data_transformer.py       # Class DataTransformer (post-split processing)
│   │
│   ├── modeling/                     # Module training mô hình
│   │   ├── __init__.py
│   │   ├── base_trainer.py           # Class BaseTrainer (abstract base)
│   │   ├── model_registry.py         # Các trainer cụ thể (RF, ET, XGB)
│   │   └── model_trainer.py          # Class ModelTrainer (orchestrator)
│   │
│   └── visualization/                # Module trực quan hóa
│       ├── __init__.py
│       └── data_visualizer.py        # Class DataVisualizer
│
├── data/                             # Dữ liệu
│   ├── taxi_price.csv                # Dữ liệu gốc
│   ├── taxi_price_processed.csv      # Dữ liệu đã xử lý
│   └── sample_input.csv              # Dữ liệu mẫu cho predict
│
├── models/                           # Mô hình đã train
│   ├── random_forest.joblib
│   ├── extra_trees.joblib
│   ├── xgboost.joblib
│   └── data_transformer.joblib       # Transformer state cho inference
│
├── results/                          # Kết quả và biểu đồ
│   ├── eda/                          # Biểu đồ EDA (6 files)
│   │   ├── 01_data_overview.png
│   │   ├── 02_numeric_distributions.png
│   │   ├── 03_categorical_distributions.png
│   │   ├── 04_correlation_heatmap.png
│   │   ├── 05_target_analysis.png
│   │   └── 06_outliers_boxplot.png
│   ├── model/                        # Biểu đồ model và kết quả
│   │   ├── metrics_summary.png
│   │   ├── predictions_combined.png
│   │   ├── feature_importance_comparison.png
│   │   └── model_results.json        # Kết quả đánh giá models
│   ├── predictions.csv               # Kết quả dự đoán
│   └── pipeline_state.json
│
├── notebooks/                        # Jupyter notebooks
│   └── taxi_price_prediction.ipynb
│
├── report/                           # Báo cáo LaTeX
│   └── bao_cao_do_an.tex
│   └── Báo cáo đồ án.pdf
├── config.py                         # File cấu hình
├── main.py                           # Script chính để chạy pipeline
├── predict.py                        # Script dự đoán với model đã train
├── requirements.txt                  # Dependencies
├── README.md                         # File này
└── yeu_cau_do_an.txt                 # Yêu cầu đồ án
```

## 🔧 Cài đặt

### Yêu cầu hệ thống:

- Python 3.8+
- pip

### Bước 1: Clone từ github hoặc download project

```bash
git clone https://github.com/drpie143/Do_An_Python.git
cd Do_An_Python
```

### Bước 2: Tạo môi trường ảo (khuyến nghị)

**Windows (PowerShell):**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**

```cmd
python -m venv venv
venv\Scripts\activate.bat
```

**Linux/macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### Bước 3: Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### Bước 4: Kiểm tra cài đặt

```bash
python -c "import pandas, sklearn, xgboost, optuna; print('Cai dat thanh cong!')"
```

> **Lưu ý:** Mỗi lần mở terminal mới, cần kích hoạt lại môi trường ảo bằng lệnh activate tương ứng.

## 🚀 Hướng dẫn sử dụng

### Cách 1: Chạy toàn bộ pipeline (Khuyến nghị)

```bash
# Chạy với hyperparameters mặc định (nhanh ~2 giây)
python main.py

# Chạy với optimization (chậm hơn nhưng kết quả tốt hơn)
python main.py --optimize

# Chạy không vẽ biểu đồ
python main.py --no-viz

# Bỏ qua download dữ liệu (nếu đã có)
python main.py --skip-download
```

Pipeline sẽ tự động:

1. Download dữ liệu từ Google Drive (nếu chưa có)
2. Tiền xử lý dữ liệu
3. Train 3 mô hình
4. Đánh giá và so sánh
5. Lưu mô hình và kết quả

### Cách 2: Dự đoán với model đã train

```bash
# Dự đoán từ file CSV (sử dụng file mẫu `data/sample_input.csv`)
python predict.py --input data/sample_input.csv --output results/predictions.csv

# Dự đoán với model cụ thể (tên model: random_forest | extra_trees | xgboost)
python predict.py --input data/sample_input.csv --model xgboost --output results/predictions.csv

# Dự đoán interactive (nhập từng giá trị)
python predict.py --interactive
```

### Cách 3: Sử dụng từng module riêng lẻ

#### Tiền xử lý dữ liệu:

```python
from src.preprocessing import DataLoader, DataTransformer

# PHASE 1: Load và clean dữ liệu (trước khi split)
loader = DataLoader.from_file('data/taxi_price.csv')
loader.drop_duplicates()
loader.unify_values()
loader.apply_constraints()
raw_df = loader.get_data()

# PHASE 2: Chia train/test
from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(raw_df, test_size=0.2, random_state=42)

# PHASE 3: Transform dữ liệu (sau khi split)
transformer = DataTransformer(data=train_df)
train_processed = transformer.fit_transform(target_col='Trip_Price')

# PHASE 4: Transform test set (dùng transformer đã fit)
test_processed = transformer.transform_new_data(test_df)

# Lưu transformer để dùng cho inference
transformer.save_state('models/data_transformer.joblib')
```

#### Training mô hình:

```python
from src.modeling import ModelTrainer

# Khởi tạo trainer
trainer = ModelTrainer(X_train, X_test, y_train, y_test)

# Train tất cả models
trainer.train_all(optimize=False)

# Hoặc train từng model
trainer.train_rf(n_estimators=100, max_depth=10)
trainer.train_extra_trees(n_estimators=200, max_depth=12)
trainer.train_xgb(max_depth=6, learning_rate=0.1)

# Đánh giá
trainer.summary()

# Lưu mô hình
trainer.save_all_models()
```

#### Visualization:

```python
# Biểu đồ metrics tổng hợp
trainer.plot_metrics_summary(save=True)

# Biểu đồ predictions tổng hợp
trainer.plot_combined_predictions(save=True)

# So sánh feature importance
trainer.compare_feature_importance(top_n=10, save=True)
```

## 🔬 Chi tiết kỹ thuật

### 1. Module Preprocessing

#### DataLoader (pre-split processing)

| Method                  | Mô tả                             |
| ----------------------- | --------------------------------- |
| `from_file()`           | Load dữ liệu từ CSV, Excel, JSON  |
| `drop_duplicates()`     | Xóa dòng trùng lặp                |
| `unify_values()`        | Chuẩn hóa text (lowercase, strip) |
| `apply_constraints()`   | Áp dụng ràng buộc dữ liệu         |
| `generate_eda_report()` | Tạo 6 biểu đồ EDA                 |
| `get_data()`            | Lấy DataFrame đã xử lý            |

#### DataTransformer (post-split processing)

| Method                          | Mô tả                                  |
| ------------------------------- | -------------------------------------- |
| `fit_transform()`               | Fit và transform trên train set        |
| `transform_new_data()`          | Transform dữ liệu mới (test/inference) |
| `fill_missing()`                | Xử lý missing values                   |
| `encode()`                      | Mã hóa biến phân loại (OneHot, Label)  |
| `scale()`                       | Chuẩn hóa features (Standard, MinMax)  |
| `remove_outliers()`             | Loại bỏ outliers (IQR, Z-score)        |
| `save_state()` / `load_state()` | Lưu/load transformer state             |

### 2. Module Modeling

#### ModelTrainer (orchestrator)

| Method                | Mô tả                             |
| --------------------- | --------------------------------- |
| `train_all()`         | Train tất cả 3 models             |
| `train_rf()`          | Train Random Forest               |
| `train_extra_trees()` | Train Extra Trees                 |
| `train_xgb()`         | Train XGBoost                     |
| `optimize_*()`        | Tối ưu hyperparameters với Optuna |
| `save_all_models()`   | Lưu tất cả models                 |
| `predict()`           | Dự đoán với model                 |
| `summary()`           | In tóm tắt kết quả                |

#### Trainer cụ thể (trong model_registry.py)

- `RandomForestTrainer` - Random Forest Regressor
- `ExtraTreesTrainer` - Extra Trees Regressor
- `XGBoostTrainer` - XGBoost Regressor

### 3. Optuna Optimization

**Hyperparameters được tối ưu:**

| Model         | Parameters                                                         |
| ------------- | ------------------------------------------------------------------ |
| Random Forest | n_estimators [50-300], max_depth [5-20]                            |
| Extra Trees   | n_estimators [50-300], max_depth [5-20]                            |
| XGBoost       | max_depth [4-10], learning_rate [0.01-0.3], n_estimators [100-500] |

### 4. Metrics đánh giá

| Metric   | Ý nghĩa                                               |
| -------- | ----------------------------------------------------- |
| **RMSE** | Root Mean Squared Error - Độ lỗi trung bình           |
| **MAE**  | Mean Absolute Error - Độ lỗi tuyệt đối trung bình     |
| **R²**   | Coefficient of Determination (0-1, càng cao càng tốt) |

## 📊 Kết quả

### So sánh hiệu suất các mô hình:

| Model          | Train RMSE | Test RMSE | Test MAE | Test R²   |
| -------------- | ---------- | --------- | -------- | --------- |
| Random Forest  | 4.47       | 7.81      | 5.02     | 0.946     |
| Extra Trees    | 3.51       | 8.17      | 4.76     | 0.941     |
| **XGBoost** ⭐ | **3.67**   | **6.67**  | **3.98** | **0.961** |

### Mô hình tốt nhất: **XGBoost**

- Test R²: **0.961** (giải thích 96.1% phương sai)
- Test RMSE: **6.67**
- Test MAE: **3.98**

### Hyperparameters tối ưu (Optuna):

**XGBoost:**

- max_depth: 5
- learning_rate: 0.0194
- n_estimators: 337
- subsample: 0.643
- colsample_bytree: 0.860
- min_child_weight: 2
- gamma: 0.868
- reg_lambda: 0.348
- reg_alpha: 0.351

**Random Forest:**

- n_estimators: 290
- max_depth: 10
- min_samples_split: 5
- min_samples_leaf: 1

**Extra Trees:**

- n_estimators: 126
- max_depth: 13
- min_samples_split: 5
- min_samples_leaf: 2

### Biểu đồ EDA (6 files trong results/eda/):

1. `01_data_overview.png` - Tổng quan dữ liệu
2. `02_numeric_distributions.png` - Phân phối biến số
3. `03_categorical_distributions.png` - Phân phối biến phân loại
4. `04_correlation_heatmap.png` - Ma trận tương quan
5. `05_target_analysis.png` - Phân tích biến mục tiêu
6. `06_outliers_boxplot.png` - Boxplot phát hiện outliers

### Biểu đồ Model (3 files trong results/model/):

1. `metrics_summary.png` - So sánh R², RMSE, MAE
2. `predictions_combined.png` - Actual vs Predicted cho tất cả models
3. `feature_importance_comparison.png` - So sánh feature importance

## 📝 Cấu hình

Chỉnh sửa file `config.py` để thay đổi các tham số:

```python
# Training
TEST_SIZE = 0.2
RANDOM_SEED = 42

# Preprocessing
MISSING_STRATEGY = {'numeric': 'median', 'categorical': 'mode'}
ENCODING_METHOD = 'onehot'
SCALING_METHOD = 'standard'

# Optuna
OPTUNA_N_TRIALS = {'random_forest': 20, 'extra_trees': 20, 'xgboost': 30}
```

## 🐛 Troubleshooting

### Lỗi download dữ liệu:

Download thủ công file `taxi_price.csv` và đặt vào thư mục `data/`

### Lỗi thiếu thư viện:

```bash
pip install -r requirements.txt --upgrade
```

### Lỗi emoji trên Windows:

Emoji có thể hiển thị sai trên PowerShell, nhưng file log (`training.log`) vẫn hiển thị đúng.

## 👥 Thành viên nhóm

Mai Quang Dũng - 23280049  
Ngô Anh Khoa - 23280065

## 📄 License

Dự án được tạo cho mục đích học tập - Môn Python cho Khoa học Dữ liệu K23.

---

**Ngày hoàn thành:** 08/12/2025

---

**Ghi chú:** Các số liệu trên được cập nhật từ kết quả chạy tối ưu hóa (`--optimize`) và lưu trong `results/pipeline_state.json`.
