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

1. **Polynomial Regression** - Mô hình tuyến tính với polynomial features
2. **Random Forest Regressor** - Ensemble learning
3. **XGBoost Regressor** - Gradient boosting (mô hình tốt nhất)

### Tối ưu hyperparameters:

- Sử dụng **Optuna** để tự động tìm kiếm hyperparameters tối ưu
- So sánh hiệu suất dựa trên RMSE, MAE, R²

## 📁 Cấu trúc dự án

```
Do_An_ver2/
│
├── src/                          # Source code chính
│   ├── __init__.py
│   ├── preprocessing/            # Module tiền xử lý dữ liệu
│   │   ├── __init__.py
│   │   └── data_preprocessor.py  # Class DataPreprocessor
│   └── modeling/                 # Module training mô hình
│       ├── __init__.py
│       └── model_trainer.py      # Class ModelTrainer
│
├── data/                         # Dữ liệu
│   ├── taxi_price.csv           # Dữ liệu gốc
│   └── taxi_price_processed.csv # Dữ liệu đã xử lý
│
├── models/                       # Mô hình đã train
│   ├── polynomial_*.joblib
│   ├── random_forest_*.joblib
│   └── xgboost_*.joblib
│
├── results/                      # Kết quả và biểu đồ
│   ├── model_results.json
│   ├── comparison_*.png
│   └── predictions_*.png
│
├── notebooks/                    # Jupyter notebooks
│   └── do_an_py_modeling.ipynb  # Notebook phân tích ban đầu
│
├── config.py                     # File cấu hình
├── main.py                       # Script chính để chạy pipeline
├── requirements.txt              # Dependencies
├── README.md                     # File này
└── yeu_cau_do_an.txt            # Yêu cầu đồ án

```

## 🔧 Cài đặt

### Yêu cầu hệ thống:

- Python 3.8+
- pip

### Bước 1: Clone hoặc download project

```bash
cd Do_An_ver2
```

### Bước 2: Cài đặt dependencies

#### Cách 1: Sử dụng môi trường Python hiện tại (Khuyến nghị nếu đã có sẵn)

```bash
# Cài đặt trực tiếp vào môi trường hiện tại
pip install -r requirements.txt
```

#### Cách 2: Tạo môi trường ảo mới (Optional)

```bash
# Tạo môi trường ảo
python -m venv venv

# Kích hoạt môi trường
# Windows PowerShell:
.\venv\Scripts\Activate.ps1
# Windows CMD:
.\venv\Scripts\activate.bat
# Linux/Mac:
source venv/bin/activate

# Cài đặt dependencies
pip install -r requirements.txt
```

#### Cách 3: Sử dụng Conda environment (Optional)

```bash
# Tạo conda environment
conda create -n taxi_price python=3.9 -y
conda activate taxi_price

# Cài đặt dependencies
pip install -r requirements.txt
```

### Bước 3: Kiểm tra cài đặt

```bash
python -c "import pandas, sklearn, xgboost, optuna; print('✅ Cài đặt thành công!')"
```

> **💡 Lưu ý:** Nếu bạn đã có môi trường Python với các thư viện cần thiết, bạn có thể sử dụng trực tiếp mà không cần tạo môi trường mới. Chỉ cần đảm bảo các thư viện trong `requirements.txt` đã được cài đặt.

## 🚀 Hướng dẫn sử dụng

### Cách 1: Chạy toàn bộ pipeline (Khuyến nghị)

```bash
# Chạy với hyperparameters mặc định (nhanh)
python main.py

# Chạy với optimization (chậm hơn nhưng kết quả tốt hơn)
python main.py --optimize

# Chạy không vẽ biểu đồ (tiết kiệm thời gian)
python main.py --no-viz
```

Pipeline sẽ tự động:

1. Download dữ liệu từ Google Drive (nếu chưa có)
2. Tiền xử lý dữ liệu
3. Train 3 mô hình
4. Đánh giá và so sánh
5. Lưu mô hình và kết quả

### Cách 2: Sử dụng từng module riêng lẻ

#### Tiền xử lý dữ liệu:

```python
from src.preprocessing.data_preprocessor import DataPreprocessor

# Khởi tạo và load data
preprocessor = DataPreprocessor()
preprocessor.load('data/taxi_price.csv')

# Xử lý missing values
preprocessor.handle_missing(strategy='auto')

# Encoding biến phân loại
preprocessor.encode_categorical(method='onehot', drop_first=True)

# Lưu dữ liệu đã xử lý
preprocessor.save_data('data/taxi_price_processed.csv')
df = preprocessor.get_processed_data()
```

#### Training mô hình:

```python
from src.modeling.model_trainer import ModelTrainer
import pandas as pd

# Load dữ liệu đã xử lý
df = pd.read_csv('data/taxi_price_processed.csv')

# Chuẩn bị dữ liệu
X_train, X_test, y_train, y_test = ModelTrainer.prepare_data(
    df, target_col='Trip_Price', test_size=0.2, random_state=42
)

# Khởi tạo trainer
trainer = ModelTrainer(X_train, X_test, y_train, y_test)

# Train XGBoost
trainer.train_xgb(max_depth=6, learning_rate=0.1, n_estimators=100)

# Đánh giá
trainer.summary()

# Lưu mô hình
trainer.save_model('xgboost', format='joblib')
```

#### Sử dụng mô hình đã train để dự đoán:

```python
from src.modeling.model_trainer import ModelTrainer
import pandas as pd

# Tạo trainer instance
trainer = ModelTrainer(X_train, X_test, y_train, y_test)

# Load mô hình đã lưu
trainer.load_model('models/xgboost_20231126_120000.joblib', model_name='xgboost')

# Dự đoán
predictions = trainer.predict(X_test, model_name='xgboost')

# Đánh giá
from sklearn.metrics import r2_score, mean_squared_error
print(f"R² Score: {r2_score(y_test, predictions):.4f}")
print(f"RMSE: {mean_squared_error(y_test, predictions, squared=False):.4f}")
```

#### Phân tích Feature Importance:

```python
# Lấy feature importance
importance_df = trainer.get_feature_importance('xgboost', top_n=10)
print(importance_df)

# Vẽ biểu đồ feature importance
trainer.plot_feature_importance('xgboost', top_n=15, save=True)

# So sánh feature importance giữa Random Forest và XGBoost
trainer.compare_feature_importance(top_n=10, save=True)
```

### Cách 3: Demo nhanh với dữ liệu mẫu

```bash
# Chạy demo để xem feature importance (nhanh, không cần download data)
python demo_feature_importance.py
```

Script này sẽ:

- Tạo dữ liệu mẫu
- Train Random Forest và XGBoost
- Phân tích và vẽ biểu đồ feature importance
- Tạo file `training.log` với đầy đủ thông tin

## 🔬 Chi tiết kỹ thuật

### 1. Class DataPreprocessor

**Vị trí:** `src/preprocessing/data_preprocessor.py`

**Chức năng chính:**

| Method                          | Mô tả                                                                   |
| ------------------------------- | ----------------------------------------------------------------------- |
| `load_data()`                   | Đọc dữ liệu từ CSV, Excel, JSON                                         |
| `eda_overview()`                | Báo cáo tổng quan (shape, missing %, skew, rare categories, correlations) |
| `apply_constraints()`           | Áp dụng ràng buộc kiểu/miền giá trị dựa trên `config.CONSTRAINT_RULES`  |
| `handle_missing()`              | Xử lý missing values (mean, median, mode, forward-fill)                 |
| `detect_outliers_*()`           | Phát hiện outliers (IQR, Z-score, Isolation Forest)                     |
| `remove_outliers()`             | Loại bỏ outliers                                                        |
| `encode_categorical()`          | Mã hóa biến phân loại (OneHot, Label Encoding)                          |
| `scale_features()`              | Chuẩn hóa dữ liệu (StandardScaler, MinMaxScaler, hỗ trợ exclude target) |
| `create_datetime_features()`    | Tạo features từ datetime                                                |
| `create_interaction_features()` | Tạo interaction features                                                |
| `save_data()`                   | Lưu dữ liệu đã xử lý                                                    |

> 📌 **Cấu hình ràng buộc**: sửa `config.CONSTRAINT_RULES` để quy định dtype, min/max và hành động (clip/drop/mean) cho từng cột. `main.py` sẽ tự động gán các rule này cho `DataPreprocessor.apply_constraints()` trước khi xử lý missing/outlier.

**Ví dụ sử dụng:**

```python
preprocessor = DataPreprocessor()
preprocessor.load('data.csv') \
            .handle_missing(strategy='auto') \
            .encode_categorical(method='onehot') \
            .scale_features(method='standard', exclude_columns=['Trip_Price']) \
            .save_data('processed.csv')

⚠️ **Lưu ý:** luôn loại cột target (`Trip_Price`) khỏi bước scale để giữ nguyên distribution của biến mục tiêu.
```

### 2. Class ModelTrainer

**Vị trí:** `src/modeling/model_trainer.py`

**Chức năng chính:**

| Method                         | Mô tả                                      |
| ------------------------------ | ------------------------------------------ |
| `prepare_data()`               | Chia và chuẩn hóa dữ liệu (static method)  |
| `optimize_polynomial()`        | Tối ưu Polynomial Regression bằng Optuna   |
| `train_polynomial()`           | Train Polynomial Regression                |
| `optimize_rf()`                | Tối ưu Random Forest bằng Optuna           |
| `train_rf()`                   | Train Random Forest                        |
| `optimize_xgb()`               | Tối ưu XGBoost bằng Optuna                 |
| `train_xgb()`                  | Train XGBoost                              |
| `save_model()`                 | Lưu mô hình (joblib/pickle)                |
| `load_model()`                 | Load mô hình                               |
| `get_best_model()`             | Tìm mô hình tốt nhất                       |
| `save_results()`               | Lưu kết quả đánh giá (JSON)                |
| `plot_comparison()`            | Vẽ biểu đồ so sánh                         |
| `plot_predictions()`           | Vẽ biểu đồ actual vs predicted             |
| `get_feature_importance()`     | **Lấy feature importance**                 |
| `plot_feature_importance()`    | **Vẽ biểu đồ feature importance**          |
| `compare_feature_importance()` | **So sánh feature importance giữa models** |
| `summary()`                    | In tóm tắt kết quả                         |
| `predict()`                    | Dự đoán với mô hình                        |

**Ví dụ workflow:**

```python
# Chuẩn bị dữ liệu
X_train, X_test, y_train, y_test = ModelTrainer.prepare_data(df)

# Khởi tạo
trainer = ModelTrainer(X_train, X_test, y_train, y_test)

# Tối ưu và train
best_params = trainer.optimize_xgb(n_trials=30)
trainer.train_xgb(**best_params)

# Đánh giá
trainer.summary()
trainer.plot_comparison()
trainer.save_all_models()
```

### 3. Optuna Optimization

Dự án sử dụng **Optuna** để tối ưu hyperparameters:

- **TPESampler**: Tree-structured Parzen Estimator
- **MedianPruner**: Dừng sớm các trial không hiệu quả
- **Objective function**: Minimize RMSE

**Hyperparameters được tối ưu:**

**Polynomial Regression:**

- degree: [2, 5]

**Random Forest:**

- n_estimators: [50, 300]
- max_depth: [5, 20]
- min_samples_split: [2, 10]
- min_samples_leaf: [1, 5]

**XGBoost:**

- max_depth: [3, 10]
- learning_rate: [0.01, 0.3]
- n_estimators: [50, 300]
- subsample: [0.5, 1.0]
- colsample_bytree: [0.5, 1.0]
- min_child_weight: [1, 5]
- lambda: [0.0, 1.0]
- alpha: [0.0, 1.0]

### 4. Metrics đánh giá

| Metric   | Ý nghĩa                                                         | Cách tính                  |
| -------- | --------------------------------------------------------------- | -------------------------- |
| **RMSE** | Root Mean Squared Error - Độ lỗi trung bình                     | √(Σ(y_true - y_pred)² / n) |
| **MAE**  | Mean Absolute Error - Độ lỗi tuyệt đối trung bình               | Σ\|y_true - y_pred\| / n   |
| **R²**   | Coefficient of Determination - Tỷ lệ phương sai được giải thích | 1 - (SS_res / SS_tot)      |

**R² Score:**

- R² = 1.0: Mô hình hoàn hảo
- R² = 0.8-1.0: Rất tốt
- R² = 0.6-0.8: Tốt
- R² < 0.6: Cần cải thiện

### 5. Feature Importance Analysis

Dự án phân tích **đặc trưng quan trọng** (Feature Importance) để hiểu:

- Features nào ảnh hưởng nhiều nhất đến giá taxi
- So sánh quan điểm của Random Forest vs XGBoost
- Loại bỏ features không quan trọng để tối ưu mô hình

**Cách sử dụng:**

```python
# Lấy top 10 features quan trọng nhất
importance_df = trainer.get_feature_importance('xgboost', top_n=10)
print(importance_df)

# Vẽ biểu đồ
trainer.plot_feature_importance('random_forest', top_n=15)

# So sánh giữa các mô hình
trainer.compare_feature_importance(top_n=10)
```

**Output:**

- `results/feature_importance_random_forest.png` - Biểu đồ RF
- `results/feature_importance_xgboost.png` - Biểu đồ XGBoost
- `results/feature_importance_comparison.png` - So sánh cả 2 models

**Ví dụ kết quả:**

Top features thường quan trọng nhất:

1. `Trip_Distance_km` - Khoảng cách
2. `Trip_Duration_Minutes` - Thời gian
3. `Per_Km_Rate` - Giá theo km
4. `Per_Minute_Rate` - Giá theo phút
5. `Base_Fare` - Giá khởi điểm

## 📊 Kết quả

### So sánh hiệu suất các mô hình:

| Model                     | Train RMSE | Test RMSE | Test MAE | Test R²   |
| ------------------------- | ---------- | --------- | -------- | --------- |
| **Polynomial Regression** | ~14.2      | ~15.9     | ~6.3     | ~0.79     |
| **Random Forest**         | ~3.5       | ~7.2      | ~5.0     | ~0.96     |
| **XGBoost** ⭐            | ~2.1       | ~6.0      | ~3.4     | **~0.97** |

### Mô hình tốt nhất: **XGBoost**

- Test R²: **0.971** (giải thích 97.1% phương sai)
- Test RMSE: **6.018** (sai lệch trung bình ~$6)
- Test MAE: **3.377** (sai lệch tuyệt đối ~$3.38)

### Nhận xét:

1. **XGBoost** cho kết quả tốt nhất với R² = 0.971
2. **Random Forest** cũng rất tốt với R² = 0.958
3. **Polynomial Regression** kém hơn đáng kể với R² = 0.794

### Visualization:

Sau khi chạy pipeline, các biểu đồ được lưu trong `results/`:

**Biểu đồ so sánh:**

- `comparison_test_r2.png` - So sánh R² score
- `comparison_test_rmse.png` - So sánh RMSE
- `comparison_test_mae.png` - So sánh MAE

**Biểu đồ predictions:**

- `predictions_polynomial.png` - Actual vs Predicted (Polynomial)
- `predictions_random_forest.png` - Actual vs Predicted (Random Forest)
- `predictions_xgboost.png` - Actual vs Predicted (XGBoost)

**Biểu đồ Feature Importance:** ⭐

- `feature_importance_random_forest.png` - Top features (RF)
- `feature_importance_xgboost.png` - Top features (XGBoost)
- `feature_importance_comparison.png` - So sánh RF vs XGBoost

**Log file:**

- `training.log` - Chi tiết quá trình training, optimization, và đánh giá

## 📝 Cấu hình

Chỉnh sửa file `config.py` để thay đổi:

```python
# Tham số training
TEST_SIZE = 0.2          # Tỷ lệ test set
RANDOM_SEED = 42         # Random seed

# Optuna optimization
OPTUNA_N_TRIALS = {
    'polynomial': 10,
    'random_forest': 20,
    'xgboost': 30
}

# Xử lý missing values
MISSING_STRATEGY = {
    'numeric': 'median',
    'categorical': 'mode'
}

# Encoding
ENCODING_METHOD = 'onehot'
```

## 🐛 Troubleshooting

### Lỗi import module:

```bash
# Thêm project vào PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"  # Linux/Mac
set PYTHONPATH=%PYTHONPATH%;%CD%          # Windows
```

### Lỗi download dữ liệu:

Nếu `gdown` không hoạt động, download thủ công:

1. Truy cập Google Drive
2. Download file `taxi_price.csv`
3. Đặt vào thư mục `data/`

### Lỗi thiếu thư viện:

```bash
pip install -r requirements.txt --upgrade
```

## 👥 Thành viên nhóm

(Điền tên và vai trò các thành viên)

## 📚 Tài liệu tham khảo

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Optuna Documentation](https://optuna.readthedocs.io/)
- [Pandas Documentation](https://pandas.pydata.org/)

## 📄 License

Dự án được tạo cho mục đích học tập - Môn Python cho Khoa học Dữ liệu K23.

---

**Ngày hoàn thành:** 26/11/2025
**Lớp:** Python cho Khoa học Dữ liệu - K23
