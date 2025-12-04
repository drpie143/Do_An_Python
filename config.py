"""
File cấu hình cho dự án Taxi Price Prediction.
"""

import os
from pathlib import Path

# ========== ĐƯỜNG DẪN ==========
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / 'data'
MODELS_DIR = PROJECT_ROOT / 'models'
RESULTS_DIR = PROJECT_ROOT / 'results'
EDA_RESULTS_DIR = RESULTS_DIR / 'eda'
MODEL_RESULTS_DIR = RESULTS_DIR / 'model'
NOTEBOOKS_DIR = PROJECT_ROOT / 'notebooks'

# Tạo thư mục nếu chưa tồn tại
for dir_path in [DATA_DIR, MODELS_DIR, RESULTS_DIR, NOTEBOOKS_DIR, EDA_RESULTS_DIR, MODEL_RESULTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ========== DỮ LIỆU ==========
DATA_FILE = DATA_DIR / 'taxi_price.csv'
PROCESSED_DATA_FILE = DATA_DIR / 'taxi_price_processed.csv'
TARGET_COLUMN = 'Trip_Price'

# Google Drive ID cho dữ liệu
GDRIVE_FILE_ID = '1w30FAYe9SU5ARU6cBYV4AxOc3FlvL7xZ'

# ========== TIỀN XỬ LÝ DỮ LIỆU ==========
# Cột số
NUMERIC_COLS = [
    'Trip_Distance_km',
    'Passenger_Count', 
    'Base_Fare',
    'Per_Km_Rate',
    'Per_Minute_Rate',
    'Trip_Duration_Minutes',
    'Speed_kmh'
]

# Cột phân loại
CATEGORICAL_COLS = [
    'Time_of_Day',
    'Day_of_Week',
    'Traffic_Conditions',
    'Weather'
]

# Chiến lược xử lý missing values
MISSING_STRATEGY = {
    'numeric': 'median',      # 'mean', 'median', 'mode'
    'categorical': 'mode'     # 'mode', 'constant'
}

# Cấu hình tạo feature tốc độ (km/h)
SPEED_FEATURE = {
    'enabled': False,  # TẮT để tránh tạo thêm missing values
    'name': 'Speed_kmh',
    'distance_col': 'Trip_Distance_km',
    'duration_col': 'Trip_Duration_Minutes',
    'min_duration_minutes': 1.0,  # tránh chia cho 0
    'round_digits': 2
}

# Quy tắc ràng buộc dữ liệu (type, miền giá trị, hành động)
CONSTRAINT_RULES = {
    # 1. Trip distance: enforce 0-140 km window
    "Trip_Distance_km": {
        "min": 0.0,
        "max": 140.0,
        "dtype": "float",
        "action": "clip",
    },
    # 2. Passenger count must be an integer between 1-6
    "Passenger_Count": {
        "min": 1,
        "max": 6,
        "dtype": "int",
        "action": "clip",
    },
    # 3. Trip duration constrained to 0-120 minutes
    "Trip_Duration_Minutes": {
        "min": 0.0,
        "max": 120.0,
        "dtype": "float",
        "action": "clip",
    },
    # 4. Base fare cannot be negative; convert violations to NaN for imputation
    "Base_Fare": {
        "min": 0.0,
        "dtype": "float",
        "action": "mean",
    },
    # 5. Distance-based rate must be non-negative
    "Per_Km_Rate": {
        "min": 0.0,
        "dtype": "float",
        "action": "mean",
    },
    # 6. Time-based rate must be non-negative
    "Per_Minute_Rate": {
        "min": 0.0,
        "dtype": "float",
        "action": "mean",
    },
    # 7. Target price cannot be negative; drop invalid rows to avoid corrupt training
    "Trip_Price": {
        "min": 0.0,
        "dtype": "float",
        "action": "drop",
    },
    # 8. Vận tốc trung bình (km/h)
    "Speed_kmh": {
        "min": 0.0,
        "max": 160.0,
        "dtype": "float",
        "action": "clip",
    },
}

# Phương pháp encoding
ENCODING_METHOD = 'onehot'    # 'onehot', 'label'
DROP_FIRST_ONEHOT = True

# Phương pháp scaling
SCALING_METHOD = 'standard'   # 'standard', 'minmax'

# Outlier detection
OUTLIER_DETECTION = False
OUTLIER_METHOD = 'iqr'        # 'iqr', 'zscore', 'isolation_forest'
OUTLIER_THRESHOLD = 1.5

# ========== TRAINING ==========
TEST_SIZE = 0.2
RANDOM_SEED = 42

# Optuna optimization
OPTUNA_N_TRIALS = {
    'polynomial': 10,
    'random_forest': 20,
    'extra_trees': 20,
    'xgboost': 30
}

OPTUNA_TIMEOUT = {
    'polynomial': 300,      # 5 phút
    'random_forest': 600,   # 10 phút
    'extra_trees': 600,     # 10 phút
    'xgboost': 900          # 15 phút
}

# Hyperparameters mặc định (nếu không dùng optimization)
DEFAULT_HYPERPARAMS = {
    'polynomial': {
        'degree': 3,
        'alpha': 1.0
    },
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 2
    },
    'extra_trees': {
        'n_estimators': 200,
        'max_depth': 12,
        'min_samples_split': 2,
        'min_samples_leaf': 1
    },
    'xgboost': {
        'max_depth': 4,
        'learning_rate': 0.05,
        'n_estimators': 150,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'min_child_weight': 5,
        'gamma': 0.1,
        'reg_lambda': 1.0,
        'reg_alpha': 0.5
    }
}

# ========== OUTPUT ==========
MODEL_FORMAT = 'joblib'       # 'joblib', 'pickle'
RESULTS_FILE = 'model_results.json'

# Logging
LOG_LEVEL = 'INFO'            # 'DEBUG', 'INFO', 'WARNING', 'ERROR'
LOG_FILE = PROJECT_ROOT / 'training.log'

# ========== VISUALIZATION ==========
PLOT_DPI = 300
PLOT_STYLE = 'seaborn-v0_8-darkgrid'
FIGURE_SIZE = (10, 6)

# ========== FEATURE SELECTION ==========
POLY_CORRELATION_THRESHOLD = 0.3

# ========== FEATURE ENGINEERING ==========
CREATE_INTERACTION_FEATURES = False
INTERACTION_PAIRS = [
    ('Trip_Distance_km', 'Per_Km_Rate'),
    ('Trip_Duration_Minutes', 'Per_Minute_Rate')
]
