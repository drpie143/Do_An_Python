"""
Script chính để chạy toàn bộ pipeline:
1. Download và load dữ liệu
2. Tiền xử lý dữ liệu (DataPreprocessor)
3. Training và tối ưu mô hình (ModelTrainer)
4. Đánh giá và visualization
5. Lưu mô hình và kết quả

Cách chạy:
    python main.py
    
    hoặc với tùy chọn:
    python main.py --optimize    # Chạy optimization với Optuna
    python main.py --no-viz      # Không vẽ biểu đồ
"""

import argparse
import logging
import sys
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

# Import từ project
from src.preprocessing.data_preprocessor import DataPreprocessor
from src.modeling.model_trainer import ModelTrainer
import config


# Cấu hình logging (force=True để ghi rõ ràng vào training.log)
config.LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.LOG_FILE, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ],
    force=True
)
logger = logging.getLogger(__name__)


def download_data():
    """Download dữ liệu từ Google Drive nếu chưa có."""
    if config.DATA_FILE.exists():
        logger.info(f"✅ Dữ liệu đã tồn tại: {config.DATA_FILE}")
        return
    
    logger.info("📥 Đang download dữ liệu từ Google Drive...")
    
    try:
        # Cài đặt gdown nếu chưa có
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown", "-q"])
        
        # Download file
        subprocess.run([
            "gdown", 
            config.GDRIVE_FILE_ID, 
            "-O", 
            str(config.DATA_FILE)
        ], check=True)
        
        logger.info(f"✅ Đã download dữ liệu vào: {config.DATA_FILE}")
        
    except Exception as e:
        logger.error(f"❌ Lỗi khi download dữ liệu: {e}")
        logger.info("💡 Vui lòng download thủ công và đặt vào thư mục data/")
        sys.exit(1)

def preprocess_data(
    generate_viz: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Optional[List[str]]]:
    """Tiền xử lý dữ liệu, tách train/test và trả về bộ dữ liệu đã sạch."""
    logger.info("\n" + "="*70)
    logger.info("📊 BƯỚC 1: TIỀN XỬ LÝ DỮ LIỆU")
    logger.info("="*70 + "\n")
    
    # Khởi tạo preprocessor và load data
    preprocessor = DataPreprocessor()
    preprocessor.load(str(config.DATA_FILE))
    preprocessor.feature_engineering(settings=getattr(config, "SPEED_FEATURE", None))
    if getattr(config, "CONSTRAINT_RULES", None):
        preprocessor.constraint_rules = config.CONSTRAINT_RULES.copy()
    
    logger.info(f"Dữ liệu gốc: {preprocessor.data.shape}")
    
    # Tạo báo cáo tổng quan nhanh để log các cảnh báo EDA
    logger.info("\n🧾 Đang chạy EDA overview...")
    overview = preprocessor.eda_overview(top_n=5)

    missing_df = None
    if overview and isinstance(overview.get("column_profile"), pd.DataFrame):
        column_profile = overview["column_profile"]
        missing_df = (
            column_profile[column_profile["missing"] > 0]
            .reset_index()
            .rename(columns={"index": "Column", "missing": "Missing_Count", "missing_pct": "Missing_Percent"})
        )
    if missing_df is not None and not missing_df.empty:
        print("\n⚠️  Missing Values:")
        print(missing_df.to_string(index=False))

    if generate_viz:
        logger.info("\n🖼️  Đang tạo các biểu đồ EDA (tự động lưu tại results/eda)...")
        preprocessor.generate_eda_report(target_col=config.TARGET_COLUMN)

    # Áp dụng ràng buộc dữ liệu (không phụ thuộc train/test)
    if preprocessor.constraint_rules:
        preprocessor.apply_constraints()

    # Tạo interaction features nếu cần (deterministic, không cần fit)
    if config.CREATE_INTERACTION_FEATURES:
        preprocessor.create_interaction_features(
            col_pairs=config.INTERACTION_PAIRS,
            operations=['multiply']
        )

    base_df = preprocessor.get_processed_data()
    logger.info("📂 Dữ liệu sau bước tiền xử lý cơ bản: %s", base_df.shape)

    logger.info("\n🔀 Chia train/test trước khi fit encoder/scaler để tránh data leakage...")
    train_df, test_df = train_test_split(
        base_df,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_SEED,
        shuffle=True
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    logger.info("Train: %s | Test: %s", train_df.shape, test_df.shape)

    train_preprocessor = DataPreprocessor(train_df)
    train_preprocessor.handle_missing(
        strategy='auto',
        numeric_strategy=config.MISSING_STRATEGY['numeric'],
        categorical_strategy=config.MISSING_STRATEGY['categorical']
    )
    if config.OUTLIER_DETECTION:
        train_preprocessor.remove_outliers(
            method=config.OUTLIER_METHOD,
            threshold=config.OUTLIER_THRESHOLD
        )
    train_preprocessor.encode_categorical(
        method=config.ENCODING_METHOD,
        drop_first=config.DROP_FIRST_ONEHOT
    )
    logger.info("\n📏 Chuẩn hóa features (fit trên train, transform cho test)...")
    train_preprocessor.scale_features(
        method='standard',
        exclude_columns=[config.TARGET_COLUMN]
    )

    train_processed = train_preprocessor.get_processed_data()
    test_processed = train_preprocessor.transform_dataset(test_df)

    # Heatmap/Correlation dựa trên train đã xử lý
    heatmap_path = config.EDA_RESULTS_DIR / 'correlation_heatmap.png'
    corr_df = train_preprocessor.plot_correlation_heatmap(
        target_col=config.TARGET_COLUMN,
        method='spearman',
        save_path=heatmap_path,
        annot=True,
        show=False
    )
    logger.info(f"📌 Heatmap tương quan (train) đã lưu tại: {heatmap_path}")
    poly_feature_subset: Optional[List[str]] = None
    if corr_df is not None and config.TARGET_COLUMN in corr_df.columns:
        corr_series = corr_df[config.TARGET_COLUMN].drop(labels=[config.TARGET_COLUMN])
        selected = corr_series[abs(corr_series) >= config.POLY_CORRELATION_THRESHOLD]
        if not selected.empty:
            poly_feature_subset = selected.index.tolist()
            logger.info(
                f"🎯 {len(poly_feature_subset)} feature có |corr| >= {config.POLY_CORRELATION_THRESHOLD}: {poly_feature_subset}"
            )
        else:
            logger.warning(
                f"⚠️  Không có feature nào đạt ngưỡng |corr| >= {config.POLY_CORRELATION_THRESHOLD}. Sử dụng toàn bộ features cho Polynomial."
            )

    train_preprocessor.print_summary()

    combined_processed = pd.concat([train_processed, test_processed], axis=0).reset_index(drop=True)
    combined_processed.to_csv(config.PROCESSED_DATA_FILE, index=False)
    logger.info("💾 Đã lưu dữ liệu đã xử lý (train+test) tại %s", config.PROCESSED_DATA_FILE)

    X_train = train_processed.drop(columns=[config.TARGET_COLUMN])
    y_train = train_processed[config.TARGET_COLUMN]
    X_test = test_processed.drop(columns=[config.TARGET_COLUMN])
    y_test = test_processed[config.TARGET_COLUMN]

    return X_train, X_test, y_train, y_test, poly_feature_subset

def train_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    optimize: bool = False,
    poly_feature_subset: Optional[List[str]] = None,
):
    """
    Huấn luyện các mô hình học máy trên bộ dữ liệu đã tách train/test.
    
    Args:
        X_train, X_test, y_train, y_test: dữ liệu sau preprocessing (không leakage)
        optimize: Có chạy optimization không
        poly_feature_subset: Danh sách feature dùng riêng cho Polynomial Regression
        
    Returns:
        ModelTrainer instance
    """
    logger.info("\n" + "="*70)
    logger.info("🤖 BƯỚC 2: HUẤN LUYỆN MÔ HÌNH")
    logger.info("="*70 + "\n")
    
    # Khởi tạo trainer
    trainer = ModelTrainer(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        output_dir=str(config.MODELS_DIR)
    )
    
    logger.info(f"Data info: {trainer.data_info}\n")
    
    # ========== POLYNOMIAL REGRESSION ==========
    if optimize:
        logger.info("🔍 Tối ưu Polynomial Regression...")
        best_poly_params = trainer.optimize_polynomial(
            n_trials=config.OPTUNA_N_TRIALS['polynomial'],
            timeout=config.OPTUNA_TIMEOUT['polynomial']
        )
        trainer.train_polynomial(
            degree=best_poly_params.get('degree', config.DEFAULT_HYPERPARAMS['polynomial']['degree']),
            alpha=best_poly_params.get('alpha', config.DEFAULT_HYPERPARAMS['polynomial']['alpha']),
            feature_subset=poly_feature_subset
        )
    else:
        default_poly = config.DEFAULT_HYPERPARAMS['polynomial']
        trainer.train_polynomial(
            degree=default_poly['degree'],
            alpha=default_poly['alpha'],
            feature_subset=poly_feature_subset
        )
    
    # ========== RANDOM FOREST ==========
    if optimize:
        logger.info("🔍 Tối ưu Random Forest...")
        best_rf_params = trainer.optimize_rf(
            n_trials=config.OPTUNA_N_TRIALS['random_forest'],
            timeout=config.OPTUNA_TIMEOUT['random_forest']
        )
        trainer.train_rf(**best_rf_params)
    else:
        trainer.train_rf(**config.DEFAULT_HYPERPARAMS['random_forest'])
    
    # ========== XGBOOST ==========
    if optimize:
        logger.info("🔍 Tối ưu XGBoost...")
        best_xgb_params = trainer.optimize_xgb(
            n_trials=config.OPTUNA_N_TRIALS['xgboost'],
            timeout=config.OPTUNA_TIMEOUT['xgboost']
        )
        trainer.train_xgb(**best_xgb_params)
    else:
        trainer.train_xgb(**config.DEFAULT_HYPERPARAMS['xgboost'])
    
    return trainer


def evaluate_and_visualize(trainer: ModelTrainer, visualize: bool = True):
    """
    Đánh giá và visualization kết quả.
    
    Args:
        trainer: ModelTrainer instance
        visualize: Có vẽ biểu đồ không
    """
    logger.info("\n" + "="*70)
    logger.info("📊 BƯỚC 3: ĐÁNH GIÁ VÀ VISUALIZATION")
    logger.info("="*70 + "\n")
    
    # In tóm tắt kết quả
    trainer.summary()
    
    # Lưu kết quả
    trainer.save_results(config.RESULTS_FILE)
    
    # Lưu tất cả mô hình
    trainer.save_all_models(format=config.MODEL_FORMAT)
    
    if visualize:
        # Vẽ biểu đồ so sánh
        logger.info("📈 Vẽ biểu đồ so sánh...")
        trainer.plot_comparison(metric='test_r2', save=True)
        trainer.plot_comparison(metric='test_rmse', save=True)
        trainer.plot_comparison(metric='test_mae', save=True)
        
        # Vẽ biểu đồ predictions
        logger.info("📈 Vẽ biểu đồ predictions...")
        trainer.plot_all_predictions(save=True)
        
        # Vẽ feature importance
        logger.info("📈 Vẽ biểu đồ feature importance...")
        trainer.plot_all_feature_importance(top_n=15, save=True)
        
        # So sánh feature importance
        logger.info("📈 So sánh feature importance...")
        trainer.compare_feature_importance(top_n=10, save=True)
    
    # Tìm mô hình tốt nhất
    best_name, best_result = trainer.get_best_model()
    
    logger.info("\n" + "="*70)
    logger.info(f"✨ MÔ HÌNH TỐT NHẤT: {best_name.upper()}")
    logger.info(f"   Test R²: {best_result['test_r2']:.6f}")
    logger.info(f"   Test RMSE: {best_result['test_rmse']:.6f}")
    logger.info(f"   Test MAE: {best_result['test_mae']:.6f}")
    logger.info("="*70 + "\n")


def main():
    """Main function để chạy toàn bộ pipeline."""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Taxi Price Prediction Pipeline')
    parser.add_argument('--optimize', action='store_true', 
                       help='Chạy optimization với Optuna')
    parser.add_argument('--no-viz', action='store_true',
                       help='Không vẽ biểu đồ visualization')
    parser.add_argument('--skip-download', action='store_true',
                       help='Bỏ qua bước download dữ liệu')
    
    args = parser.parse_args()
    
    logger.info("\n" + "="*70)
    logger.info("🚀 BẮT ĐẦU TAXI PRICE PREDICTION PIPELINE")
    logger.info("="*70 + "\n")
    
    try:
        # Bước 0: Download dữ liệu
        if not args.skip_download:
            download_data()
        
        # Bước 1: Tiền xử lý (tách train/test trước khi train)
        X_train, X_test, y_train, y_test, poly_feature_subset = preprocess_data(
            generate_viz=not args.no_viz
        )

        # Bước 2: Training
        trainer = train_models(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            optimize=args.optimize,
            poly_feature_subset=poly_feature_subset
        )
        
        # Bước 3: Đánh giá và visualization
        evaluate_and_visualize(trainer, visualize=not args.no_viz)
        
        logger.info("\n" + "="*70)
        logger.info("✅ HOÀN TẤT PIPELINE THÀNH CÔNG!")
        logger.info("="*70 + "\n")
        logger.info(f"📁 Mô hình đã lưu tại: {config.MODELS_DIR}")
        logger.info(f"📁 Kết quả đã lưu tại: {config.RESULTS_DIR}")
        
    except Exception as e:
        logger.error(f"\n❌ LỖI: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
