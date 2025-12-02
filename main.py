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
import importlib
import json
import logging
import sys
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

# Import từ project
from src.preprocessing.data_preprocessor import DataPreprocessor
from src.modeling.model_trainer import ModelTrainer
import config


# Cấu hình logging (ghi đè file log mỗi lần chạy & format rõ ràng)
config.LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
CONSOLE_FORMAT = "%(levelname)-8s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

file_handler = logging.FileHandler(config.LOG_FILE, mode='w', encoding='utf-8')
file_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(logging.Formatter(CONSOLE_FORMAT, DATE_FORMAT))

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    handlers=[file_handler, console_handler],
    force=True,
)
logger = logging.getLogger(__name__)


def _log_divider(char: str = "=", width: int = 70) -> None:
    """Log a horizontal divider with consistent width."""
    logger.info(char * width)


def log_section(title: str, icon: str = "🚀") -> None:
    """Render a bold section banner combining the legacy and sample styles."""
    logger.info("\n")
    _log_divider()
    logger.info("%s %s", icon, title.upper())
    _log_divider()
    logger.info("")


def log_step(title: str, icon: str = "🔹") -> None:
    """Highlight a sub-step within a section."""
    logger.info("%s %s", icon, title)


def log_key_value(label: str, value) -> None:
    """Align key-value summary rows for faster scanning."""
    logger.info("   %-18s: %s", label, value)


def log_stage_summary(stage_times: List[Tuple[str, float]], total_duration: float) -> None:
    """Pretty print per-stage durations and pipeline total."""
    logger.info("⏱️ PIPELINE SUMMARY")
    for stage, duration in stage_times:
        log_key_value(stage.capitalize(), f"{duration:.2f} giây")
    log_key_value("Total", f"{total_duration:.2f} giây")


def _ensure_gdown() -> Optional[object]:
    """Ensure gdown is importable without forcing installs every run."""
    try:
        return importlib.import_module("gdown")
    except ImportError:
        logger.info("gdown chưa được cài. Đang tiến hành cài đặt một lần...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
        except subprocess.CalledProcessError as exc:
            logger.warning("Không thể cài đặt gdown tự động: %s", exc)
            return None
        try:
            return importlib.import_module("gdown")
        except ImportError:
            return None


def download_data():
    """Download dữ liệu từ Google Drive nếu chưa có."""
    if config.DATA_FILE.exists():
        log_step(f"Dữ liệu đã tồn tại: {config.DATA_FILE}", icon="✅")
        return

    log_step("Đang download dữ liệu từ Google Drive (nếu khả dụng)...", icon="📥")
    gdown = _ensure_gdown()
    if gdown is None:
        logger.warning("Không thể import gdown. Vui lòng tải thủ công file taxi_price.csv và đặt vào %s", config.DATA_DIR)
        if not config.DATA_FILE.exists():
            raise FileNotFoundError(
                "Không tải được dữ liệu tự động vì thiếu gdown. Hãy đặt file taxi_price.csv vào thư mục data và chạy lại."
            )
        return

    try:
        gdown.download(id=config.GDRIVE_FILE_ID, output=str(config.DATA_FILE), quiet=False)
        log_step(f"Đã download dữ liệu vào: {config.DATA_FILE}", icon="✅")
    except Exception as exc:
        logger.error("❌ Lỗi khi download dữ liệu: %s", exc)
        logger.info("💡 Vui lòng download thủ công và đặt vào thư mục data/")

    if not config.DATA_FILE.exists():
        raise FileNotFoundError(
            "Không tìm thấy dữ liệu taxi_price.csv. Download tự động thất bại, vui lòng tải thủ công và chạy lại."
        )



def preprocess_data(generate_viz: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Optional[List[str]], DataPreprocessor]:
    """Tiền xử lý dữ liệu với train/test split trước khi fit scaler/encoder."""
    log_section("BƯỚC 1: TIỀN XỬ LÝ DỮ LIỆU", icon="🧹")
    
    preprocessor = DataPreprocessor()
    log_step("Đang nạp dữ liệu gốc", icon="📥")
    preprocessor.load(str(config.DATA_FILE))
    log_step(f"Dữ liệu gốc: {preprocessor.data.shape}", icon="📦")

    missing_df = preprocessor.check_missing()
    if len(missing_df) > 0:
        print("\n⚠️  Missing Values:")
        print(missing_df.to_string(index=False))

    if generate_viz:
        log_step("Đang tạo các biểu đồ EDA (lưu tại results/eda)", icon="🖼️")
        preprocessor.generate_eda_report(target_col=config.TARGET_COLUMN)

    log_step("Chuẩn hóa dữ liệu trước khi split", icon="🧽")
    preprocessor.apply_constraints()
    preprocessor.unify_values()
    preprocessor.feature_engineering()
    base_clean_df = preprocessor.get_processed_data()
    log_step(f"Dữ liệu sạch (pre-split): {base_clean_df.shape}", icon="📏")

    train_df, test_df = train_test_split(
        base_clean_df,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_SEED,
    )
    log_step(f"Split dữ liệu -> Train: {train_df.shape}, Test: {test_df.shape}", icon="🔀")

    train_preprocessor = DataPreprocessor(train_df)
    train_preprocessor.apply_constraints()
    train_preprocessor.unify_values()
    train_preprocessor.feature_engineering()
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
    log_step("Chuẩn hóa features (dựa trên train set)", icon="📏")
    train_preprocessor.scale_features(
        method=config.SCALING_METHOD,
        exclude_columns=[config.TARGET_COLUMN]
    )
    if config.CREATE_INTERACTION_FEATURES:
        train_preprocessor.create_interaction_features(
            col_pairs=config.INTERACTION_PAIRS,
            operations=['multiply']
        )

    heatmap_path = config.EDA_RESULTS_DIR / 'correlation_heatmap_train.png'
    corr_df = train_preprocessor.plot_correlation_heatmap(
        target_col=config.TARGET_COLUMN,
        method='spearman',
        save_path=heatmap_path,
        annot=True,
        show=False
    )
    log_step(f"Heatmap tương quan (train) đã lưu tại: {heatmap_path}", icon="📌")
    poly_feature_subset: Optional[List[str]] = None
    if corr_df is not None and config.TARGET_COLUMN in corr_df.columns:
        corr_series = corr_df[config.TARGET_COLUMN].drop(labels=[config.TARGET_COLUMN])
        selected = corr_series[abs(corr_series) >= config.POLY_CORRELATION_THRESHOLD]
        if not selected.empty:
            poly_feature_subset = selected.index.tolist()
            log_step(
                f"{len(poly_feature_subset)} feature có |corr| >= {config.POLY_CORRELATION_THRESHOLD}: {poly_feature_subset}",
                icon="🎯"
            )
        else:
            logger.warning(
                f"⚠️  Không có feature nào đạt ngưỡng |corr| >= {config.POLY_CORRELATION_THRESHOLD}. Sử dụng toàn bộ features cho Polynomial."
            )

    train_preprocessor.print_summary()
    train_preprocessor.mark_as_fitted()

    train_processed = train_preprocessor.get_processed_data()
    test_processed = train_preprocessor.transform_new_data(test_df)

    combined = pd.concat(
        [
            train_processed.assign(split='train'),
            test_processed.assign(split='test')
        ],
        ignore_index=True,
    )
    combined.to_csv(config.PROCESSED_DATA_FILE, index=False)

    X_train = train_processed.drop(columns=[config.TARGET_COLUMN])
    y_train = train_processed[config.TARGET_COLUMN]
    X_test = test_processed.drop(columns=[config.TARGET_COLUMN])
    y_test = test_processed[config.TARGET_COLUMN]

    return X_train, X_test, y_train, y_test, poly_feature_subset, train_preprocessor


def train_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    optimize: bool = False,
    poly_feature_subset: Optional[List[str]] = None,
):
    """Huấn luyện các mô hình học máy với dữ liệu đã split."""
    log_section("BƯỚC 2: HUẤN LUYỆN MÔ HÌNH", icon="🤖")
    
    # Khởi tạo trainer
    trainer = ModelTrainer(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        output_dir=str(config.MODELS_DIR)
    )
    
    log_step(f"Data info: {trainer.data_info}", icon="📊")
    
    # ========== POLYNOMIAL REGRESSION ==========
    if optimize:
        log_step("Tối ưu Polynomial Regression", icon="🔍")
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
        log_step("Tối ưu Random Forest", icon="🔍")
        best_rf_params = trainer.optimize_rf(
            n_trials=config.OPTUNA_N_TRIALS['random_forest'],
            timeout=config.OPTUNA_TIMEOUT['random_forest']
        )
        trainer.train_rf(**best_rf_params)
    else:
        trainer.train_rf(**config.DEFAULT_HYPERPARAMS['random_forest'])
    
    # ========== XGBOOST ==========
    if optimize:
        log_step("Tối ưu XGBoost", icon="🔍")
        best_xgb_params = trainer.optimize_xgb(
            n_trials=config.OPTUNA_N_TRIALS['xgboost'],
            timeout=config.OPTUNA_TIMEOUT['xgboost']
        )
        trainer.train_xgb(**best_xgb_params)
    else:
        trainer.train_xgb(**config.DEFAULT_HYPERPARAMS['xgboost'])
    
    return trainer


def evaluate_and_visualize(trainer: ModelTrainer, preprocessor: DataPreprocessor, visualize: bool = True) -> None:
    """
    Đánh giá và visualization kết quả.
    
    Args:
        trainer: ModelTrainer instance
        visualize: Có vẽ biểu đồ không
    """
    log_section("BƯỚC 3: ĐÁNH GIÁ & VISUALIZATION", icon="📊")
    
    # In tóm tắt kết quả
    trainer.summary()
    
    # Lưu kết quả
    trainer.save_results(config.RESULTS_FILE)
    
    # Lưu toàn bộ mô hình và cấu hình tiền xử lý
    saved_model_paths = trainer.save_all_models(format=config.MODEL_FORMAT)
    preprocessor_filename = f"data_preprocessor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
    preprocessor_path = config.MODELS_DIR / preprocessor_filename
    preprocessor.save_state(preprocessor_path)
    save_pipeline_metadata(preprocessor_path, saved_model_paths, trainer)
    
    if visualize:
        # Vẽ biểu đồ so sánh
        log_step("Vẽ biểu đồ so sánh", icon="📈")
        trainer.plot_comparison(metric='test_r2', save=True)
        trainer.plot_comparison(metric='test_rmse', save=True)
        trainer.plot_comparison(metric='test_mae', save=True)
        
        # Vẽ biểu đồ predictions
        log_step("Vẽ biểu đồ predictions", icon="📈")
        trainer.plot_all_predictions(save=True)
        
        # Vẽ feature importance
        log_step("Vẽ biểu đồ feature importance", icon="📈")
        trainer.plot_all_feature_importance(top_n=15, save=True)
        
        # So sánh feature importance
        log_step("So sánh feature importance", icon="📈")
        trainer.compare_feature_importance(top_n=10, save=True)
    
    # Tìm mô hình tốt nhất
    best_name, best_result = trainer.get_best_model()
    
    log_section("MÔ HÌNH TỐT NHẤT", icon="✨")
    log_key_value("Model", best_name.upper())
    log_key_value("Test R²", f"{best_result['test_r2']:.6f}")
    log_key_value("Test RMSE", f"{best_result['test_rmse']:.6f}")
    log_key_value("Test MAE", f"{best_result['test_mae']:.6f}")


def save_pipeline_metadata(preprocessor_path: Path, model_paths: Dict[str, str], trainer: ModelTrainer) -> Path:
    """Ghi lại trạng thái pipeline để phục vụ inference sau này."""
    best_name, best_result = trainer.get_best_model()
    metadata = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "preprocessor": {
            "path": str(preprocessor_path.resolve()),
            "target_column": config.TARGET_COLUMN,
            "n_features": trainer.data_info["n_features"],
        },
        "models": {},
        "best_model": best_name,
        "best_model_path": model_paths.get(best_name),
    }
    for model_name, result in trainer.results.items():
        metadata["models"][model_name] = {
            "path": model_paths.get(model_name),
            "metrics": {
                "train_rmse": float(result["train_rmse"]),
                "test_rmse": float(result["test_rmse"]),
                "test_mae": float(result["test_mae"]),
                "test_r2": float(result["test_r2"]),
            },
            "hyperparams": result["hyperparams"],
        }
    if best_name and best_result:
        metadata["best_metrics"] = {
            "test_rmse": float(best_result["test_rmse"]),
            "test_mae": float(best_result["test_mae"]),
            "test_r2": float(best_result["test_r2"]),
        }
    path = config.RESULTS_DIR / "pipeline_state.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(metadata, fp, indent=4, ensure_ascii=False)
    logger.info("✅ Đã lưu pipeline metadata: %s", path)
    return path


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
    
    log_section("BẮT ĐẦU PIPELINE DỰ ĐOÁN GIÁ TAXI", icon="🚀")
    
    stage_times = []
    pipeline_start = time.perf_counter()

    try:
        # Bước 0: Download dữ liệu
        if not args.skip_download:
            step_start = time.perf_counter()
            download_data()
            stage_times.append(("download", time.perf_counter() - step_start))
        
        # Bước 1: Tiền xử lý
        step_start = time.perf_counter()
        X_train, X_test, y_train, y_test, poly_feature_subset, preprocessor = preprocess_data(generate_viz=not args.no_viz)
        stage_times.append(("preprocess", time.perf_counter() - step_start))
        
        # Bước 2: Training
        step_start = time.perf_counter()
        trainer = train_models(
            X_train,
            X_test,
            y_train,
            y_test,
            optimize=args.optimize,
            poly_feature_subset=poly_feature_subset
        )
        stage_times.append(("train", time.perf_counter() - step_start))
        
        # Bước 3: Đánh giá và visualization
        step_start = time.perf_counter()
        evaluate_and_visualize(trainer, preprocessor, visualize=not args.no_viz)
        stage_times.append(("evaluate", time.perf_counter() - step_start))
        
        log_section("HOÀN TẤT PIPELINE", icon="✅")
        log_step(f"Mô hình đã lưu tại: {config.MODELS_DIR}", icon="📁")
        log_step(f"Kết quả đã lưu tại: {config.RESULTS_DIR}", icon="📁")

        total_duration = time.perf_counter() - pipeline_start
        logger.info("")
        log_stage_summary(stage_times, total_duration)
        
    except Exception as e:
        logger.error(f"\n❌ LỖI: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
