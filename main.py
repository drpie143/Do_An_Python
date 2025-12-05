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
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

# Import từ project
from src.preprocessing import DataLoader, DataTransformer
from src.modeling import ModelTrainer
from src.modeling.base_trainer import log_section, log_step
import config


# Cấu hình logging (ghi đè file log mỗi lần chạy & format rõ ràng)
config.LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
CONSOLE_FORMAT = "%(levelname)-8s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

file_handler = logging.FileHandler(config.LOG_FILE, mode='w', encoding='utf-8')
file_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))

# Console handler với encoding UTF-8 để hỗ trợ emoji trên Windows
console_handler = logging.StreamHandler(
    open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
)
console_handler.setFormatter(logging.Formatter(CONSOLE_FORMAT, DATE_FORMAT))

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    handlers=[file_handler, console_handler],
    force=True,
)
logger = logging.getLogger(__name__)


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


def preprocess_data(
    generate_viz: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Optional[List[str]], DataTransformer]:
    """
    Load raw data, clean it, and return processed train/test splits.
    
    Quy trình:
    1. PRE-SPLIT: Sử dụng DataLoader (xóa duplicates, unify text, apply constraints)
    2. CHIA TRAIN/TEST
    3. POST-SPLIT: Sử dụng DataTransformer (fit_transform trên train, transform trên test)
    """
    log_section("BƯỚC 1: TIỀN XỬ LÝ DỮ LIỆU", icon="🧼")

    if not config.DATA_FILE.exists():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {config.DATA_FILE}")

    # ========================================================================
    # PHASE 1: LOAD & PRE-SPLIT CLEANING (sử dụng DataLoader)
    # ========================================================================
    log_step("Đang nạp và xử lý dữ liệu gốc (DataLoader)", icon="📥")
    
    loader = DataLoader.from_file(config.DATA_FILE)
    loader.drop_duplicates()
    loader.unify_values()
    loader.apply_constraints(constraint_rules=config.CONSTRAINT_RULES)
    
    raw_df = loader.get_data()
    log_key_value("After pre-split cleaning", raw_df.shape)

    # EDA trên dữ liệu gốc
    if generate_viz:
        log_step("Đang tạo các biểu đồ EDA", icon="🖼️")
        loader.generate_eda_report(target_col=config.TARGET_COLUMN)

    # ========================================================================
    # PHASE 2: CHIA TRAIN/TEST
    # ========================================================================
    log_step("Chia train/test", icon="✂️")
    train_df, test_df = train_test_split(
        raw_df,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_SEED,
        shuffle=True,
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    log_key_value("Train shape", train_df.shape)
    log_key_value("Test shape", test_df.shape)

    # ========================================================================
    # PHASE 3: FIT_TRANSFORM trên TRAIN (sử dụng DataTransformer)
    # ========================================================================
    log_step("Xử lý TRAIN set (DataTransformer.fit_transform)", icon="🔧")
    
    transformer = DataTransformer(
        data=train_df.copy(),
        missing_strategy=config.MISSING_STRATEGY["numeric"],
        categorical_missing_strategy=config.MISSING_STRATEGY["categorical"],
        scaler_type=config.SCALING_METHOD,
        encoder_type=config.ENCODING_METHOD,
    )

    train_processed = transformer.fit_transform(
        target_col=config.TARGET_COLUMN,
        remove_outliers=config.OUTLIER_DETECTION,
        outlier_method=config.OUTLIER_METHOD,
        outlier_threshold=config.OUTLIER_THRESHOLD,
        encoding_method=config.ENCODING_METHOD,
        drop_first_onehot=config.DROP_FIRST_ONEHOT,
        scaling_method=config.SCALING_METHOD,
        interaction_pairs=config.INTERACTION_PAIRS if config.CREATE_INTERACTION_FEATURES else None,
    )

    # Chọn features cho Polynomial Regression (dùng method có sẵn)
    poly_feature_subset = transformer.get_correlated_features(
        target_col=config.TARGET_COLUMN,
        threshold=config.POLY_CORRELATION_THRESHOLD,
        method='spearman'
    ) or None

    transformer.print_summary()

    # ========================================================================
    # PHASE 4: TRANSFORM TEST (sử dụng transformer đã fit)
    # ========================================================================
    log_step("Transform TEST set (DataTransformer.transform_new_data)", icon="🔄")
    test_processed = transformer.transform_new_data(test_df)
    log_key_value("Train processed shape", train_processed.shape)
    log_key_value("Test processed shape", test_processed.shape)

    # Lưu dữ liệu đã xử lý
    combined = pd.concat([
        train_processed.assign(split='train'),
        test_processed.assign(split='test')
    ], ignore_index=True)
    combined.to_csv(config.PROCESSED_DATA_FILE, index=False)
    log_step(f"Đã lưu dữ liệu tại: {config.PROCESSED_DATA_FILE}", icon="💾")

    # Tách X và y (dùng method có sẵn cho train, pandas cho test)
    X_train, y_train = transformer.split_features_target(config.TARGET_COLUMN)
    X_test = test_processed.drop(columns=[config.TARGET_COLUMN])
    y_test = test_processed[config.TARGET_COLUMN]

    return (X_train, X_test, y_train, y_test, poly_feature_subset, transformer)


def train_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    optimize: bool = False,
    poly_feature_subset: Optional[List[str]] = None,
) -> ModelTrainer:
    """Huấn luyện các mô hình học máy."""
    log_section("BƯỚC 2: HUẤN LUYỆN MÔ HÌNH", icon="🤖")
    
    trainer = ModelTrainer(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        output_dir=str(config.MODELS_DIR)
    )
    
    # Sử dụng train_all() để huấn luyện tất cả models
    trainer.train_all(
        optimize=optimize,
        poly_feature_subset=poly_feature_subset,
        hyperparams=config.DEFAULT_HYPERPARAMS,
        optuna_config={
            'n_trials': config.OPTUNA_N_TRIALS,
            'timeout': config.OPTUNA_TIMEOUT,
        }
    )
    
    return trainer


def evaluate_and_visualize(trainer: ModelTrainer, transformer: DataTransformer, visualize: bool = True) -> None:
    """
    Đánh giá và visualization kết quả.
    
    Args:
        trainer: ModelTrainer instance
        transformer: DataTransformer instance (đã fit)
        visualize: Có vẽ biểu đồ không
    """
    log_section("BƯỚC 3: ĐÁNH GIÁ & VISUALIZATION", icon="📊")
    
    # In tóm tắt kết quả
    trainer.summary()
    
    # Lưu kết quả
    trainer.save_results(config.RESULTS_FILE)
    
    # Lưu toàn bộ mô hình và cấu hình tiền xử lý
    saved_model_paths = trainer.save_all_models(format=config.MODEL_FORMAT)
    transformer_path = config.MODELS_DIR / "data_transformer.joblib"
    transformer.save_state(transformer_path)
    save_pipeline_metadata(transformer_path, saved_model_paths, trainer)
    
    if visualize:
        # Vẽ biểu đồ metrics tổng hợp (R², RMSE, MAE trong 1 hình)
        log_step("Vẽ biểu đồ metrics tổng hợp", icon="📈")
        trainer.plot_metrics_summary(save=True)
        
        # Vẽ biểu đồ predictions tổng hợp (tất cả models trong 1 hình)
        log_step("Vẽ biểu đồ predictions tổng hợp", icon="📈")
        trainer.plot_combined_predictions(save=True)
        
        # Vẽ feature importance so sánh (1 hình cho tất cả tree-based models)
        log_step("Vẽ biểu đồ feature importance so sánh", icon="📈")
        trainer.compare_feature_importance(top_n=10, save=True)
    
    # Tìm mô hình tốt nhất
    best_name, best_result = trainer.get_best_model()
    
    log_section("MÔ HÌNH TỐT NHẤT", icon="✨")
    log_key_value("Model", best_name.upper())
    log_key_value("Test R²", f"{best_result['test_r2']:.6f}")
    log_key_value("Test RMSE", f"{best_result['test_rmse']:.6f}")
    log_key_value("Test MAE", f"{best_result['test_mae']:.6f}")


def save_pipeline_metadata(transformer_path: Path, model_paths: Dict[str, str], trainer: ModelTrainer) -> Path:
    """Ghi lại trạng thái pipeline để phục vụ inference sau này."""
    best_name, best_result = trainer.get_best_model()
    metadata = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "transformer": {
            "path": str(transformer_path.resolve()),
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
        (X_train, X_test, y_train, y_test, 
         poly_feature_subset, transformer) = preprocess_data(generate_viz=not args.no_viz)
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
        evaluate_and_visualize(trainer, transformer, visualize=not args.no_viz)
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
