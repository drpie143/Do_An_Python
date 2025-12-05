"""
Script dự đoán giá taxi sử dụng model đã train.

Sử dụng:
    # Dự đoán từ file CSV
    python predict.py --input data/new_data.csv --output results/predictions.csv
    
    # Dự đoán từ file với model cụ thể
    python predict.py --input data/new_data.csv --model random_forest
    
    # Chế độ interactive (nhập dữ liệu thủ công)
    python predict.py --interactive
    
    # Liệt kê các model khả dụng
    python predict.py --list-models
"""

import argparse
import io
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd
import joblib

from config import (
    MODELS_DIR,
    NUMERIC_COLS,
    CATEGORICAL_COLS,
    TARGET_COLUMN,
)


# Setup logging với UTF-8 support cho Windows
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.StreamHandler(io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace'))
    ]
)
logger = logging.getLogger(__name__)


# Model names mapping
MODEL_NAMES = {
    'polynomial': 'Polynomial Regression',
    'random_forest': 'Random Forest',
    'extra_trees': 'Extra Trees',
    'xgboost': 'XGBoost',
}


def list_available_models() -> Dict[str, Path]:
    """Liệt kê các model đã train có sẵn."""
    available = {}
    for model_key in MODEL_NAMES.keys():
        model_path = MODELS_DIR / f"{model_key}.joblib"
        if model_path.exists():
            available[model_key] = model_path
    return available


def load_transformer() -> Any:
    """Load DataTransformer đã fit."""
    transformer_path = MODELS_DIR / "data_transformer.joblib"
    if not transformer_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy DataTransformer tại {transformer_path}\n"
            "Vui lòng chạy main.py để train model trước."
        )
    logger.info("[LOAD] Loading DataTransformer từ %s", transformer_path)
    return joblib.load(transformer_path)


def load_model(model_name: str) -> Any:
    """Load model đã train."""
    model_path = MODELS_DIR / f"{model_name}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy model '{model_name}' tại {model_path}\n"
            f"Models khả dụng: {list(list_available_models().keys())}"
        )
    logger.info("[LOAD] Loading model '%s' từ %s", model_name, model_path)
    model_data = joblib.load(model_path)
    
    # Model được lưu dưới dạng dict với key 'model'
    if isinstance(model_data, dict) and 'model' in model_data:
        return model_data['model']
    return model_data


def prepare_input_data(
    data: pd.DataFrame, 
    transformer: Any
) -> pd.DataFrame:
    """Chuẩn bị dữ liệu input cho prediction."""
    # Remove target column if present
    if TARGET_COLUMN in data.columns:
        data = data.drop(columns=[TARGET_COLUMN])
        logger.info("[NOTE] Đã bỏ cột target '%s' khỏi input", TARGET_COLUMN)
    
    # Transform data
    logger.info("[PROCESS] Transforming dữ liệu...")
    X_transformed = transformer.transform_new_data(data)
    
    # Remove target column from transformed data (nếu có trong feature_columns)
    if TARGET_COLUMN in X_transformed.columns:
        X_transformed = X_transformed.drop(columns=[TARGET_COLUMN])
    
    return X_transformed


def predict_from_file(
    input_path: str,
    output_path: Optional[str] = None,
    model_name: str = "polynomial"
) -> pd.DataFrame:
    """Dự đoán từ file CSV."""
    # Load data
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file input: {input_path}")
    
    logger.info("[INPUT] Loading dữ liệu từ %s", input_path)
    data = pd.read_csv(input_path)
    logger.info("[DATA] Shape: %s", data.shape)
    
    # Load transformer and model
    transformer = load_transformer()
    model = load_model(model_name)
    
    # Prepare and predict
    X = prepare_input_data(data.copy(), transformer)
    logger.info("[MODEL] Đang dự đoán với model '%s'...", MODEL_NAMES.get(model_name, model_name))
    predictions = model.predict(X)
    
    # Create output DataFrame
    result = data.copy()
    result['Predicted_Price'] = predictions
    
    # Save if output path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_path, index=False)
        logger.info("[SAVE] Kết quả đã lưu tại %s", output_path)
    
    # Display summary
    logger.info("\n" + "="*50)
    logger.info("KẾT QUẢ DỰ ĐOÁN")
    logger.info("="*50)
    logger.info("Số lượng mẫu: %d", len(predictions))
    logger.info("Giá trung bình: %.2f", np.mean(predictions))
    logger.info("Giá min: %.2f", np.min(predictions))
    logger.info("Giá max: %.2f", np.max(predictions))
    
    return result


def interactive_predict(model_name: str = "polynomial") -> None:
    """Chế độ dự đoán interactive."""
    print("\n" + "="*60)
    print(" TAXI PRICE PREDICTION - INTERACTIVE MODE")
    print("="*60)
    print(f" Model: {MODEL_NAMES.get(model_name, model_name)}")
    print(" Nhập 'quit' hoặc 'q' để thoát")
    print("="*60 + "\n")
    
    # Load transformer and model
    try:
        transformer = load_transformer()
        model = load_model(model_name)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return
    
    # Feature descriptions
    feature_info = {
        'Trip_Distance_km': ('Khoảng cách (km)', float, 1.0, 100.0),
        'Time_of_Day': ('Thời gian trong ngày', str, None, ['Morning', 'Afternoon', 'Evening', 'Night']),
        'Day_of_Week': ('Ngày trong tuần', str, None, ['Weekday', 'Weekend']),
        'Passenger_Count': ('Số hành khách', int, 1, 6),
        'Traffic_Conditions': ('Tình trạng giao thông', str, None, ['Low', 'Medium', 'High']),
        'Weather': ('Thời tiết', str, None, ['Clear', 'Rain', 'Fog']),
        'Base_Fare': ('Giá cơ bản', float, 1.0, 10.0),
        'Per_Km_Rate': ('Giá mỗi km', float, 0.3, 2.0),
        'Per_Minute_Rate': ('Giá mỗi phút', float, 0.1, 1.0),
        'Trip_Duration_Minutes': ('Thời gian chuyến đi (phút)', float, 5.0, 200.0),
    }
    
    while True:
        print("\n" + "-"*40)
        print(" NHẬP THÔNG TIN CHUYẾN ĐI")
        print("-"*40)
        
        data = {}
        try:
            for col, (desc, dtype, min_val, max_val_or_options) in feature_info.items():
                while True:
                    if isinstance(max_val_or_options, list):
                        # Categorical
                        options_str = ", ".join(max_val_or_options)
                        prompt = f" {desc} [{options_str}]: "
                    else:
                        # Numeric
                        prompt = f" {desc} ({min_val}-{max_val_or_options}): "
                    
                    value = input(prompt).strip()
                    
                    if value.lower() in ['quit', 'q']:
                        print("\n[*] Tạm biệt!")
                        return
                    
                    if not value:
                        print("   [!] Vui lòng nhập giá trị")
                        continue
                    
                    try:
                        if dtype == str:
                            # Validate categorical
                            if isinstance(max_val_or_options, list):
                                # Case insensitive match
                                matched = None
                                for opt in max_val_or_options:
                                    if opt.lower() == value.lower():
                                        matched = opt
                                        break
                                if matched is None:
                                    print(f"   [!] Giá trị không hợp lệ. Chọn: {options_str}")
                                    continue
                                data[col] = matched
                            else:
                                data[col] = value
                        elif dtype == int:
                            val = int(value)
                            if min_val is not None and val < min_val:
                                print(f"   [!] Giá trị phải >= {min_val}")
                                continue
                            if max_val_or_options is not None and val > max_val_or_options:
                                print(f"   [!] Giá trị phải <= {max_val_or_options}")
                                continue
                            data[col] = val
                        else:  # float
                            val = float(value)
                            if min_val is not None and val < min_val:
                                print(f"   [!] Giá trị phải >= {min_val}")
                                continue
                            if max_val_or_options is not None and val > max_val_or_options:
                                print(f"   [!] Giá trị phải <= {max_val_or_options}")
                                continue
                            data[col] = val
                        break
                    except ValueError:
                        print(f"   [!] Giá trị không hợp lệ cho {dtype.__name__}")
                        continue
            
            # Create DataFrame and predict
            df = pd.DataFrame([data])
            
            # Calculate Speed_kmh if not provided
            if 'Speed_kmh' not in df.columns:
                duration_hours = df['Trip_Duration_Minutes'].values[0] / 60
                df['Speed_kmh'] = df['Trip_Distance_km'] / duration_hours if duration_hours > 0 else 0
            
            # Transform and predict
            X = prepare_input_data(df, transformer)
            prediction = model.predict(X)[0]
            
            # Display result
            print("\n" + "="*40)
            print(" KẾT QUẢ DỰ ĐOÁN")
            print("="*40)
            print(f" >> GIÁ DỰ ĐOÁN: ${prediction:.2f}")
            print("="*40)
            
            # Ask to continue
            cont = input("\n Dự đoán tiếp? (y/n): ").strip().lower()
            if cont not in ['y', 'yes', '']:
                print("\n[*] Tạm biệt!")
                return
                
        except KeyboardInterrupt:
            print("\n\n[*] Đã hủy. Tạm biệt!")
            return


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Dự đoán giá taxi sử dụng model đã train",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  python predict.py --input data/new_data.csv --output results/predictions.csv
  python predict.py --input data/test.csv --model random_forest
  python predict.py --interactive
  python predict.py --list-models
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        help='Đường dẫn file CSV chứa dữ liệu cần dự đoán'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Đường dẫn file CSV để lưu kết quả (optional)'
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='polynomial',
        choices=list(MODEL_NAMES.keys()),
        help='Model sử dụng để dự đoán (default: polynomial)'
    )
    
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Chế độ nhập dữ liệu thủ công'
    )
    
    parser.add_argument(
        '--list-models',
        action='store_true',
        help='Liệt kê các model đã train khả dụng'
    )
    
    args = parser.parse_args()
    
    # List models
    if args.list_models:
        print("\n" + "="*50)
        print(" MODELS KHẢ DỤNG")
        print("="*50)
        available = list_available_models()
        if not available:
            print(" [!] Chưa có model nào được train.")
            print(" Vui lòng chạy main.py để train model trước.")
        else:
            for key, path in available.items():
                status = "[OK]"
                print(f" {status} {MODEL_NAMES[key]:<25} ({key})")
        print("="*50 + "\n")
        return
    
    # Interactive mode
    if args.interactive:
        interactive_predict(args.model)
        return
    
    # File prediction
    if args.input:
        try:
            result = predict_from_file(
                input_path=args.input,
                output_path=args.output,
                model_name=args.model
            )
            
            # Print first few predictions
            print("\n Một số kết quả dự đoán:")
            print("-"*50)
            cols_to_show = ['Trip_Distance_km', 'Time_of_Day', 'Predicted_Price']
            cols_available = [c for c in cols_to_show if c in result.columns]
            print(result[cols_available].head(10).to_string(index=False))
            
        except Exception as e:
            logger.error("[ERROR] %s", e)
            sys.exit(1)
        return
    
    # No action specified
    parser.print_help()


if __name__ == "__main__":
    main()
