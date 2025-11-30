"""
Module huấn luyện mô hình học máy cho dự án Taxi Price Prediction.

Class ModelTrainer cung cấp các chức năng:
- Nạp và chia dữ liệu
- Huấn luyện nhiều mô hình (Polynomial Regression, Random Forest, XGBoost)
- Tối ưu hyperparameters bằng Optuna
- Đánh giá và so sánh mô hình
- Lưu/tải mô hình
- Trực quan hóa kết quả
"""

import logging
import json
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import xgboost as xgb
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from src.visualization import DataVisualizer
from config import MODEL_RESULTS_DIR, MODELS_DIR, PLOT_DPI, PLOT_STYLE, FIGURE_SIZE


# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Lớp xây dựng, huấn luyện và tối ưu các mô hình học máy cho bài toán Regression.
    
    Hỗ trợ:
    - 3 mô hình: Polynomial Regression, XGBoost, Random Forest
    - Tối ưu hyperparameters bằng Optuna
    - Logging quá trình huấn luyện
    - Lưu/tải mô hình
    - Đánh giá kết quả (RMSE, MAE, R²)
    
    Attributes:
        X_train, X_test: Features của train/test
        y_train, y_test: Target của train/test
        models: Dictionary lưu các mô hình đã train
        best_model: Mô hình tốt nhất
        results: Lưu kết quả đánh giá
    """
    
    RANDOM_SEED = 42
    
    def __init__(self, 
                 X_train: pd.DataFrame, 
                 X_test: pd.DataFrame,
                 y_train: pd.Series,
                 y_test: pd.Series,
                 output_dir: str = "./models"):
        """
        Khởi tạo ModelTrainer.
        
        Args:
            X_train, X_test: Features (ĐÃ SCALED từ preprocessing)
            y_train, y_test: Target
            output_dir: Thư mục lưu kết quả
        """
        self.X_train = X_train.copy()
        self.X_test = X_test.copy()
        self.y_train = y_train.copy()
        self.y_test = y_test.copy()
        
        self.output_dir = Path(output_dir) if output_dir else MODELS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.models = {}
        self.X_train_transformed = {}  # Lưu X_train đã transform (cho Polynomial)
        self.X_test_transformed = {}   # Lưu X_test đã transform (cho Polynomial)
        self.best_model = None
        self.best_model_name = None
        self.results = {}
        self.optimization_history = {}
        self.visualizer = DataVisualizer(
            output_dir=MODEL_RESULTS_DIR,
            auto_save=True,
            auto_show=False,
            dpi=PLOT_DPI,
            style=PLOT_STYLE,
            figure_size=FIGURE_SIZE,
        )
        
        # Set random seed để reproducibility
        np.random.seed(self.RANDOM_SEED)
        
        logger.info(f"✅ ModelTrainer khởi tạo thành công")
        logger.info(f"   Train: {self.X_train.shape}, Test: {self.X_test.shape}")
    
    @property
    def data_info(self) -> Dict[str, Any]:
        """Trả về thông tin dữ liệu."""
        return {
            'train_shape': self.X_train.shape,
            'test_shape': self.X_test.shape,
            'n_features': self.X_train.shape[1],
            'n_samples_train': self.X_train.shape[0],
            'n_samples_test': self.X_test.shape[0]
        }
    
    # ========== POLYNOMIAL REGRESSION ==========
    def _objective_polynomial(self, trial: optuna.Trial) -> float:
        """Objective function cho Polynomial Regression optimization."""
        degree = trial.suggest_int('degree', 2, 5)
        alpha = trial.suggest_float('alpha', 1e-3, 10, log=True)
        
        pipeline = Pipeline([
            ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
            ('scaler', StandardScaler()),
            ('regressor', Ridge(alpha=alpha))
        ])
        
        cv_scores = cross_val_score(
            pipeline,
            self.X_train,
            self.y_train,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        rmse = np.sqrt(-cv_scores.mean())
        return rmse
    
    def optimize_polynomial(self, n_trials: int = 10, timeout: int = 300) -> Dict:
        logger.info(f"\n{'='*70}")
        logger.info("🔍 Tối ưu POLYNOMIAL REGRESSION bằng Optuna")
        logger.info(f"{'='*70}")
        
        sampler = TPESampler(seed=self.RANDOM_SEED)
        pruner = MedianPruner()
        
        study = optuna.create_study(
            sampler=sampler,
            pruner=pruner,
            direction='minimize'
        )
        study.optimize(
            self._objective_polynomial,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True
        )
        
        best_params = study.best_params
        logger.info(f"✅ Best params: {best_params}")
        logger.info(f"   Best RMSE: {study.best_value:.6f}")
        
        self.optimization_history['polynomial'] = {
            'best_params': best_params,
            'best_value': study.best_value,
            'n_trials': len(study.trials)
        }
        return best_params
    
    def train_polynomial(self, degree: int = 3, alpha: float = 1.0,
                         feature_subset: Optional[List[str]] = None) -> None:
        """Huấn luyện Polynomial Regression với scaling và Ridge regularization."""
        logger.info(f"\n📊 Training POLYNOMIAL REGRESSION (degree={degree}, alpha={alpha})")
        if feature_subset:
            valid_features = [col for col in feature_subset if col in self.X_train.columns]
            missing = [col for col in feature_subset if col not in self.X_train.columns]
            if missing:
                logger.warning(f"⚠️  Các feature không tồn tại và sẽ bị bỏ qua: {missing}")
            if not valid_features:
                logger.warning("⚠️  Không còn feature hợp lệ sau khi lọc. Sử dụng toàn bộ features.")
                feature_subset = None
            else:
                feature_subset = valid_features
                logger.info(f"   Sử dụng {len(feature_subset)} feature có |corr| >= threshold")
                logger.info(f"   Features: {feature_subset}")
        
        base_X_train = self.X_train[feature_subset] if feature_subset else self.X_train
        base_X_test = self.X_test[feature_subset] if feature_subset else self.X_test
        
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_train_poly = poly.fit_transform(base_X_train)
        X_test_poly = poly.transform(base_X_test)
        
        logger.info(f"   Original features: {self.X_train.shape[1]}")
        logger.info(f"   Polynomial features: {X_train_poly.shape[1]}")
        
        poly_scaler = StandardScaler()
        X_train_poly_scaled = poly_scaler.fit_transform(X_train_poly)
        X_test_poly_scaled = poly_scaler.transform(X_test_poly)
        logger.info("   ✅ Polynomial features được scale (StandardScaler)")
        
        model = Ridge(alpha=alpha)
        model.fit(X_train_poly_scaled, self.y_train)
        
        y_pred_train = model.predict(X_train_poly_scaled)
        y_pred_test = model.predict(X_test_poly_scaled)
        
        train_rmse = np.sqrt(mean_squared_error(self.y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred_test))
        test_mae = mean_absolute_error(self.y_test, y_pred_test)
        test_r2 = r2_score(self.y_test, y_pred_test)
        
        self.models['polynomial'] = {
            'model': model,
            'poly': poly,
            'poly_scaler': poly_scaler,
            'feature_subset': feature_subset
        }
        self.X_train_transformed['polynomial'] = X_train_poly_scaled
        self.X_test_transformed['polynomial'] = X_test_poly_scaled
        
        self.results['polynomial'] = {
            'train_rmse': float(train_rmse),
            'test_rmse': float(test_rmse),
            'test_mae': float(test_mae),
            'test_r2': float(test_r2),
            'hyperparams': {
                'degree': degree,
                'alpha': alpha,
                'feature_subset': feature_subset if feature_subset else 'all'
            }
        }
        
        logger.info(f"   Train RMSE: {train_rmse:.6f}")
        logger.info(f"   Test RMSE: {test_rmse:.6f}")
        logger.info(f"   Test MAE: {test_mae:.6f}")
        logger.info(f"   Test R²: {test_r2:.6f}")
    
    # ========== RANDOM FOREST REGRESSION ==========
    def _objective_rf(self, trial: optuna.Trial) -> float:
        """Objective function cho Random Forest optimization."""
        n_estimators = trial.suggest_int('n_estimators', 50, 300)
        max_depth = trial.suggest_int('max_depth', 5, 20)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 5)
        
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=self.RANDOM_SEED,
            n_jobs=-1
        )
        
        # Dùng cross-validation TRÊN TRAIN SET (không dùng test set!)
        from sklearn.model_selection import cross_val_score
        cv_scores = cross_val_score(
            model, self.X_train, self.y_train,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        rmse = np.sqrt(-cv_scores.mean())
        
        return rmse
    
    def optimize_rf(self, n_trials: int = 20, timeout: int = 600) -> Dict:
        """
        Tối ưu hyperparameters cho Random Forest.
        
        Args:
            n_trials: Số lần thử
            timeout: Timeout tính bằng giây
            
        Returns:
            Dictionary chứa best params
        """
        logger.info(f"\n{'='*70}")
        logger.info("🔍 Tối ưu RANDOM FOREST bằng Optuna")
        logger.info(f"{'='*70}")
        
        sampler = TPESampler(seed=self.RANDOM_SEED)
        pruner = MedianPruner()
        
        study = optuna.create_study(
            sampler=sampler,
            pruner=pruner,
            direction='minimize'
        )
        
        study.optimize(
            self._objective_rf,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True
        )
        
        best_params = study.best_params
        logger.info(f"✅ Best params: {best_params}")
        logger.info(f"   Best RMSE: {study.best_value:.6f}")
        
        self.optimization_history['random_forest'] = {
            'best_params': best_params,
            'best_value': study.best_value,
            'n_trials': len(study.trials)
        }
        
        return best_params
    
    def train_rf(self, 
                 n_estimators: int = 100,
                 max_depth: int = 10,
                 min_samples_split: int = 5,
                 min_samples_leaf: int = 2) -> None:
        """Huấn luyện Random Forest."""
        logger.info(f"\n📊 Training RANDOM FOREST")
        logger.info(f"   n_estimators={n_estimators}, max_depth={max_depth}")
        
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=self.RANDOM_SEED,
            n_jobs=-1
        )
        
        # Tree-based model KHÔNG CẦN scale
        model.fit(self.X_train, self.y_train)
        
        # Đánh giá
        y_pred_train = model.predict(self.X_train)
        y_pred_test = model.predict(self.X_test)
        
        train_rmse = np.sqrt(mean_squared_error(self.y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred_test))
        test_mae = mean_absolute_error(self.y_test, y_pred_test)
        test_r2 = r2_score(self.y_test, y_pred_test)
        
        # Lưu model
        self.models['random_forest'] = {'model': model}
        self.results['random_forest'] = {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'test_r2': test_r2,
            'hyperparams': {
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf
            }
        }
        
        logger.info(f"   Train RMSE: {train_rmse:.6f}")
        logger.info(f"   Test RMSE: {test_rmse:.6f}")
        logger.info(f"   Test MAE: {test_mae:.6f}")
        logger.info(f"   Test R²: {test_r2:.6f}")
    
    # ========== XGBOOST REGRESSION ==========
    def _objective_xgb(self, trial: optuna.Trial) -> float:
        """Objective function cho XGBoost optimization."""
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 5),
            'lambda': trial.suggest_float('lambda', 0.0, 1.0),
            'alpha': trial.suggest_float('alpha', 0.0, 1.0),
            'random_state': self.RANDOM_SEED
        }
        
        model = xgb.XGBRegressor(**params)
        
        # Dùng cross-validation TRÊN TRAIN SET
        cv_scores = cross_val_score(
            model, self.X_train, self.y_train,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        rmse = np.sqrt(-cv_scores.mean())
        
        return rmse
    
    def optimize_xgb(self, n_trials: int = 30, timeout: int = 900) -> Dict:
        """
        Tối ưu hyperparameters cho XGBoost.
        
        Args:
            n_trials: Số lần thử
            timeout: Timeout tính bằng giây
            
        Returns:
            Dictionary chứa best params
        """
        logger.info(f"\n{'='*70}")
        logger.info("🔍 Tối ưu XGBOOST bằng Optuna")
        logger.info(f"{'='*70}")
        
        sampler = TPESampler(seed=self.RANDOM_SEED)
        pruner = MedianPruner()
        
        study = optuna.create_study(
            sampler=sampler,
            pruner=pruner,
            direction='minimize'
        )
        
        study.optimize(
            self._objective_xgb,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True
        )
        
        best_params = study.best_params
        logger.info(f"✅ Best params: {best_params}")
        logger.info(f"   Best RMSE: {study.best_value:.6f}")
        
        self.optimization_history['xgboost'] = {
            'best_params': best_params,
            'best_value': study.best_value,
            'n_trials': len(study.trials)
        }
        
        return best_params
    
    def train_xgb(self, **xgb_params) -> None:
        """
        Huấn luyện XGBoost.
        
        Args:
            **xgb_params: XGBoost hyperparameters
        """
        logger.info(f"\n📊 Training XGBOOST")
        
        # Default params
        default_params = {
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': self.RANDOM_SEED
        }
        
        # Update với params được truyền vào
        default_params.update(xgb_params)
        
        model = xgb.XGBRegressor(**default_params)
        
        # Tree-based không cần scale
        model.fit(self.X_train, self.y_train, verbose=False)
        
        # Đánh giá
        y_pred_train = model.predict(self.X_train)
        y_pred_test = model.predict(self.X_test)
        
        train_rmse = np.sqrt(mean_squared_error(self.y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred_test))
        test_mae = mean_absolute_error(self.y_test, y_pred_test)
        test_r2 = r2_score(self.y_test, y_pred_test)
        
        # Lưu model
        self.models['xgboost'] = {'model': model}
        self.results['xgboost'] = {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'test_r2': test_r2,
            'hyperparams': default_params
        }
        
        logger.info(f"   Train RMSE: {train_rmse:.6f}")
        logger.info(f"   Test RMSE: {test_rmse:.6f}")
        logger.info(f"   Test MAE: {test_mae:.6f}")
        logger.info(f"   Test R²: {test_r2:.6f}")
    
    # ========== SAVE & LOAD MODELS ==========
    def save_model(self, model_name: str, format: str = 'joblib') -> str:
        """
        Lưu mô hình vào file.
        
        Args:
            model_name: Tên mô hình ('polynomial', 'random_forest', 'xgboost')
            format: Định dạng ('joblib' hoặc 'pickle')
            
        Returns:
            Đường dẫn file
        """
        if model_name not in self.models:
            logger.error(f"❌ Không tìm thấy mô hình: {model_name}")
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.output_dir / f"{model_name}_{timestamp}.{format}"
        
        model_data = self.models[model_name]
        
        if format == 'joblib':
            joblib.dump(model_data, filename)
        elif format == 'pickle':
            with open(filename, 'wb') as f:
                pickle.dump(model_data, f)
        
        logger.info(f"✅ Đã lưu mô hình: {filename}")
        return str(filename)
    
    def load_model(self, filepath: str, model_name: str) -> None:
        """
        Tải mô hình từ file.
        
        Args:
            filepath: Đường dẫn file
            model_name: Tên mô hình để lưu
        """
        filepath = Path(filepath)
        
        if filepath.suffix == '.joblib':
            model_data = joblib.load(filepath)
        else:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
        
        self.models[model_name] = model_data
        logger.info(f"✅ Đã tải mô hình: {filepath}")
    
    # ========== EVALUATION & COMPARISON ==========
    def get_best_model(self) -> Tuple[str, Dict]:
        """
        Lấy mô hình tốt nhất dựa trên test R².
        
        Returns:
            (model_name, results)
        """
        best_r2 = -np.inf
        best_name = None
        
        for name, result in self.results.items():
            if result['test_r2'] > best_r2:
                best_r2 = result['test_r2']
                best_name = name
        
        self.best_model_name = best_name
        if best_name:
            self.best_model = self.models[best_name]['model']
        
        return best_name, self.results[best_name] if best_name else None
    
    def save_results(self, filename: str = 'model_results.json') -> None:
        """Lưu kết quả đánh giá ra file JSON."""
        filepath = MODEL_RESULTS_DIR / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert np types sang Python types
        results_serializable = {}
        for model_name, result in self.results.items():
            results_serializable[model_name] = {
                'train_rmse': float(result['train_rmse']),
                'test_rmse': float(result['test_rmse']),
                'test_mae': float(result['test_mae']),
                'test_r2': float(result['test_r2']),
                'hyperparams': result['hyperparams']
            }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results_serializable, f, indent=4, ensure_ascii=False)
        
        logger.info(f"✅ Đã lưu kết quả: {filepath}")
    
    def plot_comparison(self, metric: str = 'test_r2', save: bool = True) -> None:
        """
        Vẽ biểu đồ so sánh các mô hình.
        
        Args:
            metric: Metric để so sánh ('test_r2', 'test_rmse', 'test_mae')
            save: Có lưu biểu đồ không
        """
        if not self.results:
            logger.warning("❌ Chưa có kết quả để vẽ biểu đồ")
            return
        
        metric_values = {model: self.results[model][metric] for model in self.results}
        if not metric_values:
            logger.warning("❌ Không có kết quả để vẽ biểu đồ")
            return

        save_path = None
        if save:
            save_path = MODEL_RESULTS_DIR / f'comparison_{metric}.png'
        self.visualizer.plot_model_comparison(metric_values, metric, save_path=save_path, show=not save)
    
    def plot_predictions(self, model_name: str, save: bool = True) -> None:
        """Vẽ biểu đồ actual vs predicted."""
        if model_name not in self.models:
            logger.error(f"❌ Không tìm thấy mô hình: {model_name}")
            return
        
        model_obj = self.models[model_name]['model']
        
        # Xử lý đặc biệt cho Polynomial (đã được scaled)
        if model_name == 'polynomial':
            X_test_pred = self.X_test_transformed['polynomial']
        else:
            # RF và XGBoost dùng raw data
            X_test_pred = self.X_test
        
        y_pred = model_obj.predict(X_test_pred)
        
        save_path = None
        if save:
            save_path = MODEL_RESULTS_DIR / f'predictions_{model_name}.png'
        self.visualizer.plot_regression_diagnostics(
            y_true=self.y_test,
            y_pred=y_pred,
            model_name=model_name,
            save_path=save_path,
            show=not save,
        )
    
    def plot_all_predictions(self, save: bool = True) -> None:
        """Vẽ biểu đồ predictions cho tất cả các mô hình."""
        logger.info(f"\n{'='*70}")
        logger.info("📈 VẼ BIỂU ĐỒ PREDICTIONS CHO TẤT CẢ MÔ HÌNH")
        logger.info(f"{'='*70}\n")
        
        for model_name in self.models.keys():
            self.plot_predictions(model_name, save=save)
    
    def summary(self) -> None:
        """In ra tóm tắt kết quả các mô hình."""
        logger.info(f"\n{'='*70}")
        logger.info("📊 TÓM TẮT KẾT QUẢ TRAINING")
        logger.info(f"{'='*70}\n")
        
        summary_data = []
        for model_name, result in self.results.items():
            summary_data.append({
                'Model': model_name.upper(),
                'Train RMSE': f"{result['train_rmse']:.6f}",
                'Test RMSE': f"{result['test_rmse']:.6f}",
                'Test MAE': f"{result['test_mae']:.6f}",
                'Test R²': f"{result['test_r2']:.6f}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
        
        best_name, best_result = self.get_best_model()
        if best_name:
            logger.info(f"\n✨ MÔ HÌNH TỐT NHẤT: {best_name.upper()}")
            logger.info(f"   Test R²: {best_result['test_r2']:.6f}")
        
        logger.info(f"\n{'='*70}\n")
    
    # ========== DATA PREPARATION ==========
    @staticmethod
    def prepare_data(df: pd.DataFrame, 
                     target_col: str = 'Trip_Price',
                     test_size: float = 0.2,
                     random_state: int = 42,
                     scale: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Chia dữ liệu (KHÔNG scale - scale được thực hiện ở preprocessing).
        
        Args:
            df: DataFrame chứa dữ liệu
            target_col: Tên cột target
            test_size: Tỷ lệ test set
            random_state: Random seed
            scale: DEPRECATED - Không nên scale ở đây, scale ở preprocessing
            
        Returns:
            (X_train, X_test, y_train, y_test)
        """
        logger.info(f"\n{'='*70}")
        logger.info("🔄 CHUẨN BỊ DỮ LIỆU CHO TRAINING")
        logger.info(f"{'='*70}")
        
        # Tách Features và Target
        X = df.drop(target_col, axis=1)
        y = df[target_col]
        
        logger.info(f"Total samples: {len(df)}, Features: {X.shape[1]}")
        
        # Chia Train/Test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        logger.info(f"Train set: {X_train.shape[0]}, Test set: {X_test.shape[0]}")
        
        # Scale dữ liệu nếu cần (KHÔNG khuyến nghị)
        if scale:
            logger.warning("⚠️  scale=True không khuyến nghị - Nên scale ở preprocessing!")
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Chuyển thành DataFrame
            X_train = pd.DataFrame(X_train_scaled, columns=X.columns)
            X_test = pd.DataFrame(X_test_scaled, columns=X.columns)
            
            logger.info(f"✅ Dữ liệu đã được chuẩn hóa (StandardScaler)")
        
        logger.info(f"   X_train shape: {X_train.shape}")
        logger.info(f"   X_test shape: {X_test.shape}\n")
        
        return X_train, X_test, y_train, y_test
    
    # ========== SAVE ALL MODELS ==========
    def save_all_models(self, format: str = 'joblib') -> None:
        """
        Lưu tất cả các mô hình đã train.
        
        Args:
            format: Định dạng ('joblib' hoặc 'pickle')
        """
        logger.info(f"\n{'='*70}")
        logger.info("💾 LƯU TẤT CẢ CÁC MÔ HÌNH")
        logger.info(f"{'='*70}")
        
        for model_name in self.models.keys():
            self.save_model(model_name, format=format)
        
        logger.info(f"\n✅ Hoàn tất lưu {len(self.models)} mô hình!\n")
    
    def predict(self, X: pd.DataFrame, model_name: Optional[str] = None) -> np.ndarray:
        """
        Dự đoán với mô hình đã train.
        
        Args:
            X: Features cần dự đoán
            model_name: Tên mô hình (None = dùng best model)
            
        Returns:
            Array predictions
        """
        if model_name is None:
            if self.best_model_name is None:
                self.get_best_model()
            model_name = self.best_model_name
        
        if model_name not in self.models:
            raise ValueError(f"Mô hình {model_name} chưa được train")
        
        model_obj = self.models[model_name]['model']
        
        # Xử lý cho Polynomial (cần transform + scale)
        if model_name == 'polynomial':
            poly = self.models[model_name]['poly']
            poly_scaler = self.models[model_name]['poly_scaler']
            feature_subset = self.models[model_name].get('feature_subset')
            X_input = X[feature_subset] if feature_subset else X
            X_poly = poly.transform(X_input)
            X_poly = poly_scaler.transform(X_poly)
            return model_obj.predict(X_poly)
        
        # Các model khác dùng X trực tiếp (đã scaled từ preprocessing)
        return model_obj.predict(X)
    
    # ========== FEATURE IMPORTANCE ==========
    def get_feature_importance(self, model_name: str, top_n: int = 10) -> pd.DataFrame:
        """
        Lấy feature importance của mô hình.
        
        Args:
            model_name: Tên mô hình ('random_forest', 'xgboost')
            top_n: Số lượng features quan trọng nhất
            
        Returns:
            DataFrame chứa feature importance
        """
        if model_name not in self.models:
            logger.error(f"❌ Mô hình {model_name} chưa được train")
            return None
        
        if model_name == 'polynomial':
            logger.warning("⚠️  Polynomial Regression không hỗ trợ feature importance")
            return None
        
        model_obj = self.models[model_name]['model']
        
        # Lấy feature importance
        if hasattr(model_obj, 'feature_importances_'):
            importances = model_obj.feature_importances_
            feature_names = self.X_train.columns
            
            # Tạo DataFrame
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False).head(top_n)
            
            return importance_df
        else:
            logger.warning(f"⚠️  Mô hình {model_name} không hỗ trợ feature importance")
            return None
    
    def plot_feature_importance(self, model_name: str, top_n: int = 15, save: bool = True) -> None:
        """
        Vẽ biểu đồ feature importance.
        
        Args:
            model_name: Tên mô hình
            top_n: Số lượng features hiển thị
            save: Có lưu biểu đồ không
        """
        importance_df = self.get_feature_importance(model_name, top_n=top_n)
        
        if importance_df is None:
            return
        
        logger.info(f"\n📊 Feature Importance - {model_name.upper()}")
        logger.info(f"{'='*70}")
        print(importance_df.to_string(index=False))
        
        # Vẽ biểu đồ
        save_path = None
        if save:
            save_path = MODEL_RESULTS_DIR / f'feature_importance_{model_name}.png'
        self.visualizer.plot_feature_importance(
            importance_df=importance_df,
            model_name=model_name,
            save_path=save_path,
            top_n=top_n,
            show=not save,
        )
    
    def plot_all_feature_importance(self, top_n: int = 15, save: bool = True) -> None:
        """Vẽ feature importance cho tất cả mô hình hỗ trợ (bỏ qua Polynomial)."""
        logger.info(f"\n{'='*70}")
        logger.info("📊 VẼ FEATURE IMPORTANCE CHO TẤT CẢ MÔ HÌNH")
        logger.info(f"{'='*70}\n")
        
        # Lọc các models hỗ trợ feature importance
        supported_models = [m for m in self.models.keys() if m != 'polynomial']
        
        if not supported_models:
            logger.warning("⚠️  Không có mô hình nào hỗ trợ feature importance (chỉ có Polynomial)")
            return
        
        for model_name in supported_models:
            self.plot_feature_importance(model_name, top_n=top_n, save=save)
    
    def compare_feature_importance(self, top_n: int = 10, save: bool = True) -> None:
        """
        So sánh feature importance giữa các mô hình (bỏ qua Polynomial).
        
        Args:
            top_n: Số features hiển thị
            save: Có lưu biểu đồ không
        """
        logger.info(f"\n{'='*70}")
        logger.info("📊 SO SÁNH FEATURE IMPORTANCE GIỮA CÁC MÔ HÌNH")
        logger.info(f"{'='*70}\n")
        
        # Lấy feature importance từ các mô hình (chỉ RF và XGBoost)
        importances = {}
        for model_name in ['random_forest', 'xgboost']:
            if model_name in self.models:
                imp_df = self.get_feature_importance(model_name, top_n=top_n)
                if imp_df is not None:
                    importances[model_name] = imp_df
        
        if len(importances) == 0:
            logger.warning("⚠️  Không có mô hình nào hỗ trợ feature importance để so sánh")
            return
        
        if len(importances) == 1:
            logger.warning(f"⚠️  Chỉ có 1 mô hình ({list(importances.keys())[0]}), cần ít nhất 2 để so sánh")
            logger.info(f"💡 Sử dụng plot_feature_importance('{list(importances.keys())[0]}') để vẽ riêng")
            return
        
        # Vẽ biểu đồ so sánh
        save_path = None
        if save:
            save_path = MODEL_RESULTS_DIR / 'feature_importance_comparison.png'
        self.visualizer.plot_feature_importance_comparison(
            importances=importances,
            save_path=save_path,
            top_n=top_n,
            show=not save,
        )
