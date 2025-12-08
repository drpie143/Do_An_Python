"""
Model Trainer - Orchestrator cho việc huấn luyện nhiều mô hình.

Đây là lớp chính để sử dụng, cung cấp interface giống như file gốc nhưng
delegate công việc sang các trainer cụ thể trong model_registry.

Hỗ trợ:
- train_all(): Huấn luyện tất cả các mô hình
- Tối ưu hyperparameters với Optuna
- Save/Load mô hình
- Visualization và comparison
"""

import json
import pickle
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.modeling.base_trainer import log_section, log_step, log_metrics, _divider
from src.modeling.model_registry import (
    RandomForestTrainer, ExtraTreesTrainer, XGBoostTrainer,
    get_trainer, TRAINER_REGISTRY
)
from src.visualization import DataVisualizer
from config import MODEL_RESULTS_DIR, MODELS_DIR, PLOT_DPI, PLOT_STYLE, FIGURE_SIZE


logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Orchestrator cho việc huấn luyện và quản lý nhiều mô hình học máy.
    
    Hỗ trợ:
    - 3 mô hình: Random Forest, Extra Trees, XGBoost
    - Tối ưu hyperparameters bằng Optuna
    - Logging quá trình huấn luyện
    - Lưu/tải mô hình
    - Đánh giá và so sánh kết quả
    - Visualization
    
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
            X_train, X_test: Features
            y_train, y_test: Target
            output_dir: Thư mục lưu kết quả
        """
        self.X_train = X_train.copy()
        self.X_test = X_test.copy()
        self.y_train = y_train.copy()
        self.y_test = y_test.copy()
        
        self.output_dir = Path(output_dir) if output_dir else MODELS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Khởi tạo các trainers
        self._trainers: Dict[str, Any] = {}
        
        # Kết quả tổng hợp
        self.models = {}
        self.X_train_transformed = {}
        self.X_test_transformed = {}
        self.best_model = None
        self.best_model_name = None
        self.results = {}
        self.optimization_history = {}
        
        # Visualizer
        self.visualizer = DataVisualizer(
            output_dir=MODEL_RESULTS_DIR,
            auto_save=True,
            auto_show=False,
            dpi=PLOT_DPI,
            style=PLOT_STYLE,
            figure_size=FIGURE_SIZE,
        )
        
        np.random.seed(self.RANDOM_SEED)
        
        log_section("MODELTRAINER KHỞI TẠO", icon="⚙️")
        log_step("Khởi tạo thành công", icon="✅")
        log_step(f"Train: {self.X_train.shape}, Test: {self.X_test.shape}", icon="📊")
    
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
    
    def _get_trainer(self, name: str):
        """Lấy hoặc tạo trainer theo tên."""
        if name not in self._trainers:
            self._trainers[name] = get_trainer(
                name, self.X_train, self.X_test, self.y_train, self.y_test
            )
        return self._trainers[name]
    
    def _sync_trainer_results(self, name: str, trainer) -> None:
        """Đồng bộ kết quả từ trainer về ModelTrainer."""
        self.models[name] = trainer.get_model_data()
        self.results[name] = trainer.get_result()
        self.optimization_history[name] = trainer.optimization_history
    
    # ========== RANDOM FOREST ==========
    def optimize_rf(self, n_trials: int = 20, timeout: int = 600) -> Dict:
        """Tối ưu hyperparameters cho Random Forest."""
        trainer = self._get_trainer('random_forest')
        return trainer.optimize(n_trials=n_trials, timeout=timeout)
    
    def train_rf(self, n_estimators: int = 100, max_depth: int = 10,
                 min_samples_split: int = 5, min_samples_leaf: int = 2) -> None:
        """Huấn luyện Random Forest."""
        trainer = self._get_trainer('random_forest')
        trainer.train(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf
        )
        self._sync_trainer_results('random_forest', trainer)
    
    # ========== EXTRA TREES ==========
    def optimize_extra_trees(self, n_trials: int = 20, timeout: int = 600) -> Dict:
        """Tối ưu hyperparameters cho Extra Trees."""
        trainer = self._get_trainer('extra_trees')
        return trainer.optimize(n_trials=n_trials, timeout=timeout)
    
    def train_extra_trees(self, n_estimators: int = 200, max_depth: int = 12,
                          min_samples_split: int = 2, min_samples_leaf: int = 1) -> None:
        """Huấn luyện Extra Trees."""
        trainer = self._get_trainer('extra_trees')
        trainer.train(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf
        )
        self._sync_trainer_results('extra_trees', trainer)
    
    # ========== XGBOOST ==========
    def optimize_xgb(self, n_trials: int = 30, timeout: int = 900) -> Dict:
        """Tối ưu hyperparameters cho XGBoost."""
        trainer = self._get_trainer('xgboost')
        return trainer.optimize(n_trials=n_trials, timeout=timeout)
    
    def train_xgb(self, **xgb_params) -> None:
        """Huấn luyện XGBoost."""
        trainer = self._get_trainer('xgboost')
        trainer.train(**xgb_params)
        self._sync_trainer_results('xgboost', trainer)
    
    # ========== TRAIN ALL ==========
    def train_all(
        self,
        optimize: bool = False,
        hyperparams: Optional[Dict[str, Dict]] = None,
        optuna_config: Optional[Dict[str, Dict]] = None,
    ) -> "ModelTrainer":
        """
        Huấn luyện tất cả các mô hình.
        
        Args:
            optimize: Có tối ưu hyperparameters với Optuna không
            hyperparams: Dict chứa hyperparameters mặc định cho từng model
            optuna_config: Dict chứa cấu hình Optuna (n_trials, timeout)
            
        Returns:
            self để có thể chain methods
        """
        log_section("HUẤN LUYỆN TẤT CẢ MÔ HÌNH", icon="🤖")
        
        # Default hyperparams
        default_hyperparams = {
            'random_forest': {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 5, 'min_samples_leaf': 2},
            'extra_trees': {'n_estimators': 200, 'max_depth': 12, 'min_samples_split': 2, 'min_samples_leaf': 1},
            'xgboost': {'max_depth': 4, 'learning_rate': 0.05, 'n_estimators': 150, 'subsample': 0.7},
        }
        if hyperparams:
            for key in hyperparams:
                default_hyperparams[key] = hyperparams[key]
        
        # Default Optuna config
        default_optuna = {
            'n_trials': {'random_forest': 20, 'extra_trees': 20, 'xgboost': 30},
            'timeout': {'random_forest': 600, 'extra_trees': 600, 'xgboost': 900},
        }
        if optuna_config:
            for key in optuna_config:
                default_optuna[key] = optuna_config[key]
        
        # 1. Random Forest
        if optimize:
            log_step("Tối ưu Random Forest", icon="🔍")
            best_rf = self.optimize_rf(
                n_trials=default_optuna['n_trials']['random_forest'],
                timeout=default_optuna['timeout']['random_forest']
            )
            self.train_rf(**best_rf)
        else:
            self.train_rf(**default_hyperparams['random_forest'])
        
        # 2. Extra Trees
        if optimize:
            log_step("Tối ưu Extra Trees", icon="🔍")
            best_et = self.optimize_extra_trees(
                n_trials=default_optuna['n_trials']['extra_trees'],
                timeout=default_optuna['timeout']['extra_trees']
            )
            self.train_extra_trees(**best_et)
        else:
            self.train_extra_trees(**default_hyperparams['extra_trees'])
        
        # 3. XGBoost
        if optimize:
            log_step("Tối ưu XGBoost", icon="🔍")
            best_xgb = self.optimize_xgb(
                n_trials=default_optuna['n_trials']['xgboost'],
                timeout=default_optuna['timeout']['xgboost']
            )
            self.train_xgb(**best_xgb)
        else:
            self.train_xgb(**default_hyperparams['xgboost'])
        
        log_step("Đã hoàn tất huấn luyện tất cả mô hình!", icon="✅")
        return self
    
    # ========== SAVE & LOAD ==========
    def save_model(self, model_name: str, format: str = 'joblib', 
                   use_timestamp: bool = False) -> str:
        """
        Lưu mô hình vào file.
        
        Args:
            model_name: Tên mô hình
            format: Định dạng ('joblib' hoặc 'pickle')
            use_timestamp: Nếu True thì thêm timestamp vào tên file,
                          Nếu False thì ghi đè file cũ (mặc định)
        """
        if model_name not in self.models:
            logger.error(f"❌ Không tìm thấy mô hình: {model_name}")
            return None
        
        # Chỉ thêm timestamp nếu được yêu cầu
        if use_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = self.output_dir / f"{model_name}_{timestamp}.{format}"
        else:
            filename = self.output_dir / f"{model_name}.{format}"
        
        model_data = self.models[model_name]
        
        if format == 'joblib':
            joblib.dump(model_data, filename)
        elif format == 'pickle':
            with open(filename, 'wb') as f:
                pickle.dump(model_data, f)
        
        log_step(f"Đã lưu mô hình: {filename}", icon="💾")
        return str(filename)
    
    def load_model(self, filepath: str, model_name: str) -> None:
        """Tải mô hình từ file."""
        filepath = Path(filepath)
        
        if filepath.suffix == '.joblib':
            model_data = joblib.load(filepath)
        else:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
        
        self.models[model_name] = model_data
        log_step(f"Đã tải mô hình: {filepath}", icon="📂")
    
    def save_all_models(self, format: str = 'joblib', 
                        use_timestamp: bool = False) -> Dict[str, str]:
        """
        Lưu tất cả các mô hình đã train.
        
        Args:
            format: Định dạng ('joblib' hoặc 'pickle')
            use_timestamp: Nếu True thì thêm timestamp, False thì ghi đè (mặc định)
        """
        log_section("LƯU TẤT CẢ CÁC MÔ HÌNH", icon="💾")
        
        saved_paths: Dict[str, str] = {}
        for model_name in self.models.keys():
            path = self.save_model(model_name, format=format, use_timestamp=use_timestamp)
            if path:
                saved_paths[model_name] = path
        
        log_step(f"Hoàn tất lưu {len(saved_paths)} mô hình!", icon="✅")
        return saved_paths
    
    # ========== EVALUATION ==========
    def get_best_model(self) -> Tuple[str, Dict]:
        """Lấy mô hình tốt nhất dựa trên test R²."""
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
        
        log_step(f"Đã lưu kết quả: {filepath}", icon="💾")
    
    def summary(self) -> None:
        """In ra tóm tắt kết quả các mô hình."""
        log_section("TÓM TẮT KẾT QUẢ TRAINING", icon="📊")
        
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
            log_section("MÔ HÌNH TỐT NHẤT", icon="✨")
            log_step(f"Model: {best_name.upper()}", icon="🏆")
            log_metrics({
                "Test R²": best_result['test_r2'],
                "Test RMSE": best_result['test_rmse'],
                "Test MAE": best_result['test_mae']
            })
        
        logger.info("%s\n", _divider())
    
    # ========== PREDICTIONS ==========
    def predict(self, X: pd.DataFrame, model_name: Optional[str] = None) -> np.ndarray:
        """Dự đoán với mô hình đã train."""
        if model_name is None:
            if self.best_model_name is None:
                self.get_best_model()
            model_name = self.best_model_name
        
        if model_name not in self.models:
            raise ValueError(f"Mô hình {model_name} chưa được train")
        
        model_obj = self.models[model_name]['model']
        return model_obj.predict(X)
    
    # ========== VISUALIZATION ==========
    def plot_comparison(self, metric: str = 'test_r2', save: bool = True) -> None:
        """Vẽ biểu đồ so sánh các mô hình."""
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
        y_pred = model_obj.predict(self.X_test)
        
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
        """Vẽ biểu đồ predictions cho tất cả các mô hình (riêng lẻ - deprecated)."""
        log_section("VẼ BIỂU ĐỒ PREDICTIONS CHO TẤT CẢ MÔ HÌNH", icon="📈")
        
        for model_name in self.models.keys():
            self.plot_predictions(model_name, save=save)
    
    def plot_combined_predictions(self, save: bool = True) -> None:
        """Vẽ tất cả predictions trong 1 figure để so sánh."""
        log_section("VẼ BIỂU ĐỒ PREDICTIONS TỔNG HỢP", icon="📈")
        
        predictions = {}
        for model_name in self.models.keys():
            model_obj = self.models[model_name]['model']
            y_pred = model_obj.predict(self.X_test)
            predictions[model_name] = (self.y_test.values, y_pred)
        
        save_path = None
        if save:
            save_path = MODEL_RESULTS_DIR / 'predictions_combined.png'
        
        self.visualizer.plot_combined_predictions(
            predictions=predictions,
            save_path=save_path,
            show=not save,
        )
    
    def plot_metrics_summary(self, save: bool = True) -> None:
        """Vẽ biểu đồ tổng hợp tất cả metrics."""
        log_section("VẼ BIỂU ĐỒ METRICS TỔNG HỢP", icon="📊")
        
        if not self.results:
            logger.warning("❌ Chưa có kết quả để vẽ biểu đồ")
            return
        
        save_path = None
        if save:
            save_path = MODEL_RESULTS_DIR / 'metrics_summary.png'
        
        self.visualizer.plot_metrics_summary(
            results=self.results,
            save_path=save_path,
            show=not save,
        )
    
    # ========== FEATURE IMPORTANCE ==========
    def get_feature_importance(self, model_name: str, top_n: int = 10) -> pd.DataFrame:
        """Lấy feature importance của mô hình."""
        if model_name not in self.models:
            logger.error(f"❌ Mô hình {model_name} chưa được train")
            return None
        
        model_obj = self.models[model_name]['model']
        
        if hasattr(model_obj, 'feature_importances_'):
            importances = model_obj.feature_importances_
            feature_names = self.X_train.columns
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False).head(top_n)
            
            return importance_df
        else:
            logger.warning(f"⚠️  Mô hình {model_name} không hỗ trợ feature importance")
            return None
    
    def plot_feature_importance(self, model_name: str, top_n: int = 15, save: bool = True) -> None:
        """Vẽ biểu đồ feature importance."""
        importance_df = self.get_feature_importance(model_name, top_n=top_n)
        
        if importance_df is None:
            return
        
        logger.info(f"\n📊 Feature Importance - {model_name.upper()}")
        logger.info(f"{'='*70}")
        print(importance_df.to_string(index=False))
        
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
        """Vẽ feature importance cho tất cả mô hình hỗ trợ."""
        log_section("VẼ FEATURE IMPORTANCE CHO TẤT CẢ MÔ HÌNH", icon="📊")
        
        supported_models = list(self.models.keys())
        
        if not supported_models:
            logger.warning("⚠️  Không có mô hình nào hỗ trợ feature importance")
            return
        
        for model_name in supported_models:
            self.plot_feature_importance(model_name, top_n=top_n, save=save)
    
    def compare_feature_importance(self, top_n: int = 10, save: bool = True) -> None:
        """So sánh feature importance giữa các mô hình."""
        log_section("SO SÁNH FEATURE IMPORTANCE GIỮA CÁC MÔ HÌNH", icon="📊")
        
        importances = {}
        for model_name in ['random_forest', 'extra_trees', 'xgboost']:
            if model_name in self.models:
                imp_df = self.get_feature_importance(model_name, top_n=top_n)
                if imp_df is not None:
                    importances[model_name] = imp_df
        
        if len(importances) == 0:
            logger.warning("⚠️  Không có mô hình nào hỗ trợ feature importance để so sánh")
            return
        
        if len(importances) == 1:
            logger.warning(f"⚠️  Chỉ có 1 mô hình ({list(importances.keys())[0]}), cần ít nhất 2 để so sánh")
            return
        
        save_path = None
        if save:
            save_path = MODEL_RESULTS_DIR / 'feature_importance_comparison.png'
        self.visualizer.plot_feature_importance_comparison(
            importances=importances,
            save_path=save_path,
            top_n=top_n,
            show=not save,
        )
    
    # ========== DATA PREPARATION (Static) ==========
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
            scale: DEPRECATED - Không nên scale ở đây
            
        Returns:
            (X_train, X_test, y_train, y_test)
        """
        log_section("CHUẨN BỊ DỮ LIỆU CHO TRAINING", icon="🔄")
        
        X = df.drop(target_col, axis=1)
        y = df[target_col]
        
        log_step(f"Total samples: {len(df)}, Features: {X.shape[1]}", icon="📦")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        log_step(f"Train set: {X_train.shape[0]}, Test set: {X_test.shape[0]}", icon="🔀")
        
        if scale:
            logger.warning("⚠️  scale=True không khuyến nghị - Nên scale ở preprocessing!")
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            X_train = pd.DataFrame(X_train_scaled, columns=X.columns)
            X_test = pd.DataFrame(X_test_scaled, columns=X.columns)
            
            log_step("Dữ liệu đã được chuẩn hóa (StandardScaler)", icon="✅")
        
        log_step(f"X_train shape: {X_train.shape}", icon="📊")
        log_step(f"X_test shape: {X_test.shape}", icon="📊")
        
        return X_train, X_test, y_train, y_test
