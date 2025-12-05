"""
Model Registry - Chứa các Trainer cụ thể cho từng loại mô hình.

Bao gồm:
- PolynomialTrainer: Polynomial Regression với Ridge
- RandomForestTrainer: Random Forest Regressor
- ExtraTreesTrainer: Extra Trees Regressor
- XGBoostTrainer: XGBoost Regressor
"""

import time
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import cross_val_score, train_test_split

import xgboost as xgb
import optuna

from src.modeling.base_trainer import (
    BaseTrainer, log_section, log_step, log_metrics, logger
)


# ========== POLYNOMIAL REGRESSION TRAINER ==========
class PolynomialTrainer(BaseTrainer):
    """Trainer cho Polynomial Regression với Ridge regularization."""
    
    @property
    def model_name(self) -> str:
        return 'polynomial'
    
    def _objective(self, trial: optuna.Trial) -> float:
        """Objective function cho Polynomial Regression optimization."""
        degree = trial.suggest_int('degree', 2, 5)
        alpha = trial.suggest_float('alpha', 1e-3, 10, log=True)
        
        pipeline = Pipeline([
            ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
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
    
    def optimize(self, n_trials: int = 10, timeout: int = 300) -> Dict:
        """Tối ưu hyperparameters cho Polynomial Regression."""
        log_section("TỐI ƯU POLYNOMIAL REGRESSION", icon="🔍")
        
        study = self._create_optuna_study()
        study.optimize(
            self._objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True
        )
        
        best_params = study.best_params
        log_step(f"Best params: {best_params}", icon="✅")
        log_metrics({"Best RMSE": study.best_value})
        
        self.optimization_history = {
            'best_params': best_params,
            'best_value': study.best_value,
            'n_trials': len(study.trials)
        }
        return best_params
    
    def train(self, degree: int = 3, alpha: float = 1.0,
              feature_subset: Optional[List[str]] = None) -> None:
        """
        Huấn luyện Polynomial Regression.
        
        Args:
            degree: Bậc của polynomial
            alpha: Hệ số regularization cho Ridge
            feature_subset: Danh sách features sử dụng (None = tất cả)
        """
        start_time = time.perf_counter()
        log_section("TRAINING POLYNOMIAL REGRESSION", icon="📊")
        log_step(f"degree={degree}, alpha={alpha}")
        
        # Xử lý feature subset
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
                log_step(f"Sử dụng {len(feature_subset)} feature có |corr| >= threshold", icon="📌")
                log_step(f"Features: {feature_subset}", icon="🧮")
        
        # Chuẩn bị dữ liệu
        base_X_train = self.X_train[feature_subset] if feature_subset else self.X_train
        base_X_test = self.X_test[feature_subset] if feature_subset else self.X_test
        
        # Tạo polynomial features
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_train_poly = poly.fit_transform(base_X_train)
        X_test_poly = poly.transform(base_X_test)
        
        log_step(f"Original features: {self.X_train.shape[1]}", icon="📊")
        log_step(f"Polynomial features: {X_train_poly.shape[1]}", icon="🧱")
        
        # Train model
        model = Ridge(alpha=alpha)
        model.fit(X_train_poly, self.y_train)
        
        # Evaluate
        y_pred_train = model.predict(X_train_poly)
        y_pred_test = model.predict(X_test_poly)
        
        train_metrics = self.evaluate(self.y_train, y_pred_train, prefix='train')
        test_metrics = self.evaluate(self.y_test, y_pred_test, prefix='test')
        
        # Lưu model và kết quả
        self.model = model
        self.model_data = {
            'model': model,
            'poly': poly,
            'feature_subset': feature_subset
        }
        
        # Lưu thêm transformed data để dùng cho prediction
        self.X_train_transformed = X_train_poly
        self.X_test_transformed = X_test_poly
        
        self.result = {
            'train_rmse': train_metrics['train_rmse'],
            'test_rmse': test_metrics['test_rmse'],
            'test_mae': test_metrics['test_mae'],
            'test_r2': test_metrics['test_r2'],
            'hyperparams': {
                'degree': degree,
                'alpha': alpha,
                'feature_subset': feature_subset if feature_subset else 'all'
            }
        }
        
        log_metrics({
            "Train RMSE": train_metrics['train_rmse'],
            "Test RMSE": test_metrics['test_rmse'],
            "Test MAE": test_metrics['test_mae'],
            "Test R²": test_metrics['test_r2'],
        })
        log_step(f"Thời gian train: {time.perf_counter() - start_time:.2f} giây", icon="⏱️")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Dự đoán với Polynomial model (cần transform features)."""
        if self.model is None:
            raise ValueError("Mô hình Polynomial chưa được train")
        
        poly = self.model_data['poly']
        feature_subset = self.model_data.get('feature_subset')
        
        X_input = X[feature_subset] if feature_subset else X
        X_poly = poly.transform(X_input)
        
        return self.model.predict(X_poly)


# ========== RANDOM FOREST TRAINER ==========
class RandomForestTrainer(BaseTrainer):
    """Trainer cho Random Forest Regressor."""
    
    @property
    def model_name(self) -> str:
        return 'random_forest'
    
    def _objective(self, trial: optuna.Trial) -> float:
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
        
        return self._run_cross_validation(model)
    
    def optimize(self, n_trials: int = 20, timeout: int = 600) -> Dict:
        """Tối ưu hyperparameters cho Random Forest."""
        log_section("TỐI ƯU RANDOM FOREST", icon="🔍")
        
        study = self._create_optuna_study()
        study.optimize(
            self._objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True
        )
        
        best_params = study.best_params
        log_step(f"Best params: {best_params}", icon="✅")
        log_metrics({"Best RMSE": study.best_value})
        
        self.optimization_history = {
            'best_params': best_params,
            'best_value': study.best_value,
            'n_trials': len(study.trials)
        }
        return best_params
    
    def train(self, n_estimators: int = 100, max_depth: int = 10,
              min_samples_split: int = 5, min_samples_leaf: int = 2) -> None:
        """Huấn luyện Random Forest."""
        start_time = time.perf_counter()
        log_section("TRAINING RANDOM FOREST", icon="🌲")
        log_step(f"n_estimators={n_estimators}, max_depth={max_depth}")
        
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=self.RANDOM_SEED,
            n_jobs=-1
        )
        
        model.fit(self.X_train, self.y_train)
        
        # Evaluate
        y_pred_train = model.predict(self.X_train)
        y_pred_test = model.predict(self.X_test)
        
        train_metrics = self.evaluate(self.y_train, y_pred_train, prefix='train')
        test_metrics = self.evaluate(self.y_test, y_pred_test, prefix='test')
        
        # Lưu model
        self.model = model
        self.model_data = {'model': model}
        
        self.result = {
            'train_rmse': train_metrics['train_rmse'],
            'test_rmse': test_metrics['test_rmse'],
            'test_mae': test_metrics['test_mae'],
            'test_r2': test_metrics['test_r2'],
            'hyperparams': {
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf
            }
        }
        
        log_metrics({
            "Train RMSE": train_metrics['train_rmse'],
            "Test RMSE": test_metrics['test_rmse'],
            "Test MAE": test_metrics['test_mae'],
            "Test R²": test_metrics['test_r2'],
        })
        log_step(f"Thời gian train: {time.perf_counter() - start_time:.2f} giây", icon="⏱️")


# ========== EXTRA TREES TRAINER ==========
class ExtraTreesTrainer(BaseTrainer):
    """Trainer cho Extra Trees (Extremely Randomized Trees) Regressor."""
    
    @property
    def model_name(self) -> str:
        return 'extra_trees'
    
    def _objective(self, trial: optuna.Trial) -> float:
        """Objective function cho Extra Trees optimization."""
        n_estimators = trial.suggest_int('n_estimators', 50, 300)
        max_depth = trial.suggest_int('max_depth', 5, 20)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 5)
        
        model = ExtraTreesRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=self.RANDOM_SEED,
            n_jobs=-1
        )
        
        return self._run_cross_validation(model)
    
    def optimize(self, n_trials: int = 20, timeout: int = 600) -> Dict:
        """Tối ưu hyperparameters cho Extra Trees."""
        log_section("TỐI ƯU EXTRA TREES", icon="🔍")
        
        study = self._create_optuna_study()
        study.optimize(
            self._objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True
        )
        
        best_params = study.best_params
        log_step(f"Best params: {best_params}", icon="✅")
        log_metrics({"Best RMSE": study.best_value})
        
        self.optimization_history = {
            'best_params': best_params,
            'best_value': study.best_value,
            'n_trials': len(study.trials)
        }
        return best_params
    
    def train(self, n_estimators: int = 200, max_depth: int = 12,
              min_samples_split: int = 2, min_samples_leaf: int = 1) -> None:
        """Huấn luyện Extra Trees."""
        start_time = time.perf_counter()
        log_section("TRAINING EXTRA TREES", icon="🌳")
        log_step(f"n_estimators={n_estimators}, max_depth={max_depth}")
        
        model = ExtraTreesRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=self.RANDOM_SEED,
            n_jobs=-1
        )
        
        model.fit(self.X_train, self.y_train)
        
        # Evaluate
        y_pred_train = model.predict(self.X_train)
        y_pred_test = model.predict(self.X_test)
        
        train_metrics = self.evaluate(self.y_train, y_pred_train, prefix='train')
        test_metrics = self.evaluate(self.y_test, y_pred_test, prefix='test')
        
        # Lưu model
        self.model = model
        self.model_data = {'model': model}
        
        self.result = {
            'train_rmse': train_metrics['train_rmse'],
            'test_rmse': test_metrics['test_rmse'],
            'test_mae': test_metrics['test_mae'],
            'test_r2': test_metrics['test_r2'],
            'hyperparams': {
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf
            }
        }
        
        log_metrics({
            "Train RMSE": train_metrics['train_rmse'],
            "Test RMSE": test_metrics['test_rmse'],
            "Test MAE": test_metrics['test_mae'],
            "Test R²": test_metrics['test_r2'],
        })
        log_step(f"Thời gian train: {time.perf_counter() - start_time:.2f} giây", icon="⏱️")


# ========== XGBOOST TRAINER ==========
class XGBoostTrainer(BaseTrainer):
    """Trainer cho XGBoost Regressor."""
    
    @property
    def model_name(self) -> str:
        return 'xgboost'
    
    def _objective(self, trial: optuna.Trial) -> float:
        """Objective function cho XGBoost optimization."""
        params = {
            'max_depth': trial.suggest_int('max_depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 10.0, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'random_state': self.RANDOM_SEED,
        }
        
        model = xgb.XGBRegressor(**params)
        return self._run_cross_validation(model)
    
    def optimize(self, n_trials: int = 30, timeout: int = 900) -> Dict:
        """Tối ưu hyperparameters cho XGBoost."""
        log_section("TỐI ƯU XGBOOST", icon="🔍")
        
        study = self._create_optuna_study()
        study.optimize(
            self._objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True
        )
        
        best_params = study.best_params
        log_step(f"Best params: {best_params}", icon="✅")
        log_metrics({"Best RMSE": study.best_value})
        
        self.optimization_history = {
            'best_params': best_params,
            'best_value': study.best_value,
            'n_trials': len(study.trials)
        }
        return best_params
    
    def train(self, **xgb_params) -> None:
        """
        Huấn luyện XGBoost.
        
        Args:
            **xgb_params: XGBoost hyperparameters
        """
        start_time = time.perf_counter()
        log_section("TRAINING XGBOOST", icon="⚡")
        
        # Default params
        default_params = {
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 300,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'gamma': 0,
            'reg_lambda': 1.0,
            'reg_alpha': 0,
            'random_state': self.RANDOM_SEED,
            'n_jobs': -1
        }
        default_params.update(xgb_params)
        
        # Xử lý early stopping
        early_stopping = default_params.pop('early_stopping_rounds', None)
        
        model = xgb.XGBRegressor(**default_params)
        
        if early_stopping:
            X_tr, X_val, y_tr, y_val = train_test_split(
                self.X_train, self.y_train,
                test_size=0.2,
                random_state=self.RANDOM_SEED
            )
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        else:
            model.fit(self.X_train, self.y_train, verbose=False)
        
        # Evaluate
        y_pred_train = model.predict(self.X_train)
        y_pred_test = model.predict(self.X_test)
        
        train_metrics = self.evaluate(self.y_train, y_pred_train, prefix='train')
        test_metrics = self.evaluate(self.y_test, y_pred_test, prefix='test')
        
        # Log gap
        gap = train_metrics['train_rmse'] - test_metrics['test_rmse']
        log_step(f"Train-Test Gap: {abs(gap):.2f} (target < 3.0)", icon="📊")
        
        # Lưu model
        self.model = model
        self.model_data = {'model': model}
        
        self.result = {
            'train_rmse': train_metrics['train_rmse'],
            'test_rmse': test_metrics['test_rmse'],
            'test_mae': test_metrics['test_mae'],
            'test_r2': test_metrics['test_r2'],
            'hyperparams': default_params
        }
        
        log_metrics({
            "Train RMSE": train_metrics['train_rmse'],
            "Test RMSE": test_metrics['test_rmse'],
            "Test MAE": test_metrics['test_mae'],
            "Test R²": test_metrics['test_r2'],
        })
        log_step(f"Thời gian train: {time.perf_counter() - start_time:.2f} giây", icon="⏱️")


# ========== TRAINER REGISTRY ==========
TRAINER_REGISTRY = {
    'polynomial': PolynomialTrainer,
    'random_forest': RandomForestTrainer,
    'extra_trees': ExtraTreesTrainer,
    'xgboost': XGBoostTrainer,
}


def get_trainer(name: str, X_train: pd.DataFrame, X_test: pd.DataFrame,
                y_train: pd.Series, y_test: pd.Series) -> BaseTrainer:
    """
    Factory function để tạo trainer theo tên.
    
    Args:
        name: Tên mô hình ('polynomial', 'random_forest', 'extra_trees', 'xgboost')
        X_train, X_test: Features
        y_train, y_test: Target
        
    Returns:
        Trainer instance
    """
    if name not in TRAINER_REGISTRY:
        raise ValueError(f"Unknown trainer: {name}. Available: {list(TRAINER_REGISTRY.keys())}")
    
    return TRAINER_REGISTRY[name](X_train, X_test, y_train, y_test)
