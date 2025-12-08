"""
Model Registry - Chứa các Trainer cụ thể cho từng loại mô hình.

Bao gồm:
- RandomForestTrainer: Random Forest Regressor
- ExtraTreesTrainer: Extra Trees Regressor
- XGBoostTrainer: XGBoost Regressor
"""

import time
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import cross_val_score, train_test_split

import xgboost as xgb
import optuna

from src.modeling.base_trainer import (
    BaseTrainer, log_section, log_step, log_metrics, logger
)


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
    'random_forest': RandomForestTrainer,
    'extra_trees': ExtraTreesTrainer,
    'xgboost': XGBoostTrainer,
}


def get_trainer(name: str, X_train: pd.DataFrame, X_test: pd.DataFrame,
                y_train: pd.Series, y_test: pd.Series) -> BaseTrainer:
    """
    Factory function để tạo trainer theo tên.
    
    Args:
        name: Tên mô hình ('random_forest', 'extra_trees', 'xgboost')
        X_train, X_test: Features
        y_train, y_test: Target
        
    Returns:
        Trainer instance
    """
    if name not in TRAINER_REGISTRY:
        raise ValueError(f"Unknown trainer: {name}. Available: {list(TRAINER_REGISTRY.keys())}")
    
    return TRAINER_REGISTRY[name](X_train, X_test, y_train, y_test)
