"""
Base Trainer - Lớp cơ sở cho các Model Trainer.

Cung cấp:
- Abstract base class cho các trainer cụ thể
- Các hàm tiện ích logging
- Methods evaluate chung
"""

import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler


logger = logging.getLogger(__name__)

# Kiểm tra xem terminal có hỗ trợ emoji không (Windows cmd/powershell thường không)
_SUPPORTS_EMOJI = os.name != 'nt' or os.environ.get('WT_SESSION')  # Windows Terminal hỗ trợ

# Mapping emoji -> text fallback
_ICON_MAP = {
    "📘": "[INFO]",
    "🔸": ">>",
    "✅": "[OK]",
    "❌": "[ERROR]",
    "⚙️": "[CONFIG]",
    "📊": "[DATA]",
    "🔍": "[SEARCH]",
    "🌲": "[RF]",
    "🌳": "[ET]",
    "⚡": "[XGB]",
    "💾": "[SAVE]",
    "📂": "[LOAD]",
    "🎯": "[TARGET]",
    "📌": "[NOTE]",
    "🧮": "[CALC]",
    "🧱": "[BUILD]",
    "⏱️": "[TIME]",
    "🤖": "[MODEL]",
    "📈": "[CHART]",
    "✨": "[BEST]",
    "🏆": "[WINNER]",
    "🔄": "[SYNC]",
    "🚀": "[START]",
    "🧼": "[CLEAN]",
    "📥": "[INPUT]",
    "✂️": "[SPLIT]",
    "🔧": "[PROCESS]",
}

def _get_icon(icon: str) -> str:
    """Trả về icon phù hợp với terminal."""
    if _SUPPORTS_EMOJI:
        return icon
    return _ICON_MAP.get(icon, "[*]")


# ========== LOGGING UTILITIES ==========
def _divider(width: int = 70, char: str = "=") -> str:
    """Tạo dòng phân cách."""
    return char * width


def log_section(title: str, icon: str = "📘") -> None:
    """Log tiêu đề section."""
    logger.info("\n%s", _divider())
    logger.info("%s %s", _get_icon(icon), title.upper())
    logger.info("%s", _divider())


def log_step(message: str, icon: str = "🔸") -> None:
    """Log một bước thực hiện."""
    logger.info("%s %s", _get_icon(icon), message)


def log_metrics(metrics: Dict[str, float]) -> None:
    """Log các metrics."""
    for label, value in metrics.items():
        logger.info("   %-12s: %.6f", label, value)


class BaseTrainer(ABC):
    """
    Lớp cơ sở abstract cho các model trainer.
    
    Cung cấp:
    - Interface chung cho train/optimize
    - Phương thức evaluate
    - Quản lý random seed
    
    Subclasses phải implement:
    - _objective(): Objective function cho Optuna
    - optimize(): Tối ưu hyperparameters
    - train(): Huấn luyện mô hình
    """
    
    RANDOM_SEED = 42
    
    def __init__(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                 y_train: pd.Series, y_test: pd.Series):
        """
        Khởi tạo BaseTrainer.
        
        Args:
            X_train, X_test: Features
            y_train, y_test: Target
        """
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        self.model = None
        self.model_data = {}  # Chứa model + các objects liên quan (poly, etc.)
        self.result = {}
        self.optimization_history = {}
        
        np.random.seed(self.RANDOM_SEED)
    
    @property
    def model_name(self) -> str:
        """Tên mô hình (phải được override)."""
        raise NotImplementedError
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, 
                 prefix: str = 'test') -> Dict[str, float]:
        """
        Đánh giá mô hình với các metrics chuẩn.
        
        Args:
            y_true: Giá trị thực
            y_pred: Giá trị dự đoán
            prefix: Prefix cho tên metrics ('train' hoặc 'test')
            
        Returns:
            Dict chứa các metrics
        """
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        return {
            f'{prefix}_rmse': float(rmse),
            f'{prefix}_mae': float(mae),
            f'{prefix}_r2': float(r2)
        }
    
    def _create_optuna_study(self, direction: str = 'minimize') -> optuna.Study:
        """Tạo Optuna study với cấu hình chuẩn."""
        sampler = TPESampler(seed=self.RANDOM_SEED)
        pruner = MedianPruner()
        
        return optuna.create_study(
            sampler=sampler,
            pruner=pruner,
            direction=direction
        )
    
    def _run_cross_validation(self, model, cv: int = 5, 
                              scoring: str = 'neg_mean_squared_error') -> float:
        """
        Chạy cross-validation và trả về RMSE.
        
        Args:
            model: Model sklearn
            cv: Số folds
            scoring: Metric scoring
            
        Returns:
            RMSE trung bình
        """
        cv_scores = cross_val_score(
            model, self.X_train, self.y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1
        )
        return np.sqrt(-cv_scores.mean())
    
    @abstractmethod
    def _objective(self, trial: optuna.Trial) -> float:
        """
        Objective function cho Optuna optimization.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Metric cần minimize (thường là RMSE)
        """
        pass
    
    @abstractmethod
    def optimize(self, n_trials: int = 20, timeout: int = 600) -> Dict[str, Any]:
        """
        Tối ưu hyperparameters với Optuna.
        
        Args:
            n_trials: Số lần thử
            timeout: Timeout (giây)
            
        Returns:
            Best hyperparameters
        """
        pass
    
    @abstractmethod
    def train(self, **kwargs) -> None:
        """
        Huấn luyện mô hình với hyperparameters cho trước.
        
        Args:
            **kwargs: Hyperparameters
        """
        pass
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Dự đoán với mô hình đã train.
        
        Args:
            X: Features
            
        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError(f"Mô hình {self.model_name} chưa được train")
        return self.model.predict(X)
    
    def get_result(self) -> Dict[str, Any]:
        """Trả về kết quả training."""
        return self.result
    
    def get_model_data(self) -> Dict[str, Any]:
        """Trả về model data (model + các objects liên quan)."""
        return self.model_data
