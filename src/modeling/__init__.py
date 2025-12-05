"""
Module huấn luyện mô hình.

Cấu trúc:
- base_trainer.py: Lớp cơ sở và utilities
- model_registry.py: Các trainer cụ thể (Polynomial, RF, ET, XGBoost)
- model_trainer.py: Orchestrator chính
"""

from .model_trainer import ModelTrainer
from .base_trainer import BaseTrainer, log_section, log_step, log_metrics
from .model_registry import (
    PolynomialTrainer,
    RandomForestTrainer,
    ExtraTreesTrainer,
    XGBoostTrainer,
    get_trainer,
    TRAINER_REGISTRY,
)

__all__ = [
    'ModelTrainer',
    'BaseTrainer',
    'PolynomialTrainer',
    'RandomForestTrainer',
    'ExtraTreesTrainer',
    'XGBoostTrainer',
    'get_trainer',
    'TRAINER_REGISTRY',
    'log_section',
    'log_step',
    'log_metrics',
]
