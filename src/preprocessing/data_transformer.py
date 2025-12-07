"""
Data transformation utilities cho post-split processing.

Module này xử lý các bước SAU khi chia train/test:
- Handle missing values (fit trên train, transform trên test)
- Remove outliers (làm trên tập train , model có thể gặp khó khăn khi áp dụng thực tế nên không xử lí trên tập test  )
- Encode categorical (fit trên train, transform trên test)
- Scale features (fit trên train, transform trên test)
- Interaction features
- fit_transform() pipeline
- transform_new_data() cho test/inference
- Save/load state
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, Set

import numpy as np
import pandas as pd
import joblib
from sklearn.base import clone
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import (
	LabelEncoder,
	MaxAbsScaler,
	MinMaxScaler,
	Normalizer,
	OneHotEncoder,
	OrdinalEncoder,
	RobustScaler,
	StandardScaler,
)

from src.visualization import DataVisualizer
from config import EDA_RESULTS_DIR, PLOT_DPI, PLOT_STYLE, FIGURE_SIZE


logger = logging.getLogger(__name__)

MissingRule = Union[str, Tuple[str, Any]]


class DataTransformer:
	"""Transform dữ liệu sau khi chia train/test (fit trên train, transform trên test)."""

	def __init__(
		self,
		data: Optional[pd.DataFrame] = None,
		missing_strategy: str = "median",
		categorical_missing_strategy: str = "mode",
		scaler_type: str = "standard",
		encoder_type: str = "onehot",
	) -> None:
		"""
		Khởi tạo DataTransformer.
		
		Args:
			data: DataFrame (thường là train data)
			missing_strategy: Chiến lược fill missing cho numeric ('mean', 'median', 'mode')
			categorical_missing_strategy: Chiến lược fill missing cho categorical ('mode', 'constant')
			scaler_type: Loại scaler ('standard', 'minmax', 'robust')
			encoder_type: Loại encoder ('onehot', 'label', 'ordinal')
		"""
		self.data = data.copy() if data is not None else None
		self.numeric_cols: List[str] = []
		self.categorical_cols: List[str] = []
		self.types_: Dict[str, List[str]] = {}
		self.preprocessing_steps: List[str] = []

		# Missing value strategies
		self.missing_strategy = missing_strategy
		self.categorical_missing_strategy = categorical_missing_strategy
		self.missing_rules: Dict[str, MissingRule] = {}
		self.impute_values: Dict[str, Any] = {}

		# Constraint rules (for transform_new_data)
		self.constraint_rules: Dict[str, Dict[str, Any]] = {}

		# Outlier rules
		self.outlier_rules: Dict[str, str] = {}

		# Scaler
		self.scaler_rules: Dict[str, str] = {}
		self.scalers: Dict[str, Any] = {}
		self.default_scaler = self._init_default_scaler(scaler_type)
		self._scale_exclude: List[str] = []
		self._scaling_method: Optional[str] = None

		# Encoder
		self.encoder_rules: Dict[str, str] = {}
		self.encoders: Dict[str, Any] = {}
		self.default_encoder = self._init_default_encoder(encoder_type)
		self._onehot_drop_first = True
		self._encoding_method: Optional[str] = None
		self.onehot_feature_names: Dict[str, List[str]] = {}

		# Fitted state
		self.feature_columns: List[str] = []
		self._is_fitted: bool = False
		self._capture_config: bool = True

		# Config capture for transform_new_data
		self._unify_applied: bool = False
		self._text_unify_rules: Optional[Dict[str, Dict[str, str]]] = None
		self._feature_engineering_config: Optional[Dict[str, Any]] = None
		self._datetime_feature_configs: List[Dict[str, Any]] = []
		self._datetime_config_keys: Set[Tuple[str, Tuple[str, ...], bool]] = set()
		self._interaction_configs: List[Dict[str, Any]] = []
		self._interaction_config_keys: Set[Tuple[Tuple[Tuple[str, str], ...], Tuple[str, ...]]] = set()

		# Logging
		self.log: Dict[str, List[str]] = {}

		# Visualizer
		self._visualizer_init_args = {
			"target_col": None,
			"output_dir": EDA_RESULTS_DIR,
			"auto_save": True,
			"auto_show": False,
			"dpi": PLOT_DPI,
			"style": PLOT_STYLE,
			"figure_size": FIGURE_SIZE,
		}
		self.visualizer = DataVisualizer(data=self.data, **self._visualizer_init_args)

		if self.data is not None:
			self.detect_types()
			logger.info("✅ DataTransformer khởi tạo thành công - Shape: %s", self.data.shape)

	# ------------------------------------------------------------------
	# Internal helpers
	# ------------------------------------------------------------------
	def _init_default_scaler(self, scaler_type: str) -> Optional[Any]:
		mapping = {
			"standard": StandardScaler,
			"minmax": MinMaxScaler,
			"robust": RobustScaler,
			"maxabs": MaxAbsScaler,
			"normalize": lambda: Normalizer(norm="l2"),
		}
		constructor = mapping.get(scaler_type.lower()) if isinstance(scaler_type, str) else None
		return constructor() if constructor else None

	def _init_default_encoder(self, encoder_type: str) -> Optional[Any]:
		mapping = {
			"label": LabelEncoder,
			"onehot": lambda: OneHotEncoder(sparse_output=False, handle_unknown="ignore", dtype=int),
			"ordinal": lambda: OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
		}
		constructor = mapping.get(encoder_type.lower()) if isinstance(encoder_type, str) else None
		return constructor() if constructor else None

	def _update_visualizer(self) -> None:
		self.visualizer.set_data(self.data)

	def _log(self, key: str, message: str) -> None:
		self.log.setdefault(key, []).append(message)

	def detect_types(self) -> Dict[str, List[str]]:
		"""Phân loại các cột theo kiểu dữ liệu."""
		if self.data is None:
			raise ValueError("Chưa có dữ liệu để phân loại kiểu.")

		df = self.data
		numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
		datetime_cols = df.select_dtypes(include=["datetime", "datetime64[ns]"]).columns.tolist()
		bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()
		categorical_cols = [col for col in df.columns if col not in set(numeric_cols + datetime_cols + bool_cols)]

		self.types_ = {
			"numeric_data": numeric_cols,
			"categorical_data": categorical_cols,
			"datetime_data": datetime_cols,
			"boolean_data": bool_cols,
		}
		self.numeric_cols = numeric_cols
		self.categorical_cols = categorical_cols
		return self.types_

	# ------------------------------------------------------------------
	# Missing value handling
	# ------------------------------------------------------------------
	def fill_missing(self, missing_rules: Optional[Dict[str, MissingRule]] = None, *, fit: bool = True) -> "DataTransformer":
		"""Xử lý missing values."""
		if self.data is None:
			raise ValueError("Chưa có dữ liệu")
		if missing_rules:
			self.missing_rules.update(missing_rules)

		df = self.data
		cols_with_nan = [col for col in df.columns if df[col].isna().any()]
		if not cols_with_nan:
			logger.info("Không có missing value cần xử lý")
			return self

		self.detect_types()
		for col in cols_with_nan:
			rule = self.missing_rules.get(col)
			strategy = None
			custom_value = None
			if isinstance(rule, (list, tuple)):
				strategy, custom_value = rule[0], rule[1]
			elif isinstance(rule, str):
				strategy = rule

			if strategy is None:
				strategy = self.missing_strategy if pd.api.types.is_numeric_dtype(df[col]) else self.categorical_missing_strategy

			try:
				if strategy == "drop":
					if fit:
						before = len(df)
						df = df.dropna(subset=[col])
						logger.info("%s: Drop %s dòng", col, before - len(df))
					self.impute_values[col] = "__drop__"
				elif strategy == "mean" and pd.api.types.is_numeric_dtype(df[col]):
					value = df[col].mean() if fit else self.impute_values.get(col)
					if value is None:
						value = df[col].mean()
					df[col] = df[col].fillna(value)
					if fit:
						self.impute_values[col] = value
				elif strategy == "median" and pd.api.types.is_numeric_dtype(df[col]):
					value = df[col].median() if fit else self.impute_values.get(col)
					if value is None:
						value = df[col].median()
					df[col] = df[col].fillna(value)
					if fit:
						self.impute_values[col] = value
				elif strategy == "mode":
					if fit:
						mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else custom_value
						self.impute_values[col] = mode_val
					else:
						mode_val = self.impute_values.get(col)
						if mode_val is None and not df[col].mode().empty:
							mode_val = df[col].mode().iloc[0]
					df[col] = df[col].fillna(mode_val)
				elif strategy == "constant":
					if fit:
						fill_val = custom_value
						if fill_val is None:
							fill_val = 0 if pd.api.types.is_numeric_dtype(df[col]) else "Unknown"
						self.impute_values[col] = fill_val
					else:
						fill_val = self.impute_values.get(col, custom_value)
						if fill_val is None:
							fill_val = 0 if pd.api.types.is_numeric_dtype(df[col]) else "Unknown"
					df[col] = df[col].fillna(fill_val)
				elif strategy == "ffill":
					df[col] = df[col].ffill()
				elif strategy == "bfill":
					df[col] = df[col].bfill()
				else:
					if fit:
						default_value = df[col].median() if pd.api.types.is_numeric_dtype(df[col]) else df[col].mode().iloc[0]
						self.impute_values[col] = default_value
					else:
						default_value = self.impute_values.get(col)
						if default_value is None:
							default_value = df[col].median() if pd.api.types.is_numeric_dtype(df[col]) else df[col].mode().iloc[0]
					df[col] = df[col].fillna(default_value)
			except Exception as exc:
				self._log("missing_error", f"{col}: {exc}")
				logger.error("Lỗi xử lý missing cho %s: %s", col, exc)

		self.data = df.reset_index(drop=True)
		if fit:
			self.preprocessing_steps.append("fill_missing")
		self._update_visualizer()
		return self

	def handle_missing(
		self,
		strategy: str = "auto",
		numeric_strategy: str = "median",
		categorical_strategy: str = "mode",
		fill_value: Optional[Any] = None,
	) -> "DataTransformer":
		"""Xử lý missing values với chiến lược cho numeric và categorical."""
		if strategy == "drop":
			self.missing_strategy = "drop"
			self.categorical_missing_strategy = "drop"
			return self.fill_missing()

		self.missing_strategy = numeric_strategy
		self.categorical_missing_strategy = categorical_strategy
		if fill_value is not None:
			for col in self.numeric_cols:
				self.missing_rules.setdefault(col, ("constant", fill_value))
		return self.fill_missing()

	# ------------------------------------------------------------------
	# Outlier handling
	# ------------------------------------------------------------------
	def handle_outliers(self, outlier_rules: Optional[Dict[str, str]] = None) -> "DataTransformer":
		"""Xử lý outliers theo từng cột."""
		if self.data is None:
			return self
		rules = outlier_rules or self.outlier_rules
		if not rules:
			return self

		df = self.data
		for col, method in rules.items():
			if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
				continue
			method = method.lower()
			mask_outlier = None

			try:
				if method == "iqr":
					q1 = df[col].quantile(0.25)
					q3 = df[col].quantile(0.75)
					iqr = q3 - q1
					lower = q1 - 1.5 * iqr
					upper = q3 + 1.5 * iqr
					mask_outlier = (df[col] < lower) | (df[col] > upper)
				elif method == "zscore":
					z_scores = np.abs((df[col] - df[col].mean()) / df[col].std(ddof=0))
					mask_outlier = z_scores > 3
				elif method == "iforest":
					clf = IsolationForest(random_state=42, contamination="auto")
					preds = clf.fit_predict(df[[col]].fillna(df[col].mean()))
					mask_outlier = preds == -1
				else:
					logger.warning("Method %s không hợp lệ cho cột %s", method, col)
					continue

				if mask_outlier is not None and mask_outlier.any():
					logger.info("%s: Drop %s outliers (%s)", col, int(mask_outlier.sum()), method)
					df = df.loc[~mask_outlier]
			except Exception as exc:
				self._log("outlier_error", f"{col}: {exc}")

		self.data = df.reset_index(drop=True)
		self._update_visualizer()
		self.preprocessing_steps.append("handle_outliers")
		return self

	def remove_outliers(self, method: str = "iqr", threshold: float = 1.5) -> "DataTransformer":
		"""Loại bỏ outliers từ tất cả cột số."""
		self.outlier_rules = {col: method for col in self.numeric_cols}
		return self.handle_outliers(outlier_rules=self.outlier_rules)

	# ------------------------------------------------------------------
	# Scaling
	# ------------------------------------------------------------------
	def scale(self, scaler_rules: Optional[Dict[str, str]] = None, *, fit: bool = True) -> "DataTransformer":
		"""Scale các cột số."""
		if self.data is None:
			return self
		self.detect_types()
		num_cols = self.types_.get("numeric_data", [])
		if not num_cols:
			return self

		if scaler_rules:
			self.scaler_rules.update(scaler_rules)

		df = self.data
		logger.info(
			"⚙️  SCALE FEATURES | Tổng cột số: %s | Có rule riêng: %s",
			len(num_cols),
			len(self.scaler_rules),
		)
		scaled_cols = set()
		for col, method in self.scaler_rules.items():
			if col not in num_cols:
				continue
			if fit:
				scaler = self._init_default_scaler(method)
				if scaler is None:
					continue
				df[[col]] = scaler.fit_transform(df[[col]])
				self.scalers[col] = scaler
			else:
				scaler = self.scalers.get(col)
				if scaler is None:
					continue
				df[[col]] = scaler.transform(df[[col]])
			scaled_cols.add(col)
			logger.info("   • %s: dùng scaler %s", col, scaler.__class__.__name__)

		remaining_cols = [col for col in num_cols if col not in scaled_cols]
		if self.default_scaler and remaining_cols:
			logger.info(
				"   • %s cột còn lại dùng scaler mặc định %s",
				len(remaining_cols),
				self.default_scaler.__class__.__name__,
			)
			for col in remaining_cols:
				if fit:
					scaler = clone(self.default_scaler)
					df[[col]] = scaler.fit_transform(df[[col]])
					self.scalers[col] = scaler
				else:
					scaler = self.scalers.get(col)
					if scaler is None:
						continue
					df[[col]] = scaler.transform(df[[col]])

		self.data = df
		return self

	def scale_features(
		self,
		method: str = "standard",
		columns: Optional[List[str]] = None,
		exclude_columns: Optional[List[str]] = None,
		*,
		fit: bool = True,
	) -> "DataTransformer":
		"""Scale features với phương pháp chỉ định."""
		if not fit and self._scaling_method is not None:
			method = self._scaling_method
		self.detect_types()
		target_cols = columns or self.types_.get("numeric_data", [])
		if exclude_columns:
			target_cols = [col for col in target_cols if col not in exclude_columns]
		if fit:
			self._scaling_method = method
			self._scale_exclude = exclude_columns or []
			self.scaler_rules = {col: method for col in target_cols}
			self.default_scaler = None
		return self.scale(fit=fit)

	# ------------------------------------------------------------------
	# Encoding
	# ------------------------------------------------------------------
	def encode(self, encoder_rules: Optional[Dict[str, str]] = None, *, fit: bool = True) -> "DataTransformer":
		"""Encode các cột categorical."""
		if self.data is None:
			return self
		self.detect_types()
		cat_cols = self.types_.get("categorical_data", [])
		if not cat_cols:
			return self
		if encoder_rules:
			self.encoder_rules.update(encoder_rules)

		df = self.data
		logger.info(
			"🧩 ENCODING FEATURES | Tổng cột phân loại: %s | Rule riêng: %s",
			len(cat_cols),
			len(self.encoder_rules),
		)
		onehot_frames = []
		cols_to_drop: List[str] = []

		for col in cat_cols:
			method = self.encoder_rules.get(col)
			if method is None and self.default_encoder is not None:
				if isinstance(self.default_encoder, OneHotEncoder):
					method = "onehot"
				elif isinstance(self.default_encoder, OrdinalEncoder):
					method = "ordinal"
				else:
					method = "label"

			try:
				if method == "label":
					if fit:
						le = LabelEncoder()
						df[col] = le.fit_transform(df[col].astype(str))
						self.encoders[col] = le
					else:
						le = self.encoders.get(col)
						if le is not None:
							df[col] = df[col].astype(str).apply(
								lambda x: le.transform([x])[0] if x in le.classes_ else -1
							)
					logger.info("   • %s: LabelEncoder", col)
				elif method == "onehot":
					if fit:
						ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore", dtype=int, drop="first" if self._onehot_drop_first else None)
						encoded = ohe.fit_transform(df[[col]])
						self.encoders[col] = ohe
						feature_names = [f"{col}_{cat}" for cat in ohe.categories_[0][1:]] if self._onehot_drop_first else [f"{col}_{cat}" for cat in ohe.categories_[0]]
						self.onehot_feature_names[col] = feature_names
					else:
						ohe = self.encoders.get(col)
						if ohe is not None:
							encoded = ohe.transform(df[[col]])
							feature_names = self.onehot_feature_names.get(col, [])
						else:
							continue
					ohe_df = pd.DataFrame(encoded, columns=feature_names, index=df.index)
					onehot_frames.append(ohe_df)
					cols_to_drop.append(col)
					logger.info("   • %s: OneHotEncoder -> %s cột mới", col, len(feature_names))
				elif method == "ordinal":
					if fit:
						oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
						df[[col]] = oe.fit_transform(df[[col]])
						self.encoders[col] = oe
					else:
						oe = self.encoders.get(col)
						if oe is not None:
							df[[col]] = oe.transform(df[[col]])
					logger.info("   • %s: OrdinalEncoder", col)
			except Exception as exc:
				self._log("encode_error", f"{col}: {exc}")

		if onehot_frames:
			df = pd.concat([df] + onehot_frames, axis=1)
			df.drop(columns=cols_to_drop, inplace=True)
			logger.info("   • Tổng cột one-hot mới: %s", sum(frame.shape[1] for frame in onehot_frames))

		self.data = df
		return self

	def encode_categorical(
		self,
		method: str = "onehot",
		columns: Optional[List[str]] = None,
		drop_first: bool = True,
		*,
		fit: bool = True,
	) -> "DataTransformer":
		"""Encode categorical với phương pháp chỉ định."""
		self._onehot_drop_first = drop_first
		if fit:
			self._encoding_method = method
			if method == "onehot" and columns:
				self.encoder_rules = {col: "onehot" for col in columns}
			elif method == "label" and columns:
				self.encoder_rules = {col: "label" for col in columns}
		if fit:
			self.default_encoder = self._init_default_encoder(method)
		return self.encode(fit=fit)

	# ------------------------------------------------------------------
	# Feature engineering (post-split)
	# ------------------------------------------------------------------
	def create_datetime_features(
		self,
		datetime_col: str,
		features: Sequence[str] = ("hour", "day", "month", "dayofweek"),
		drop_original: bool = False,
	) -> "DataTransformer":
		"""Tạo features từ cột datetime."""
		if self.data is None or datetime_col not in self.data.columns:
			return self
		df = self.data
		if not pd.api.types.is_datetime64_any_dtype(df[datetime_col]):
			df[datetime_col] = pd.to_datetime(df[datetime_col], errors="coerce")
		if self._capture_config:
			config_key = (datetime_col, tuple(features), drop_original)
			if config_key not in self._datetime_config_keys:
				self._datetime_config_keys.add(config_key)
				self._datetime_feature_configs.append(
					{"datetime_col": datetime_col, "features": tuple(features), "drop_original": drop_original}
				)

		mapping = {
			"hour": df[datetime_col].dt.hour,
			"day": df[datetime_col].dt.day,
			"month": df[datetime_col].dt.month,
			"year": df[datetime_col].dt.year,
			"dayofweek": df[datetime_col].dt.dayofweek,
			"quarter": df[datetime_col].dt.quarter,
		}
		for feat in features:
			if feat in mapping:
				df[f"{datetime_col}_{feat}"] = mapping[feat]
		if drop_original:
			df.drop(columns=[datetime_col], inplace=True)
		self.data = df
		self.detect_types()
		if self._capture_config:
			self.preprocessing_steps.append("create_datetime_features")
		return self

	def create_interaction_features(
		self,
		col_pairs: List[Tuple[str, str]],
		operations: Sequence[str] = ("multiply",),
	) -> "DataTransformer":
		"""Tạo interaction features từ các cặp cột."""
		if self.data is None:
			return self
		df = self.data
		if self._capture_config:
			normalized_pairs = [tuple(pair) for pair in col_pairs]
			config_key = (tuple(normalized_pairs), tuple(operations))
			if config_key not in self._interaction_config_keys:
				self._interaction_config_keys.add(config_key)
				self._interaction_configs.append(
					{"col_pairs": normalized_pairs, "operations": tuple(operations)}
				)
		for col1, col2 in col_pairs:
			if col1 not in df.columns or col2 not in df.columns:
				continue
			for op in operations:
				if op == "multiply":
					df[f"{col1}_x_{col2}"] = df[col1] * df[col2]
				elif op == "add":
					df[f"{col1}_plus_{col2}"] = df[col1] + df[col2]
				elif op == "subtract":
					df[f"{col1}_minus_{col2}"] = df[col1] - df[col2]
				elif op == "divide":
					df[f"{col1}_div_{col2}"] = df[col1] / (df[col2] + 1e-6)
		self.data = df
		self.detect_types()
		if self._capture_config:
			self.preprocessing_steps.append("create_interaction_features")
		return self

	# ------------------------------------------------------------------
	# Visualization bridge
	# ------------------------------------------------------------------
	def plot_correlation_heatmap(
		self,
		target_col: Optional[str] = None,
		method: str = "pearson",
		save_path: Optional[Union[str, Path]] = None,
		figsize: Tuple[int, int] = (12, 10),
		annot: bool = False,
		show: bool = False,
	) -> Optional[pd.DataFrame]:
		"""Vẽ heatmap tương quan."""
		if self.data is None:
			logger.warning("❌ Chưa có dữ liệu để vẽ heatmap")
			return None
		self.visualizer.set_data(self.data)
		self.visualizer.set_target(target_col)
		return self.visualizer.plot_correlation_heatmap(
			method=method,
			annot=annot,
			figsize=figsize,
			save_path=Path(save_path) if save_path else None,
			show=show,
		)

	def generate_eda_report(
		self,
		target_col: Optional[str] = None,
		numeric_cols: Optional[Sequence[str]] = None,
		categorical_cols: Optional[Sequence[str]] = None,
	) -> None:
		"""
		Tạo báo cáo EDA tối ưu với 5 biểu đồ chính:
		1. Data Overview - Tổng quan dataset
		2. Numeric Distributions - Phân phối các features số
		3. Categorical Distributions - Phân phối các features phân loại
		4. Correlation Heatmap - Ma trận tương quan
		5. Target Analysis - Phân tích biến mục tiêu
		"""
		if self.data is None:
			logger.warning("❌ Chưa có dữ liệu để tạo biểu đồ EDA")
			return

		self.detect_types()
		self.visualizer.set_data(self.data)
		self.visualizer.set_target(target_col)

		logger.info("📊 Đang tạo báo cáo EDA...")
		
		# 1. Data Overview
		self.visualizer.plot_data_overview()
		
		# 2. Numeric distributions (grid)
		self.visualizer.plot_numeric_grid(cols=numeric_cols or self.numeric_cols)
		
		# 3. Categorical distributions (grid)
		self.visualizer.plot_categorical_grid(cols=categorical_cols or self.categorical_cols)
		
		# 4. Correlation heatmap
		corr_save_path = self.visualizer.output_dir / "04_correlation_heatmap.png"
		self.visualizer.plot_correlation_heatmap(method="spearman", annot=True, save_path=corr_save_path, show=False)
		
		# 5. Target analysis
		if target_col:
			self.visualizer.plot_target_analysis(top_n_corr=10)
		
		# 6. Outliers boxplot
		self.visualizer.plot_outliers_boxplot(cols=numeric_cols or self.numeric_cols)
		
		logger.info("✅ Đã tạo xong báo cáo EDA (6 biểu đồ)")

	# ------------------------------------------------------------------
	# Pipeline orchestration
	# ------------------------------------------------------------------
	def mark_as_fitted(self) -> None:
		"""Đánh dấu transformer đã được fit."""
		if self.data is None:
			return
		self.feature_columns = self.data.columns.tolist()
		self._is_fitted = True

	def fit_transform(
		self,
		target_col: str,
		remove_outliers: bool = False,
		outlier_method: str = "iqr",
		outlier_threshold: float = 1.5,
		encoding_method: str = "onehot",
		drop_first_onehot: bool = True,
		scaling_method: str = "standard",
		exclude_scale_cols: Optional[List[str]] = None,
		interaction_pairs: Optional[List[Tuple[str, str]]] = None,
		generate_eda: bool = False,
	) -> pd.DataFrame:
		"""
		Pipeline hoàn chỉnh để fit và transform dữ liệu train.
		
		Thứ tự xử lý: Outliers → Fill Missing → Encode → Scale → Interaction
		
		Args:
			target_col: Tên cột target (sẽ không scale)
			remove_outliers: Có loại bỏ outliers không
			outlier_method: Phương pháp detect outliers ('iqr', 'zscore')
			outlier_threshold: Ngưỡng detect outliers
			encoding_method: Phương pháp encoding ('onehot', 'label')
			drop_first_onehot: Có drop first column trong onehot không
			scaling_method: Phương pháp scaling ('standard', 'minmax', 'robust')
			exclude_scale_cols: Các cột không scale (ngoài target)
			interaction_pairs: Các cặp cột để tạo interaction features
			generate_eda: Có tạo EDA report không
			
		Returns:
			DataFrame đã được xử lý
		"""
		if self.data is None:
			raise ValueError("Chưa có dữ liệu để xử lý")
		
		logger.info("=" * 70)
		logger.info("🔧 BẮT ĐẦU FIT_TRANSFORM PIPELINE")
		logger.info("=" * 70)
		
		# EDA trước khi xử lý
		if generate_eda:
			logger.info("📊 Đang tạo EDA report...")
			self.generate_eda_report(target_col=target_col)
		
		# 1. Remove outliers
		if remove_outliers:
			logger.info("🔸 Bước 1: Loại bỏ outliers (%s)", outlier_method)
			self.remove_outliers(method=outlier_method, threshold=outlier_threshold)
			logger.info("   Shape sau outliers: %s", self.data.shape)
		
		# 2. Fill missing values
		logger.info("🔸 Bước 2: Xử lý missing values")
		self.handle_missing(
			strategy='auto',
			numeric_strategy=self.missing_strategy,
			categorical_strategy=self.categorical_missing_strategy
		)
		
		# 3. Encode categorical
		logger.info("🔸 Bước 3: Mã hóa biến phân loại (%s)", encoding_method)
		self._onehot_drop_first = drop_first_onehot
		self.encode_categorical(method=encoding_method, drop_first=drop_first_onehot)
		
		# 4. Scale features
		logger.info("🔸 Bước 4: Chuẩn hóa features (%s)", scaling_method)
		auto_exclude = [target_col] if target_col else []
		if exclude_scale_cols:
			auto_exclude.extend(exclude_scale_cols)
		# Thêm các cột binary vào exclude list
		for col in self.data.columns:
			if col not in auto_exclude and self.data[col].nunique() <= 2:
				if set(self.data[col].dropna().unique()).issubset({0, 1, 0.0, 1.0}):
					auto_exclude.append(col)
		
		self.scale_features(method=scaling_method, exclude_columns=auto_exclude)
		
		# 5. Interaction features
		if interaction_pairs:
			logger.info("🔸 Bước 5: Tạo interaction features")
			self.create_interaction_features(col_pairs=interaction_pairs, operations=['multiply'])
		
		# Mark as fitted
		self.mark_as_fitted()
		
		logger.info("✅ FIT_TRANSFORM HOÀN TẤT - Shape: %s", self.data.shape)
		return self.data.copy()

	def transform_new_data(self, df: pd.DataFrame) -> pd.DataFrame:
		"""Transform dữ liệu mới (test/inference) dùng params đã fit từ train."""
		if not self._is_fitted:
			raise ValueError("Transformer chưa được fit trên dữ liệu train")
		original_data = self.data
		original_numeric = self.numeric_cols.copy()
		original_categorical = self.categorical_cols.copy()
		try:
			self._capture_config = False
			self.data = df.copy()
			self.detect_types()
			
			# Apply constraints nếu có
			if self.constraint_rules:
				for col, rule in self.constraint_rules.items():
					if col not in self.data.columns or not isinstance(rule, dict):
						continue
					action = rule.get("action", "drop")
					min_val, max_val = rule.get("min"), rule.get("max")
					if action == "clip" and (min_val is not None or max_val is not None):
						self.data[col] = self.data[col].clip(lower=min_val, upper=max_val)
			
			# Unify values nếu đã áp dụng
			if self._unify_applied or self._text_unify_rules:
				cat_cols = self.types_.get("categorical_data", [])
				for col in cat_cols:
					self.data[col] = self.data[col].astype(str).str.lower().str.strip()
					if self._text_unify_rules and col in self._text_unify_rules:
						self.data[col] = self.data[col].replace(self._text_unify_rules[col])
			
			# Datetime features
			if self._datetime_feature_configs:
				for config in self._datetime_feature_configs:
					self.create_datetime_features(
						datetime_col=config["datetime_col"],
						features=config["features"],
						drop_original=config["drop_original"],
					)
			
			# Feature engineering
			if self._feature_engineering_config:
				# Áp dụng feature engineering đơn giản
				pass
			
			# Fill missing
			self.fill_missing(fit=False)
			
			# Encode
			encoding_method = self._encoding_method or "onehot"
			self.encode_categorical(method=encoding_method, fit=False)
			
			# Scale
			exclude_cols = self._scale_exclude
			self.scale_features(method=self._scaling_method or "standard", exclude_columns=exclude_cols, fit=False)
			
			# Interaction features
			if self._interaction_configs:
				for config in self._interaction_configs:
					self.create_interaction_features(
						col_pairs=[tuple(pair) for pair in config["col_pairs"]],
						operations=config["operations"],
					)
			
			# Reindex để match columns với train
			if self.feature_columns:
				self.data = self.data.reindex(columns=self.feature_columns, fill_value=0)
			
			return self.data.copy()
		finally:
			self._capture_config = True
			self.data = original_data
			self.numeric_cols = original_numeric
			self.categorical_cols = original_categorical
			if self.data is not None:
				self.detect_types()

	# ------------------------------------------------------------------
	# Persistence & reporting
	# ------------------------------------------------------------------
	def get_processed_data(self) -> pd.DataFrame:
		"""Lấy dữ liệu đã xử lý."""
		if self.data is None:
			raise ValueError("Chưa có dữ liệu")
		return self.data.copy()

	def split_features_target(self, target_col: Optional[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
		"""
		Tách features (X) và target (y) từ dữ liệu đã xử lý.
		
		Args:
			target_col: Tên cột target (mặc định dùng self.target_col)
			
		Returns:
			(X, y) tuple
		"""
		if self.data is None:
			raise ValueError("Chưa có dữ liệu")
		
		target = target_col or self.target_col
		if target is None:
			raise ValueError("Chưa xác định target column")
		if target not in self.data.columns:
			raise ValueError(f"Target column '{target}' không tồn tại trong data")
		
		X = self.data.drop(columns=[target])
		y = self.data[target]
		return X, y

	def get_correlated_features(
		self, 
		target_col: Optional[str] = None,
		threshold: float = 0.1,
		method: str = "spearman"
	) -> List[str]:
		"""
		Lấy danh sách features có correlation với target >= threshold.
		
		Args:
			target_col: Tên cột target (mặc định dùng self.target_col)
			threshold: Ngưỡng |correlation| tối thiểu
			method: Phương pháp tính correlation ('pearson', 'spearman', 'kendall')
			
		Returns:
			Danh sách tên features thỏa mãn điều kiện
		"""
		if self.data is None:
			raise ValueError("Chưa có dữ liệu")
		
		target = target_col or self.target_col
		if target is None:
			raise ValueError("Chưa xác định target column")
		
		numeric_data = self.data.select_dtypes(include=['number'])
		if target not in numeric_data.columns:
			logger.warning(f"Target '{target}' không phải numeric, không thể tính correlation")
			return []
		
		corr_matrix = numeric_data.corr(method=method)
		if target not in corr_matrix.columns:
			return []
		
		corr_series = corr_matrix[target].drop(labels=[target], errors='ignore')
		selected = corr_series[abs(corr_series) >= threshold]
		
		features = selected.index.tolist()
		logger.info(f"🎯 {len(features)} features có |corr| >= {threshold}: {features}")
		return features

	def summary(self) -> Dict[str, Any]:
		"""Tóm tắt thông tin dữ liệu."""
		if self.data is None:
			return {"status": "No data loaded"}
		return {
			"shape": self.data.shape,
			"n_numeric_cols": len(self.numeric_cols),
			"n_categorical_cols": len(self.categorical_cols),
			"total_missing": int(self.data.isnull().sum().sum()),
			"preprocessing_steps": self.preprocessing_steps,
			"memory_usage_mb": float(self.data.memory_usage(deep=True).sum() / 1024 ** 2),
			"is_fitted": self._is_fitted,
		}

	def print_summary(self) -> None:
		"""In tóm tắt ra log."""
		info = self.summary()
		logger.info("%s", "=" * 70)
		logger.info("📊 TÓM TẮT DỮ LIỆU")
		logger.info("%s", info)
		logger.info("%s", "=" * 70)

	def save_state(self, path: Union[str, Path]) -> str:
		"""Lưu transformer state để dùng cho inference."""
		path = Path(path)
		path.parent.mkdir(parents=True, exist_ok=True)
		joblib.dump(self, path)
		logger.info("✅ Đã lưu transformer state tại %s", path)
		return str(path)

	@staticmethod
	def load_state(path: Union[str, Path]) -> "DataTransformer":
		"""Load transformer state đã lưu."""
		return joblib.load(Path(path))

	def __getstate__(self) -> Dict[str, Any]:
		state = self.__dict__.copy()
		state["visualizer"] = None
		state["_visualizer_state"] = self._visualizer_init_args
		return state

	def __setstate__(self, state: Dict[str, Any]) -> None:
		visualizer_state = state.pop("_visualizer_state", None)
		self.__dict__.update(state)
		viz_args = visualizer_state or {
			"target_col": None,
			"output_dir": EDA_RESULTS_DIR,
			"auto_save": True,
			"auto_show": False,
			"dpi": PLOT_DPI,
			"style": PLOT_STYLE,
			"figure_size": FIGURE_SIZE,
		}
		self.visualizer = DataVisualizer(data=self.data, **viz_args)
