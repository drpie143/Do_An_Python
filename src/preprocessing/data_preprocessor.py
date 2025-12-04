"""Comprehensive data preprocessing utilities for the Taxi Price Prediction project."""

from __future__ import annotations

import logging
import os
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


class DataPreprocessor:
	"""Data preparation pipeline inspired by the notebook EDA flow."""

	def __init__(
		self,
		data: Optional[pd.DataFrame] = None,
		missing_strategy: str = "mean",
		categorical_missing_strategy: str = "mode",
		scaler_type: str = "standard",
		encoder_type: str = "label",
	) -> None:
		self.data = data.copy() if data is not None else None
		self.original_data = data.copy() if data is not None else None
		self.numeric_cols: List[str] = []
		self.categorical_cols: List[str] = []
		self.types_: Dict[str, List[str]] = {}
		self.preprocessing_steps: List[str] = []

		self.missing_strategy = missing_strategy
		self.categorical_missing_strategy = categorical_missing_strategy
		
		self.missing_rules: Dict[str, MissingRule] = {}
		self.constraint_rules: Dict[str, Dict[str, Any]] = {}
		self.outlier_rules: Dict[str, str] = {}
		self.scaler_rules: Dict[str, str] = {}
		self.encoder_rules: Dict[str, str] = {}
		self._onehot_drop_first = True

		self.scalers: Dict[str, Any] = {}
		self.encoders: Dict[str, Any] = {}
		self.log: Dict[str, List[str]] = {}

		self.default_scaler = self._init_default_scaler(scaler_type)
		self.default_encoder = self._init_default_encoder(encoder_type)

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

		self.impute_values: Dict[str, Any] = {}
		self.onehot_feature_names: Dict[str, List[str]] = {}
		self.feature_columns: List[str] = []
		self._scale_exclude: List[str] = []
		self._scaling_method: Optional[str] = None
		self._encoding_method: Optional[str] = None
		self._is_fitted: bool = False
		self._capture_config: bool = True
		self._unify_applied: bool = False
		self._text_unify_rules: Optional[Dict[str, Dict[str, str]]] = None
		self._feature_engineering_config: Optional[Dict[str, Any]] = None
		self._datetime_feature_configs: List[Dict[str, Any]] = []
		self._datetime_config_keys: Set[Tuple[str, Tuple[str, ...], bool]] = set()
		self._interaction_configs: List[Dict[str, Any]] = []
		self._interaction_config_keys: Set[Tuple[Tuple[Tuple[str, str], ...], Tuple[str, ...]]] = set()

		if self.data is not None:
			self.detect_types()
			logger.info("✅ DataPreprocessor khởi tạo thành công - Shape: %s", self.data.shape)

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

	# ------------------------------------------------------------------
	# Data loading utilities
	# ------------------------------------------------------------------
	@staticmethod
	def load_data(filepath: Union[str, Path], **kwargs) -> pd.DataFrame:
		filepath = Path(filepath)
		if not filepath.exists():
			raise FileNotFoundError(f"File không tồn tại: {filepath}")

		ext = filepath.suffix.lower()
		try:
			if ext == ".csv":
				df = pd.read_csv(filepath, **kwargs)
			elif ext in {".xls", ".xlsx"}:
				df = pd.read_excel(filepath, **kwargs)
			elif ext == ".json":
				df = pd.read_json(filepath, **kwargs)
			else:
				raise ValueError(f"Định dạng file không được hỗ trợ: {ext}")
		except Exception as exc:
			logger.error("❌ Lỗi khi đọc file %s: %s", filepath, exc)
			raise

		logger.info("✅ Đọc dữ liệu thành công từ %s - Shape: %s", filepath, df.shape)
		return df

	def load(self, filepath: Union[str, Path], **kwargs) -> "DataPreprocessor":
		self.data = self.load_data(filepath, **kwargs)
		self.original_data = self.data.copy()
		self.detect_types()
		self._update_visualizer()
		return self

	@classmethod
	def from_file(cls, filepath: Union[str, Path], **kwargs) -> "DataPreprocessor":
		instance = cls()
		instance.load(filepath, **kwargs)
		return instance

	# ------------------------------------------------------------------
	# Profiling utilities
	# ------------------------------------------------------------------
	def detect_types(self) -> Dict[str, List[str]]:
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

	def eda_overview(self, top_n: int = 10) -> Dict[str, Any]:
		if self.data is None:
			raise ValueError("Chưa có dữ liệu")

		self.detect_types()
		df = self.data
		overview: Dict[str, Any] = {}

		logger.info("%s", "=" * 70)
		logger.info("📊 BÁO CÁO TỔNG QUAN DỮ LIỆU")
		logger.info("%s", "=" * 70)

		# ----------------------------------------------------------------------
		# 1) SHAPE - DUPLICATES - MEMORY - MISSING SNAPSHOT
		# ----------------------------------------------------------------------
		n_rows, n_cols = df.shape
		n_dups = df.duplicated().sum()
		memory_usage = df.memory_usage(deep=True).sum() / 1024 ** 2
		total_missing = int(df.isna().sum().sum())
		missing_pct = round((total_missing / (n_rows * n_cols)) * 100, 2) if n_rows and n_cols else 0.0

		overview["shape"] = (n_rows, n_cols)
		overview["duplicates"] = int(n_dups)
		overview["memory_mb"] = float(memory_usage)
		overview["missing_summary"] = {
			"total_missing": total_missing,
			"missing_pct": missing_pct,
		}

		logger.info("[1] Rows: %s | Columns: %s | Duplicates: %s", n_rows, n_cols, n_dups)
		logger.info("Memory usage: %.2f MB | Missing: %s (%.2f%%)", memory_usage, total_missing, missing_pct)
		if missing_pct >= 10:
			logger.warning("⚠ Tỉ lệ thiếu dữ liệu toàn bảng đang ở mức %.2f%%", missing_pct)

		# ----------------------------------------------------------------------
		# 2) DTYPES SUMMARY
		# ----------------------------------------------------------------------
		dtype_counts = df.dtypes.value_counts().to_dict()
		overview["dtype_counts"] = dtype_counts
		logger.info("[2] Dtypes summary: %s", dtype_counts)

		# ----------------------------------------------------------------------
		# 3) COLUMN METADATA
		# ----------------------------------------------------------------------
		info_df = pd.DataFrame(
			{
				"dtype": df.dtypes,
				"missing": df.isna().sum(),
				"missing_pct": (df.isna().sum() / len(df) * 100).round(2),
				"unique": df.nunique(),
				"constant": [df[col].nunique() <= 1 for col in df.columns],
			}
		)

		overview["column_profile"] = info_df
		logger.info(
			"[3] Top %s columns with highest missing pct:\n%s",
			top_n,
			info_df.sort_values("missing_pct", ascending=False).head(top_n),
		)

		high_missing = info_df[info_df["missing_pct"] > 50]
		if not high_missing.empty:
			logger.warning("⚠ Cột thiếu dữ liệu >50%%: %s", high_missing.index.tolist())
		constant_cols = info_df[info_df["constant"] == True]
		if not constant_cols.empty:
			logger.warning("⚠ Cột hằng số (unique=1): %s", constant_cols.index.tolist())

		# ----------------------------------------------------------------------
		# 4) THỐNG KÊ SỐ - DESCRIPTIVE + SKEWNESS
		# ----------------------------------------------------------------------
		num_cols = self.types_.get("numeric_data", [])
		numeric_summary = None

		if num_cols:
			desc = df[num_cols].describe().T
			desc["skewness"] = df[num_cols].skew()
			numeric_summary = desc[["mean", "std", "min", "50%", "max", "skewness"]]

			overview["numeric_summary"] = numeric_summary
			logger.info("[4] Numeric statistics with skewness:\n%s", numeric_summary)
			skewed_cols = numeric_summary[numeric_summary["skewness"].abs() > 1].index.tolist()
			if skewed_cols:
				logger.warning("⚠ Các cột có phân phối lệch mạnh (|skew| > 1): %s", skewed_cols)
		else:
			logger.info("[4] Không có cột số.")

		# ----------------------------------------------------------------------
		# 5) CATEGORICAL VALUE COUNTS
		# ----------------------------------------------------------------------
		cat_cols = self.types_.get("categorical_data", [])
		cat_summary = {}
		rare_categories: Dict[str, List[str]] = {}

		if cat_cols:
			for col in cat_cols:
				vc = df[col].value_counts(dropna=False).head(top_n)
				cat_summary[col] = vc
				logger.info("[5] Top %s values for column '%s':\n%s", top_n, col, vc)
				normalized = df[col].value_counts(normalize=True, dropna=False)
				rare = normalized[normalized < 0.05].index.tolist()
				if rare:
					rare_categories[col] = rare
		else:
			logger.info("[5] Không có cột phân loại.")

		overview["categorical_summary"] = cat_summary
		if rare_categories:
			overview["rare_categories"] = rare_categories
			logger.warning("⚠ Phát hiện giá trị hiếm (<5%%) ở các cột: %s", rare_categories)

		# ----------------------------------------------------------------------
		# 6) CORRELATION SNAPSHOT
		# ----------------------------------------------------------------------
		if num_cols and len(num_cols) > 1:
			corr_matrix = df[num_cols].corr(numeric_only=True).abs()
			if corr_matrix.notna().any().any():
				upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
				stacked = upper.stack().sort_values(ascending=False)
				if not stacked.empty:
					top_corr = stacked.head(top_n)
					overview["top_correlations"] = top_corr
					logger.info("[6] Top %s strong feature correlations:\n%s", top_n, top_corr)

		logger.info("=" * 70)
		return overview

	# ------------------------------------------------------------------
	# Cleaning steps
	# ------------------------------------------------------------------
	def apply_constraints(self) -> "DataPreprocessor":
		if not self.constraint_rules or self.data is None:
			return self

		df = self.data
		logger.info("%s", "=" * 70)
		logger.info("ÁP DỤNG %s RÀNG BUỘC DỮ LIỆU", len(self.constraint_rules))
		logger.info("%s", "=" * 70)

		for col, rule in self.constraint_rules.items():
			if col not in df.columns:
				self._log("constraints_warning", f"Cột {col} không tồn tại")
				continue
			if not isinstance(rule, dict):
				self._log("constraints_error", f"Rule cho cột {col} phải là dict")
				continue

			if "dtype" in rule:
				desired_type = rule["dtype"]
				converted = pd.to_numeric(df[col], errors="coerce") if desired_type in {"int", "float"} else df[col]
				mask_bad = converted.isna() & df[col].notna()
				if mask_bad.any():
					bad_rows = df.index[mask_bad].tolist()
					df = df.loc[~mask_bad]
					df[col] = converted.loc[df.index]
					logger.info("%s: Đã xóa các dòng %s do sai dtype", col, bad_rows)
				else:
					df[col] = converted

			min_val = rule.get("min", -float("inf"))
			max_val = rule.get("max", float("inf"))
			action = rule.get("action", "drop")
			mask_invalid = (df[col] < min_val) | (df[col] > max_val)
			n_invalid = int(mask_invalid.sum())
			if n_invalid == 0:
				continue

			if action == "drop":
				violating_rows = df.index[mask_invalid].tolist()
				df = df.loc[~mask_invalid]
				logger.info(
					"%s: Đã xóa các dòng %s vì vi phạm miền giá trị [%s, %s]",
					col,
					violating_rows,
					min_val,
					max_val,
				)
			elif action == "clip":
				df[col] = df[col].clip(lower=min_val, upper=max_val)
			elif action == "mean":
				df.loc[mask_invalid, col] = np.nan
			self._log("constraints_action", f"{col}: {n_invalid} dòng -> {action}")

		self.data = df.reset_index(drop=True)
		self._update_visualizer()
		return self

	def fill_missing(self, missing_rules: Optional[Dict[str, MissingRule]] = None, *, fit: bool = True) -> "DataPreprocessor":
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
	) -> "DataPreprocessor":
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

	def unify_values(self, text_rules: Optional[Dict[str, Dict[str, str]]] = None) -> "DataPreprocessor":
		if self.data is None:
			return self
		self.detect_types()
		cat_cols = self.types_.get("categorical_data", [])
		df = self.data
		if self._capture_config:
			self._unify_applied = True
			if text_rules:
				self._text_unify_rules = text_rules

		for col in cat_cols:
			try:
				df[col] = df[col].astype(str).str.lower().str.strip()
				if text_rules and col in text_rules:
					df[col] = df[col].replace(text_rules[col])
			except Exception as exc:
				self._log("unify_error", f"{col}: {exc}")

		self.data = df
		if self._capture_config:
			self.preprocessing_steps.append("unify_values")
		return self

	def handle_outliers(self, outlier_rules: Optional[Dict[str, str]] = None) -> "DataPreprocessor":
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

	def remove_outliers(self, method: str = "iqr", **kwargs) -> "DataPreprocessor":
		self.outlier_rules = {col: method for col in self.numeric_cols}
		return self.handle_outliers(outlier_rules=self.outlier_rules)

	# ------------------------------------------------------------------
	# Scaling and encoding
	# ------------------------------------------------------------------
	def scale(self, scaler_rules: Optional[Dict[str, str]] = None, *, fit: bool = True) -> "DataPreprocessor":
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
	) -> "DataPreprocessor":
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

	def encode(self, encoder_rules: Optional[Dict[str, str]] = None, *, fit: bool = True) -> "DataPreprocessor":
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
	) -> "DataPreprocessor":
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
	# Feature engineering
	# ------------------------------------------------------------------
	def feature_engineering(self, min_speed: float = 0.0, max_speed: float = 150.0) -> "DataPreprocessor":
		if self.data is None:
			return self
		df = self.data
		if self._capture_config:
			self._feature_engineering_config = {"min_speed": min_speed, "max_speed": max_speed}
		if {"Trip_Distance_km", "Trip_Duration_Minutes"}.issubset(df.columns):
			duration_hours = df["Trip_Duration_Minutes"] / 60
			df["Speed_kmh"] = np.where(duration_hours > 0, df["Trip_Distance_km"] / duration_hours, 0).round(2)
			mask_invalid = (df["Speed_kmh"] < min_speed) | (df["Speed_kmh"] > max_speed)
			if mask_invalid.any():
				logger.info("Speed_kmh: Drop %s dòng ngoài miền", int(mask_invalid.sum()))
				df = df.loc[~mask_invalid]
		self.data = df.reset_index(drop=True)
		self.detect_types()
		if self._capture_config:
			self.preprocessing_steps.append("feature_engineering")
		return self

	def create_datetime_features(
		self,
		datetime_col: str,
		features: Sequence[str] = ("hour", "day", "month", "dayofweek"),
		drop_original: bool = False,
	) -> "DataPreprocessor":
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
	) -> "DataPreprocessor":
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
		"""Chạy bộ EDA mặc định và tự động lưu vào results/eda."""
		if self.data is None:
			logger.warning("❌ Chưa có dữ liệu để tạo biểu đồ EDA")
			return

		self.detect_types()
		self.visualizer.set_data(self.data)
		self.visualizer.set_target(target_col)

		self.visualizer.plot_missing_heatmap()
		self.visualizer.plot_numerical(cols=numeric_cols or self.numeric_cols)
		self.visualizer.plot_categorical(cols=categorical_cols or self.categorical_cols)
		self.visualizer.plot_correlation_heatmap(method="spearman", annot=False, show=False)
		if target_col:
			self.visualizer.plot_relationship_with_target(cols=self.numeric_cols + self.categorical_cols)

	# ------------------------------------------------------------------
	# Pipeline orchestration
	# ------------------------------------------------------------------
	def run(self, target_col: Optional[str] = None, perform_eda: bool = True) -> "DataPreprocessor":
		logger.info("%s", "#" * 70)
		logger.info("🚀 BẮT ĐẦU DATA PIPELINE")
		logger.info("%s", "#" * 70)

		# --- BƯỚC 1: PRE-EDA (Khám phá dữ liệu thô) ---
		if perform_eda:
			logger.info("📊 Đang tạo báo cáo EDA dữ liệu thô (Pre-processing)...")
			self.print_summary()
			self.visualizer.output_dir = Path(EDA_RESULTS_DIR) / "raw"
			self.generate_eda_report(target_col=target_col)

		# --- BƯỚC 2: XỬ LÝ DỮ LIỆU (PIPELINE) ---
		self.detect_types()
		self.apply_constraints()
		self.fill_missing()
		self.unify_values()
		self.feature_engineering()
		self.handle_outliers()
		self.scale()
		self.encode()

		# --- BƯỚC 3: POST-EDA (Kiểm tra lại dữ liệu sạch) ---
		if perform_eda:
			logger.info("📊 Đang tạo báo cáo EDA dữ liệu sạch (Post-processing)...")
			self._update_visualizer()
			self.visualizer.output_dir = Path(EDA_RESULTS_DIR) / "processed"
			self.plot_correlation_heatmap(target_col=target_col, annot=False)
			self.print_summary()

		logger.info("✅ PIPELINE HOÀN TẤT - Steps: %s", self.preprocessing_steps)
		return self

	# ------------------------------------------------------------------
	# Persistence & reporting
	# ------------------------------------------------------------------
	def save(self, path: Union[str, Path], index: bool = False) -> None:
		if self.data is None:
			logger.warning("❌ Không có dữ liệu để lưu")
			return
		path = Path(path)
		path.parent.mkdir(parents=True, exist_ok=True)
		ext = path.suffix.lower()
		if ext == ".csv":
			self.data.to_csv(path, index=index)
		elif ext in {".xls", ".xlsx"}:
			self.data.to_excel(path, index=index)
		elif ext == ".json":
			self.data.to_json(path, orient="records", indent=4)
		else:
			raise ValueError(f"Định dạng '{ext}' không được hỗ trợ")
		logger.info("✅ Đã lưu dữ liệu: %s", path.resolve())

	def save_data(self, filepath: str, index: bool = False, **kwargs) -> None:
		self.save(filepath, index=index)

	def save_state(self, path: Union[str, Path]) -> str:
		"""Persist the fitted preprocessor so inference pipelines can reload it."""
		path = Path(path)
		path.parent.mkdir(parents=True, exist_ok=True)
		joblib.dump(self, path)
		logger.info("✅ Đã lưu cấu hình tiền xử lý tại %s", path)
		return str(path)

	@staticmethod
	def load_state(path: Union[str, Path]) -> "DataPreprocessor":
		"""Load a previously saved preprocessor state."""
		return joblib.load(Path(path))

	def get_processed_data(self) -> pd.DataFrame:
		if self.data is None:
			raise ValueError("Chưa có dữ liệu")
		return self.data.copy()

	def summary(self) -> Dict[str, Any]:
		if self.data is None:
			return {"status": "No data loaded"}
		return {
			"shape": self.data.shape,
			"n_numeric_cols": len(self.numeric_cols),
			"n_categorical_cols": len(self.categorical_cols),
			"total_missing": int(self.data.isnull().sum().sum()),
			"preprocessing_steps": self.preprocessing_steps,
			"memory_usage_mb": float(self.data.memory_usage(deep=True).sum() / 1024 ** 2),
		}

	def print_summary(self) -> None:
		info = self.summary()
		logger.info("%s", "=" * 70)
		logger.info("📊 TÓM TẮT DỮ LIỆU")
		logger.info("%s", info)
		logger.info("%s", "=" * 70)

	def mark_as_fitted(self) -> None:
		if self.data is None:
			return
		self.feature_columns = self.data.columns.tolist()
		self._is_fitted = True

	def transform_new_data(self, df: pd.DataFrame) -> pd.DataFrame:
		if not self._is_fitted:
			raise ValueError("Preprocessor chưa được fit trên dữ liệu train")
		original_data = self.data
		original_numeric = self.numeric_cols.copy()
		original_categorical = self.categorical_cols.copy()
		try:
			self._capture_config = False
			self.data = df.copy()
			self.detect_types()
			self.apply_constraints()
			if self._unify_applied or self._text_unify_rules:
				self.unify_values(text_rules=self._text_unify_rules)
			if self._datetime_feature_configs:
				for config in self._datetime_feature_configs:
					self.create_datetime_features(
						datetime_col=config["datetime_col"],
						features=config["features"],
						drop_original=config["drop_original"],
					)
			if self._feature_engineering_config:
				self.feature_engineering(**self._feature_engineering_config)
			self.fill_missing(fit=False)
			encoding_method = self._encoding_method or "onehot"
			self.encode_categorical(method=encoding_method, fit=False)
			exclude_cols = self._scale_exclude
			self.scale_features(method=self._scaling_method or "standard", exclude_columns=exclude_cols, fit=False)
			if self._interaction_configs:
				for config in self._interaction_configs:
					self.create_interaction_features(
						col_pairs=[tuple(pair) for pair in config["col_pairs"]],
						operations=config["operations"],
					)
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

