"""Comprehensive data preprocessing utilities for the Taxi Price Prediction project."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
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


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
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

		self.visualizer = DataVisualizer(
			data=self.data,
			target_col=None,
			output_dir=EDA_RESULTS_DIR,
			auto_save=True,
			auto_show=False,
			dpi=PLOT_DPI,
			style=PLOT_STYLE,
			figure_size=FIGURE_SIZE,
		)

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

		n_rows, n_cols = df.shape
		n_dups = df.duplicated().sum()
		memory_usage = df.memory_usage(deep=True).sum() / 1024 ** 2
		overview["shape"] = (n_rows, n_cols)
		overview["duplicates"] = int(n_dups)
		overview["memory_mb"] = float(memory_usage)

		logger.info("Rows: %s | Columns: %s | Duplicates: %s", n_rows, n_cols, n_dups)
		logger.info("Memory usage: %.2f MB", memory_usage)

		dtype_counts = df.dtypes.value_counts().to_dict()
		overview["dtype_counts"] = dtype_counts
		logger.info("Dtypes summary: %s", dtype_counts)

		info_df = pd.DataFrame(
			{
				"dtype": df.dtypes,
				"missing": df.isna().sum(),
				"missing_pct": (df.isna().sum() / len(df) * 100).round(2),
				"unique": df.nunique(),
			}
		)
		overview["column_profile"] = info_df
		logger.info("Top columns with missing data:\n%s", info_df.sort_values("missing_pct", ascending=False).head(top_n))

		return overview

	def check_missing(self) -> Optional[pd.DataFrame]:
		if self.data is None:
			logger.warning("❌ Chưa load dữ liệu")
			return None

		missing_count = self.data.isnull().sum()
		missing_percent = (missing_count / len(self.data)) * 100
		missing_df = pd.DataFrame(
			{
				"Column": missing_count.index,
				"Missing_Count": missing_count.values,
				"Missing_Percent": missing_percent.values,
			}
		)
		missing_df = missing_df[missing_df["Missing_Count"] > 0].sort_values("Missing_Count", ascending=False)
		if not missing_df.empty:
			logger.info("⚠️  Phát hiện %s cột có missing values", len(missing_df))
		else:
			logger.info("✅ Không có missing values")
		return missing_df

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
					df = df.loc[~mask_bad]
					df[col] = converted.loc[df.index]
					logger.info("%s: Đã loại bỏ %s dòng sai kiểu", col, mask_bad.sum())
				else:
					df[col] = converted

			min_val = rule.get("min", -float("inf"))
			max_val = rule.get("max", float("inf"))
			action = rule.get("action", "drop")
			mask_invalid = (df[col] < min_val) | (df[col] > max_val)
			n_invalid = int(mask_invalid.sum())
			if n_invalid == 0:
				continue

			logger.info("%s: %s dòng vi phạm miền giá trị", col, n_invalid)
			if action == "drop":
				df = df.loc[~mask_invalid]
			elif action == "clip":
				df[col] = df[col].clip(lower=min_val, upper=max_val)
			elif action == "mean":
				df.loc[mask_invalid, col] = np.nan
			self._log("constraints_action", f"{col}: {n_invalid} dòng -> {action}")

		self.data = df.reset_index(drop=True)
		self._update_visualizer()
		return self

	def fill_missing(self, missing_rules: Optional[Dict[str, MissingRule]] = None) -> "DataPreprocessor":
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
					before = len(df)
					df = df.dropna(subset=[col])
					logger.info("%s: Drop %s dòng", col, before - len(df))
				elif strategy == "mean" and pd.api.types.is_numeric_dtype(df[col]):
					df[col] = df[col].fillna(df[col].mean())
				elif strategy == "median" and pd.api.types.is_numeric_dtype(df[col]):
					df[col] = df[col].fillna(df[col].median())
				elif strategy == "mode":
					mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else custom_value
					df[col] = df[col].fillna(mode_val)
				elif strategy == "constant":
					fill_val = custom_value
					if fill_val is None:
						fill_val = 0 if pd.api.types.is_numeric_dtype(df[col]) else "Unknown"
					df[col] = df[col].fillna(fill_val)
				elif strategy == "ffill":
					df[col] = df[col].ffill()
				elif strategy == "bfill":
					df[col] = df[col].bfill()
				else:
					# fallback
					default_value = df[col].median() if pd.api.types.is_numeric_dtype(df[col]) else df[col].mode().iloc[0]
					df[col] = df[col].fillna(default_value)
			except Exception as exc:
				self._log("missing_error", f"{col}: {exc}")
				logger.error("Lỗi xử lý missing cho %s: %s", col, exc)

		self.data = df.reset_index(drop=True)
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

		for col in cat_cols:
			try:
				df[col] = df[col].astype(str).str.lower().str.strip()
				if text_rules and col in text_rules:
					df[col] = df[col].replace(text_rules[col])
			except Exception as exc:
				self._log("unify_error", f"{col}: {exc}")

		self.data = df
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
	def scale(self, scaler_rules: Optional[Dict[str, str]] = None) -> "DataPreprocessor":
		if self.data is None:
			return self
		self.detect_types()
		num_cols = self.types_.get("numeric_data", [])
		if not num_cols:
			return self

		if scaler_rules:
			self.scaler_rules.update(scaler_rules)

		df = self.data
		scaled_cols = set()
		for col, method in self.scaler_rules.items():
			if col not in num_cols:
				continue
			scaler = self._init_default_scaler(method)
			if scaler is None:
				continue
			df[[col]] = scaler.fit_transform(df[[col]])
			self.scalers[col] = scaler
			scaled_cols.add(col)

		remaining_cols = [col for col in num_cols if col not in scaled_cols]
		if self.default_scaler and remaining_cols:
			for col in remaining_cols:
				scaler = clone(self.default_scaler)
				df[[col]] = scaler.fit_transform(df[[col]])
				self.scalers[col] = scaler

		self.data = df
		self.preprocessing_steps.append("scale")
		return self

	def scale_features(
		self,
		method: str = "standard",
		columns: Optional[List[str]] = None,
		exclude_columns: Optional[List[str]] = None,
	) -> "DataPreprocessor":
		self.detect_types()
		target_cols = columns or self.types_.get("numeric_data", [])
		if exclude_columns:
			target_cols = [col for col in target_cols if col not in exclude_columns]
		self.scaler_rules = {col: method for col in target_cols}
		self.default_scaler = None
		return self.scale()

	def encode(self, encoder_rules: Optional[Dict[str, str]] = None) -> "DataPreprocessor":
		if self.data is None:
			return self
		self.detect_types()
		cat_cols = self.types_.get("categorical_data", [])
		if not cat_cols:
			return self
		if encoder_rules:
			self.encoder_rules.update(encoder_rules)

		df = self.data
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
					encoder = LabelEncoder()
					df[col] = encoder.fit_transform(df[col].astype(str))
					self.encoders[col] = encoder
				elif method == "ordinal":
					encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
					df[[col]] = encoder.fit_transform(df[[col]].astype(str))
					self.encoders[col] = encoder
				elif method == "onehot":
					encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore", dtype=int)
					matrix = encoder.fit_transform(df[[col]].astype(str))
					cols = encoder.get_feature_names_out([col])
					temp_df = pd.DataFrame(matrix, columns=cols, index=df.index)
					if self._onehot_drop_first and temp_df.shape[1] > 1:
						temp_df = temp_df.iloc[:, 1:]
					onehot_frames.append(temp_df)
					cols_to_drop.append(col)
					self.encoders[col] = encoder
			except Exception as exc:
				self._log("encode_error", f"{col}: {exc}")

		if onehot_frames:
			df = pd.concat([df] + onehot_frames, axis=1)
			df.drop(columns=cols_to_drop, inplace=True)

		self.data = df
		self.preprocessing_steps.append("encode")
		return self

	def encode_categorical(
		self,
		method: str = "onehot",
		columns: Optional[List[str]] = None,
		drop_first: bool = True,
	) -> "DataPreprocessor":
		self._onehot_drop_first = drop_first
		if method == "onehot" and columns:
			self.encoder_rules = {col: "onehot" for col in columns}
		elif method == "label" and columns:
			self.encoder_rules = {col: "label" for col in columns}
		self.default_encoder = self._init_default_encoder(method)
		return self.encode()

	# ------------------------------------------------------------------
	# Feature engineering
	# ------------------------------------------------------------------
	def feature_engineering(self, min_speed: float = 0.0, max_speed: float = 150.0) -> "DataPreprocessor":
		if self.data is None:
			return self
		df = self.data
		if {"Trip_Distance_km", "Trip_Duration_Minutes"}.issubset(df.columns):
			duration_hours = df["Trip_Duration_Minutes"] / 60
			df["Speed_kmh"] = np.where(duration_hours > 0, df["Trip_Distance_km"] / duration_hours, 0).round(2)
			mask_invalid = (df["Speed_kmh"] < min_speed) | (df["Speed_kmh"] > max_speed)
			if mask_invalid.any():
				logger.info("Speed_kmh: Drop %s dòng ngoài miền", int(mask_invalid.sum()))
				df = df.loc[~mask_invalid]
		self.data = df.reset_index(drop=True)
		self.detect_types()
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
	def run(self, target_col: Optional[str] = None) -> "DataPreprocessor":
		logger.info("%s", "#" * 70)
		logger.info("BẮT ĐẦU DATA PIPELINE")
		logger.info("%s", "#" * 70)
		self.detect_types()
		self.apply_constraints()
		self.fill_missing()
		self.unify_values()
		self.feature_engineering()
		self.handle_outliers()
		self.scale()
		self.encode()
		logger.info("PIPELINE HOÀN TẤT - Steps: %s", self.preprocessing_steps)
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

