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
from config import EDA_RESULTS_DIR, PLOT_DPI, PLOT_STYLE, FIGURE_SIZE, SPEED_FEATURE


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
		self.outlier_rules: Dict[str, Union[str, Dict[str, Any]]] = {}
		self.scaler_rules: Dict[str, str] = {}
		self.encoder_rules: Dict[str, str] = {}
		self._onehot_drop_first = True
		self.impute_values: Dict[str, Any] = {}
		self.outlier_boundaries: Dict[str, Dict[str, Any]] = {}
		self.outlier_models: Dict[str, Any] = {}
		self.processed_columns: List[str] = []

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
		self.impute_values = {}
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
					fill_val = df[col].mean()
					df[col] = df[col].fillna(fill_val)
					self.impute_values[col] = fill_val
				elif strategy == "median" and pd.api.types.is_numeric_dtype(df[col]):
					fill_val = df[col].median()
					df[col] = df[col].fillna(fill_val)
					self.impute_values[col] = fill_val
				elif strategy == "mode":
					mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else custom_value
					df[col] = df[col].fillna(mode_val)
					self.impute_values[col] = mode_val
				elif strategy == "constant":
					fill_val = custom_value
					if fill_val is None:
						fill_val = 0 if pd.api.types.is_numeric_dtype(df[col]) else "Unknown"
					df[col] = df[col].fillna(fill_val)
					self.impute_values[col] = fill_val
				elif strategy == "ffill":
					df[col] = df[col].ffill()
				elif strategy == "bfill":
					df[col] = df[col].bfill()
				else:
					# fallback
					default_value = df[col].median() if pd.api.types.is_numeric_dtype(df[col]) else df[col].mode().iloc[0]
					df[col] = df[col].fillna(default_value)
					self.impute_values[col] = default_value
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

	def handle_outliers(self, outlier_rules: Optional[Dict[str, Union[str, Dict[str, Any]]]] = None) -> "DataPreprocessor":
		if self.data is None:
			return self
		rules = outlier_rules or self.outlier_rules
		if not rules:
			return self

		df = self.data
		for col, rule in rules.items():
			if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
				continue
			if isinstance(rule, dict):
				method = str(rule.get("method", "iqr")).lower()
				threshold = rule.get("threshold")
				extra_params = rule.get("params") or {}
			else:
				method = str(rule).lower()
				threshold = None
				extra_params = {}
			mask_outlier = None

			try:
				if method == "iqr":
					q1 = df[col].quantile(0.25)
					q3 = df[col].quantile(0.75)
					iqr = q3 - q1
					multiplier = threshold if threshold is not None else 1.5
					lower = q1 - multiplier * iqr
					upper = q3 + multiplier * iqr
					mask_outlier = (df[col] < lower) | (df[col] > upper)
					self.outlier_boundaries[col] = {
						"method": "iqr",
						"lower": lower,
						"upper": upper,
					}
				elif method == "zscore":
					mean_val = df[col].mean()
					std_val = df[col].std(ddof=0)
					z_scores = np.abs((df[col] - mean_val) / std_val)
					z_limit = threshold if threshold is not None else 3.0
					mask_outlier = z_scores > z_limit
					self.outlier_boundaries[col] = {
						"method": "zscore",
						"mean": mean_val,
						"std": std_val,
						"limit": z_limit,
					}
				elif method in {"iforest", "isolation_forest"}:
					contamination = extra_params.get("contamination")
					if contamination is None and threshold is not None:
						contamination = threshold
					clf = IsolationForest(random_state=42, contamination=contamination or "auto")
					preds = clf.fit_predict(df[[col]].fillna(df[col].mean()))
					mask_outlier = preds == -1
					self.outlier_boundaries[col] = {"method": "iforest"}
					self.outlier_models[col] = clf
				else:
					logger.warning("Method %s không hợp lệ cho cột %s", method, col)
					continue

				if mask_outlier is not None and mask_outlier.any():
					default_threshold = {"iqr": 1.5, "zscore": 3.0}.get(method, "auto")
					threshold_display = threshold if threshold is not None else default_threshold
					if method in {"iforest", "isolation_forest"}:
						threshold_display = contamination or "auto"
					logger.info(
						"%s: Drop %s outliers (%s, threshold=%s)",
						col,
						int(mask_outlier.sum()),
						method,
						threshold_display,
					)
					df = df.loc[~mask_outlier]
			except Exception as exc:
				self._log("outlier_error", f"{col}: {exc}")

		self.data = df.reset_index(drop=True)
		self._update_visualizer()
		self.preprocessing_steps.append("handle_outliers")
		return self

	def remove_outliers(self, method: str = "iqr", threshold: Optional[float] = None, **kwargs) -> "DataPreprocessor":
		self.outlier_rules = {
			col: {"method": method, "threshold": threshold, "params": kwargs}
			for col in self.numeric_cols
		}
		self.outlier_boundaries = {}
		self.outlier_models = {}
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
		logger.info(
			"⚙️  SCALE FEATURES | Tổng cột số: %s | Có rule riêng: %s",
			len(num_cols),
			len(self.scaler_rules),
		)
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
			logger.info("   • %s: dùng scaler %s", col, scaler.__class__.__name__)

		remaining_cols = [col for col in num_cols if col not in scaled_cols]
		if self.default_scaler and remaining_cols:
			logger.info(
				"   • %s cột còn lại dùng scaler mặc định %s",
				len(remaining_cols),
				self.default_scaler.__class__.__name__,
			)
			for col in remaining_cols:
				scaler = clone(self.default_scaler)
				df[[col]] = scaler.fit_transform(df[[col]])
				self.scalers[col] = scaler

		self.data = df
		logger.info("✅ Scale hoàn tất - tổng cột đã scale: %s", len(self.scalers))
		self.preprocessing_steps.append("scale")
		self.processed_columns = df.columns.tolist()
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

	def transform_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
		"""Áp dụng lại các bước đã fit trên dataset mới (ví dụ: test set)."""
		if self.data is None:
			raise ValueError("Cần fit DataPreprocessor trên tập train trước khi transform")
		if not self.processed_columns:
			self.processed_columns = self.data.columns.tolist()

		transformed = df.copy()
		for col, fill_val in self.impute_values.items():
			if col in transformed.columns:
				transformed[col] = transformed[col].fillna(fill_val)
			else:
				transformed[col] = fill_val

		if self.outlier_boundaries:
			keep_mask = pd.Series(True, index=transformed.index)
			for col, info in self.outlier_boundaries.items():
				if col not in transformed.columns:
					continue
				method = info.get("method")
				col_mask = pd.Series(False, index=transformed.index)
				if method == "iqr":
					lower = info.get("lower", -np.inf)
					upper = info.get("upper", np.inf)
					col_mask = (transformed[col] < lower) | (transformed[col] > upper)
				elif method == "zscore":
					std_val = info.get("std", 0)
					if std_val and std_val != 0:
						mean_val = info.get("mean", 0)
						limit = info.get("limit", 3.0)
						z_scores = np.abs((transformed[col] - mean_val) / std_val)
						col_mask = z_scores > limit
				elif method in {"iforest", "isolation_forest"}:
					model = self.outlier_models.get(col)
					if model is not None:
						filled = transformed[[col]].fillna(self.impute_values.get(col, transformed[col].mean()))
						preds = model.predict(filled)
						col_mask = preds == -1
				keep_mask &= ~col_mask
			if (~keep_mask).any():
				logger.info("Test set: drop %s dòng do vi phạm outlier rule", int((~keep_mask).sum()))
				transformed = transformed.loc[keep_mask]

		onehot_frames: List[pd.DataFrame] = []
		cols_to_drop: List[str] = []
		for col, encoder in self.encoders.items():
			if isinstance(encoder, LabelEncoder):
				if col not in transformed.columns:
					transformed[col] = encoder.classes_[0]
				series = transformed[col].astype(str)
				unknown_mask = ~series.isin(encoder.classes_)
				if unknown_mask.any():
					series.loc[unknown_mask] = encoder.classes_[0]
				transformed[col] = encoder.transform(series)
			elif isinstance(encoder, OrdinalEncoder):
				if col not in transformed.columns:
					transformed[col] = ""
				transformed[[col]] = encoder.transform(transformed[[col]].astype(str))
			elif isinstance(encoder, OneHotEncoder):
				if col not in transformed.columns:
					transformed[col] = ""
				matrix = encoder.transform(transformed[[col]].astype(str))
				cols = encoder.get_feature_names_out([col])
				temp_df = pd.DataFrame(matrix, columns=cols, index=transformed.index)
				if self._onehot_drop_first and temp_df.shape[1] > 1:
					temp_df = temp_df.iloc[:, 1:]
				temp_df = temp_df.astype(int)
				onehot_frames.append(temp_df)
				cols_to_drop.append(col)

		if onehot_frames:
			transformed = pd.concat([transformed] + onehot_frames, axis=1)
			transformed.drop(columns=cols_to_drop, inplace=True)

		for col, scaler in self.scalers.items():
			if col not in transformed.columns:
				transformed[col] = 0.0
			try:
				transformed[[col]] = scaler.transform(transformed[[col]])
			except Exception as exc:
				logger.warning("Không thể scale cột %s cho tập transform: %s", col, exc)

		for col in self.processed_columns:
			if col not in transformed.columns:
				transformed[col] = 0
		extra_cols = [col for col in transformed.columns if col not in self.processed_columns]
		if extra_cols:
			transformed = transformed.drop(columns=extra_cols)

		return transformed[self.processed_columns].reset_index(drop=True)

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
					encoder = LabelEncoder()
					df[col] = encoder.fit_transform(df[col].astype(str))
					self.encoders[col] = encoder
					logger.info("   • %s: LabelEncoder -> %s lớp", col, len(encoder.classes_))
				elif method == "ordinal":
					encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
					df[[col]] = encoder.fit_transform(df[[col]].astype(str))
					self.encoders[col] = encoder
					logger.info("   • %s: OrdinalEncoder", col)
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
					logger.info("   • %s: OneHotEncoder -> thêm %s cột", col, temp_df.shape[1])
			except Exception as exc:
				self._log("encode_error", f"{col}: {exc}")

		if onehot_frames:
			df = pd.concat([df] + onehot_frames, axis=1)
			df.drop(columns=cols_to_drop, inplace=True)
			logger.info("   • Tổng cột one-hot mới: %s", sum(frame.shape[1] for frame in onehot_frames))

		self.data = df
		logger.info("✅ Encoding hoàn tất - shape mới: %s", df.shape)
		self.preprocessing_steps.append("encode")
		self.processed_columns = df.columns.tolist()
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
	def feature_engineering(self, settings: Optional[Dict[str, Any]] = None) -> "DataPreprocessor":
		if self.data is None:
			return self

		speed_settings = settings or SPEED_FEATURE or {}
		if not speed_settings.get("enabled", False):
			logger.info("⚙️  Bỏ qua feature_engineering vì SPEED_FEATURE.disabled")
			return self

		distance_col = speed_settings.get("distance_col", "Trip_Distance_km")
		duration_col = speed_settings.get("duration_col", "Trip_Duration_Minutes")
		feature_name = speed_settings.get("name", "Speed_kmh")
		min_duration = max(float(speed_settings.get("min_duration_minutes", 1.0)), 1e-6)
		round_digits = speed_settings.get("round_digits")

		df = self.data
		missing_cols = [col for col in (distance_col, duration_col) if col not in df.columns]
		if missing_cols:
			logger.warning("⚠️  Không thể tạo %s vì thiếu cột: %s", feature_name, missing_cols)
			return self

		logger.info(
			"🚗 Feature engineering: %s = %s / (%s / 60) | min_duration=%.2f phút",
			feature_name,
			distance_col,
			duration_col,
			min_duration,
		)

		distance_series = pd.to_numeric(df[distance_col], errors="coerce")
		duration_series = pd.to_numeric(df[duration_col], errors="coerce")
		invalid_duration_mask = duration_series < min_duration
		if invalid_duration_mask.any():
			logger.warning(
				"⚠️  %s dòng có %s < %.2f phút -> clip về ngưỡng an toàn",
				int(invalid_duration_mask.sum()),
				duration_col,
				min_duration,
			)
			duration_series = duration_series.clip(lower=min_duration)

		speed_series = distance_series / (duration_series / 60.0)
		if isinstance(round_digits, int):
			speed_series = speed_series.round(round_digits)

		df[feature_name] = speed_series
		if self.original_data is not None:
			self.original_data[feature_name] = speed_series

		stats = speed_series.describe()
		logger.info(
			"✅ %s tạo xong | min=%.2f | max=%.2f | mean=%.2f | non-null=%s",
			feature_name,
			stats.get("min", float("nan")),
			stats.get("max", float("nan")),
			stats.get("mean", float("nan")),
			int(speed_series.notna().sum()),
		)

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

	def get_processed_data(self) -> pd.DataFrame:
		if self.data is None:
			raise ValueError("Chưa có dữ liệu")
		if not self.processed_columns:
			self.processed_columns = self.data.columns.tolist()
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

