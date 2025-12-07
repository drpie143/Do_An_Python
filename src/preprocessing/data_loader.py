"""
Data loading và pre-split cleaning utilities.

Module này xử lý các bước TRƯỚC khi chia train/test:
- Load data từ file (CSV, Excel, JSON)
- Detect column types
- EDA overview & visualization
- Xóa duplicates, unify text values
- Apply constraints cơ bản (clip, drop invalid rows)
- Feature engineering đơn giản (không dùng thống kê)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from src.visualization import DataVisualizer
from config import EDA_RESULTS_DIR, PLOT_DPI, PLOT_STYLE, FIGURE_SIZE


logger = logging.getLogger(__name__)


class DataLoader:
	"""Load và xử lý nhẹ dữ liệu trước khi split train/test."""

	def __init__(
		self,
		data: Optional[pd.DataFrame] = None,
	) -> None:
		"""
		Khởi tạo DataLoader.
		
		Args:
			data: DataFrame ban đầu (optional)
		"""
		self.data = data.copy() if data is not None else None
		self.original_data = data.copy() if data is not None else None
		self.numeric_cols: List[str] = []
		self.categorical_cols: List[str] = []
		self.types_: Dict[str, List[str]] = {}
		self.preprocessing_steps: List[str] = []
		self.log: Dict[str, List[str]] = {}
		
		# Constraint rules
		self.constraint_rules: Dict[str, Dict[str, Any]] = {}
		
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
			logger.info("✅ DataLoader khởi tạo thành công - Shape: %s", self.data.shape)

	# ------------------------------------------------------------------
	# Internal helpers
	# ------------------------------------------------------------------
	def _update_visualizer(self) -> None:
		self.visualizer.set_data(self.data)

	def _log(self, key: str, message: str) -> None:
		self.log.setdefault(key, []).append(message)

	# ------------------------------------------------------------------
	# Data loading utilities
	# ------------------------------------------------------------------
	@staticmethod
	def load_data(filepath: Union[str, Path], **kwargs) -> pd.DataFrame:
		"""Load data từ file CSV, Excel hoặc JSON."""
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

	def load(self, filepath: Union[str, Path], **kwargs) -> "DataLoader":
		"""Load data và cập nhật instance."""
		self.data = self.load_data(filepath, **kwargs)
		self.original_data = self.data.copy()
		self.detect_types()
		self._update_visualizer()
		return self

	@classmethod
	def from_file(cls, filepath: Union[str, Path], **kwargs) -> "DataLoader":
		"""Tạo instance mới từ file."""
		instance = cls()
		instance.load(filepath, **kwargs)
		return instance

	# ------------------------------------------------------------------
	# Profiling utilities
	# ------------------------------------------------------------------
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

	def eda_overview(self, top_n: int = 10) -> Dict[str, Any]:
		"""Tạo báo cáo tổng quan EDA."""
		if self.data is None:
			raise ValueError("Chưa có dữ liệu")

		self.detect_types()
		df = self.data
		overview: Dict[str, Any] = {}

		logger.info("%s", "=" * 70)
		logger.info("📊 BÁO CÁO TỔNG QUAN DỮ LIỆU")
		logger.info("%s", "=" * 70)

		# 1) SHAPE - DUPLICATES - MEMORY - MISSING SNAPSHOT
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

		# 2) DTYPES SUMMARY
		dtype_counts = df.dtypes.value_counts().to_dict()
		overview["dtype_counts"] = dtype_counts
		logger.info("[2] Dtypes summary: %s", dtype_counts)

		# 3) COLUMN METADATA
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

		# 4) THỐNG KÊ SỐ - DESCRIPTIVE + SKEWNESS
		num_cols = self.types_.get("numeric_data", [])
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

		# 5) CATEGORICAL VALUE COUNTS
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

		# 6) CORRELATION SNAPSHOT
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
	# Pre-split cleaning (không dùng thống kê để tránh data leakage)
	# ------------------------------------------------------------------
	def drop_duplicates(self) -> "DataLoader":
		"""Xóa các dòng trùng lặp."""
		if self.data is None:
			return self
		n_before = len(self.data)
		self.data = self.data.drop_duplicates().reset_index(drop=True)
		n_dropped = n_before - len(self.data)
		if n_dropped > 0:
			logger.info("🗑️ Đã xóa %s dòng trùng lặp", n_dropped)
			self.preprocessing_steps.append("drop_duplicates")
		self._update_visualizer()
		return self

	def unify_values(self, text_rules: Optional[Dict[str, Dict[str, str]]] = None) -> "DataLoader":
		"""Thống nhất text values (lowercase, strip)."""
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
		logger.info("🔤 Đã thống nhất text values (lowercase, strip)")
		return self

	def apply_constraints(self, constraint_rules: Optional[Dict[str, Dict[str, Any]]] = None) -> "DataLoader":
		"""
		Áp dụng ràng buộc dữ liệu (clip, drop).
		
		CHỈ dùng các action không cần thống kê: clip, drop
		Không dùng mean/median vì sẽ gây data leakage
		"""
		if constraint_rules:
			self.constraint_rules = constraint_rules
		if not self.constraint_rules or self.data is None:
			return self

		df = self.data
		logger.info("📏 Áp dụng %s ràng buộc dữ liệu", len(self.constraint_rules))

		for col, rule in self.constraint_rules.items():
			if col not in df.columns:
				continue
			if not isinstance(rule, dict):
				continue

			# Chuyển đổi dtype nếu cần
			if "dtype" in rule:
				desired_type = rule["dtype"]
				if desired_type in {"int", "float"}:
					converted = pd.to_numeric(df[col], errors="coerce")
					mask_bad = converted.isna() & df[col].notna()
					if mask_bad.any():
						df = df.loc[~mask_bad]
						df[col] = converted.loc[df.index]
					else:
						df[col] = converted

			min_val = rule.get("min")
			max_val = rule.get("max")
			action = rule.get("action", "drop")

			if min_val is None and max_val is None:
				continue

			# Chỉ xử lý clip và drop (không dùng mean/median)
			if action == "clip":
				df[col] = df[col].clip(lower=min_val, upper=max_val)
				logger.info("   • %s: clip to [%s, %s]", col, min_val, max_val)
			elif action == "drop":
				mask = pd.Series(True, index=df.index)
				if min_val is not None:
					mask &= df[col] >= min_val
				if max_val is not None:
					mask &= df[col] <= max_val
				n_dropped = int((~mask).sum())
				if n_dropped > 0:
					df = df.loc[mask]
					logger.info("   • %s: dropped %s rows outside [%s, %s]", col, n_dropped, min_val, max_val)

		self.data = df.reset_index(drop=True)
		self.preprocessing_steps.append("apply_constraints")
		self._update_visualizer()
		return self

	def feature_engineering(self, min_speed: float = 0.0, max_speed: float = 150.0) -> "DataLoader":
		"""Tạo feature Speed_kmh từ distance và duration."""
		if self.data is None:
			return self
		df = self.data
		
		if {"Trip_Distance_km", "Trip_Duration_Minutes"}.issubset(df.columns):
			duration_hours = df["Trip_Duration_Minutes"] / 60
			df["Speed_kmh"] = np.where(duration_hours > 0, df["Trip_Distance_km"] / duration_hours, 0).round(2)
			mask_invalid = (df["Speed_kmh"] < min_speed) | (df["Speed_kmh"] > max_speed)
			if mask_invalid.any():
				logger.info("🏎️ Speed_kmh: Drop %s dòng ngoài miền [%s, %s]", int(mask_invalid.sum()), min_speed, max_speed)
				df = df.loc[~mask_invalid]
			self.data = df.reset_index(drop=True)
			self.detect_types()
			self.preprocessing_steps.append("feature_engineering")
			logger.info("🏎️ Đã tạo feature Speed_kmh")
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
	# Reporting
	# ------------------------------------------------------------------
	def get_data(self) -> pd.DataFrame:
		"""Lấy dữ liệu đã xử lý."""
		if self.data is None:
			raise ValueError("Chưa có dữ liệu")
		return self.data.copy()

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
		}

	def print_summary(self) -> None:
		"""In tóm tắt ra log."""
		info = self.summary()
		logger.info("%s", "=" * 70)
		logger.info("📊 TÓM TẮT DỮ LIỆU")
		logger.info("%s", info)
		logger.info("%s", "=" * 70)

	def save(self, path: Union[str, Path], index: bool = False) -> None:
		"""Lưu dữ liệu ra file."""
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
