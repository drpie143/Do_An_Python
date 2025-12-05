"""Utility module providing reusable visualization helpers for the project."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)


class DataVisualizer:
    """Generic plotting helper that keeps all figures in one place."""

    def __init__(
        self,
        data: Optional[pd.DataFrame] = None,
        target_col: Optional[str] = None,
        output_dir: Optional[Path] = None,
        auto_save: bool = True,
        auto_show: bool = False,
        dpi: int = 300,
        style: Optional[str] = None,
        figure_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        self.data = data
        self.target_col = target_col
        self.auto_save = auto_save
        self.auto_show = auto_show
        self.dpi = dpi
        self.output_dir = Path(output_dir) if output_dir else Path("results/eda")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if style:
            try:
                plt.style.use(style)
            except OSError:
                logger.warning("Không tìm thấy style '%s', sử dụng mặc định", style)
        sns.set_theme(style="whitegrid")
        plt.rcParams.setdefault("figure.figsize", figure_size or (10, 6))
        plt.rcParams.setdefault("font.size", 11)

    # ------------------------------------------------------------------
    # Generic setters
    # ------------------------------------------------------------------
    def set_data(self, data: Optional[pd.DataFrame]) -> None:
        self.data = data

    def set_target(self, target_col: Optional[str]) -> None:
        self.target_col = target_col

    def set_output_dir(self, output_dir: Optional[Path]) -> None:
        if output_dir is None:
            return
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _sanitize_name(self, name: str) -> str:
        cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", name).strip("_")
        return cleaned or "plot"

    def _resolve_save_path(self, save_path: Optional[Path], default_name: str) -> Optional[Path]:
        if save_path:
            path = Path(save_path)
        elif self.auto_save and self.output_dir:
            base = self._sanitize_name(default_name)
            path = self.output_dir / f"{base}.png"
        else:
            return None
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _finalize_plot(
        self,
        figure: plt.Figure,
        default_name: str,
        save_path: Optional[Path] = None,
        show: Optional[bool] = None,
    ) -> None:
        if figure is None:
            return
        path = self._resolve_save_path(save_path, default_name)
        if path:
            figure.savefig(path, dpi=self.dpi, bbox_inches="tight")
            logger.info("Visualization saved to %s", path)
        if show is None:
            show = self.auto_show
        if show:
            plt.show()
        else:
            plt.close(figure)

    def _is_numeric_series(self, series: pd.Series) -> bool:
        if series is None:
            return False
        if pd.api.types.is_numeric_dtype(series):
            return True
        converted = pd.to_numeric(series.dropna(), errors="coerce")
        return converted.notna().all() and not converted.empty

    def _coerce_numeric_series(self, series: pd.Series) -> pd.Series:
        if series is None:
            return series
        if pd.api.types.is_numeric_dtype(series):
            return series
        converted = pd.to_numeric(series, errors="coerce")
        return converted

    def _validate_cols(
        self,
        cols: Optional[Sequence[str]],
        dtype_include: Optional[Iterable[str]] = None,
    ) -> List[str]:
        if self.data is None:
            logger.warning("No data is set for visualization.")
            return []

        if cols is None:
            selected = self.data.columns.tolist()
            if dtype_include is not None:
                selected = self.data.select_dtypes(include=dtype_include).columns.tolist()
        else:
            selected = [col for col in cols if col in self.data.columns]
            missing = set(cols) - set(selected)
            if missing:
                logger.warning("Columns ignored because they do not exist: %s", sorted(missing))

        if self.target_col in selected:
            selected.remove(self.target_col)
        return selected

    # ------------------------------------------------------------------
    # Exploratory plots
    # ------------------------------------------------------------------
    def plot_numerical(self, cols: Optional[Sequence[str]] = None, bins: int = 30) -> None:
        """Vẽ riêng từng numeric column (deprecated - dùng plot_numeric_grid)."""
        numeric_cols = self._validate_cols(cols, dtype_include=["number"])
        if not numeric_cols:
            return

        for col in numeric_cols:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            sns.histplot(self.data[col], kde=True, ax=axes[0], bins=bins, color="royalblue")
            axes[0].set_title(f"Distribution: {col}")

            sns.boxplot(x=self.data[col], ax=axes[1], color="salmon")
            axes[1].set_title(f"Outliers: {col}")
            plt.tight_layout()
            self._finalize_plot(fig, f"numeric_{col}")

    def plot_numeric_grid(
        self, 
        cols: Optional[Sequence[str]] = None, 
        bins: int = 30,
        show: Optional[bool] = None,
    ) -> None:
        """
        Vẽ tất cả numeric features trong 1 figure dạng grid.
        Mỗi feature có histogram với KDE.
        """
        numeric_cols = self._validate_cols(cols, dtype_include=["number"])
        if not numeric_cols:
            return
        
        n_cols_plot = len(numeric_cols)
        n_cols_grid = 3  # 3 features per row
        n_rows = (n_cols_plot + n_cols_grid - 1) // n_cols_grid
        
        fig, axes = plt.subplots(n_rows, n_cols_grid, figsize=(16, 4 * n_rows))
        if n_cols_plot == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for idx, col in enumerate(numeric_cols):
            ax = axes[idx]
            
            # Histogram với KDE
            sns.histplot(self.data[col], kde=True, ax=ax, bins=bins, color="steelblue", alpha=0.7)
            
            # Thêm thông tin thống kê - chỉ vẽ đường mean
            mean_val = self.data[col].mean()
            median_val = self.data[col].median()
            std_val = self.data[col].std()
            
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=1.5)
            ax.axvline(median_val, color='green', linestyle='-', linewidth=1.5)
            
            # Title với stats gọn gàng
            ax.set_title(f"{col}\n(Mean={mean_val:.1f}, Median={median_val:.1f}, Std={std_val:.1f})", 
                        fontweight='bold', fontsize=10)
            ax.set_xlabel('')
            ax.set_ylabel('')
        
        # Ẩn các subplot trống
        for idx in range(n_cols_plot, len(axes)):
            axes[idx].set_visible(False)
        
        # Legend chung ở ngoài
        fig.legend(['Mean', 'Median'], loc='upper right', fontsize=10, 
                   framealpha=0.9, bbox_to_anchor=(0.99, 0.99))
        
        plt.suptitle("Numeric Features Distribution", fontweight='bold', fontsize=14, y=1.02)
        plt.tight_layout()
        self._finalize_plot(fig, "02_numeric_distributions", show=show)

    def plot_outliers_boxplot(
        self, 
        cols: Optional[Sequence[str]] = None,
        show: Optional[bool] = None,
    ) -> None:
        """
        Vẽ boxplot để phát hiện outliers cho tất cả numeric features.
        """
        numeric_cols = self._validate_cols(cols, dtype_include=["number"])
        if not numeric_cols:
            return
        
        n_cols_plot = len(numeric_cols)
        n_cols_grid = 3
        n_rows = (n_cols_plot + n_cols_grid - 1) // n_cols_grid
        
        fig, axes = plt.subplots(n_rows, n_cols_grid, figsize=(16, 4 * n_rows))
        if n_cols_plot == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for idx, col in enumerate(numeric_cols):
            ax = axes[idx]
            
            # Boxplot
            box_data = self.data[col].dropna()
            bp = ax.boxplot(box_data, vert=False, patch_artist=True,
                           boxprops=dict(facecolor='lightblue', color='navy'),
                           medianprops=dict(color='red', linewidth=2),
                           whiskerprops=dict(color='navy'),
                           capprops=dict(color='navy'),
                           flierprops=dict(marker='o', markerfacecolor='red', markersize=4, alpha=0.5))
            
            # Tính IQR và số outliers
            q1, q3 = box_data.quantile([0.25, 0.75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            n_outliers = ((box_data < lower_bound) | (box_data > upper_bound)).sum()
            outlier_pct = n_outliers / len(box_data) * 100
            
            # Title với thông tin outlier
            ax.set_title(f"{col}\n(Outliers: {n_outliers} = {outlier_pct:.1f}%)", 
                        fontweight='bold', fontsize=10,
                        color='red' if outlier_pct > 5 else 'black')
            ax.set_xlabel('')
            ax.set_yticks([])
            
            # Thêm text IQR bounds
            ax.text(0.02, 0.85, f'Q1={q1:.1f}\nQ3={q3:.1f}\nIQR={iqr:.1f}', 
                   transform=ax.transAxes, fontsize=8,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        # Ẩn các subplot trống
        for idx in range(n_cols_plot, len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle("Outlier Detection (Boxplot)", fontweight='bold', fontsize=14, y=1.02)
        plt.tight_layout()
        self._finalize_plot(fig, "06_outliers_boxplot", show=show)

    def plot_categorical_grid(
        self, 
        cols: Optional[Sequence[str]] = None, 
        top_n: int = 10,
        show: Optional[bool] = None,
    ) -> None:
        """Vẽ tất cả categorical features trong 1 figure dạng grid."""
        cat_cols = self._validate_cols(cols, dtype_include=["object", "category"])
        if not cat_cols:
            return
        
        # Lọc bỏ columns có quá nhiều unique values
        valid_cols = [c for c in cat_cols if self.data[c].nunique() <= 50]
        if not valid_cols:
            logger.warning("No valid categorical columns to plot")
            return
        
        n_cols_plot = len(valid_cols)
        n_cols_grid = min(n_cols_plot, 2)
        n_rows = (n_cols_plot + n_cols_grid - 1) // n_cols_grid
        
        fig, axes = plt.subplots(n_rows, n_cols_grid, figsize=(7 * n_cols_grid, 5 * n_rows))
        if n_cols_plot == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        colors = sns.color_palette("viridis", top_n)
        
        for idx, col in enumerate(valid_cols):
            ax = axes[idx]
            counts = self.data[col].value_counts().head(top_n)
            
            bars = ax.barh(counts.index[::-1], counts.values[::-1], color=colors[:len(counts)])
            ax.set_xlabel("Count")
            ax.set_title(col, fontweight='bold', fontsize=11)
            
            # Thêm số count trên mỗi bar
            for bar in bars:
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height()/2, f'{int(width):,}',
                       ha='left', va='center', fontsize=9)
        
        # Ẩn các subplot trống
        for idx in range(n_cols_plot, len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle("Categorical Features Distribution", fontweight='bold', fontsize=14, y=1.02)
        plt.tight_layout()
        self._finalize_plot(fig, "03_categorical_distributions", show=show)

    def plot_data_overview(self, show: Optional[bool] = None) -> None:
        """
        Vẽ biểu đồ tổng quan dataset: shape, missing, dtypes.
        """
        if self.data is None:
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        
        # 1. Missing values
        ax1 = axes[0]
        missing = self.data.isnull().sum()
        missing_pct = (missing / len(self.data) * 100).sort_values(ascending=True)
        colors = ['#e74c3c' if v > 0 else '#2ecc71' for v in missing_pct.values]
        ax1.barh(missing_pct.index, missing_pct.values, color=colors)
        ax1.set_xlabel("Missing %")
        ax1.set_title("Missing Values by Column", fontweight='bold')
        ax1.axvline(x=5, color='orange', linestyle='--', alpha=0.7, label='5% threshold')
        
        # 2. Data types
        ax2 = axes[1]
        dtype_counts = self.data.dtypes.astype(str).value_counts()
        colors2 = sns.color_palette("Set2", len(dtype_counts))
        wedges, texts, autotexts = ax2.pie(
            dtype_counts.values, 
            labels=dtype_counts.index,
            autopct='%1.1f%%',
            colors=colors2,
            explode=[0.05] * len(dtype_counts)
        )
        ax2.set_title("Column Data Types", fontweight='bold')
        
        # 3. Dataset info text
        ax3 = axes[2]
        ax3.axis('off')
        
        n_rows, n_cols = self.data.shape
        n_numeric = len(self.data.select_dtypes(include=['number']).columns)
        n_categorical = len(self.data.select_dtypes(include=['object', 'category']).columns)
        total_missing = self.data.isnull().sum().sum()
        missing_pct_total = total_missing / (n_rows * n_cols) * 100
        memory_mb = self.data.memory_usage(deep=True).sum() / 1024**2
        n_duplicates = self.data.duplicated().sum()
        
        info_text = f"""
        DATASET OVERVIEW
        {'='*40}
        
        Shape: {n_rows:,} rows x {n_cols} columns
        
        Column Types:
           - Numeric: {n_numeric}
           - Categorical: {n_categorical}
        
        Missing Values:
           - Total: {total_missing:,} ({missing_pct_total:.2f}%)
           - Columns with missing: {(missing > 0).sum()}
        
        Duplicates: {n_duplicates:,} rows
        
        Memory: {memory_mb:.2f} MB
        """
        
        ax3.text(0.1, 0.5, info_text, transform=ax3.transAxes, fontsize=12,
                verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
        
        plt.suptitle("Data Overview", fontweight='bold', fontsize=14, y=1.02)
        plt.tight_layout()
        self._finalize_plot(fig, "01_data_overview", show=show)

    def plot_target_analysis(
        self, 
        top_n_corr: int = 10,
        show: Optional[bool] = None,
    ) -> None:
        """
        Vẽ phân tích target: distribution + top correlations.
        """
        if self.data is None or not self.target_col:
            logger.warning("Target column required for target analysis")
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        
        target_data = self.data[self.target_col]
        
        # 1. Target distribution
        ax1 = axes[0]
        sns.histplot(target_data, kde=True, ax=ax1, color='#3498db', bins=30)
        ax1.axvline(target_data.mean(), color='red', linestyle='--', label=f'Mean: {target_data.mean():.2f}')
        ax1.axvline(target_data.median(), color='green', linestyle='-', label=f'Median: {target_data.median():.2f}')
        ax1.set_title(f"Target Distribution: {self.target_col}", fontweight='bold')
        ax1.legend()
        
        # 2. Target boxplot
        ax2 = axes[1]
        sns.boxplot(x=target_data, ax=ax2, color='#e74c3c')
        ax2.set_title(f"Target Outliers: {self.target_col}", fontweight='bold')
        
        # Thêm stats
        q1, q3 = target_data.quantile([0.25, 0.75])
        iqr = q3 - q1
        textstr = f'Q1: {q1:.2f}\nQ3: {q3:.2f}\nIQR: {iqr:.2f}\nStd: {target_data.std():.2f}'
        ax2.text(0.97, 0.97, textstr, transform=ax2.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 3. Top correlations with target
        ax3 = axes[2]
        numeric_cols = self.data.select_dtypes(include=['number']).columns.tolist()
        if self.target_col in numeric_cols:
            corr_with_target = self.data[numeric_cols].corr()[self.target_col].drop(self.target_col)
            top_corr = corr_with_target.abs().sort_values(ascending=False).head(top_n_corr)
            top_corr_values = corr_with_target[top_corr.index]
            
            colors = ['#27ae60' if v > 0 else '#e74c3c' for v in top_corr_values.values]
            bars = ax3.barh(top_corr_values.index[::-1], top_corr_values.values[::-1], color=colors[::-1])
            ax3.axvline(x=0, color='black', linewidth=0.5)
            ax3.set_xlabel("Correlation")
            ax3.set_title(f"Top {top_n_corr} Correlations with {self.target_col}", fontweight='bold')
            
            # Thêm giá trị trên bars
            for bar in bars:
                width = bar.get_width()
                ax3.text(width, bar.get_y() + bar.get_height()/2, f'{width:.3f}',
                        ha='left' if width > 0 else 'right', va='center', fontsize=9)
        
        plt.suptitle(f"Target Analysis: {self.target_col}", fontweight='bold', fontsize=14, y=1.02)
        plt.tight_layout()
        self._finalize_plot(fig, "05_target_analysis", show=show)

    def plot_categorical(self, cols: Optional[Sequence[str]] = None, top_n: int = 10) -> None:
        cat_cols = self._validate_cols(cols, dtype_include=["object", "category"])
        if not cat_cols:
            return

        for col in cat_cols:
            if self.data[col].nunique() > 50:
                logger.warning("Skip %s because it has too many unique values", col)
                continue
            fig, ax = plt.subplots(figsize=(10, 5))
            order = self.data[col].value_counts().iloc[:top_n].index
            counts = self.data[col].value_counts().loc[order].iloc[::-1]
            colors = sns.color_palette("coolwarm", len(counts))
            ax.barh(counts.index, counts.values, color=colors)
            ax.set_xlabel("Count")
            ax.set_title(f"Frequency: {col} (Top {top_n})")
            plt.tight_layout()
            self._finalize_plot(fig, f"categorical_{col}")

    def plot_correlation_heatmap(
        self,
        cols: Optional[Sequence[str]] = None,
        method: str = "pearson",
        annot: bool = False,
        figsize: Tuple[int, int] = (12, 10),
        save_path: Optional[Path] = None,
        show: Optional[bool] = None,
    ) -> Optional[pd.DataFrame]:
        if self.data is None:
            logger.warning("Cannot draw correlation heatmap without data")
            return None

        target_cols = self._validate_cols(cols, dtype_include=["number"])
        if cols is None:
            target_cols = self.data.select_dtypes(include=["number"]).columns.tolist()

        if len(target_cols) < 2:
            logger.warning("Need at least two numeric columns for correlation heatmap")
            return None

        corr_df = self.data[target_cols].corr(method=method)
        fig = plt.figure(figsize=figsize)
        mask = np.triu(np.ones_like(corr_df, dtype=bool))
        sns.heatmap(
            corr_df,
            mask=mask,
            annot=annot,
            fmt=".2f",
            cmap="RdBu_r",
            center=0,
            square=True,
            cbar=True,
        )
        plt.title(f"Feature Correlation Heatmap ({method.title()})", fontweight="bold")
        self._finalize_plot(fig, f"correlation_{method}", save_path=save_path, show=show)
        return corr_df

    def plot_relationship_with_target(
        self,
        cols: Optional[Sequence[str]] = None,
        show: Optional[bool] = None,
    ) -> None:
        if self.data is None or not self.target_col:
            logger.warning("Target column is required for this visualization")
            return

        target_cols = self._validate_cols(cols)
        target_series = self.data[self.target_col]
        target_is_numeric = self._is_numeric_series(target_series)
        if target_is_numeric:
            target_series = self._coerce_numeric_series(target_series)
        for col in target_cols:
            fig, ax = plt.subplots(figsize=(10, 6))
            feature_series = self.data[col]
            feature_is_numeric = self._is_numeric_series(feature_series)
            if feature_is_numeric:
                feature_series = self._coerce_numeric_series(feature_series)

            if feature_is_numeric and target_is_numeric:
                sns.regplot(x=feature_series, y=target_series, scatter_kws={"alpha": 0.4}, line_kws={"color": "red"}, ax=ax)
                ax.set_xlabel(col)
                ax.set_ylabel(self.target_col)
            elif feature_is_numeric and not target_is_numeric:
                sns.violinplot(x=target_series, y=feature_series, color=sns.color_palette("muted")[0], ax=ax)
                ax.set_xlabel(self.target_col)
                ax.set_ylabel(col)
            elif not feature_is_numeric and target_is_numeric:
                sns.boxplot(x=feature_series, y=target_series, ax=ax)
                ax.set_xlabel(col)
                ax.set_ylabel(self.target_col)
            else:
                crosstab = pd.crosstab(feature_series, target_series, normalize="index")
                crosstab.plot(kind="bar", stacked=True, colormap="viridis", ax=ax)
                ax.set_ylabel("Ratio")
            ax.set_title(f"Relationship: {col} vs {self.target_col}")
            self._finalize_plot(fig, f"relationship_{col}_vs_{self.target_col}", show=show)

    def plot_missing_heatmap(self, show: Optional[bool] = None) -> None:
        if self.data is None:
            logger.warning("Cannot draw missing heatmap without data")
            return
        if self.data.isna().sum().sum() == 0:
            logger.info("Dataset is clean. No missing values to plot.")
            return

        fig = plt.figure(figsize=(12, 6))
        sns.heatmap(self.data.isna(), cbar=False, yticklabels=False, cmap="viridis")
        plt.title("Missing Value Map")
        self._finalize_plot(fig, "missing_heatmap", show=show)

    def plot_qq(self, col: str, show: Optional[bool] = None) -> None:
        from scipy import stats

        if self.data is None or col not in self.data.columns:
            logger.warning("Column %s is not available for QQ plot", col)
            return
        fig = plt.figure(figsize=(8, 6))
        stats.probplot(self.data[col].dropna(), dist="norm", plot=plt)
        plt.title(f"QQ Plot: {col}")
        self._finalize_plot(fig, f"qq_{col}", show=show)

    def plot_joint_distribution(self, col_x: str, col_y: str, kind: str = "scatter", show: Optional[bool] = None) -> None:
        if self.data is None:
            return
        if not (pd.api.types.is_numeric_dtype(self.data[col_x]) and pd.api.types.is_numeric_dtype(self.data[col_y])):
            logger.warning("Joint plot requires numeric columns")
            return
        g = sns.jointplot(data=self.data, x=col_x, y=col_y, kind=kind, color="teal", height=8)
        g.fig.suptitle(f"Joint Plot: {col_x} vs {col_y}", y=1.02)
        self._finalize_plot(g.fig, f"joint_{col_x}_{col_y}", show=show)

    def plot_hexbin_density(self, col_x: str, col_y: str, show: Optional[bool] = None) -> None:
        if self.data is None:
            return
        fig = plt.figure(figsize=(10, 8))
        plt.hexbin(self.data[col_x], self.data[col_y], gridsize=30, cmap="Blues", mincnt=1)
        plt.colorbar(label="Observation count")
        plt.xlabel(col_x)
        plt.ylabel(col_y)
        plt.title(f"Hexbin Density: {col_x} vs {col_y}")
        self._finalize_plot(fig, f"hexbin_{col_x}_{col_y}", show=show)

    # ------------------------------------------------------------------
    # Modeling visualizations
    # ------------------------------------------------------------------
    def plot_model_comparison(
        self,
        metric_values: Dict[str, float],
        metric: str,
        save_path: Optional[Path] = None,
        show: Optional[bool] = None,
    ) -> None:
        if not metric_values:
            logger.warning("No metric values supplied for comparison plot")
            return

        fig = plt.figure(figsize=(10, 6))
        colors = sns.color_palette("viridis", len(metric_values))
        bars = plt.bar(metric_values.keys(), metric_values.values(), color=colors)
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height, f"{height:.4f}", ha="center", va="bottom", fontweight="bold")
        plt.xlabel("Model", fontweight="bold")
        plt.ylabel(metric, fontweight="bold")
        plt.title(f"Model Comparison - {metric}", fontweight="bold")
        plt.grid(axis="y", alpha=0.3)
        self._finalize_plot(fig, f"model_comparison_{metric}", save_path=save_path, show=show)

    def plot_regression_diagnostics(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        model_name: str,
        save_path: Optional[Path] = None,
        show: Optional[bool] = None,
    ) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        axes[0].scatter(y_true, y_pred, alpha=0.6)
        min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
        axes[0].plot([min_val, max_val], [min_val, max_val], "r--", lw=2)
        axes[0].set_xlabel("Actual", fontweight="bold")
        axes[0].set_ylabel("Predicted", fontweight="bold")
        axes[0].set_title(f"{model_name} - Actual vs Predicted", fontweight="bold")
        axes[0].grid(alpha=0.3)

        residuals = y_true - y_pred
        axes[1].scatter(y_pred, residuals, alpha=0.6)
        axes[1].axhline(y=0, color="r", linestyle="--", lw=2)
        axes[1].set_xlabel("Predicted", fontweight="bold")
        axes[1].set_ylabel("Residuals", fontweight="bold")
        axes[1].set_title(f"{model_name} - Residuals", fontweight="bold")
        axes[1].grid(alpha=0.3)
        plt.tight_layout()
        self._finalize_plot(fig, f"regression_diag_{model_name}", save_path=save_path, show=show)

    def plot_feature_importance(
        self,
        importance_df: pd.DataFrame,
        model_name: str,
        save_path: Optional[Path] = None,
        top_n: int = 15,
        show: Optional[bool] = None,
    ) -> None:
        if importance_df.empty:
            logger.warning("Feature importance DataFrame is empty")
            return

        top_features = importance_df.sort_values("importance", ascending=False).head(top_n)
        fig = plt.figure(figsize=(10, 8))
        plt.barh(top_features["feature"], top_features["importance"], color="steelblue")
        plt.xlabel("Importance", fontweight="bold")
        plt.ylabel("Feature", fontweight="bold")
        plt.title(f"Feature Importance - {model_name.upper()}", fontweight="bold")
        plt.gca().invert_yaxis()
        plt.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        self._finalize_plot(fig, f"feature_importance_{model_name}", save_path=save_path, show=show)

    def plot_feature_importance_comparison(
        self,
        importances: Dict[str, pd.DataFrame],
        save_path: Optional[Path] = None,
        top_n: int = 10,
        show: Optional[bool] = None,
    ) -> None:
        if not importances:
            logger.warning("No feature importance data to compare")
            return

        n_models = len(importances)
        fig, axes = plt.subplots(1, n_models, figsize=(8 * n_models, 6))
        if n_models == 1:
            axes = [axes]

        for ax, (model_name, df_imp) in zip(axes, importances.items()):
            subset = df_imp.sort_values("importance", ascending=False).head(top_n)
            ax.barh(subset["feature"], subset["importance"], color="steelblue")
            ax.set_title(model_name.upper(), fontweight="bold")
            ax.set_xlabel("Importance", fontweight="bold")
            ax.set_ylabel("Feature", fontweight="bold")
            ax.invert_yaxis()
            ax.grid(axis="x", alpha=0.3)

        plt.tight_layout()
        self._finalize_plot(fig, "feature_importance_comparison", save_path=save_path, show=show)

    def evaluate_classification(
        self,
        model,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        show: Optional[bool] = None,
    ) -> None:
        from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_curve, auc

        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(1, 2, figsize=(16, 6))

        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap="Blues", ax=ax[0], colorbar=False)
        ax[0].set_title("Confusion Matrix")

        if hasattr(model, "predict_proba") and len(np.unique(y_test)) == 2:
            y_probs = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_probs)
            roc_auc = auc(fpr, tpr)
            ax[1].plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc:.2f}")
            ax[1].plot([0, 1], [0, 1], "k--", lw=2)
            ax[1].set_xlabel("False Positive Rate")
            ax[1].set_ylabel("True Positive Rate")
            ax[1].set_title("ROC Curve")
            ax[1].legend(loc="lower right")
        else:
            ax[1].text(0.5, 0.5, "ROC không khả dụng", ha="center")

        self._finalize_plot(fig, "classification_report", show=show)

    def evaluate_regression(
        self,
        model,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        show: Optional[bool] = None,
    ) -> None:
        y_pred = model.predict(X_test)
        self.plot_regression_diagnostics(y_test, y_pred, model.__class__.__name__, show=show)

    def plot_combined_predictions(
        self,
        predictions: Dict[str, Tuple[np.ndarray, np.ndarray]],
        save_path: Optional[Path] = None,
        show: Optional[bool] = None,
    ) -> None:
        """
        Vẽ biểu đồ Actual vs Predicted cho tất cả mô hình trong 1 figure.
        
        Args:
            predictions: Dict {model_name: (y_true, y_pred)}
            save_path: Đường dẫn lưu file
            show: Hiển thị biểu đồ
        """
        if not predictions:
            logger.warning("Không có predictions để vẽ")
            return
        
        n_models = len(predictions)
        n_cols = min(n_models, 2)
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 6 * n_rows))
        if n_models == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']
        
        for idx, (model_name, (y_true, y_pred)) in enumerate(predictions.items()):
            ax = axes[idx]
            color = colors[idx % len(colors)]
            
            # Scatter plot
            ax.scatter(y_true, y_pred, alpha=0.5, s=20, c=color, label='Predictions')
            
            # Perfect prediction line
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Perfect Fit')
            
            # Calculate metrics
            from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
            r2 = r2_score(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            
            # Add text box with metrics
            textstr = f'R² = {r2:.4f}\nRMSE = {rmse:.2f}\nMAE = {mae:.2f}'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=props)
            
            ax.set_xlabel("Actual", fontweight="bold")
            ax.set_ylabel("Predicted", fontweight="bold")
            ax.set_title(f"{model_name.upper()}", fontweight="bold", fontsize=12)
            ax.grid(alpha=0.3)
            ax.legend(loc='lower right')
        
        # Hide empty subplots
        for idx in range(n_models, len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle("Actual vs Predicted - Model Comparison", fontweight="bold", fontsize=14, y=1.02)
        plt.tight_layout()
        self._finalize_plot(fig, "predictions_combined", save_path=save_path, show=show)

    def plot_metrics_summary(
        self,
        results: Dict[str, Dict[str, float]],
        save_path: Optional[Path] = None,
        show: Optional[bool] = None,
    ) -> None:
        """
        Vẽ biểu đồ tổng hợp các metrics (R², RMSE, MAE) cho tất cả models.
        
        Args:
            results: Dict {model_name: {'test_r2': ..., 'test_rmse': ..., 'test_mae': ...}}
            save_path: Đường dẫn lưu file
            show: Hiển thị biểu đồ
        """
        if not results:
            logger.warning("Không có kết quả để vẽ")
            return
        
        models = list(results.keys())
        metrics = ['test_r2', 'test_rmse', 'test_mae']
        metric_labels = ['R² Score', 'RMSE', 'MAE']
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        colors = sns.color_palette("viridis", len(models))
        
        for ax, metric, label in zip(axes, metrics, metric_labels):
            values = [results[m][metric] for m in models]
            bars = ax.bar(models, values, color=colors)
            
            # Add value labels
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, height,
                       f'{val:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
            
            ax.set_xlabel("Model", fontweight="bold")
            ax.set_ylabel(label, fontweight="bold")
            ax.set_title(label, fontweight="bold", fontsize=12)
            ax.set_xticklabels([m.upper() for m in models], rotation=15)
            ax.grid(axis="y", alpha=0.3)
            
            # Highlight best model
            if metric == 'test_r2':
                best_idx = np.argmax(values)
            else:
                best_idx = np.argmin(values)
            bars[best_idx].set_color('#27ae60')
            bars[best_idx].set_edgecolor('black')
            bars[best_idx].set_linewidth(2)
        
        plt.suptitle("Model Performance Comparison", fontweight="bold", fontsize=14, y=1.02)
        plt.tight_layout()
        self._finalize_plot(fig, "metrics_summary", save_path=save_path, show=show)
