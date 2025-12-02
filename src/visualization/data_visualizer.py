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
        if path.exists():
            try:
                path.unlink()
            except OSError as exc:
                logger.warning("Không thể xóa file cũ %s: %s", path, exc)
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
