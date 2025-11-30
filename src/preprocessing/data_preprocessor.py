"""
Module tiền xử lý dữ liệu cho dự án Taxi Price Prediction.

Class DataPreprocessor cung cấp các chức năng:
- Đọc dữ liệu từ nhiều định dạng (csv, xlsx, json)
- Xử lý giá trị thiếu
- Phát hiện và xử lý outliers
- Mã hóa biến phân loại
- Chuẩn hóa dữ liệu
- Feature engineering
- Lưu dữ liệu đã xử lý
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import IsolationForest


# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Lớp tiền xử lý dữ liệu cho bài toán Taxi Price Prediction.
    
    Cung cấp đầy đủ các chức năng tiền xử lý dữ liệu bao gồm:
    - Đọc dữ liệu từ nhiều nguồn
    - Xử lý missing values
    - Phát hiện và xử lý outliers
    - Encoding biến phân loại
    - Scaling/Normalization
    - Feature engineering
    
    Attributes:
        data (pd.DataFrame): DataFrame chứa dữ liệu
        original_data (pd.DataFrame): Bản sao dữ liệu gốc
        numeric_cols (List[str]): Danh sách cột số
        categorical_cols (List[str]): Danh sách cột phân loại
        scaler: Scaler đã fit (StandardScaler hoặc MinMaxScaler)
        encoders (Dict): Dictionary lưu các encoder
    """
    
    def __init__(self, data: Optional[pd.DataFrame] = None):
        """
        Khởi tạo DataPreprocessor.
        
        Args:
            data: DataFrame dữ liệu (optional)
        """
        self.data = data.copy() if data is not None else None
        self.original_data = data.copy() if data is not None else None
        self.numeric_cols = []
        self.categorical_cols = []
        self.scaler = None
        self.encoders = {}
        self.preprocessing_steps = []  # Track các bước đã thực hiện
        
        if self.data is not None:
            self._identify_column_types()
            logger.info("✅ DataPreprocessor khởi tạo thành công")
            logger.info(f"   Shape: {self.data.shape}")
    
    def __repr__(self) -> str:
        """Representation của DataPreprocessor."""
        if self.data is not None:
            return (f"DataPreprocessor(shape={self.data.shape}, "
                   f"numeric_cols={len(self.numeric_cols)}, "
                   f"categorical_cols={len(self.categorical_cols)})")
        return "DataPreprocessor(no data loaded)"
    
    # ========== LOAD DATA ==========
    @staticmethod
    def load_data(filepath: str, **kwargs) -> pd.DataFrame:
        """
        Đọc dữ liệu từ file (csv, xlsx, json).
        
        Args:
            filepath: Đường dẫn file
            **kwargs: Các tham số bổ sung cho pandas read functions
            
        Returns:
            DataFrame chứa dữ liệu
            
        Raises:
            ValueError: Nếu định dạng file không được hỗ trợ
            FileNotFoundError: Nếu file không tồn tại
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File không tồn tại: {filepath}")
        
        file_ext = filepath.suffix.lower()
        
        try:
            if file_ext == '.csv':
                df = pd.read_csv(filepath, **kwargs)
            elif file_ext in ['.xlsx', '.xls']:
                df = pd.read_excel(filepath, **kwargs)
            elif file_ext == '.json':
                df = pd.read_json(filepath, **kwargs)
            else:
                raise ValueError(f"Định dạng file không được hỗ trợ: {file_ext}")
            
            logger.info(f"✅ Đọc dữ liệu thành công từ: {filepath}")
            logger.info(f"   Shape: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"❌ Lỗi khi đọc file: {e}")
            raise
    
    def load(self, filepath: str, **kwargs) -> 'DataPreprocessor':
        """
        Load dữ liệu vào instance.
        
        Args:
            filepath: Đường dẫn file
            **kwargs: Tham số cho load_data
            
        Returns:
            self cho method chaining
        """
        self.data = self.load_data(filepath, **kwargs)
        self.original_data = self.data.copy()
        self._identify_column_types()
        return self
    
    def _identify_column_types(self) -> None:
        """Tự động nhận diện kiểu dữ liệu các cột."""
        if self.data is None:
            return
        
        self.numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = self.data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        logger.info(f"📊 Phát hiện {len(self.numeric_cols)} cột số và {len(self.categorical_cols)} cột phân loại")
    
    # ========== MISSING VALUES ==========
    def check_missing(self) -> pd.DataFrame:
        """
        Kiểm tra missing values.
        
        Returns:
            DataFrame chứa thống kê missing values
        """
        if self.data is None:
            logger.warning("❌ Chưa load dữ liệu")
            return None
        
        missing_count = self.data.isnull().sum()
        missing_percent = (missing_count / len(self.data)) * 100
        
        missing_df = pd.DataFrame({
            'Column': missing_count.index,
            'Missing_Count': missing_count.values,
            'Missing_Percent': missing_percent.values
        })
        
        missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values(
            'Missing_Count', ascending=False
        )
        
        if len(missing_df) > 0:
            logger.info(f"⚠️  Phát hiện {len(missing_df)} cột có missing values")
        else:
            logger.info("✅ Không có missing values")
        
        return missing_df
    
    def handle_missing(self, 
                      strategy: str = 'auto',
                      numeric_strategy: str = 'median',
                      categorical_strategy: str = 'mode',
                      fill_value: Optional[Any] = None) -> 'DataPreprocessor':
        """
        Xử lý missing values.
        
        Args:
            strategy: Chiến lược xử lý ('auto', 'drop', 'fill')
            numeric_strategy: Chiến lược cho cột số ('mean', 'median', 'mode', 'forward_fill')
            categorical_strategy: Chiến lược cho cột phân loại ('mode', 'constant')
            fill_value: Giá trị dùng để fill (nếu strategy='fill' và constant)
            
        Returns:
            self cho method chaining
        """
        if self.data is None:
            logger.warning("❌ Chưa load dữ liệu")
            return self
        
        logger.info(f"\n{'='*70}")
        logger.info("🔧 XỬ LÝ MISSING VALUES")
        logger.info(f"{'='*70}")
        
        initial_missing = self.data.isnull().sum().sum()
        logger.info(f"Tổng missing trước xử lý: {initial_missing}")
        
        if strategy == 'drop':
            self.data = self.data.dropna()
            logger.info(f"✅ Đã xóa các dòng có missing values")
        
        elif strategy in ['auto', 'fill']:
            # Xử lý cột số
            for col in self.numeric_cols:
                if self.data[col].isnull().any():
                    if numeric_strategy == 'mean':
                        fill_val = self.data[col].mean()
                    elif numeric_strategy == 'median':
                        fill_val = self.data[col].median()
                    elif numeric_strategy == 'mode':
                        fill_val = self.data[col].mode()[0]
                    elif numeric_strategy == 'forward_fill':
                        self.data[col] = self.data[col].fillna(method='ffill')
                        continue
                    else:
                        fill_val = fill_value if fill_value is not None else 0
                    
                    self.data[col] = self.data[col].fillna(fill_val)
                    logger.info(f"   Filled {col} ({numeric_strategy}): {fill_val:.2f}")
            
            # Xử lý cột phân loại
            for col in self.categorical_cols:
                if self.data[col].isnull().any():
                    if categorical_strategy == 'mode':
                        fill_val = self.data[col].mode()[0]
                    else:
                        fill_val = fill_value if fill_value is not None else 'Unknown'
                    
                    self.data[col] = self.data[col].fillna(fill_val)
                    logger.info(f"   Filled {col} ({categorical_strategy}): {fill_val}")
        
        final_missing = self.data.isnull().sum().sum()
        logger.info(f"✅ Tổng missing sau xử lý: {final_missing}")
        
        self.preprocessing_steps.append('handle_missing')
        return self
    
    # ========== OUTLIERS ==========
    def detect_outliers_iqr(self, columns: Optional[List[str]] = None, 
                           threshold: float = 1.5) -> Dict[str, pd.Series]:
        """
        Phát hiện outliers bằng IQR method.
        
        Args:
            columns: Danh sách cột cần kiểm tra (None = tất cả cột số)
            threshold: IQR multiplier (mặc định 1.5)
            
        Returns:
            Dictionary {column: outlier_indices}
        """
        if self.data is None:
            logger.warning("❌ Chưa load dữ liệu")
            return {}
        
        if columns is None:
            columns = self.numeric_cols
        
        outliers = {}
        
        for col in columns:
            Q1 = self.data[col].quantile(0.25)
            Q3 = self.data[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            outlier_mask = (self.data[col] < lower_bound) | (self.data[col] > upper_bound)
            outliers[col] = self.data[outlier_mask].index
            
            if len(outliers[col]) > 0:
                logger.info(f"   {col}: {len(outliers[col])} outliers detected")
        
        return outliers
    
    def detect_outliers_zscore(self, columns: Optional[List[str]] = None,
                              threshold: float = 3.0) -> Dict[str, pd.Series]:
        """
        Phát hiện outliers bằng Z-score method.
        
        Args:
            columns: Danh sách cột cần kiểm tra
            threshold: Z-score threshold (mặc định 3.0)
            
        Returns:
            Dictionary {column: outlier_indices}
        """
        if self.data is None:
            logger.warning("❌ Chưa load dữ liệu")
            return {}
        
        if columns is None:
            columns = self.numeric_cols
        
        outliers = {}
        
        for col in columns:
            z_scores = np.abs((self.data[col] - self.data[col].mean()) / self.data[col].std())
            outlier_mask = z_scores > threshold
            outliers[col] = self.data[outlier_mask].index
            
            if len(outliers[col]) > 0:
                logger.info(f"   {col}: {len(outliers[col])} outliers detected (Z-score)")
        
        return outliers
    
    def detect_outliers_isolation_forest(self, 
                                        columns: Optional[List[str]] = None,
                                        contamination: float = 0.1) -> np.ndarray:
        """
        Phát hiện outliers bằng Isolation Forest.
        
        Args:
            columns: Danh sách cột cần kiểm tra
            contamination: Tỷ lệ outliers ước tính (0-0.5)
            
        Returns:
            Array indices của outliers
        """
        if self.data is None:
            logger.warning("❌ Chưa load dữ liệu")
            return np.array([])
        
        if columns is None:
            columns = self.numeric_cols
        
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        predictions = iso_forest.fit_predict(self.data[columns])
        
        outlier_indices = np.where(predictions == -1)[0]
        logger.info(f"   Isolation Forest: {len(outlier_indices)} outliers detected")
        
        return outlier_indices
    
    def remove_outliers(self, method: str = 'iqr', **kwargs) -> 'DataPreprocessor':
        """
        Loại bỏ outliers.
        
        Args:
            method: Phương pháp ('iqr', 'zscore', 'isolation_forest')
            **kwargs: Tham số cho detection method
            
        Returns:
            self cho method chaining
        """
        if self.data is None:
            logger.warning("❌ Chưa load dữ liệu")
            return self
        
        logger.info(f"\n{'='*70}")
        logger.info(f"🔍 PHÁT HIỆN VÀ XÓA OUTLIERS (method={method})")
        logger.info(f"{'='*70}")
        
        initial_shape = self.data.shape
        
        if method == 'iqr':
            outliers_dict = self.detect_outliers_iqr(**kwargs)
            # Lấy union của tất cả outlier indices
            all_outlier_indices = set()
            for indices in outliers_dict.values():
                all_outlier_indices.update(indices)
            self.data = self.data.drop(list(all_outlier_indices))
        
        elif method == 'zscore':
            outliers_dict = self.detect_outliers_zscore(**kwargs)
            all_outlier_indices = set()
            for indices in outliers_dict.values():
                all_outlier_indices.update(indices)
            self.data = self.data.drop(list(all_outlier_indices))
        
        elif method == 'isolation_forest':
            outlier_indices = self.detect_outliers_isolation_forest(**kwargs)
            self.data = self.data.drop(self.data.index[outlier_indices])
        
        self.data = self.data.reset_index(drop=True)
        final_shape = self.data.shape
        
        logger.info(f"✅ Đã xóa {initial_shape[0] - final_shape[0]} outliers")
        logger.info(f"   Shape: {initial_shape} → {final_shape}")
        
        self.preprocessing_steps.append('remove_outliers')
        return self
    
    # ========== ENCODING ==========
    def encode_categorical(self, 
                          method: str = 'onehot',
                          columns: Optional[List[str]] = None,
                          drop_first: bool = True) -> 'DataPreprocessor':
        """
        Mã hóa biến phân loại.
        
        Args:
            method: Phương pháp ('onehot', 'label')
            columns: Danh sách cột cần encode (None = tất cả cột phân loại)
            drop_first: Có drop cột đầu tiên khi onehot không (tránh multicollinearity)
            
        Returns:
            self cho method chaining
        """
        if self.data is None:
            logger.warning("❌ Chưa load dữ liệu")
            return self
        
        logger.info(f"\n{'='*70}")
        logger.info(f"🔤 MÃ HÓA BIẾN PHÂN LOẠI (method={method})")
        logger.info(f"{'='*70}")
        
        if columns is None:
            columns = self.categorical_cols
        
        if method == 'onehot':
            self.data = pd.get_dummies(self.data, columns=columns, drop_first=drop_first)
            logger.info(f"✅ OneHot Encoding hoàn tất")
            logger.info(f"   Shape sau encoding: {self.data.shape}")
        
        elif method == 'label':
            for col in columns:
                le = LabelEncoder()
                self.data[col] = le.fit_transform(self.data[col])
                self.encoders[col] = le
                logger.info(f"   {col}: {len(le.classes_)} classes encoded")
            logger.info(f"✅ Label Encoding hoàn tất")
        
        # Update column types
        self._identify_column_types()
        
        self.preprocessing_steps.append('encode_categorical')
        return self
    
    # ========== SCALING ==========
    def scale_features(self, 
                      method: str = 'standard',
                      columns: Optional[List[str]] = None,
                      exclude_columns: Optional[List[str]] = None) -> 'DataPreprocessor':
        """
        Chuẩn hóa dữ liệu.
        
        Args:
            method: Phương pháp ('standard', 'minmax')
            columns: Danh sách cột cần scale (None = tất cả cột số)
            exclude_columns: Danh sách cột không scale
            
        Returns:
            self cho method chaining
        """
        if self.data is None:
            logger.warning("❌ Chưa load dữ liệu")
            return self
        
        logger.info(f"\n{'='*70}")
        logger.info(f"📏 CHUẨN HÓA DỮ LIỆU (method={method})")
        logger.info(f"{'='*70}")
        
        if columns is None:
            columns = self.numeric_cols
        
        if exclude_columns:
            columns = [col for col in columns if col not in exclude_columns]
        
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Method không hợp lệ: {method}")
        
        self.data[columns] = self.scaler.fit_transform(self.data[columns])
        
        logger.info(f"✅ Scaling hoàn tất cho {len(columns)} cột")
        
        self.preprocessing_steps.append('scale_features')
        return self
    
    # ========== VISUALIZATION ==========
    def plot_correlation_heatmap(self,
                                 target_col: Optional[str] = None,
                                 method: str = 'pearson',
                                 save_path: Optional[Union[str, Path]] = None,
                                 figsize: Tuple[int, int] = (12, 10),
                                 annot: bool = False,
                                 show: bool = False) -> Optional[pd.DataFrame]:
        """Vẽ heatmap tương quan để hỗ trợ chọn feature."""
        if self.data is None:
            logger.warning("❌ Chưa có dữ liệu để vẽ heatmap")
            return None
        
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            logger.warning("❌ Không có cột số để tính tương quan")
            return None
        
        corr_df = self.data[numeric_cols].corr(method=method)
        if target_col and target_col in corr_df.columns:
            ordered_cols = [target_col] + [col for col in corr_df.columns if col != target_col]
            corr_df = corr_df.loc[ordered_cols, ordered_cols]
        
        plt.figure(figsize=figsize)
        sns.heatmap(
            corr_df,
            cmap='RdYlBu_r',
            annot=annot,
            fmt='.2f',
            square=True,
            cbar=True
        )
        plt.title(f'Feature Correlation Heatmap ({method.title()})', fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"✅ Đã lưu heatmap tương quan: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return corr_df
    
    # ========== FEATURE ENGINEERING ==========
    def create_datetime_features(self, 
                                 datetime_col: str,
                                 features: List[str] = ['hour', 'day', 'month', 'dayofweek'],
                                 drop_original: bool = False) -> 'DataPreprocessor':
        """
        Tạo features từ cột datetime.
        
        Args:
            datetime_col: Tên cột datetime
            features: Danh sách features cần tạo
            drop_original: Có xóa cột gốc không
            
        Returns:
            self cho method chaining
        """
        if self.data is None:
            logger.warning("❌ Chưa load dữ liệu")
            return self
        
        logger.info(f"\n{'='*70}")
        logger.info("📅 TẠO DATETIME FEATURES")
        logger.info(f"{'='*70}")
        
        # Convert to datetime nếu chưa
        if not pd.api.types.is_datetime64_any_dtype(self.data[datetime_col]):
            self.data[datetime_col] = pd.to_datetime(self.data[datetime_col])
        
        created_features = []
        
        if 'hour' in features:
            self.data[f'{datetime_col}_hour'] = self.data[datetime_col].dt.hour
            created_features.append('hour')
        
        if 'day' in features:
            self.data[f'{datetime_col}_day'] = self.data[datetime_col].dt.day
            created_features.append('day')
        
        if 'month' in features:
            self.data[f'{datetime_col}_month'] = self.data[datetime_col].dt.month
            created_features.append('month')
        
        if 'year' in features:
            self.data[f'{datetime_col}_year'] = self.data[datetime_col].dt.year
            created_features.append('year')
        
        if 'dayofweek' in features:
            self.data[f'{datetime_col}_dayofweek'] = self.data[datetime_col].dt.dayofweek
            created_features.append('dayofweek')
        
        if 'quarter' in features:
            self.data[f'{datetime_col}_quarter'] = self.data[datetime_col].dt.quarter
            created_features.append('quarter')
        
        if drop_original:
            self.data = self.data.drop(datetime_col, axis=1)
            logger.info(f"   Đã xóa cột gốc: {datetime_col}")
        
        logger.info(f"✅ Đã tạo {len(created_features)} features: {created_features}")
        
        # Update column types
        self._identify_column_types()
        
        self.preprocessing_steps.append('create_datetime_features')
        return self
    
    def create_interaction_features(self, 
                                   col_pairs: List[Tuple[str, str]],
                                   operations: List[str] = ['multiply']) -> 'DataPreprocessor':
        """
        Tạo interaction features từ các cặp cột.
        
        Args:
            col_pairs: List các tuple (col1, col2)
            operations: Các phép toán ('multiply', 'add', 'subtract', 'divide')
            
        Returns:
            self cho method chaining
        """
        if self.data is None:
            logger.warning("❌ Chưa load dữ liệu")
            return self
        
        logger.info(f"\n{'='*70}")
        logger.info("🔗 TẠO INTERACTION FEATURES")
        logger.info(f"{'='*70}")
        
        created_count = 0
        
        for col1, col2 in col_pairs:
            if col1 not in self.data.columns or col2 not in self.data.columns:
                logger.warning(f"   Bỏ qua: {col1}, {col2} không tồn tại")
                continue
            
            for op in operations:
                if op == 'multiply':
                    feature_name = f'{col1}_x_{col2}'
                    self.data[feature_name] = self.data[col1] * self.data[col2]
                elif op == 'add':
                    feature_name = f'{col1}_plus_{col2}'
                    self.data[feature_name] = self.data[col1] + self.data[col2]
                elif op == 'subtract':
                    feature_name = f'{col1}_minus_{col2}'
                    self.data[feature_name] = self.data[col1] - self.data[col2]
                elif op == 'divide':
                    feature_name = f'{col1}_div_{col2}'
                    # Tránh chia cho 0
                    self.data[feature_name] = self.data[col1] / (self.data[col2] + 1e-6)
                
                logger.info(f"   Created: {feature_name}")
                created_count += 1
        
        logger.info(f"✅ Đã tạo {created_count} interaction features")
        
        # Update column types
        self._identify_column_types()
        
        self.preprocessing_steps.append('create_interaction_features')
        return self
    
    # ========== SAVE & EXPORT ==========
    def save_data(self, filepath: str, index: bool = False, **kwargs) -> None:
        """
        Lưu dữ liệu đã xử lý ra file.
        
        Args:
            filepath: Đường dẫn file output
            index: Có lưu index không
            **kwargs: Tham số cho pandas to_csv/to_excel/to_json
        """
        if self.data is None:
            logger.warning("❌ Chưa có dữ liệu để lưu")
            return
        
        filepath = Path(filepath)
        file_ext = filepath.suffix.lower()
        
        try:
            if file_ext == '.csv':
                self.data.to_csv(filepath, index=index, **kwargs)
            elif file_ext in ['.xlsx', '.xls']:
                self.data.to_excel(filepath, index=index, **kwargs)
            elif file_ext == '.json':
                self.data.to_json(filepath, **kwargs)
            else:
                raise ValueError(f"Định dạng file không được hỗ trợ: {file_ext}")
            
            logger.info(f"✅ Đã lưu dữ liệu vào: {filepath}")
            
        except Exception as e:
            logger.error(f"❌ Lỗi khi lưu file: {e}")
            raise
    
    def get_processed_data(self) -> pd.DataFrame:
        """
        Lấy DataFrame đã xử lý.
        
        Returns:
            DataFrame đã xử lý
        """
        return self.data.copy()
    
    def summary(self) -> Dict[str, Any]:
        """
        Trả về thống kê tổng quan về dữ liệu.
        
        Returns:
            Dictionary chứa thông tin tổng quan
        """
        if self.data is None:
            return {'status': 'No data loaded'}
        
        return {
            'shape': self.data.shape,
            'n_numeric_cols': len(self.numeric_cols),
            'n_categorical_cols': len(self.categorical_cols),
            'total_missing': self.data.isnull().sum().sum(),
            'preprocessing_steps': self.preprocessing_steps,
            'memory_usage_mb': self.data.memory_usage(deep=True).sum() / 1024**2
        }
    
    def print_summary(self) -> None:
        """In ra tóm tắt dữ liệu."""
        summary = self.summary()
        
        logger.info(f"\n{'='*70}")
        logger.info("📊 TÓM TẮT DỮ LIỆU")
        logger.info(f"{'='*70}")
        logger.info(f"Shape: {summary.get('shape', 'N/A')}")
        logger.info(f"Numeric columns: {summary.get('n_numeric_cols', 0)}")
        logger.info(f"Categorical columns: {summary.get('n_categorical_cols', 0)}")
        logger.info(f"Total missing: {summary.get('total_missing', 0)}")
        logger.info(f"Memory usage: {summary.get('memory_usage_mb', 0):.2f} MB")
        logger.info(f"Preprocessing steps: {', '.join(summary.get('preprocessing_steps', []))}")
        logger.info(f"{'='*70}\n")
