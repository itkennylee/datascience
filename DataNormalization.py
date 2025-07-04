
# Import các thư viện cần thiết
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler


# Định nghĩa class DataNormalizer để chuẩn hóa dữ liệu số
class DataNormalizer:
    def __init__(self, dataframe, scaler_type='standard', exclude_columns=None, verbose=True):
        """
        Khởi tạo đối tượng DataNormalizer.
        Args:
            dataframe (pd.DataFrame): Dữ liệu đầu vào.
            scaler_type (str): Loại scaler sử dụng ('standard' hoặc 'robust').
            exclude_columns (list): Danh sách các cột không chuẩn hóa.
            verbose (bool): Có in log ra màn hình hay không.
        """
        self.df = dataframe.copy()  # Dữ liệu sẽ được chuẩn hóa
        self.scaler_type = scaler_type
        self.exclude_columns = exclude_columns if exclude_columns else []
        self.verbose = verbose
        self.scalers = {}  # Lưu các scaler đã fit cho từng cột
        self.report = {}   # Báo cáo kết quả chuẩn hóa

    def log(self, message):
        """In log nếu chế độ verbose bật."""
        if self.verbose:
            print(message)

    def detect_columns_to_scale(self):
        """
        Xác định các cột số cần chuẩn hóa, loại trừ các cột nhị phân, phần trăm, tỉ lệ, v.v.
        Returns:
            list: Danh sách các cột sẽ được chuẩn hóa.
        """
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        skip_keywords = ['percent', 'ratio', 'rate', 'return']
        cols_to_scale = []

        for col in numeric_cols:
            if col in self.exclude_columns:
                continue
            if any(kw in col.lower() for kw in skip_keywords):
                continue
            if self.df[col].nunique() <= 2:
                continue  # Bỏ qua cột nhị phân
            cols_to_scale.append(col)

        self.report['columns_to_normalize'] = cols_to_scale
        return cols_to_scale

    def get_scaler(self):
        """
        Lấy scaler phù hợp theo lựa chọn.
        Returns:
            sklearn Scaler object
        """
        if self.scaler_type == 'robust':
            return RobustScaler()
        return StandardScaler()

    def normalize(self):
        """
        Chuẩn hóa các cột số đã xác định bằng scaler tương ứng.
        Returns:
            self
        """
        cols = self.detect_columns_to_scale()
        for col in cols:
            scaler = self.get_scaler()
            original_vals = self.df[[col]].values
            # Chuẩn hóa giá trị cột
            self.df[col] = scaler.fit_transform(original_vals)
            self.scalers[col] = scaler
            self.report[col] = {
                'scaler': self.scaler_type,
                'min_before': float(np.min(original_vals)),
                'max_before': float(np.max(original_vals)),
                'min_after': float(np.min(self.df[col])),
                'max_after': float(np.max(self.df[col]))
            }
        self.log("Đã chuẩn hóa các cột sau: " + ", ".join(cols))
        return self

    def summary(self):
        """
        In ra tổng kết quá trình chuẩn hóa dữ liệu.
        """
        self.log("\nTỔNG KẾT CHUẨN HÓA DỮ LIỆU:")
        for col in self.report.get('columns_to_normalize', []):
            info = self.report[col]
            print(f"- {col}: [{info['min_before']:.2f} → {info['min_after']:.2f}], "
                  f"{info['max_before']:.2f} → {info['max_after']:.2f}] using {info['scaler']}")
        return self

    def run_all(self):
        """
        Chạy toàn bộ các bước chuẩn hóa dữ liệu.
        Returns:
            pd.DataFrame: Dữ liệu đã được chuẩn hóa.
        """
        return self.normalize().summary().df
