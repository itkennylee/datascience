import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler

class NLGDataNormalizer:
    def __init__(self, dataframe, scaler_type='standard', exclude_columns=None, verbose=True):
        self.df = dataframe.copy()
        self.scaler_type = scaler_type
        self.exclude_columns = exclude_columns if exclude_columns else []
        self.verbose = verbose
        self.scalers = {}
        self.report = {}

    def log(self, message):
        if self.verbose:
            print(message)

    def detect_columns_to_scale(self):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        skip_keywords = ['percent', 'ratio', 'rate', 'return']
        cols_to_scale = []

        for col in numeric_cols:
            if col in self.exclude_columns:
                continue
            if any(kw in col.lower() for kw in skip_keywords):
                continue
            if self.df[col].nunique() <= 2:
                continue  # nhị phân
            cols_to_scale.append(col)

        self.report['columns_to_normalize'] = cols_to_scale
        return cols_to_scale

    def get_scaler(self):
        if self.scaler_type == 'robust':
            return RobustScaler()
        return StandardScaler()

    def normalize(self):
        cols = self.detect_columns_to_scale()
        for col in cols:
            scaler = self.get_scaler()
            original_vals = self.df[[col]].values
            self.df[col] = scaler.fit_transform(original_vals)
            self.scalers[col] = scaler
            self.report[col] = {
                'scaler': self.scaler_type,
                'min_before': float(np.min(original_vals)),
                'max_before': float(np.max(original_vals)),
                'min_after': float(np.min(self.df[col])),
                'max_after': float(np.max(self.df[col]))
            }
        self.log("✅ Đã chuẩn hóa các cột sau: " + ", ".join(cols))
        return self

    def summary(self):
        self.log("\n📋 TỔNG KẾT CHUẨN HÓA DỮ LIỆU:")
        for col in self.report.get('columns_to_normalize', []):
            info = self.report[col]
            print(f"- {col}: [{info['min_before']:.2f} → {info['min_after']:.2f}], "
                  f"{info['max_before']:.2f} → {info['max_after']:.2f}] using {info['scaler']}")
        return self

    def run_all(self):
        return self.normalize().summary().df
