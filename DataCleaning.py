import pandas as pd
import numpy as np

class NLGDataCleaner:
    def __init__(self, dataframe, verbose=True):
        self.original_df = dataframe.copy()
        self.df = dataframe.copy()
        self.verbose = verbose
        self.report = {}

    def log(self, message):
        if self.verbose:
            print(message)

    def convert_datetime(self):
        self.log("🔄 Chuyển đổi cột 'time' sang datetime...")
        self.df['time'] = pd.to_datetime(self.df['time'], errors='coerce')
        return self

    def handle_missing_values(self):
        missing_before = self.df.isnull().sum().sum()
        self.log(f"🧹 Xử lý missing values: {missing_before} giá trị thiếu trước khi xử lý.")
        self.df.ffill(inplace=True)
        self.df.bfill(inplace=True)
        missing_after = self.df.isnull().sum().sum()
        self.log(f"✅ Còn lại {missing_after} missing sau xử lý.")
        self.report['missing_values_fixed'] = missing_before - missing_after
        return self

    def remove_duplicates(self):
        duplicates = self.df.duplicated().sum()
        self.log(f"🗑️ Số dòng trùng lặp cần loại bỏ: {duplicates}")
        self.df.drop_duplicates(inplace=True)
        self.report['duplicates_removed'] = duplicates
        return self

    def clean_outliers_iqr(self, columns=None):
        self.log("📊 Phát hiện và xử lý outlier bằng IQR...")
        outlier_cols = []
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        for col in columns:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            outliers = ((self.df[col] < lower) | (self.df[col] > upper)).sum()
            if outliers > 0:
                outlier_cols.append(col)
                self.df[col] = np.where(
                    (self.df[col] < lower) | (self.df[col] > upper),
                    np.nan,
                    self.df[col]
                )
        self.df.ffill(inplace=True)
        self.df.bfill(inplace=True)
        self.log(f"📌 Cột có outliers được xử lý: {outlier_cols}")
        self.report['outlier_columns'] = outlier_cols
        return self

    def standardize_units(self):
        self.log("🧮 Kiểm tra và chuẩn hóa đơn vị giá trị (triệu đồng)...")
        value_cols = [col for col in self.df.columns if 'value' in col]
        scaled_cols = []
        for col in value_cols:
            if self.df[col].max() > 1e9:
                self.df[col] = self.df[col] / 1e6
                scaled_cols.append(col)
        self.log(f"✅ Đã chuẩn hóa đơn vị cho các cột: {scaled_cols}")
        self.report['scaled_columns'] = scaled_cols
        return self

    def add_date_features(self):
        self.log("📅 Thêm đặc trưng thời gian từ cột 'time'...")
        self.df['year'] = self.df['time'].dt.year
        self.df['month'] = self.df['time'].dt.month
        self.df['dayofweek'] = self.df['time'].dt.dayofweek
        self.report['date_features_added'] = ['year', 'month', 'dayofweek']
        return self

    def summary(self):
        self.report['rows_before'] = self.original_df.shape[0]
        self.report['rows_after'] = self.df.shape[0]
        self.report['columns'] = self.df.shape[1]
        self.log("\n📋 TỔNG KẾT LÀM SẠCH DỮ LIỆU:")
        for key, value in self.report.items():
            print(f"- {key}: {value}")
        return self

    def run_all(self):
        return (self.convert_datetime()
                    .handle_missing_values()
                    .remove_duplicates()
                    .clean_outliers_iqr()
                    .standardize_units()
                    .add_date_features()
                    .summary()
                    .df)
