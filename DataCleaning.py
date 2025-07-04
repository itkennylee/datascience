
# Import các thư viện cần thiết lần 2
import pandas as pd
import numpy as np


# Định nghĩa class DataCleaner để làm sạch dữ liệu 
class DataCleaner:
    def __init__(self, dataframe, verbose=True):
        """
        Khởi tạo đối tượng NLGDataCleaner.
        Args:
            dataframe (pd.DataFrame): Dữ liệu đầu vào.
            verbose (bool): Có in log ra màn hình hay không.
        """
        self.original_df = dataframe.copy()  # Lưu bản gốc
        self.df = dataframe.copy()           # Dữ liệu sẽ được xử lý
        self.verbose = verbose
        self.report = {}                    # Báo cáo kết quả làm sạch

    def log(self, message):
        """In log nếu chế độ verbose bật."""
        if self.verbose:
            print(message)

    def convert_datetime(self):
        """
        Chuyển đổi cột 'time' sang kiểu datetime.
        """
        self.log("Chuyển đổi cột 'time' sang datetime...")
        self.df['time'] = pd.to_datetime(self.df['time'], errors='coerce')
        return self

    def handle_missing_values(self):
        """
        Xử lý các giá trị thiếu bằng phương pháp forward fill và backward fill.
        """
        missing_before = self.df.isnull().sum().sum()
        self.log(f"Xử lý missing values: {missing_before} giá trị thiếu trước khi xử lý.")
        self.df.ffill(inplace=True)
        self.df.bfill(inplace=True)
        missing_after = self.df.isnull().sum().sum()
        self.log(f"Còn lại {missing_after} missing sau xử lý.")
        self.report['missing_values_fixed'] = missing_before - missing_after
        return self

    def remove_duplicates(self):
        """
        Loại bỏ các dòng trùng lặp trong DataFrame.
        """
        duplicates = self.df.duplicated().sum()
        self.log(f"Số dòng trùng lặp cần loại bỏ: {duplicates}")
        self.df.drop_duplicates(inplace=True)
        self.report['duplicates_removed'] = duplicates
        return self

    def clean_outliers_iqr(self, columns=None):
        """
        Phát hiện và xử lý outlier bằng phương pháp IQR cho các cột số.
        Args:
            columns (list, optional): Danh sách các cột cần xử lý. Nếu None sẽ tự động chọn các cột số.
        """
        self.log("Phát hiện và xử lý outlier bằng IQR...")
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
                # Gán các giá trị outlier thành NaN
                self.df[col] = np.where(
                    (self.df[col] < lower) | (self.df[col] > upper),
                    np.nan,
                    self.df[col]
                )
        # Điền lại các giá trị NaN sau khi loại bỏ outlier
        self.df.ffill(inplace=True)
        self.df.bfill(inplace=True)
        self.log(f"Cột có outliers được xử lý: {outlier_cols}")
        self.report['outlier_columns'] = outlier_cols
        return self

    def standardize_units(self):
        """
        Chuẩn hóa đơn vị các cột giá trị về triệu đồng nếu phát hiện đơn vị lớn hơn.
        """
        self.log("Kiểm tra và chuẩn hóa đơn vị giá trị (triệu đồng)...")
        value_cols = [col for col in self.df.columns if 'value' in col]
        scaled_cols = []
        for col in value_cols:
            if self.df[col].max() > 1e9:
                self.df[col] = self.df[col] / 1e6
                scaled_cols.append(col)
        self.log(f"Đã chuẩn hóa đơn vị cho các cột: {scaled_cols}")
        self.report['scaled_columns'] = scaled_cols
        return self

    def add_date_features(self):
        """
        Thêm các đặc trưng thời gian (year, month, dayofweek) từ cột 'time'.
        """
        self.log("Thêm đặc trưng thời gian từ cột 'time'...")
        self.df['year'] = self.df['time'].dt.year
        self.df['month'] = self.df['time'].dt.month
        self.df['dayofweek'] = self.df['time'].dt.dayofweek
        self.report['date_features_added'] = ['year', 'month', 'dayofweek']
        return self

    def summary(self):
        """
        In ra tổng kết quá trình làm sạch dữ liệu.
        """
        self.report['rows_before'] = self.original_df.shape[0]
        self.report['rows_after'] = self.df.shape[0]
        self.report['columns'] = self.df.shape[1]
        self.log("\nTỔNG KẾT LÀM SẠCH DỮ LIỆU:")

        for key, value in self.report.items():
            print(f"- {key}: {value}")
        return self

    def run_all(self):
        """
        Chạy toàn bộ các bước làm sạch dữ liệu theo thứ tự.
        Returns:
            pd.DataFrame: Dữ liệu đã được làm sạch.
        """
        return (self.convert_datetime()
                    .handle_missing_values()
                    .remove_duplicates()
                    .clean_outliers_iqr()
                    .standardize_units()
                    .add_date_features()
                    .summary()
                    .df)
