import pandas as pd
import numpy as np
import missingno as msno
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.decomposition import PCA
import seaborn as sns
from imblearn.over_sampling import SMOTE

class Datapreprocessor:

    # 1️⃣ Load Data
    def load_data(self, filepath):
        try:
            if filepath.endswith(".csv"):
                return pd.read_csv(filepath)
            elif filepath.endswith(".parquet"):
                return pd.read_parquet(filepath)
            elif filepath.endswith(".xlsx") or filepath.endswith(".xls"):
                return pd.read_excel(filepath)
            else:
                raise ValueError("Unsupported file format. Only CSV, Parquet, XLSX, or XLS are allowed.")
        except Exception as e:
            print(f"Error loading file '{filepath}': {e}")
            raise

    # 2️⃣ Phase 1: Exploration
    def explore_data(self, data):
        print("\n--- Dataset Overview ---")
        print(data.head())
        print("\n--- Info ---")
        data.info()
        print("\n--- Summary Statistics ---")
        print(data.describe())
        print("\n--- Missing Values ---")
        print(data.isnull().sum())
        print("\n--- Dataset Shape ---")
        print(f"Rows: {data.shape[0]}, Columns: {data.shape[1]}")

    def identify_issues(self, data):
        cols_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        print("\n--- Biological Impossibilities (Zeros) ---")
        for col in cols_with_zero:
            if col in data.columns:
                print(f"{col}: {(data[col] == 0).sum()} zeros")
        print("\n--- Data Types ---")
        print(data.dtypes)

        print("\n--- Potential Outliers (IQR method) ---")
        for col in cols_with_zero:
            if col in data.columns and pd.api.types.is_numeric_dtype(data[col]):
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                outliers = ((data[col] < lower) | (data[col] > upper)).sum()
                print(f"{col}: {outliers} potential outliers")

    # 3️⃣ Missing Value Analysis
    def missing_value_analysis(self, data):
        cols_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        for col in cols_with_zero:
            if col in data.columns:
                data[col] = data[col].replace(0, np.nan)

        missing_percent = (data.isnull().sum() / len(data)) * 100
        print("\n--- Missing Value Percentages ---")
        print(missing_percent.round(2))

        # Visualize missing data
        print("\n--- Missing Data Matrix ---")
        msno.matrix(data)
        plt.show()

        print("\n--- Missing Data Heatmap ---")
        msno.heatmap(data)
        plt.show()

        return data

    # 4️⃣ Imputation
    def impute_missing(self, data, strategy='median'):
        numeric_cols = data.select_dtypes(include=np.number).columns
        for col in numeric_cols:
            if strategy == 'median':
                data[col] = data[col].fillna(data[col].median())
            elif strategy == 'mean':
                data[col] = data[col].fillna(data[col].mean())
            elif strategy == 'mode':
                data[col] = data[col].fillna(data[col].mode()[0])
            else:
                raise ValueError("Use 'median', 'mean', or 'mode'")
        return data

    # 5️⃣ Outlier Detection & Treatment
    def treat_outliers(self, data, method='IQR'):
        numeric_cols = data.select_dtypes(include=np.number).columns
        for col in numeric_cols:
            if method == 'IQR':
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                data[col] = np.where(data[col] < lower, lower, data[col])
                data[col] = np.where(data[col] > upper, upper, data[col])
            elif method == 'Z-score':
                z_scores = np.abs(stats.zscore(data[col]))
                data = data[z_scores < 3]  # remove rows with z-score > 3
            else:
                raise ValueError("Method must be 'IQR' or 'Z-score'.")
        return data

    # 6️⃣ Feature Engineering
    def feature_engineering(self, data):
        bins_age = [0, 34, 60, np.inf]
        labels_age = ['young', 'middle-aged', 'senior']
        data['AgeGroup'] = pd.cut(data['Age'], bins=bins_age, labels=labels_age)

        bins_bmi = [0, 18.5, 24.9, 29.9, np.inf]
        labels_bmi = ['underweight', 'normal', 'overweight', 'obese']
        data['BMICategory'] = pd.cut(data['BMI'], bins_bmi, labels=labels_bmi)

        bins_glucose = [0, 139, 199, np.inf]
        labels_glucose = ['normal', 'prediabetes', 'diabetes']
        data['GlucoseCategory'] = pd.cut(data['Glucose'], bins=bins_glucose, labels=labels_glucose)
        return data

    # 7️⃣ Encoding
    def encode_features(self, data):
        le = LabelEncoder()
        for col in ['AgeGroup', 'BMICategory', 'GlucoseCategory']:
            if col in data.columns:
                data[col] = le.fit_transform(data[col])
        return data

    # 8️⃣ Scaling
    def scale_features(self, data, method='standard'):
        numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError("Use 'standard' or 'minmax' scaling")
        data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
        return data

    # 9️⃣ Correlation Matrix
    def correlation_matrix(self, data):
        corr = data.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm')
        plt.title('Correlation Matrix')
        plt.show()
        return corr

    # 🔟 Feature Selection
    def select_k_best_features(self, X, y, k=5):
        if y.dtype != int:
            y = y.astype(int)
        selector = SelectKBest(score_func=mutual_info_classif, k=k)
        selector.fit(X, y)
        selected_features = X.columns[selector.get_support()]
        print(f"Top {k} features selected by mutual information: {list(selected_features)}")
        return selected_features

    # 1️⃣1️⃣ PCA
    def perform_pca(self, X, n_components=None):
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)
        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = explained_variance.cumsum()

        plt.figure(figsize=(8,5))
        plt.plot(range(1, len(explained_variance)+1), cumulative_variance, marker='o', linestyle='--')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('PCA Explained Variance')
        plt.grid(True)
        plt.show()

        print("Explained variance by component:", explained_variance)
        return X_pca, explained_variance, cumulative_variance

    # 1️⃣2️⃣ Class Distribution
    def analyze_class_distribution(self, y):
        class_counts = y.value_counts()
        print("Class distribution:\n", class_counts)
        imbalance_ratio = class_counts.min() / class_counts.max()
        print(f"Imbalance ratio (minority/majority): {imbalance_ratio:.2f}")

        plt.figure(figsize=(6,4))
        class_counts.plot(kind='bar', color=['skyblue', 'salmon'])
        plt.title('Target Class Distribution')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.show()

        return imbalance_ratio

    # 1️⃣3️⃣ Balancing
    def balance_data(self, X, y, method='SMOTE'):
        if method == 'SMOTE':
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)
        else:
            raise ValueError("Currently only 'SMOTE' is supported for balancing")
        print("Class distribution after balancing:\n", y_resampled.value_counts())
        return X_resampled, y_resampled

    # ✅ 1️⃣4️⃣ Save Clean Data
    def save_clean_data(self, data, filepath):
        """
        Save the cleaned dataset to a specified file format.
        Supported: .csv, .xlsx, .parquet
        """
        try:
            if filepath.endswith(".csv"):
                data.to_csv(filepath, index=False)
            elif filepath.endswith(".xlsx") or filepath.endswith(".xls"):
                data.to_excel(filepath, index=False)
            elif filepath.endswith(".parquet"):
                data.to_parquet(filepath, index=False)
            else:
                raise ValueError("Unsupported file format. Use .csv, .xlsx, or .parquet")
            print(f"✅ Cleaned data successfully saved to '{filepath}'")
        except Exception as e:
            print(f"❌ Error saving cleaned data: {e}")
    def save_data_dictionary(self, data, filepath):
        
        try:
            data_dict = pd.DataFrame({
                "Feature": data.columns,
                "Data Type": data.dtypes.astype(str),
                "Unique Values": [data[col].nunique() for col in data.columns],
                "Missing Values": [data[col].isnull().sum() for col in data.columns],
                "Min Value": [data[col].min() if pd.api.types.is_numeric_dtype(data[col]) else "" for col in data.columns],
                "Max Value": [data[col].max() if pd.api.types.is_numeric_dtype(data[col]) else "" for col in data.columns],
                "Mean": [data[col].mean() if pd.api.types.is_numeric_dtype(data[col]) else "" for col in data.columns],
                "Median": [data[col].median() if pd.api.types.is_numeric_dtype(data[col]) else "" for col in data.columns],
                "Std": [data[col].std() if pd.api.types.is_numeric_dtype(data[col]) else "" for col in data.columns],
                "Example Value": [data[col].iloc[0] for col in data.columns],
               
            })

            if filepath.endswith(".csv"):
                data_dict.to_csv(filepath, index=False)
            elif filepath.endswith(".xlsx") or filepath.endswith(".xls"):
                data_dict.to_excel(filepath, index=False)
            else:
                raise ValueError("Unsupported file format. Use .csv or .xlsx")

            print(f"✅ Data dictionary successfully saved to '{filepath}'")
        except Exception as e:
            print(f"❌ Error saving data dictionary: {e}")