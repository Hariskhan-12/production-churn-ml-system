from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from src.monitoring.logger import logger
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import pandas as pd


class DataPreprocessor:

    def __init__(self, config: dict):

        self.config = config
        self.target = config["feature"]["target"]

    def build_pipeline(self, df):

        # remove duplicates
        duplicates = df.duplicated().sum()

        if duplicates > 0:
            df = df.drop_duplicates()

        logger.info(f"Duplicates removed: {duplicates}")

        # check target
        if self.target not in df.columns:
            raise ValueError("Target variable not found in dataframe")

        # feature separation
        num_cols = df.select_dtypes(
            include=["int64", "float64"]
        ).columns.tolist()

        cat_cols = df.select_dtypes(
            include=["object"]
        ).columns.tolist()

        # remove target from features
        if self.target in num_cols:
            num_cols.remove(self.target)

        if self.target in cat_cols:
            cat_cols.remove(self.target)

        logger.info(f"Numeric columns: {num_cols}")

        logger.info(f"Categorical columns: {cat_cols}")

        # numerical pipeline
        num_pipeline = Pipeline([

            ("imputer", SimpleImputer(strategy="median")),

            ("scaler", StandardScaler())

        ])

        # categorical pipeline
        cat_pipeline = Pipeline([

            ("imputer", SimpleImputer(strategy="most_frequent")),

            ("onehot", OneHotEncoder(handle_unknown="ignore"))

        ])

        # full preprocessor
        preprocessor = ColumnTransformer([

            ("num", num_pipeline, num_cols),

            ("cat", cat_pipeline, cat_cols)

        ])

        return preprocessor

    def preprocess(self, df: pd.DataFrame):

        logger.info("Starting preprocessing")

        # convert TotalCharges safely
        if "TotalCharges" in df.columns:

            df["TotalCharges"] = pd.to_numeric(
                df["TotalCharges"],
                errors="coerce"
            )

        # split features and target
        X = df.drop(self.target, axis=1)

        y = df[self.target]

        # build preprocessing pipeline
        preprocessor = self.build_pipeline(df)

        logger.info("Preprocessing completed")

        return X, y, preprocessor    