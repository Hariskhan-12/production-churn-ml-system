import pandas as pd
from pathlib import Path
from src.monitoring.logger import logger


class DataIngestion:
    def __init__(self, config: dict):
        self.config = config
        self.file_path = Path(config["data"]["path"])
        self.expected_columns = set(config["schema"]["expected_columns"])
        self.expected_dtype=config['schema']['expected_dtypes']
    def load_data(self) -> pd.DataFrame:

        # 1. Check file exists
        if not self.file_path.exists():
            logger.error(f"File not found: {self.file_path}")
            raise FileNotFoundError(f"File not found: {self.file_path}")

        # 2. Load data safely
        try:
            logger.info(f"Loading data from {self.file_path}")
            df = pd.read_csv(self.file_path)
        except Exception as e:
            logger.error(f"Error reading file: {e}")
            raise

        # 3. Check empty
        if df.empty:
            logger.error("Data file is empty")
            raise ValueError("Data file is empty")

        # 4. Validate schema
        actual_columns = set(df.columns)

        missing_columns = self.expected_columns - actual_columns
        if missing_columns:
            logger.error(f"Missing columns: {missing_columns}")
            raise ValueError(f"Missing columns: {missing_columns}")

        #log unexpected columns
        extra_columns = actual_columns - self.expected_columns
        if extra_columns:
            logger.warning(f"Unexpected columns found: {extra_columns}")

        logger.info(f"Data loaded successfully with shape {df.shape}")

       
        return df

