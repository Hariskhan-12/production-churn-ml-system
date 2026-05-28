import logging
import os
import joblib
import pandas as pd


from src.data.ingest import DataIngestion
from src.features.preprocess import DataPreprocessor
from sklearn.model_selection import train_test_split
from src.models.trainer import train_model
from src.models.evaluate import evalute_model
from src.utils.config import ConfigLoader


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class TraningPipeline:
    def __init__(self):
        self.config = ConfigLoader().load_config()

        self.model_dir = self.config['artifacts']['model_dir']

        self.model_path = os.path.join(
            self.model_dir,
            self.config['artifacts']['model_name']
        )

        self.preprocessor_path = os.path.join(
            self.model_dir,
            self.config["artifacts"]["preprocessor_name"]
        )

    def run(self):

        logging.info("Training started")

        # 1. Data ingestion
        ingestion = DataIngestion(self.config)
        df = ingestion.load_data()

        logging.info(f"Raw data shape: {df.shape}")

        # 2. Preprocessing
        preprocessor_obj = DataPreprocessor(self.config)
        X, y, preprocessor = preprocessor_obj.preprocess(df)

        logging.info(f"Feature shape: {X.shape}")

        # 3. Split
        test_size = self.config['model']['test_size']
        random_state = self.config['model']['random_state']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state
        )

        logging.info(f"X_train shape: {X_train.shape}")
        logging.info(f"X_test shape: {X_test.shape}")

        # 4. Transform
        X_train_transformed = preprocessor.fit_transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)

        # 5. Train
        model = train_model(
            X_train_transformed,
            y_train,
            self.config
        )

        # 6. Evaluate
        metrics = evalute_model(
            model,
            X_test_transformed,
            y_test
        )

        logging.info(f"Metrics: {metrics}")

        # 7. Save
        os.makedirs(self.model_dir, exist_ok=True)

        joblib.dump(model, self.model_path)
        joblib.dump(preprocessor, self.preprocessor_path)

        logging.info("Training completed successfully")


if __name__ == "__main__":
    pipeline = TraningPipeline()
    pipeline.run()