import logging
from sklearn.linear_model import LogisticRegression


def train_model(X_train, y_train, config: dict):

    logging.info("Starting model training")

    # safety checks
    if X_train is None or X_train.shape[0] == 0:
        raise ValueError("X_train is empty")

    if y_train is None or len(y_train) == 0:
        raise ValueError("y_train is empty")

    model_config = config.get("model", {})
    model_type = model_config.get("type", "XGBoost")

    if model_type == "XGBoost":

        # ONLY valid sklearn params
        model_params = {
            "max_iter": 1000
        }

        model = LogisticRegression(**model_params)

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    try:
        model.fit(X_train, y_train)

    except Exception as e:
        logging.error(f"Training failed: {e}")
        raise

    logging.info("Model training completed successfully")

    return model