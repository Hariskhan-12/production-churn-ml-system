import pandas as pd

from app.service.model_loader import (
    get_model,
    get_preprocessor
)


class PredictionService:

    @staticmethod
    def predict(data):

        model = get_model()

        preprocessor = get_preprocessor()

        df = pd.DataFrame([data])

        required_columns = [
            "customerID",
            "gender",
            "SeniorCitizen",
            "Partner",
            "Dependents",
            "tenure",
            "PhoneService",
            "MultipleLines",
            "InternetService",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
            "Contract",
            "PaperlessBilling",
            "PaymentMethod",
            "MonthlyCharges",
            "TotalCharges"
        ]

        for col in required_columns:

            if col not in df.columns:

                if col in [
                    "SeniorCitizen",
                    "tenure",
                    "MonthlyCharges",
                    "TotalCharges"
                ]:
                    df[col] = 0

                else:
                    df[col] = "Unknown"

        df = df[required_columns]

        transformed_data = preprocessor.transform(df)

        prediction = model.predict(
            transformed_data
        )[0]

        probability = model.predict_proba(
            transformed_data
        )[0][1]

        return {
            "prediction": str(prediction),
            "probability": float(probability)
        }