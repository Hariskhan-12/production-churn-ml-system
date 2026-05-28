from pydantic import BaseModel, Field
from typing import Literal


class ChurnRequest(BaseModel):

    gender: Literal["Male", "Female"]

    SeniorCitizen: int = Field(ge=0, le=1)

    Partner: Literal["Yes", "No"]

    Dependents: Literal["Yes", "No"]

    tenure: int = Field(ge=0)

    PhoneService: Literal["Yes", "No"]

    MultipleLines: str

    InternetService: str

    OnlineSecurity: str

    OnlineBackup: str

    DeviceProtection: str

    TechSupport: str

    StreamingTV: str

    StreamingMovies: str

    Contract: str

    PaperlessBilling: Literal["Yes", "No"]

    PaymentMethod: str

    MonthlyCharges: float = Field(ge=0)

    TotalCharges: float = Field(ge=0)


class PredictionResponse(BaseModel):

    prediction: str
    probability: float