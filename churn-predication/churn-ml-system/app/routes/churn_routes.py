import logging

from fastapi import APIRouter, HTTPException

from app.schema.schema import (
    ChurnRequest,
    PredictionResponse
)

from app.service.predication_service import (
    PredictionService
)

logger = logging.getLogger(__name__)


router = APIRouter()



@router.post(
    "/predict",
    response_model=PredictionResponse
)
def predict_churn(
    request: ChurnRequest
):

    try:

        logger.info(
            "Prediction request received"
        )

        result = PredictionService.predict(
            request
        )

        logger.info(
            "Prediction completed successfully"
        )

        return result

    except Exception as e:

        logger.error(
            f"Prediction API failed: {str(e)}"
        )

        raise HTTPException(
            status_code=500,
            detail="Internal Server Error"
        )