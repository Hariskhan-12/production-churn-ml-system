from fastapi import FastAPI
from app.service.model_loader import load_artifacts
from app.routes.churn_routes import router
import logging

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s - "
        "%(levelname)s - "
        "%(message)s"
    )
)

app=FastAPI(title='ML Churn Predication API',
        description=(
        "Production-grade ML inference API "
        "for customer churn prediction"
    ),
    version="1.0.0"
            )
@app.on_event('startup')

def startup_event():
    logging.info(
        "Application startup initiated"
    )

    load_artifacts()

    logging.info(
        "Artifacts loaded successfully"
    )


@app.get("/health")
def health_check():

    return {
        "status": "healthy"
    }

app.include_router(router)

@app.get("/")
def root():

    return {
        "message": (
            "Churn Prediction API Running"
        )
    }