from src.monitoring.logger import logger
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

def evalute_model(model, X_test , y_test)->dict:
    """
    Evaluate trained ML model on test data.

    Args:
        model: Trained machine learning model
        X_test: Transformed test features
        y_test: True labels

    Returns:
        dict: Evaluation metrics
    """

    logger.info("Starting model evaluation")

    try:
        y_pred=model.predict( X_test)

    except Exception as e:
         logger.error(f"Prediction failed: {e}")
         raise
     
    accuracy=accuracy_score(y_pred, y_test)

    precision=precision_score(
       y_test,
       y_pred,
       average='weighted'
    )

    recall=recall_score(
       y_test,
       y_pred,
       average='weighted'
    )

    f1=f1_score(
       y_test,
       y_pred,
       average='weighted'
    )

    report = classification_report(
        y_test,
        y_pred
    )

    matrix = confusion_matrix(
        y_test,
        y_pred
    )


    logger.info(f"Accuracy: {accuracy:.4f}")

    logger.info(f"Precision: {precision:.4f}")

    logger.info(f"Recall: {recall:.4f}")

    logger.info(f"F1 Score: {f1:.4f}")

    logger.info(
        "\nClassification Report:\n"
        f"{report}"
    )

    logger.info(
        "\nConfusion Matrix:\n"
        f"{matrix}"
    )

    logger.info("Model evaluation completed")
    

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "classification_report": report,
        "confusion_matrix": matrix.tolist()
    }