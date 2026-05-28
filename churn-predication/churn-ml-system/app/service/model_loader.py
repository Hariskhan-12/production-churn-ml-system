import joblib

from pathlib import Path



BASE_DIR = (Path(__file__).resolve().parent.parent.parent)



MODEL_PATH = (
    BASE_DIR
    / "models"
    / "v1"
    / "model.pkl"
)

PREPROCESSOR_PATH = (
    BASE_DIR
    / "models"
    / "v1"
    / "preprocessor.pkl"
)



model = None

preprocessor = None


def load_artifacts():

    global model
    global preprocessor

    model = joblib.load(
        MODEL_PATH
    )

    preprocessor = joblib.load(
        PREPROCESSOR_PATH
    )

    print("Artifacts loaded successfully")



def get_model():

    return model


def get_preprocessor():

    return preprocessor