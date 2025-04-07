from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import Client
import numpy as np
import os
import cloudpickle as pickle


router = APIRouter(prefix="/model", tags=["Model Management"])

# Define directory for pickled models
MODEL_DIR = "app/clients/service/MLmodels"


def load_client_training_data(db):
    clients = db.query(Client).all()
    X, y = [], []

    for c in clients:
        try:
            features = [
                c.age or 0,
                c.gender or 0,
                c.work_experience or 0,
                c.canada_workex or 0,
                c.dep_num or 0,
                c.reading_english_scale or 0,
                c.speaking_english_scale or 0,
                c.writing_english_scale or 0,
                c.numeracy_scale or 0,
                c.computer_scale or 0,
            ]
            label = 1 if c.currently_employed else 0
            X.append(features)
            y.append(label)
        except:
            continue  # skip bad records
    return X, y


class ModelManager:
    available_models = {
        "logistic_regression": LogisticRegression(),
        "random_forest": RandomForestClassifier(),
        "neural_network": MLPClassifier(),
    }
    current_model_name = "logistic_regression"
    current_model = available_models[current_model_name]

    @classmethod
    def load_pickled_models(cls):
        if not os.path.exists(MODEL_DIR):
            return
        for file in os.listdir(MODEL_DIR):
            if file.endswith(".pkl"):
                model_name = file.replace(".pkl", "")
                with open(os.path.join(MODEL_DIR, file), "rb") as f:
                    model = pickle.load(f)
                    cls.available_models[model_name] = model

    @classmethod
    def set_model(cls, model_name: str):
        if model_name not in cls.available_models:
            raise ValueError("Model not found")
        cls.current_model_name = model_name
        cls.current_model = cls.available_models[model_name]

    @classmethod
    def get_current_model(cls) -> str:
        return cls.current_model_name

    @classmethod
    def list_models(cls):
        return list(cls.available_models.keys())

    @classmethod
    def predict(cls, input_data: list):
        try:
            return cls.current_model.predict(np.array(input_data).reshape(1, -1)).tolist()
        except Exception as e:
            return str(e)


# Load models when module loads
#ModelManager.load_pickled_models()


@router.post("/train_models")
def train_models(db: Session = Depends(get_db)):
    X, y = load_client_training_data(db)
    if len(X) == 0:
        return {"error": "No training data found in the database."}
    for model in ModelManager.available_models.values():
        try:
            model.fit(X, y)
        except Exception as e:
            print(f"Error training {model}: {e}")
    return {"message": f"Trained {len(ModelManager.available_models)} models on {len(X)} samples."}


class PredictRequest(BaseModel):
    input: List[float]


@router.post("/predict")
def predict(request: PredictRequest):
    prediction = ModelManager.predict(request.input)
    return {"input": request.input, "prediction": prediction}


class SetModelRequest(BaseModel):
    model_name: str


@router.post("/set_model")
def set_model(request: SetModelRequest):
    try:
        ModelManager.set_model(request.model_name)
        return {"message": f"Switched to model: {request.model_name}"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/current_model")
def current_model():
    return {"current_model": ModelManager.get_current_model()}


@router.get("/list_models")
def list_models():
    return {"available_models": ModelManager.list_models()}
