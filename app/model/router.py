# from fastapi import APIRouter, HTTPException, Depends
# from pydantic import BaseModel
# from typing import List
# # from sklearn.linear_model import LogisticRegression
# # from sklearn.ensemble import RandomForestClassifier
# # from sklearn.neural_network import MLPClassifier

# from sklearn.ensemble import RandomForestRegressor
# from sklearn.neural_network import MLPRegressor
# from sklearn.linear_model import LinearRegression

# from sqlalchemy.orm import Session
# from app.database import get_db
# from app.models import Client
# from app.models import ClientCase
# import numpy as np
# import os
# import cloudpickle as pickle


# router = APIRouter(prefix="/model", tags=["Model Management"])

# # Define directory for pickled models
# MODEL_DIR = "app/clients/service/MLmodels"


# def load_client_training_data(db):
#     # clients = db.query(Client).all()
    
#     clients = db.query(Client).all()  # <-- all clients
#     print(f"Found {len(clients)} clients in database")  # 👈 debug here

#     clients_with_cases = [c for c in clients if c.cases]
#     print(f"Found {len(clients_with_cases)} clients with at least one case")  # 👈 debug here
    
#     clients = db.query(Client).join(ClientCase).all()
#     X, y = [], []

#     for c in clients:
#         try:
#             if not c.cases:
#                 continue  # skip clients without cases
#             features = [
#                 # c.age or 0,
#                 # c.gender or 0,
#                 # c.work_experience or 0,
#                 # c.canada_workex or 0,
#                 # c.dep_num or 0,
#                 # c.reading_english_scale or 0,
#                 # c.speaking_english_scale or 0,
#                 # c.writing_english_scale or 0,
#                 # c.numeracy_scale or 0,
#                 # c.computer_scale or 0,
#                 c.age or 0,
#                 c.gender or 0,
#                 c.work_experience or 0,
#                 c.canada_workex or 0,
#                 c.dep_num or 0,
#                 c.canada_born or 0,
#                 c.citizen_status or 0,
#                 c.level_of_schooling or 0,
#                 c.fluent_english or 0,
#                 c.reading_english_scale or 0,
#                 c.speaking_english_scale or 0,
#                 c.writing_english_scale or 0,
#                 c.numeracy_scale or 0,
#                 c.computer_scale or 0,
#                 c.transportation_bool or 0,
#                 c.caregiver_bool or 0,
#                 c.housing or 0,
#                 c.income_source or 0,
#                 c.felony_bool or 0,
#                 c.attending_school or 0,
#                 c.currently_employed or 0,
#                 c.substance_use or 0,
#                 c.time_unemployed or 0,
#                 c.need_mental_health_support_bool or 0,
#                 c.client_case.employment_assistance or 0,
#                 c.client_case.life_stabilization or 0,
#                 c.client_case.retention_services or 0,
#                 c.client_case.specialized_services or 0,
#                 c.client_case.employment_related_financial_supports or 0,
#                 c.client_case.employer_financial_supports or 0,
#                 c.client_case.enhanced_referrals or 0,
#             ]
#             # label = 1 if c.currently_employed else 0
#             # label = 1 if c.success_rate > 0 else 0
#             label = c.cases[0].success_rate or 0 
#             X.append(features)
#             y.append(label)
#         except:
#             continue  # skip bad records
#     return X, y


# class ModelManager:
#     available_models = {
#         # "logistic_regression": LogisticRegression(),
#         # "random_forest": RandomForestClassifier(),
#         # "neural_network": MLPClassifier(),
#         "linear_regression": LinearRegression(),
#         "random_forest_regressor": RandomForestRegressor(),
#         "mlp_regressor": MLPRegressor(),
#     }
#     current_model_name = "linear_regression"
#     current_model = available_models[current_model_name]

#     @classmethod
#     def load_pickled_models(cls):
#         if not os.path.exists(MODEL_DIR):
#             return
#         for file in os.listdir(MODEL_DIR):
#             if file.endswith(".pkl"):
#                 model_name = file.replace(".pkl", "")
#                 with open(os.path.join(MODEL_DIR, file), "rb") as f:
#                     model = pickle.load(f)
#                     cls.available_models[model_name] = model

#     @classmethod
#     def set_model(cls, model_name: str):
#         if model_name not in cls.available_models:
#             raise ValueError("Model not found")
#         cls.current_model_name = model_name
#         cls.current_model = cls.available_models[model_name]

#     @classmethod
#     def get_current_model(cls) -> str:
#         return cls.current_model_name

#     @classmethod
#     def list_models(cls):
#         return list(cls.available_models.keys())

#     @classmethod
#     def predict(cls, input_data: list):
#         try:
#             return cls.current_model.predict(np.array(input_data).reshape(1, -1)).tolist()
#         except Exception as e:
#             return str(e)


# # Load models when module loads
# ModelManager.load_pickled_models()


# # @router.post("/train_models")
# # def train_models(db: Session = Depends(get_db)):
# #     X, y = load_client_training_data(db)
# #     if len(X) == 0:
# #         return {"error": "No training data found in the database."}
# #     for model in ModelManager.available_models.values():
# #         try:
# #             model.fit(X, y)
# #         except Exception as e:
# #             print(f"Error training {model}: {e}")
# #     return {"message": f"Trained {len(ModelManager.available_models)} models on {len(X)} samples."}

# @router.post("/train_models")
# def train_models(db: Session = Depends(get_db)):
#     X, y = load_client_training_data(db)
#     print(f"Loaded {len(X)} training samples")  # Debug

#     if len(X) == 0:
#         return {"error": "No training data found in the database."}
    
#     if not os.path.exists(MODEL_DIR):
#         os.makedirs(MODEL_DIR)  # Make sure directory exists

#     for model_name, model in ModelManager.available_models.items():
#         try:
#             model.fit(X, y)
#             # Save each trained model
#             with open(os.path.join(MODEL_DIR, f"{model_name}.pkl"), "wb") as f:
#                 pickle.dump(model, f)
#             print(f"Trained and saved model: {model_name}")
#         except Exception as e:
#             print(f"Error training {model_name}: {e}")
#     return {"message": f"Trained and saved {len(ModelManager.available_models)} models on {len(X)} samples."}


# class PredictRequest(BaseModel):
#     input: List[float]


# @router.post("/predict")
# def predict(request: PredictRequest):
#     prediction = ModelManager.predict(request.input)
#     return {"input": request.input, "prediction": prediction}


# class SetModelRequest(BaseModel):
#     model_name: str


# @router.post("/set_model")
# def set_model(request: SetModelRequest):
#     try:
#         ModelManager.set_model(request.model_name)
#         return {"message": f"Switched to model: {request.model_name}"}
#     except ValueError as e:
#         raise HTTPException(status_code=400, detail=str(e))


# @router.get("/current_model")
# def current_model():
#     return {"current_model": ModelManager.get_current_model()}


# @router.get("/list_models")
# def list_models():
#     return {"available_models": ModelManager.list_models()}


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
            # Make sure client has at least one case
            if not c.cases:
                continue

            # Build feature list exactly matching your CSV example
            features = [
                c.age or 0,
                c.gender or 0,
                c.work_experience or 0,
                c.canada_workex or 0,
                c.dep_num or 0,
                c.canada_born or 0,
                c.citizen_status or 0,
                c.level_of_schooling or 0,
                c.fluent_english or 0,
                c.reading_english_scale or 0,
                c.speaking_english_scale or 0,
                c.writing_english_scale or 0,
                c.numeracy_scale or 0,
                c.computer_scale or 0,
                c.transportation_bool or 0,
                c.caregiver_bool or 0,
                c.housing or 0,
                c.income_source or 0,
                c.felony_bool or 0,
                c.attending_school or 0,
                c.currently_employed or 0,
                c.substance_use or 0,
                c.time_unemployed or 0,
                c.need_mental_health_support_bool or 0,
                # Intervention fields - from first linked case
                c.cases[0].employment_assistance or 0,
                c.cases[0].life_stabilization or 0,
                c.cases[0].retention_services or 0,
                c.cases[0].specialized_services or 0,
                c.cases[0].employment_related_financial_supports or 0,
                c.cases[0].employer_financial_supports or 0,
                c.cases[0].enhanced_referrals or 0,
            ]

            # For y/label: now predict the *success_rate* instead of employment
            label = c.cases[0].success_rate or 0

            X.append(features)
            y.append(label)

        except Exception as e:
            print(f"Skipping bad record: {e}")
            continue

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
ModelManager.load_pickled_models()


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