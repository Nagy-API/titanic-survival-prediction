from pathlib import Path
from typing import Literal, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "model.pkl"

model_data = joblib.load(MODEL_PATH)

model = model_data["model"]
scaler = model_data["scaler"]
columns = model_data["columns"]


app = FastAPI(
    title="Titanic Survival Prediction API",
    description="An API that predicts whether a Titanic passenger is likely to survive or not.",
    version="1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TitanicInput(BaseModel):
    Pclass: Literal[1, 2, 3]
    Name: str = Field(..., min_length=1, description="Passenger name, including title such as Mr, Mrs, or Miss")
    Sex: Literal["male", "female"]
    Age: float = Field(..., ge=0, le=100)
    SibSp: int = Field(..., ge=0)
    Parch: int = Field(..., ge=0)
    Fare: float = Field(..., ge=0)
    Cabin: Optional[str] = None
    Embarked: Literal["S", "C", "Q"]


@app.get("/")
def home():
    return {"message": "Titanic Survival Prediction API is running."}


@app.post("/predict")
def predict(data: TitanicInput):
    input_data = pd.DataFrame([data.model_dump()])

    input_data["Title"] = input_data["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)

    input_data["Title"] = input_data["Title"].replace({
        "Mlle": "Miss",
        "Ms": "Miss",
        "Mme": "Mrs",
        "Dr": "Rare",
        "Rev": "Rare",
        "Major": "Rare",
        "Col": "Rare",
        "Don": "Rare",
        "Lady": "Rare",
        "Sir": "Rare",
        "Capt": "Rare",
        "Countess": "Rare",
        "Jonkheer": "Rare",
    })

    input_data["HasCabin"] = input_data["Cabin"].notnull().astype(int)
    input_data["FamilySize"] = input_data["SibSp"] + input_data["Parch"] + 1
    input_data["IsAlone"] = (input_data["FamilySize"] == 1).astype(int)
    input_data["LogFare"] = np.log1p(input_data["Fare"])

    final_data = pd.DataFrame(0, index=[0], columns=columns)

    final_data["Sex"] = input_data["Sex"].map({"male": 0, "female": 1}).iloc[0]
    final_data["Age"] = input_data["Age"].iloc[0]
    final_data["HasCabin"] = input_data["HasCabin"].iloc[0]
    final_data["FamilySize"] = input_data["FamilySize"].iloc[0]
    final_data["IsAlone"] = input_data["IsAlone"].iloc[0]
    final_data["LogFare"] = input_data["LogFare"].iloc[0]

    embarked = input_data["Embarked"].iloc[0]

    if embarked == "Q" and "Embarked_Q" in final_data.columns:
        final_data["Embarked_Q"] = 1

    if embarked == "S" and "Embarked_S" in final_data.columns:
        final_data["Embarked_S"] = 1

    title = input_data["Title"].iloc[0]

    if title == "Miss" and "Title_Miss" in final_data.columns:
        final_data["Title_Miss"] = 1

    if title == "Mr" and "Title_Mr" in final_data.columns:
        final_data["Title_Mr"] = 1

    if title == "Mrs" and "Title_Mrs" in final_data.columns:
        final_data["Title_Mrs"] = 1

    if title == "Rare" and "Title_Rare" in final_data.columns:
        final_data["Title_Rare"] = 1

    pclass = input_data["Pclass"].iloc[0]

    if pclass == 2 and "Pclass_2" in final_data.columns:
        final_data["Pclass_2"] = 1

    if pclass == 3 and "Pclass_3" in final_data.columns:
        final_data["Pclass_3"] = 1

    final_data_scaled = scaler.transform(final_data)

    prediction = model.predict(final_data_scaled)[0]
    probability = model.predict_proba(final_data_scaled)[0][1]

    result = "Survived" if prediction == 1 else "Not Survived"

    return {
        "prediction": result,
        "prediction_code": int(prediction),
        "survival_probability": round(float(probability), 4),
    }
