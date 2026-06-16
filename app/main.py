from pathlib import Path
import sqlite3
from typing import Literal, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

# Importing this module is important because the saved Joblib pipeline uses
# TitanicFeatureEngineer from app.ml_pipeline.
from app.ml_pipeline import TitanicFeatureEngineer  # noqa: F401


BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
MODEL_PATH = ROOT_DIR / "model" / "model.pkl"
DB_DIR = ROOT_DIR / "database"
DB_PATH = DB_DIR / "predictions.db"

model = joblib.load(MODEL_PATH)

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

app = FastAPI(
    title="Titanic Survival Prediction API",
    description="A FastAPI app that predicts Titanic passenger survival and stores prediction history.",
    version="3.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount(
    "/static",
    StaticFiles(directory=str(BASE_DIR / "static")),
    name="static",
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


def to_dict(data: TitanicInput) -> dict:
    """Support both Pydantic v1 and v2."""
    if hasattr(data, "model_dump"):
        return data.model_dump()
    return data.dict()


def init_db() -> None:
    DB_DIR.mkdir(exist_ok=True)

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pclass INTEGER NOT NULL,
                name TEXT NOT NULL,
                sex TEXT NOT NULL,
                age REAL NOT NULL,
                sibsp INTEGER NOT NULL,
                parch INTEGER NOT NULL,
                fare REAL NOT NULL,
                cabin TEXT,
                embarked TEXT NOT NULL,
                prediction_code INTEGER NOT NULL,
                prediction_label TEXT NOT NULL,
                survival_probability REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()


def normalize_form_data(
    pclass: int,
    name: str,
    sex: str,
    age: float,
    sibsp: int,
    parch: int,
    fare: float,
    cabin: Optional[str],
    embarked: str,
) -> dict:
    clean_cabin = cabin.strip() if cabin else None
    if clean_cabin == "":
        clean_cabin = None

    return {
        "Pclass": int(pclass),
        "Name": name.strip(),
        "Sex": sex,
        "Age": float(age),
        "SibSp": int(sibsp),
        "Parch": int(parch),
        "Fare": float(fare),
        "Cabin": clean_cabin,
        "Embarked": embarked,
    }


def run_prediction(passenger_data: dict) -> dict:
    input_data = pd.DataFrame([passenger_data])

    prediction_code = int(model.predict(input_data)[0])
    survival_probability = float(model.predict_proba(input_data)[0][1])
    prediction_label = "Survived" if prediction_code == 1 else "Not Survived"

    return {
        "prediction": prediction_label,
        "prediction_code": prediction_code,
        "survival_probability": round(survival_probability, 4),
        "survival_percentage": round(survival_probability * 100, 1),
    }


def save_prediction(passenger_data: dict, prediction_result: dict) -> None:
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO predictions (
                pclass,
                name,
                sex,
                age,
                sibsp,
                parch,
                fare,
                cabin,
                embarked,
                prediction_code,
                prediction_label,
                survival_probability
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                passenger_data["Pclass"],
                passenger_data["Name"],
                passenger_data["Sex"],
                passenger_data["Age"],
                passenger_data["SibSp"],
                passenger_data["Parch"],
                passenger_data["Fare"],
                passenger_data["Cabin"],
                passenger_data["Embarked"],
                prediction_result["prediction_code"],
                prediction_result["prediction"],
                prediction_result["survival_probability"],
            ),
        )
        conn.commit()


def get_recent_predictions(limit: int = 10) -> list[dict]:
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT
                id,
                pclass,
                name,
                sex,
                age,
                sibsp,
                parch,
                fare,
                cabin,
                embarked,
                prediction_code,
                prediction_label,
                survival_probability,
                created_at
            FROM predictions
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = cursor.fetchall()

    predictions = []
    for row in rows:
        item = dict(row)
        item["survival_percentage"] = round(float(item["survival_probability"]) * 100, 1)
        item["fare_formatted"] = f"${float(item['fare']):,.2f}"
        item["cabin_display"] = item["cabin"] or "Unknown"
        item["status_class"] = "survived" if item["prediction_code"] == 1 else "not-survived"
        predictions.append(item)

    return predictions


def clear_prediction_history() -> None:
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM predictions")
        conn.commit()


def validate_passenger_data(passenger_data: dict) -> Optional[str]:
    if passenger_data["Name"] == "":
        return "Passenger name is required."

    if passenger_data["Age"] < 0 or passenger_data["Age"] > 100:
        return "Age must be between 0 and 100."

    if passenger_data["Fare"] < 0:
        return "Fare cannot be negative."

    if passenger_data["SibSp"] < 0 or passenger_data["Parch"] < 0:
        return "Family counts cannot be negative."

    return None


def render_home(
    request: Request,
    prediction_result: Optional[dict] = None,
    form_data: Optional[dict] = None,
    error: Optional[str] = None,
):
    recent_predictions = get_recent_predictions()

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "prediction_result": prediction_result,
            "form_data": form_data or {},
            "error": error,
            "recent_predictions": recent_predictions,
        },
    )


@app.on_event("startup")
def startup_event() -> None:
    init_db()


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return render_home(request)


@app.post("/predict", response_class=HTMLResponse)
def predict_from_form(
    request: Request,
    pclass: int = Form(...),
    name: str = Form(...),
    sex: str = Form(...),
    age: float = Form(...),
    sibsp: int = Form(...),
    parch: int = Form(...),
    fare: float = Form(...),
    cabin: Optional[str] = Form(None),
    embarked: str = Form(...),
):
    form_data = normalize_form_data(
        pclass=pclass,
        name=name,
        sex=sex,
        age=age,
        sibsp=sibsp,
        parch=parch,
        fare=fare,
        cabin=cabin,
        embarked=embarked,
    )

    validation_error = validate_passenger_data(form_data)
    if validation_error:
        return render_home(request, form_data=form_data, error=validation_error)

    prediction_result = run_prediction(form_data)
    save_prediction(form_data, prediction_result)

    return render_home(
        request,
        prediction_result=prediction_result,
        form_data=form_data,
    )


@app.post("/api/predict")
def predict_api(data: TitanicInput):
    passenger_data = to_dict(data)
    prediction_result = run_prediction(passenger_data)
    save_prediction(passenger_data, prediction_result)

    return prediction_result


@app.get("/api/history")
def history_api():
    return {"recent_predictions": get_recent_predictions()}


@app.post("/clear-history")
def clear_history():
    clear_prediction_history()
    return RedirectResponse(url="/", status_code=303)


@app.get("/health")
def health():
    return {"status": "ok", "model_version": "pipeline-v3", "history_enabled": True}
