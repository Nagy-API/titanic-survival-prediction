"""Stateless inference for the Titanic portfolio demo."""

import hashlib
import io
import json
import logging
from importlib.metadata import version
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.concurrency import run_in_threadpool
from starlette.requests import ClientDisconnect

APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR.parent / "model"
MAX_REQUEST_BYTES = 2048
MODEL_PACKAGES = ("scikit-learn", "numpy", "pandas", "scipy", "joblib", "threadpoolctl")
LOG = logging.getLogger(__name__)

SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; base-uri 'none'; object-src 'none'; "
        "frame-ancestors 'none'; form-action 'self'"
    ),
    "Referrer-Policy": "no-referrer",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
}


class Passenger(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False)

    Pclass: int = Field(ge=1, le=3)
    Sex: Literal["male", "female"]
    Age: float | None = Field(ge=0, le=100)
    SibSp: int = Field(ge=0, le=20)
    Parch: int = Field(ge=0, le=20)
    Fare: float | None = Field(ge=0, le=600)
    Embarked: Literal["S", "C", "Q"] | None
    Title: Literal["Mr", "Mrs", "Miss", "Master", "Rare"]
    HasCabin: bool


class Prediction(BaseModel):
    prediction: Literal["Survived", "Not Survived"]
    prediction_code: Literal[0, 1]
    survival_probability: float
    survival_percentage: float


def load_model(model_dir: Path):
    """Verify the bundled, repository-controlled artifact before deserializing."""
    metrics = json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))
    artifact = (model_dir / "model.pkl").read_bytes()
    digest = hashlib.sha256(artifact).hexdigest()
    if digest != metrics["artifact_sha256"]:
        raise ValueError("Model checksum mismatch")
    if any(metrics["versions"][name] != version(name) for name in MODEL_PACKAGES):
        raise ValueError("Model dependency version mismatch")

    model = joblib.load(io.BytesIO(artifact))
    if model.classes_.tolist() != [0, 1]:
        raise ValueError("Unexpected model classes")
    return model, metrics


async def read_passenger(request: Request) -> Passenger:
    if request.headers.get("content-type", "").split(";", 1)[0].strip().lower() != "application/json":
        raise HTTPException(415, "Send an application/json request.")
    content_length = request.headers.get("content-length")
    if content_length is not None:
        if not content_length.isascii() or not content_length.isdecimal():
            raise HTTPException(400, "Invalid request size.")
        if len(content_length) > 10 or int(content_length) > MAX_REQUEST_BYTES:
            raise HTTPException(413, "Request body is too large.")

    body = bytearray()
    try:
        async for chunk in request.stream():
            if len(body) + len(chunk) > MAX_REQUEST_BYTES:
                raise HTTPException(413, "Request body is too large.")
            body.extend(chunk)
    except ClientDisconnect:
        raise HTTPException(400, "Incomplete request.") from None

    try:
        return Passenger.model_validate_json(body)
    except ValidationError:
        raise HTTPException(422, "Check the passenger fields and try again.") from None


def infer(model, passenger: Passenger) -> Prediction:
    frame = pd.DataFrame([passenger.model_dump()])
    probability = float(model.predict_proba(frame)[0, 1])
    if not 0 <= probability <= 1:
        raise ValueError("Invalid model probability")
    code = int(model.predict(frame)[0])
    return Prediction(
        prediction="Survived" if code else "Not Survived",
        prediction_code=code,
        survival_probability=probability,
        survival_percentage=round(probability * 100, 1),
    )


def create_app(model_dir: Path = MODEL_DIR) -> FastAPI:
    api = FastAPI(
        title="Titanic Survival Prediction",
        description="Classify a passenger scenario with a historical-data model.",
        version="4.0.0",
        docs_url=None,
        redoc_url=None,
    )
    try:
        model, metrics = load_model(model_dir)
    except Exception:
        LOG.error("Titanic model unavailable during initialization")
        model, metrics = None, {}

    @api.middleware("http")
    async def add_security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers.update(SECURITY_HEADERS)
        if request.url.path.startswith("/api/") or request.url.path == "/health":
            response.headers["Cache-Control"] = "no-store"
        return response

    templates = Jinja2Templates(directory=APP_DIR / "templates")
    api.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")

    @api.get("/", response_class=HTMLResponse, include_in_schema=False)
    def home(request: Request):
        return templates.TemplateResponse(request, "index.html", {"metrics": metrics})

    @api.get("/health")
    def health():
        if model is None:
            return JSONResponse({"status": "unavailable"}, status_code=503)
        return {"status": "ok", "model_sha256": metrics["artifact_sha256"]}

    @api.post(
        "/api/predict",
        response_model=Prediction,
        openapi_extra={
            "requestBody": {
                "required": True,
                "content": {"application/json": {"schema": Passenger.model_json_schema()}},
            }
        },
    )
    async def predict(request: Request):
        passenger = await read_passenger(request)
        if model is None:
            raise HTTPException(503, "The prediction model is unavailable.")
        try:
            return await run_in_threadpool(infer, model, passenger)
        except Exception:
            LOG.error("Titanic inference failed")
            raise HTTPException(503, "Prediction is temporarily unavailable.") from None

    return api


app = create_app()
