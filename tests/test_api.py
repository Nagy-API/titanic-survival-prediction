import json
import logging
from pathlib import Path
from shutil import copyfile

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import main
from app.main import create_app

ROOT = Path(__file__).resolve().parents[1]

SAMPLE = {
    "Pclass": 1,
    "Sex": "female",
    "Title": "Mrs",
    "Age": 29,
    "Fare": 80,
    "SibSp": 1,
    "Parch": 0,
    "HasCabin": True,
    "Embarked": "C",
}


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app()) as client:
        yield client


def test_runtime_health_and_assets(client):
    assert client.get("/health").status_code == 200
    home = client.get("/")
    assert home.status_code == 200
    assert "default-src 'self'" in home.headers["content-security-policy"]
    assert home.headers["x-content-type-options"] == "nosniff"
    assert client.get("/static/script.js").status_code == 200
    assert client.get("/static/style.css").status_code == 200


@pytest.mark.parametrize(
    ("attributes", "label", "code"),
    [
        ({}, "Survived", 1),
        ({"Pclass": 3, "Sex": "male", "Title": "Mr", "HasCabin": False, "Fare": 8},
         "Not Survived", 0),
    ],
)
def test_api_matches_fitted_pipeline(client, attributes, label, code):
    passenger = SAMPLE | attributes
    model_file = ROOT / "model" / "model.pkl"
    # The app verifies the bundled artifact before reporting readiness.
    assert client.get("/health").status_code == 200
    pipeline = joblib.load(model_file)
    sample = pd.DataFrame([passenger])
    positive_index = list(pipeline.classes_).index(1)
    probability = float(pipeline.predict_proba(sample)[0, positive_index])
    assert pipeline.predict(sample).tolist() == [code]

    response = client.post("/api/predict", json=passenger)
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert response.json() == {
        "prediction": label,
        "prediction_code": code,
        "survival_probability": pytest.approx(probability, abs=0.000051),
        "survival_percentage": pytest.approx(probability * 100, abs=0.051),
    }


@pytest.mark.parametrize("attributes", [
    {"Age": None, "Fare": None, "Embarked": None},
    {"Pclass": 3, "Age": 0, "Fare": 0, "SibSp": 0, "Parch": 0},
    {"Age": 100, "Fare": 600, "SibSp": 20, "Parch": 20},
])
def test_nullable_and_boundary_values(client, attributes):
    response = client.post("/api/predict", json=SAMPLE | attributes)
    assert response.status_code == 200
    assert 0 <= response.json()["survival_probability"] <= 1


@pytest.mark.parametrize(("field", "value"), [
    ("Pclass", True), ("Pclass", 1.0), ("Pclass", "1"), ("Pclass", 4),
    ("Sex", "unknown"), ("Title", "unlisted"), ("Embarked", "D"),
    ("Age", -1), ("Age", 100.1), ("Age", "28"), ("Age", False),
    ("Fare", -1), ("Fare", 600.01), ("Fare", "9"),
    ("SibSp", -1), ("SibSp", 20.1), ("SibSp", 1.5),
    ("Parch", 21), ("Parch", True), ("HasCabin", "false"),
])
def test_invalid_values_are_rejected(client, field, value):
    response = client.post("/api/predict", json=SAMPLE | {field: value})
    assert response.status_code == 422
    assert isinstance(response.json()["detail"], str)


@pytest.mark.parametrize("field", list(SAMPLE))
def test_explicit_fields_are_required(client, field):
    response = client.post("/api/predict", json={
        key: value for key, value in SAMPLE.items() if key != field
    })
    assert response.status_code == 422


def test_invalid_json_has_sanitized_error(client):
    marker = "PRIVATE PASSENGER PLACEHOLDER"
    response = client.post(
        "/api/predict",
        content=json.dumps(SAMPLE)[:-1] + ", " + marker,
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code in (400, 422)
    assert isinstance(response.json()["detail"], str)
    assert marker not in response.text


def test_prediction_rejects_non_json(client):
    response = client.post("/api/predict", data={"Pclass": "1"})
    assert response.status_code == 415


def test_prediction_limits_body_size(client):
    encoded = json.dumps(SAMPLE).encode()
    response = client.post(
        "/api/predict",
        content=encoded.ljust(2049),
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 413


def test_prediction_limits_streamed_body_size(client):
    chunks = iter([json.dumps(SAMPLE).encode(), b" " * 2048])
    response = client.post(
        "/api/predict", content=chunks, headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 413


def test_submitted_details_are_never_echoed_or_logged(client, caplog, capsys):
    caplog.set_level(logging.DEBUG)
    marker = "PRIVATE PASSENGER PLACEHOLDER"
    response = client.post("/api/predict", json=SAMPLE | {"Name": marker})
    assert response.status_code == 422
    assert marker not in response.text
    assert marker not in caplog.text
    out, err = capsys.readouterr()
    assert marker not in out + err


def test_inference_failure_does_not_echo_or_log_details(client, monkeypatch, caplog):
    def failing_inference(model, passenger):
        raise RuntimeError(f"Inference failed for age {passenger.Age}")

    monkeypatch.setattr(main, "infer", failing_inference)
    response = client.post("/api/predict", json=SAMPLE | {"Age": 12.345678})
    assert response.status_code == 503
    assert response.json() == {"detail": "Prediction is temporarily unavailable."}
    assert "Titanic inference failed" in caplog.text
    assert "12.345678" not in caplog.text


@pytest.mark.parametrize(("method", "path"), [
    ("GET", "/api/history"),
    ("POST", "/clear-history"),
    ("POST", "/predict"),
    ("GET", "/model/model.pkl"),
    ("GET", "/data/Titanic-Dataset.csv"),
    ("GET", "/notebook/Titanic.ipynb"),
])
def test_private_or_obsolete_routes_are_absent(client, method, path):
    assert client.request(method, path).status_code == 404


@pytest.mark.parametrize("artifact_state", ["missing", "invalid_checksum"])
def test_model_loading_failure_is_sanitized(tmp_path, artifact_state):
    if artifact_state == "invalid_checksum":
        copyfile(ROOT / "model" / "metrics.json", tmp_path / "metrics.json")
        (tmp_path / "model.pkl").write_bytes(b"intentionally invalid model")
    with TestClient(create_app(model_dir=tmp_path)) as failing_client:
        health = failing_client.get("/health")
        response = failing_client.post("/api/predict", json=SAMPLE)

    assert health.status_code == response.status_code == 503
    for reply in (health, response):
        assert str(tmp_path) not in reply.text
        assert "model.pkl" not in reply.text
