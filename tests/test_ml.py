import hashlib
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from app.ml_pipeline import TitanicFeatureEngineer
from train import load_dataset, run_experiment, save_artifacts, split_data

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "Titanic-Dataset.csv"
MODEL_DIR = ROOT / "model"


def test_historical_and_serving_features_are_equivalent():
    base = {"Pclass": 2, "Sex": "female", "Age": 21, "Embarked": "Q",
            "SibSp": 1, "Parch": 2, "Fare": 12}
    historical = base | {"Name": "Example, Mlle. Fictional", "Cabin": "B12"}
    scenario = base | {"Title": "Miss", "HasCabin": True}
    features = TitanicFeatureEngineer().fit_transform(pd.DataFrame([historical, scenario]))
    pd.testing.assert_series_equal(features.iloc[0], features.iloc[1], check_names=False)

    row = features.iloc[0]
    assert row["Title"] == "Miss"
    assert row["HasCabin"] == 1
    assert row["FamilySize"] == 4
    assert row["LogFare"] == pytest.approx(math.log(13))
    assert {"Name", "Cabin", "PassengerId", "Ticket", "Survived"}.isdisjoint(features)


def test_missing_raw_features_produce_expected_defaults():
    features = TitanicFeatureEngineer().fit_transform(pd.DataFrame([{"Pclass": 1}]))
    row = features.iloc[0]
    assert row["Title"] == "Rare"
    assert row["HasCabin"] == 0
    assert row["FamilySize"] == 1
    assert pd.isna(row["Age"])
    assert pd.isna(row["LogFare"])


def test_bundled_artifact_matches_recorded_checksum():
    metrics = json.loads((MODEL_DIR / "metrics.json").read_text(encoding="utf-8"))
    assert hashlib.sha256((MODEL_DIR / "model.pkl").read_bytes()).hexdigest() == metrics[
        "artifact_sha256"
    ]


@pytest.fixture(scope="module")
def synthetic_data():
    rows = []
    for index in range(50):
        survived = index % 2
        rows.append({
            "PassengerId": index + 1,
            "Survived": survived,
            "Pclass": 1 + index % 3,
            "Name": "Example, Mrs. Fictional" if survived else "Example, Mr. Fictional",
            "Sex": "female" if survived else "male",
            "Age": 20 + index % 7,
            "SibSp": index % 2,
            "Parch": index % 3,
            "Ticket": "Example-" + str(index),
            "Fare": 20 + index,
            "Cabin": "B12" if survived else None,
            "Embarked": "S",
        })
    return pd.DataFrame(rows)


def test_split_is_deterministic_and_disjoint(synthetic_data):
    first = split_data(synthetic_data)
    second = split_data(synthetic_data)
    for left, right in zip(first, second, strict=True):
        assert left.equals(right)
    x_train, x_test, y_train, y_test = first
    assert len(x_train) == len(y_train) == 40
    assert len(x_test) == len(y_test) == 10
    assert set(x_train.index).isdisjoint(x_test.index)
    assert y_test.value_counts().to_dict() == {0: 5, 1: 5}


def test_retraining_is_deterministic_and_handles_unknown_categories(tmp_path, synthetic_data):
    dataset_path = tmp_path / "train.csv"
    synthetic_data.to_csv(dataset_path, index=False)
    dataset = load_dataset(dataset_path)
    pipeline, metrics = run_experiment(dataset)
    repeated, repeated_metrics = run_experiment(dataset)
    assert metrics == repeated_metrics

    _, x_test, _, _ = split_data(dataset)
    expected = pipeline.predict_proba(x_test)
    np.testing.assert_allclose(repeated.predict_proba(x_test), expected, rtol=0, atol=1e-12)
    assert metrics["test_accuracy"] == 1.0
    assert metrics["confusion_matrix"] == [[5, 0], [0, 5]]

    output = tmp_path / "model"
    original_digest = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    dataset_path.write_text("Replaced after training")
    save_artifacts(pipeline, metrics, output)
    saved = joblib.load(output / "model.pkl")
    np.testing.assert_array_equal(saved.predict_proba(x_test), expected)
    saved_report = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert saved_report["dataset_sha256"] == original_digest
    assert saved_report["artifact_sha256"] == hashlib.sha256(
        (output / "model.pkl").read_bytes()
    ).hexdigest()

    missing = pd.DataFrame([{
        "Pclass": 2, "Sex": "female", "Title": "Rare", "HasCabin": False,
        "Age": None, "Fare": None, "Embarked": None, "SibSp": 0, "Parch": 0,
    }])
    probabilities = saved.predict_proba(missing)
    assert np.isfinite(probabilities).all()
    assert probabilities.sum() == pytest.approx(1.0)


@pytest.mark.skipif(not DATASET.exists(), reason="local historical dataset required")
def test_published_historical_metrics_are_reproducible():
    recorded = json.loads((MODEL_DIR / "metrics.json").read_text(encoding="utf-8"))
    assert hashlib.sha256(DATASET.read_bytes()).hexdigest() == recorded["dataset_sha256"]
    dataset = load_dataset(DATASET)
    rebuilt, metrics = run_experiment(dataset)
    assert metrics["selected_model"] == recorded["selected_model"]
    assert metrics["confusion_matrix"] == recorded["confusion_matrix"]
    for metric in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        key = f"test_{metric}"
        assert metrics[key] == pytest.approx(recorded[key], abs=1e-9)

    bundled = joblib.load(MODEL_DIR / "model.pkl")
    _, x_test, _, _ = split_data(dataset)
    np.testing.assert_allclose(
        rebuilt.predict_proba(x_test), bundled.predict_proba(x_test), rtol=0, atol=1e-10
    )
