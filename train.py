"""Train, evaluate, and export the complete Titanic inference pipeline."""

import argparse
import hashlib
import json
import platform
from importlib.metadata import version
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

from app.ml_pipeline import TitanicFeatureEngineer

ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = ROOT / "data" / "Titanic-Dataset.csv"
SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
HISTORICAL_INPUTS = ["Pclass", "Sex", "Age", "Embarked", "Name", "Cabin", "SibSp", "Parch", "Fare"]
NUMERIC = ["Age", "HasCabin", "FamilySize", "IsAlone", "LogFare"]
CATEGORICAL = ["Pclass", "Sex", "Embarked", "Title"]
SCORES = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
    "roc_auc": "roc_auc",
}


def load_dataset(path: Path) -> pd.DataFrame:
    source = path.read_bytes()
    data = pd.read_csv(BytesIO(source), dtype={"Cabin": str, "Embarked": str, "Ticket": str})
    missing = set([*HISTORICAL_INPUTS, "Survived"]).difference(data)
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(sorted(missing))}")
    if data["Survived"].isna().any() or set(data["Survived"]) != {0, 1}:
        raise ValueError("Survived must contain both binary labels and no missing values.")
    if "PassengerId" in data and data["PassengerId"].duplicated().any():
        raise ValueError("PassengerId must identify unique rows.")
    data.attrs["dataset_sha256"] = hashlib.sha256(source).hexdigest()
    return data


def split_data(data: pd.DataFrame) -> tuple:
    return train_test_split(
        data[HISTORICAL_INPUTS],
        data["Survived"],
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=data["Survived"],
    )


def make_pipeline(classifier) -> Pipeline:
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric, NUMERIC),
        ("categorical", categorical, CATEGORICAL),
    ])
    return Pipeline([
        ("features", TitanicFeatureEngineer()),
        ("preprocessor", preprocessor),
        ("model", classifier),
    ])


def compare_models(X_train: pd.DataFrame, y_train: pd.Series) -> tuple[dict, list[dict]]:
    candidates = {
        "Dummy Baseline": make_pipeline(DummyClassifier(strategy="most_frequent")),
        "Logistic Regression": make_pipeline(LogisticRegression(max_iter=1000, random_state=SEED)),
        "Random Forest": make_pipeline(RandomForestClassifier(
            n_estimators=80, max_depth=5, min_samples_leaf=2, random_state=SEED, n_jobs=1
        )),
    }
    folds = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    comparison = []
    for name, pipeline in candidates.items():
        scores = cross_validate(
            pipeline, X_train, y_train, cv=folds, scoring=SCORES, error_score="raise"
        )
        row = {"model": name}
        for metric in SCORES:
            fold_scores = scores[f"test_{metric}"]
            row[metric] = float(fold_scores.mean())
            row[f"{metric}_std"] = float(fold_scores.std())
        comparison.append(row)
    comparison.sort(key=lambda row: (-row["roc_auc"], row["model"]))
    return candidates, comparison


def error_breakdown(X_test: pd.DataFrame, actual: pd.Series, predicted) -> list[dict]:
    results = pd.DataFrame({"actual": actual, "predicted": predicted}, index=X_test.index)
    results["Passenger class"] = X_test["Pclass"]
    results["Sex recorded in dataset"] = X_test["Sex"]
    results["Age recorded"] = X_test["Age"].notna().map({True: "Yes", False: "No"})
    results["false_positive"] = (results["actual"] == 0) & (results["predicted"] == 1)
    results["false_negative"] = (results["actual"] == 1) & (results["predicted"] == 0)
    groups = []
    for column in ["Passenger class", "Sex recorded in dataset", "Age recorded"]:
        for value, group in results.groupby(column, dropna=False):
            groups.append({
                "field": column,
                "group": str(value),
                "passengers": len(group),
                "survivors": int(group["actual"].sum()),
                "false_positives": int(group["false_positive"].sum()),
                "false_negatives": int(group["false_negative"].sum()),
            })
    return groups


def run_experiment(data: pd.DataFrame) -> tuple[Pipeline, dict]:
    """Select on development folds, then evaluate the fitted winner on the holdout."""
    X_train, X_test, y_train, y_test = split_data(data)
    with threadpool_limits(limits=1):
        candidates, comparison = compare_models(X_train, y_train)
        winner = next(row["model"] for row in comparison if row["model"] != "Dummy Baseline")
        model = candidates[winner].fit(X_train, y_train)
        predicted = model.predict(X_test)
        survival = model.predict_proba(X_test)[:, list(model.classes_).index(1)]

    ticket_overlap = None
    if "Ticket" in data:
        train_tickets = data.loc[X_train.index, "Ticket"].dropna()
        ticket_overlap = int(data.loc[X_test.index, "Ticket"].isin(train_tickets).sum())

    metrics = {
        "selected_model": winner,
        "selection_rule": "Highest mean development CV ROC AUC among learned candidates; exact ties favor Logistic Regression.",
        "cross_validation": comparison,
        "seed": SEED,
        "cv_folds": CV_FOLDS,
        "holdout_fraction": TEST_SIZE,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "test_previously_examined": True,
        "holdout_rows_sharing_a_train_ticket": ticket_overlap,
        "test_accuracy": accuracy_score(y_test, predicted),
        "test_precision": precision_score(y_test, predicted, zero_division=0),
        "test_recall": recall_score(y_test, predicted, zero_division=0),
        "test_f1": f1_score(y_test, predicted, zero_division=0),
        "test_roc_auc": roc_auc_score(y_test, survival),
        "confusion_matrix": confusion_matrix(y_test, predicted, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y_test, predicted, labels=[0, 1], output_dict=True, zero_division=0
        ),
        "error_breakdown": error_breakdown(X_test, y_test, predicted),
        "dataset_sha256": data.attrs.get("dataset_sha256"),
    }
    return model, metrics


def save_artifacts(pipeline: Pipeline, metrics: dict, output_dir: Path) -> None:
    if not metrics.get("dataset_sha256"):
        raise ValueError("Load the input through load_dataset before exporting artifacts.")
    runtime_packages = ["scikit-learn", "numpy", "pandas", "scipy", "joblib", "threadpoolctl"]
    versions = {"python": platform.python_version()}
    versions.update({package: version(package) for package in runtime_packages})

    output_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output_dir) as temporary:
        temporary = Path(temporary)
        artifact = temporary / "model.pkl"
        joblib.dump(pipeline, artifact, compress=3)
        report = {
            **metrics,
            "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            "versions": versions,
        }
        metadata = temporary / "metrics.json"
        metadata.write_text(json.dumps(report, allow_nan=False, indent=2) + "\n", encoding="utf-8")
        artifact.replace(output_dir / artifact.name)
        metadata.replace(output_dir / metadata.name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "model")
    args = parser.parse_args()
    if not args.data.is_file():
        parser.error(f"CSV not found at {args.data}. See the README for how to get the Titanic training data.")

    pipeline, metrics = run_experiment(load_dataset(args.data))
    save_artifacts(pipeline, metrics, args.output_dir)
    print(f"Saved {metrics['selected_model']} pipeline and evaluation to {args.output_dir}")
    print(json.dumps({key: metrics[f"test_{key}"] for key in SCORES}, indent=2))


if __name__ == "__main__":
    main()
