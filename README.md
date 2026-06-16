# Titanic Survival Prediction

A machine learning classification project that predicts whether a Titanic passenger was likely to survive based on passenger details such as class, sex, age, fare, family size, cabin availability, and embarked port.

This upgraded version includes a clean scikit-learn training workflow, a saved full pipeline model, a FastAPI web app, a premium Titanic-themed dashboard UI, and SQLite prediction history.

## Project Highlights

- End-to-end binary classification workflow
- Professional preprocessing with `Pipeline` and `ColumnTransformer`
- Reusable feature engineering transformer
- `DummyClassifier` baseline comparison
- Cross-validation on the training set only
- Final test evaluation with classification metrics and ROC AUC
- Saved full pipeline model using Joblib
- FastAPI backend
- Titanic-themed web dashboard
- SQLite prediction history
- Use-again and details actions for previous predictions

## Dataset

The project uses the Titanic passenger dataset.

Important input columns:

- `Pclass`
- `Name`
- `Sex`
- `Age`
- `SibSp`
- `Parch`
- `Fare`
- `Cabin`
- `Embarked`

Target column:

- `Survived`

## Feature Engineering

The custom feature engineering transformer is stored in:

```text
app/ml_pipeline.py
```

Engineered features include:

- `Title`: extracted from passenger names
- `HasCabin`: whether cabin information exists
- `FamilySize`: `SibSp + Parch + 1`
- `IsAlone`: whether the passenger traveled alone
- `LogFare`: log-transformed fare

Keeping this logic in one module makes the notebook and FastAPI app consistent.

## Model Comparison

Models were compared using 5-fold stratified cross-validation on the training set.

| Model | CV Accuracy | CV F1 | CV ROC AUC |
|---|---:|---:|---:|
| Dummy Baseline | ~0.617 | ~0.000 | ~0.500 |
| Logistic Regression | ~0.824 | ~0.768 | ~0.875 |
| Random Forest | ~0.830 | ~0.765 | ~0.874 |

## Final Model

Final selected model:

```text
Logistic Regression Pipeline
```

It was selected because it achieved strong performance while staying simple, explainable, and suitable for a portfolio classification project.

Final test results:

| Metric | Score |
|---|---:|
| Accuracy | ~0.827 |
| Precision | ~0.788 |
| Recall | ~0.754 |
| F1-score | ~0.770 |
| ROC AUC | ~0.865 |

The saved model file is:

```text
model/model.pkl
```

The model file contains the full pipeline:

- Feature engineering
- Missing value handling
- Scaling
- One-hot encoding
- Classifier

## Web App

The web app now runs through FastAPI instead of opening a plain static HTML file. This allows the app to save predictions and display history.

Main features:

- Passenger input form
- Survival prediction label
- Survival probability gauge
- SQLite prediction history
- Details button for each saved prediction
- Use Again button to refill the form from history
- Clear History action
- JSON API endpoint for programmatic predictions

## Project Structure

```text
Titanic/
│
├── app/
│   ├── __init__.py
│   ├── index.html                 # Local helper page
│   ├── main.py                    # FastAPI app
│   ├── ml_pipeline.py             # Reusable feature engineering
│   ├── static/
│   │   ├── script.js
│   │   └── style.css
│   └── templates/
│       └── index.html             # Main dashboard UI
│
├── data/
│   └── Titanic-Dataset.csv
│
├── database/
│   └── .gitkeep                   # Runtime DB is generated locally
│
├── model/
│   ├── model.pkl
│   ├── metrics.json
│   └── model_comparison.csv
│
├── notebook/
│   ├── Titanic.ipynb
│   └── model.pkl
│
│
├── requirements.txt
├── README.md
└── .gitignore
```

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the FastAPI server

From the main project folder, run:

```bash
uvicorn app.main:app --reload
```

### 3. Open the web app

Open:

```text
http://127.0.0.1:8000
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

Health check:

```text
http://127.0.0.1:8000/health
```

## API Example

Endpoint:

```text
POST /api/predict
```

Example input:

```json
{
  "Pclass": 3,
  "Name": "Nagy, Mr. Youssef",
  "Sex": "male",
  "Age": 23,
  "SibSp": 1,
  "Parch": 0,
  "Fare": 12.5,
  "Cabin": null,
  "Embarked": "S"
}
```

Example output:

```json
{
  "prediction": "Not Survived",
  "prediction_code": 0,
  "survival_probability": 0.2146,
  "survival_percentage": 21.5
}
```

## Notes

- The SQLite database is created automatically at runtime in the `database/` folder.
- Runtime database files are ignored by Git using `.gitignore`.
- To preserve consistency, the API uses the saved full pipeline directly instead of manually repeating preprocessing steps.
