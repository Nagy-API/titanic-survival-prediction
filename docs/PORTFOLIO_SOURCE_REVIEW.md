# Titanic Survival Prediction — Portfolio Source Review

**Reviewed:** 2026-11-05  
**Reviewed source:** [main at `48275e85a970f1dfb8b5111ced48b064f4f51b56`](https://github.com/Nagy-API/titanic-survival-prediction/tree/48275e85a970f1dfb8b5111ced48b064f4f51b56)  
**Scope:** Public-source static audit of the root project, saved result files and notebook outputs. No model retraining, API execution, dependency installation, deployment or data mutation was performed.

## Canonical version — use the repository root

The current app is implemented in **root** `app/main.py`, `app/ml_pipeline.py`, `app/templates/index.html`, `app/static/`, `model/model.pkl`, `notebook/Titanic.ipynb` and root `requirements.txt`.

The repository also contains **two nested historical copies**:

- `titanic-survival-prediction/`
- `titanic-survival-prediction/titanic-survival-prediction/`

Their documentation and `app/main.py` describe a **legacy v1** architecture with in-route feature transformation and no SQLite prediction dashboard. Neither nested version should be used for the current portfolio case study. Their older README references `Img/frontend-preview.png` and `Img/api-docs.png`, but the inspected repository tree did **not** contain those assets. Preserve any wanted history first, then optionally clean the nested archives as a separate approved change; do not delete anything from this documentation PR.

A compiled `app/__pycache__/main.cpython-311.pyc` was also tracked even though the current `.gitignore` excludes Python bytecode. This is a source-hygiene follow-up; no files were deleted.

## Modeling methodology verified from notebook and code

1. Source data: `data/Titanic-Dataset.csv`, 891 rows; binary `Survived` target.
2. Holdout split: `train_test_split(test_size=0.2, stratify=y, random_state=42)` → **712 train / 179 test**.
3. `TitanicFeatureEngineer` in `app/ml_pipeline.py` creates `Title`, `HasCabin`, `FamilySize`, `IsAlone` and `LogFare`. It does not fit statistical parameters outside the training pipeline.
4. `ColumnTransformer` contains median imputation and `StandardScaler` for numerical features; most-frequent imputation and `OneHotEncoder(handle_unknown="ignore")` for categoricals.
5. Training uses full Scikit-learn `Pipeline(features → preprocessor → model)`.
6. **Five-fold StratifiedKFold** cross-validation on **training data only**, with accuracy, precision, recall, F1 and ROC AUC; comparison against `DummyClassifier` and Random Forest.
7. Final selection: **Logistic Regression** based on strong CV ROC AUC and simpler interpretation, trained on training split and evaluated on held-out test data.
8. Notebook saves the full fitted pipeline to `model/model.pkl` and `notebook/model.pkl`; FastAPI loads `model/model.pkl` via Joblib and imports the custom transformer class.

### Cross-validation evidence (`model/model_comparison.csv`)

| Model | CV accuracy | CV F1 | CV ROC AUC |
| --- | ---: | ---: | ---: |
| Dummy Baseline | 0.61658 | 0.00000 | 0.50000 |
| Logistic Regression | 0.82445 | 0.76826 | **0.87465** |
| Random Forest | **0.83008** | 0.76490 | 0.87354 |

Cross-validation values are **means across five training folds**, not held-out test results. No extensive GridSearchCV or externally validated generalization claim was evidenced in the current notebook.

### Held-out test (`model/metrics.json`)

| Metric | Result |
| --- | ---: |
| Accuracy | **0.8268156425** (82.68%) |
| Precision (survived) | **0.7878787879** |
| Recall (survived) | **0.7536231884** |
| F1 (survived) | **0.7703703704** |
| ROC AUC | **0.8653491436** |
| Support | **179** |
| Confusion matrix | **[[96, 14], [17, 52]]** (actual [0,1] × predicted [0,1]) |

These values are directly recorded in the notebook and metrics file. They were **not independently recomputed** in this review. Do not describe them as cross-validation results or claim test accuracy was computed on the full dataset.

## Root FastAPI app — code inventory

- `GET /`: Jinja2-rendered styled prediction form and last 10 SQLite history entries.
- `POST /predict`: standard HTML form processing → `run_prediction` → save to SQLite → render result and history.
- `POST /api/predict`: Pydantic-validated JSON input, model output and save to SQLite.
- `GET /api/history`: returns last 10 saved predictions.
- `POST /clear-history`: deletes **all** rows from local `database/predictions.db` and redirects to home.
- `GET /health`: simple process status and model version label `pipeline-v3`.
- `app/static/script.js`: sample passenger presets, history detail toggle and refill form (“Use Again”).
- `app/ml_pipeline.py`: common feature construction for training/inference; no separate duplicated inference preprocessing in root app.
- Startup creates database/table if absent; SQLite is intended as **local demo history**, not multi-user production data management.

## Important publication/security and reproducibility boundaries

1. Root `requirements.txt` lists **unpinned package names**, so reproducibility across future dependency versions is not guaranteed (especially when reloading a serialized Scikit-learn pipeline). Do not imply the model artifact is portable to arbitrary Scikit-learn versions.
2. Root API has `allow_origins=["*"]` and no authentication. `GET /api/history` returns stored passenger inputs, while `POST /clear-history` globally clears history. Acceptable only as a trusted/local demo **without sensitive user data**. Before public shared deployment, isolate histories/authenticate or disable the history endpoints, restrict CORS and address state-changing CSRF/abuse.
3. HTML `POST /predict` validates a subset of fields explicitly, while JSON `POST /api/predict` uses strict Pydantic `Literal`/range validations. If the HTML route were published publicly, align validation paths and error handling. This is a scoped engineering improvement, not an established active incident.
4. `joblib.load(model/model.pkl)` deserializes a Python pickle-based artifact; load it **only from trusted, version-controlled sources**, never arbitrary uploads.
5. Nested legacy projects and checked-in bytecode may mislead readers. Preserve or archive them only after confirming there are no needed references; the README identifies the root as canonical.
6. No automated tests/CI workflow, published hosted demo URL, production service-level monitoring, dependency version lock or independent security audit were found in the inspected source tree. The project **can still be considered a finished educational demo**.
7. Treat survival probabilities as outputs of a model on a historical demonstration dataset, **not** safety-critical predictions or causal explanations of survival.

## Portfolio positioning and evidence

**Category:** Supporting AI/ML case study (tabular binary classification, reproducible preprocessing, evaluation discipline, and API integration).

**Safe claim:** “Built a Titanic survival-classification demo with a leakage-conscious Scikit-learn pipeline, evaluated Logistic Regression and Random Forest against a baseline using training-only stratified CV, achieved 82.68% held-out accuracy and 0.865 ROC AUC, and served inference through FastAPI with a local SQLite history dashboard.”

**Not supported:** Company/client delivery, live production deployment, unique trained deep learning model, optimized hyperparameters by GridSearchCV, independent reproduction of results, live traffic, business impact or multi-user security readiness.

**Evidence still needed before visual publication:** approved screenshot of locally running current root UI and results, optional non-sensitive API Swagger screenshot, reproducible environment and deterministic smoke test if publishing a runnable online demo. Individual authorship / team context was not independently established from code alone.

## Decision

Preserve the user's **development-complete** status. Documentation changes only; no code, model, data, nested files, dependencies or production deployment modified in this review. Any cleanup or API hardening is a *separate optional change*, not required to preserve the completed project status.
