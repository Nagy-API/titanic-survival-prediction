"""Reusable preprocessing objects for the Titanic ML pipeline.

Keeping this transformer in a normal Python module makes the saved Joblib
pipeline importable from both the notebook and the FastAPI app.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class TitanicFeatureEngineer(BaseEstimator, TransformerMixin):
    """Create Titanic-specific features without learning from the test set.

    The transformer does not use the target and does not calculate statistics
    from the full dataset. Missing values are handled later inside the sklearn
    preprocessing pipeline.
    """

    title_mapping = {
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
        "Dona": "Rare",
    }

    output_columns = [
        "Pclass",
        "Sex",
        "Age",
        "Embarked",
        "Title",
        "HasCabin",
        "FamilySize",
        "IsAlone",
        "LogFare",
    ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()

        for col in ["Name", "Cabin", "SibSp", "Parch", "Fare"]:
            if col not in X.columns:
                X[col] = np.nan

        X["Title"] = X["Name"].fillna("").astype(str).str.extract(r" ([A-Za-z]+)\.", expand=False)
        X["Title"] = X["Title"].replace(self.title_mapping).fillna("Rare")

        X["HasCabin"] = X["Cabin"].notna().astype(int)
        X["HasCabin"] = np.where(X["Cabin"].astype(str).str.strip().isin(["", "nan", "None"]), 0, X["HasCabin"])

        X["FamilySize"] = X["SibSp"].fillna(0) + X["Parch"].fillna(0) + 1
        X["IsAlone"] = (X["FamilySize"] == 1).astype(int)
        X["LogFare"] = np.log1p(X["Fare"].clip(lower=0))

        for col in self.output_columns:
            if col not in X.columns:
                X[col] = np.nan

        return X[self.output_columns]
