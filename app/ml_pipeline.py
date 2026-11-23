"""Shared, stateless feature engineering for historical rows and API scenarios."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class TitanicFeatureEngineer(BaseEstimator, TransformerMixin):
    """Convert raw historical or explicit inference inputs to one feature schema."""

    title_mapping = {
        "Mlle": "Miss",
        "Ms": "Miss",
        "Mme": "Mrs",
    }
    titles = ("Mr", "Mrs", "Miss", "Master", "Rare")

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
        inputs = pd.DataFrame(X).reindex(columns=[
            "Pclass", "Sex", "Age", "Embarked", "Name", "Cabin",
            "Title", "HasCabin", "SibSp", "Parch", "Fare",
        ])
        features = inputs[["Pclass", "Sex", "Age", "Embarked"]].copy()
        for column in ["Pclass", "Age"]:
            features[column] = pd.to_numeric(features[column], errors="coerce").astype(float)
        for column in ["Sex", "Embarked"]:
            features[column] = features[column].astype(object)
            features[column] = features[column].where(features[column].notna(), np.nan)

        extracted_title = inputs["Name"].astype("string").str.extract(
            r" ([A-Za-z]+)\.", expand=False
        )
        title = inputs["Title"].astype("string").fillna(extracted_title)
        title = title.replace(self.title_mapping)
        features["Title"] = title.where(title.isin(self.titles), "Rare").astype(object)

        cabin = inputs["Cabin"].astype("string").str.strip()
        cabin_recorded = (cabin.notna() & ~cabin.isin(["", "nan", "None"])).astype(int)
        explicit_cabin = pd.to_numeric(inputs["HasCabin"], errors="coerce")
        features["HasCabin"] = explicit_cabin.fillna(cabin_recorded).astype(int)

        relatives = inputs[["SibSp", "Parch"]].apply(pd.to_numeric, errors="coerce")
        features["FamilySize"] = relatives.fillna(0).sum(axis=1) + 1
        features["IsAlone"] = (features["FamilySize"] == 1).astype(int)
        fare = pd.to_numeric(inputs["Fare"], errors="coerce")
        features["LogFare"] = np.log1p(fare.clip(lower=0))
        return features[self.output_columns]
