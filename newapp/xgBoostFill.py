import pandas as pd
import dask.dataframe as dd
import xgboost as xgb
import os
import numpy as np
import lightgbm as lgb
import InputData

class xgBoostFill:
    """
    A class to perform missing data imputation using XGBoost.
    It trains a separate XGBoost model for each target column with missing values,
    using other specified features to predict the missing data.
    """
    def __init__(self, target_columns, features, random_state=42,feature_map=None):
        self.target_columns = target_columns
        self.features = features
        self.feature_map = feature_map or {}
        self.random_state = random_state
        self.models = {}
    
    def clean_dtypes(self, df):
        df = df.copy()
        numeric_cols = [
            "admission_age", "los_hospital", "los_icu",
            "sbp", "dbp", "pulse_pressure",
            "heart_rate", "resp_rate", "mbp", "temperature", "spo2"
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32").round(2)

        if "hospstay_seq" in df.columns:
            df["hospstay_seq"] = pd.to_numeric(df["hospstay_seq"], errors="coerce").astype("int8")
        if "icustay_seq" in df.columns:
            df["icustay_seq"] = pd.to_numeric(df["icustay_seq"], errors="coerce").astype("int8")
        if "gender" in df.columns:
            df["gender"] = pd.to_numeric(df["gender"], errors="coerce").fillna(0).astype("int8")
        return df

    def short_gap_fill(self, g, col, edge_limit=4):
        """Fill only small gaps forward/backward."""
        g[col] = g[col].ffill(limit=edge_limit)
        g[col] = g[col].bfill(limit=edge_limit)
        return g

    def fit(self, data):
        print("Training hybrid imputers...")

        data = self.clean_dtypes(data)

        for col in self.target_columns:
            # # handle temperature separately
            # if col == "temperature":
            #     print("Skipping model training for temperature (hybrid strategy).")
            #     continue
            # if col == "spo2":
            #     print("Skipping model training for SpO₂ (will use simple fill).")
            #     continue

            current_features = [f for f in self.features if f != col]
            current_features = self.feature_map.get(col, [f for f in self.features if f != col])
            train_data = data.dropna(subset=[col])
            if train_data.empty:
                continue

            X_train = self.clean_dtypes(train_data[current_features])
            y_train = pd.to_numeric(train_data[col], errors="coerce").astype("float32").round(2)

            model = lgb.LGBMRegressor(
                objective="regression",
                n_estimators=500,
                learning_rate=0.01,
                max_depth=6,
                random_state=self.random_state,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            #self.models[col] = model
            self.models[col] = (model, current_features)


        return self

    def transform(self, data):
        filled = data.copy()
        filled = self.clean_dtypes(filled)

        for col in self.target_columns:
            missing_idx = filled[filled[col].isnull()].index
            if missing_idx.empty:
                continue

            # if col == "temperature":
            #     # First short-gap ffill/bfill
            #     filled = self.short_gap_fill(filled, col)
            #     # If still missing, use median
            #     if filled[col].isnull().sum() > 0:
            #         filled[col] = filled[col].fillna(filled[col].median())

            # elif col == "spo2":
            #     # Simple median fill for spo2
            #     filled[col] = filled[col].fillna(filled[col].median())

            else:
                # ✅ Retrieve (model, feature set) from self.models
                model_tuple = self.models.get(col)
                if model_tuple:
                    model, feature_cols = model_tuple
                    X_pred = self.clean_dtypes(filled.loc[missing_idx, feature_cols])
                    preds = model.predict(X_pred).astype(filled[col].dtype)
                    filled.loc[missing_idx, col] = preds

        return filled

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)