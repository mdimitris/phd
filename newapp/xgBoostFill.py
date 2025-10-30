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
    def __init__(self, target_columns, features, short_gap_targets=None, feature_map=None, random_state=42):
        self.target_columns = target_columns
        self.features = features
        self.short_gap_targets = short_gap_targets or []
        self.feature_map = feature_map or {}
        self.random_state = random_state
        self.models = {}  # stores (model, features) per target

    @staticmethod
    def short_gap_fill(df, col, limit=1):
        """Forward-fill and backward-fill small gaps."""
        df[col] = df[col].ffill(limit=limit).bfill(limit=limit)
        return df

    @staticmethod
    def clean_dtypes(df):
        """Ensure consistent dtypes for numeric and categorical features."""
        df = df.copy()
        numeric_cols = ["admission_age","los_hospital","los_icu","sbp","dbp","pulse_pressure",
                        "heart_rate","resp_rate","mbp","temperature","spo2"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
        int_cols = ["hospstay_seq","icustay_seq","gender"]
        for col in int_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int8")
        return df

    def fit(self, df: pd.DataFrame):
        """
        Fit LightGBM models on each target column.
        Input df must be a pandas DataFrame.
        """
        df = self.clean_dtypes(df)
        for target in self.target_columns:
            features = self.feature_map.get(target, [f for f in self.features if f != target])
            train_df = df.dropna(subset=[target])
            if train_df.empty:
                continue
            X_train = self.clean_dtypes(train_df[features])
            y_train = train_df[target].astype("float32")
            model = lgb.LGBMRegressor(
                n_estimators=500,
                learning_rate=0.01,
                max_depth=6,
                random_state=self.random_state,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            self.models[target] = (model, features)
        return self

    def _fill_partition(self, pdf: pd.DataFrame):
        """Fill a single partition of a Dask DataFrame."""
        pdf = self.clean_dtypes(pdf)
        for target in self.target_columns:
            # Short-gap fill
            if target in self.short_gap_targets:
                pdf = self.short_gap_fill(pdf, target, limit=4)
            # Model-based fill
            missing_idx = pdf[pd.isna(pdf[target])].index
            if len(missing_idx) > 0 and target in self.models:
                model, features = self.models[target]
                X_pred = self.clean_dtypes(pdf.loc[missing_idx, features])
                pdf.loc[missing_idx, target] = model.predict(X_pred).astype(pdf[target].dtype)
        return pdf

    def transform(self, df):
        """
        Apply imputation to a Dask or pandas DataFrame.
        """
        if isinstance(df, dd.DataFrame):
            filled_ddf = df.map_partitions(self._fill_partition, meta=df._meta)
            return filled_ddf
        else:  # pandas fallback
            return self._fill_partition(df)

    def fit_transform(self, df: pd.DataFrame, ddf: dd.DataFrame):
        """
        Convenience method:
        - fit on pandas DataFrame (small sample)
        - transform Dask DataFrame (large)
        """
        self.fit(df)
        return self.transform(ddf)