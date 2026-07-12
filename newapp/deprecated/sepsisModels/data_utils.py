# sepsis_models/data_utils.py

import numpy as np
import pandas as pd
from typing import List, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def encode_and_scale(
    df: pd.DataFrame,
    features: List[str],
    label_col: str,
) -> tuple[pd.DataFrame, StandardScaler]:
    df = df.copy()

    if "gender" in df.columns and df["gender"].dtype == "object":
        df["gender"] = df["gender"].map({"M": 1, "F": 0}).astype("float32")

    df[label_col] = df[label_col].astype("float32")

    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])

    return df, scaler


def patient_wise_split(
    df: pd.DataFrame,
    test_size: float = 0.3,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    unique_stays = df["stay_id"].unique()
    train_stays, val_stays = train_test_split(
        unique_stays, test_size=test_size, random_state=random_state
    )
    df_train = df[df["stay_id"].isin(train_stays)].copy()
    df_val = df[df["stay_id"].isin(val_stays)].copy()
    return df_train, df_val


def create_sequences_per_stay(
    df: pd.DataFrame,
    features: List[str],
    label_col: str,
    seq_len: int,
    time_col: str,
) -> tuple[np.ndarray, np.ndarray]:
    df = df.sort_values(["stay_id", time_col])
    X_seqs, y_seqs = [], []

    for stay_id, g in df.groupby("stay_id"):
        X_vals = g[features].values
        y_vals = g[label_col].values

        if len(g) <= seq_len:
            continue

        for i in range(len(g) - seq_len):
            X_seqs.append(X_vals[i : i + seq_len])
            y_seqs.append(y_vals[i + seq_len])

    X = np.array(X_seqs, dtype=np.float32)
    y = np.array(y_seqs, dtype=np.float32).reshape(-1, 1)
    return X, y


def aggregate_for_tabular(
    df: pd.DataFrame,
    features: List[str],
    label_col: str,
    time_col: str,
) -> tuple[pd.DataFrame, pd.Series]:
    df = df.sort_values(["stay_id", time_col])
    agg_df = df.groupby("stay_id").tail(1)
    X = agg_df[features].copy()
    y = agg_df[label_col].astype("int")
    return X, y
