import InputData
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import psutil, os
import os
import time
import tracemalloc
import gc
import dask.dataframe as dd
import shutil


class vitalsImputeNew:

    def __init__(self, vitals,checkingColumns, interval):
        self.vitals = vitals
        self.checkingColumns = checkingColumns
        self.interval = interval

    def set_vitals(self, vitals):
        self.vitals = vitals

    def get_vitals(self):
        return self.vitals

    def set_interval(self, interval):
        self.interval = interval

    def get_interval(self):
        return self.interval

    def set_checkingColumns(self, checkingColumns):
        self.checkingColumns = checkingColumns

    def get_checkingColumns(self):
        return self.checkingColumns

    @staticmethod
    def fillVitals_partition(df, vital_cols):
        # Sort charttime **within each stay_id**
        df = df.groupby("stay_id", group_keys=False).apply(lambda g: g.sort_values("charttime"))

        # Fill missing values for numeric vital columns
        df[vital_cols] = (
            df[vital_cols]
            .bfill()
            .ffill()
            .interpolate(method="linear", limit_direction="both")
        )
        return df

    def prepareVitals(self):
        start_time = time.time()

        # Convert to datetime
        self.vitals["charttime"] = dd.to_datetime(self.vitals["charttime"], errors="coerce")

        # Create 15-min bins
        self.vitals["time_bin"] = self.vitals["charttime"].dt.floor("15min")

        # Set index (not sorted to avoid expensive operation)
        self.vitals = self.vitals.set_index("charttime", sorted=False)

        # Drop unnecessary columns
        cols_to_del = ["race", "hadm_id", "gcs", "dod", "gcs_unable", "gcs_time", "gcs_calc"]
        self.vitals = self.vitals.drop(columns=cols_to_del, errors='ignore')

        # Optimize datatypes
        self.vitals = self.vitals.astype({
            "subject_id": pd.Int32Dtype(),
            "stay_id": pd.Int32Dtype(),
            "gender": pd.Int8Dtype(),
            "hospstay_seq": pd.Int8Dtype(),
            "icustay_seq": pd.Int8Dtype(),
            "hospital_expire_flag": pd.Int8Dtype(),
            "label_sepsis_within_6h": pd.Int8Dtype(),
            "label_sepsis_within_8h": pd.Int8Dtype(),
            "label_sepsis_within_12h": pd.Int8Dtype(),
            "label_sepsis_within_24h": pd.Int8Dtype(),
            "sepsis_label": pd.Int8Dtype(),
            "heart_rate": pd.Float32Dtype(),
            "resp_rate": pd.Float32Dtype(),
            "sbp": pd.Float32Dtype(),
            "dbp": pd.Float32Dtype(),
            "mbp": pd.Float32Dtype(),
            "spo2": pd.Float32Dtype(),
            "pulse_pressure": pd.Float32Dtype(),
            "temperature": pd.Float32Dtype(),
            "admission_age": pd.Float32Dtype(),
            "los_hospital": pd.Float32Dtype(),
            "los_icu": pd.Float32Dtype(),
            "hours_before_sepsis": pd.Float32Dtype(),
        })

        # Repartition to a reasonable number of partitions for 4 cores
        self.vitals = self.vitals.repartition(npartitions=16)

        # Count NaNs before filling
        empties_before = self.vitals[self.checkingColumns].isna().sum().compute()
        print("Empties before fill:")
        print(empties_before)

        # Fill missing values using map_partitions
        self.vitals = self.vitals.map_partitions(
            vitalsImputeNew.fillVitals_partition,
            self.checkingColumns,
            meta=self.vitals._meta
        )

        # Persist to memory, triggers computation
        start_persist = time.time()
        print("start persist")
        self.vitals = self.vitals.persist()
        print(f"Persist took {time.time() - start_persist:.2f} seconds")

        # Count NaNs after filling
        empties_after = self.vitals[self.checkingColumns].isna().sum().compute()
        print("Empties after fill:")
        print(empties_after)

        # Total elapsed time
        elapsed = time.time() - start_time
        print(f"⏱️ Total preprocessing time: {elapsed:.2f} seconds")

        return self.vitals
        
        # print("sorted_groups")
        # sorted_groups = sorted_groups.reset_index("charttime")
        # print(sorted_groups.head(200, npartitions=1))

        # print('check for index in preparevitals...')
        # print(self.vitals.dtypes)

        # return self.find_duplicate_indices()

    def find_duplicate_indices(self):
        """
        Find and return all duplicate index values in the Dask dataframe self.vitals
        """
        print("Finding duplicate index values...")

        # Convert index to series
        index_series = self.vitals.index.to_series()

        # Mark duplicates per partition
        duplicates_per_partition = index_series.map_partitions(
            lambda s: s[s.duplicated()],
            meta=index_series._meta
        )

        # Compute the results
        duplicated_indices = duplicates_per_partition.compute()

        # Get unique duplicate values
        unique_dupes = duplicated_indices.unique()
        print(f"Found {len(unique_dupes)} unique duplicate index values.")
        print("Example duplicates:", unique_dupes[:20])  # show first 20

        return unique_dupes
