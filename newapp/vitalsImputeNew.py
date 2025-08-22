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
    
    
    def prepareVitals(self):
        
        print(self.vitals.info())
      
        gc.collect()
        print('Optimizing dtypes and preprocessing...')

        # Optimize types
        self.vitals[["subject_id", "stay_id"]] = self.vitals[["subject_id", "stay_id"]].astype(pd.Int32Dtype())
        self.vitals["charttime"] = dd.to_datetime(self.vitals["charttime"], errors="coerce")

        categorical_cols = [
            "gender", "hospstay_seq", "icustay_seq", "gcs_unable", "hospital_expire_flag",
            "label_sepsis_within_6h", "label_sepsis_within_8h", "label_sepsis_within_12h",
            "label_sepsis_within_24h", "sepsis_label"
        ]
        self.vitals[categorical_cols] = self.vitals[categorical_cols].astype(pd.Int8Dtype())

        # Drop unnecessary columns
        self.vitals.drop(columns=["race", "hadm_id", "gcs", "dod", "gcs_unable", "gcs_time", "gcs_calc"], errors='ignore')

        # Remove empty rows
        self.vitals = InputData.clearEmpties_ddf(self.vitals, self.checkingColumns, "charttime", 4)

        # Float optimization
        float_cols = self.checkingColumns + ["temperature", "admission_age", "los_hospital", "los_icu", "hours_before_sepsis"]
        self.vitals[float_cols] = self.vitals[float_cols].round(2).astype(pd.Float32Dtype())

        # Time-based processing
        self.vitals.sort_values(by=["subject_id", "stay_id", "charttime"], inplace=True)
        
        self.vitals["relative_time_min"] = self.vitals.groupby(["subject_id", "stay_id"])["charttime"].transform(
            lambda x: (x - x.min()).dt.total_seconds() / 60,
            meta=('relative_time_min', 'f4')  # f8 = float64
        ).fillna(0)
        
        self.vitals["time_gap_min"] = self.vitals.groupby(["subject_id", "stay_id"])["charttime"].transform(
            lambda x: x.diff().dt.total_seconds() / 60,
            meta=('relative_time_min', 'f4')  # f8 = float64
        ).fillna(0)
        
        self.vitals["group"] = self.vitals.groupby(["subject_id", "stay_id"])["charttime"].transform(
            lambda x: (x.diff() > pd.Timedelta(minutes=self.interval)).cumsum(),
    meta=('group', 'i4')  # specify output type
        )

        
        return self.DaskFill(self.vitals)
    
    
    def DaskFill(self,df):
        df[self.checkingColumns] = df[self.checkingColumns].ffill().bfill().interpolate(method="linear", limit_direction="both")
        return self.vitals