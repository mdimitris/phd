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
        print('Optimizing dtypes and preprocessing...')

        # Your dtypes conversion
        self.vitals = self.vitals.astype({
            "subject_id": pd.Int32Dtype(),
            "stay_id": pd.Int32Dtype(),
            # ... (all other dtypes)
        })

        self.vitals["charttime"] = dd.to_datetime(self.vitals["charttime"], errors="coerce")
        self.vitals = self.vitals.drop(columns=["race", "hadm_id", "gcs", "dod", "gcs_unable", "gcs_time", "gcs_calc"], errors='ignore')
        self.vitals = self.vitals.dropna(how='all', subset=self.checkingColumns)

        # 🚀 The Correct and Final Sequence: No Multi-Index
        
        # 1. Ensure a unique, default RangeIndex.
        self.vitals = self.vitals.reset_index(drop=True)

        # 2. Set the single-column index to `stay_id` and persist.
        # This is a critical step that partitions and sorts the data by `stay_id`,
        # placing all data for each stay in a single partition.
        self.vitals = self.vitals.set_index("stay_id", sorted=True).persist()

        # 3. Use `map_partitions` to perform time-series operations locally.
        # The `map_partitions` method operates on each Dask partition (which is a Pandas DataFrame),
        # where the data is already sorted by `stay_id`. Within each partition, we can
        # perform a Pandas sort by `charttime` to handle the time-series logic correctly.
        def _impute_and_transform_partition(df):
            # We need to sort by charttime within each partition for `diff` and other time-series ops
            # This is a local operation on a Pandas DataFrame, so it's efficient here.
            df = df.sort_values(by="charttime")

            # Now, you can use Pandas `groupby` and `transform` on the sorted local data.
            df["relative_time_min"] = (df.groupby("stay_id")["charttime"]
                                        .transform(lambda x: (x - x.min()).dt.total_seconds() / 60)
                                        .fillna(0))
            
            df["time_gap_min"] = (df.groupby("stay_id")["charttime"]
                                    .transform(lambda x: x.diff().dt.total_seconds() / 60)
                                    .fillna(0))
            
            df["group"] = (df.groupby("stay_id")["charttime"]
                            .transform(lambda x: (x.diff() > pd.Timedelta(minutes=self.interval)).cumsum()))
            
            # Finally, perform the imputation using `groupby().apply()` on the Pandas DataFrame.
            # This is the most reliable way to handle ffill/bfill/interpolate with Dask.
            df[self.checkingColumns] = df.groupby("stay_id")[self.checkingColumns].apply(
                lambda group: group.ffill().bfill().interpolate(method="linear", limit_direction="both")
            )
            
            return df

        # Apply the partition function to the entire Dask DataFrame.
        # We must provide `meta` to tell Dask the output types.
        # Let's rebuild `meta` to include our new columns and ensure dtypes are correct.
        meta = self.vitals.copy()._meta
        meta["relative_time_min"] = pd.Series([], dtype="f4")
        meta["time_gap_min"] = pd.Series([], dtype="f4")
        meta["group"] = pd.Series([], dtype="i4")
        
        self.vitals = self.vitals.map_partitions(
            _impute_and_transform_partition,
            meta=meta
        )
        
        # 4. Reset the index back to a standard RangeIndex
        self.vitals = self.vitals.reset_index()

        # The DaskFill method can now be simplified as the imputation is done here.
        return self.DaskFill()
    
   
    
    def DaskFill(self):
        # print('dask vitals columns info:')
        # print(self.vitals)
        print('dask vitals datatypes and info:')
      
        print(self.vitals.dtypes)
        unique_stay_ids = self.vitals["stay_id"].nunique().compute()
        print("Number of unique stay_id:", unique_stay_ids)

        # print("Number of partitions:")
        # print(self.vitals.npartitions)
        # print("Missing values BEFORE filling:")
        # empties_before=self.vitals.reset_index(drop=True)[self.checkingColumns].isna().sum().compute()
        # print(empties_before)
        # del empties_before
        print("Start filling:")

        self.vitals = self.vitals.groupby(["stay_id"]).apply(
            lambda df: df.ffill().bfill().interpolate(method="linear", limit_direction="both"),
            meta=self.vitals
        )
        print("Finished filling, Check for duplicate columns now:")
        
        print(self.vitals.dtypes)
        exit()
        empties_after=self.vitals.loc[:, self.checkingColumns].isna().sum().compute()
        print (empties_after)
        # print(self.vitals.compute())
        exit()
        empties_after=self.vitals.loc[:, self.checkingColumns].isna().sum().compute()
        print("Missing values AFTER filling:")
        print(empties_after)
        exit()
        print("Missing values BEFORE filling:")
        print(self.vitals.npartitions)
        # pandas_vitals=self.vitals.head(100)
        print('test')
        # print(pandas_vitals)
        #print(self.vitals[self.checkingColumns].isna().sum().compute())
        exit()
        
        # Create 15-min bins
        df["charttime_bin"] = df["charttime"].dt.floor(f"{self.interval}min")

        # Sort by patient/stay/time (important for ffill/bfill/interpolation)
        df = df.map_partitions(lambda pdf: pdf.sort_values(["subject_id", "stay_id", "charttime"]))

        # Fill function per partition
        def fill_partition(pdf, columns):
            # Apply fill per patient-stay within the partition
            for (sub_id, stay_id), group in pdf.groupby(["subject_id", "stay_id"]):
                idx = group.index
                filled = (
                    group[columns]
                    .ffill()
                    .bfill()
                    .interpolate(method="linear", limit_direction="both")
                )
                pdf.loc[idx, columns] = filled
            return pdf

        df = df.map_partitions(fill_partition, self.checkingColumns)

        print("Missing values AFTER filling:")
        print(df[self.checkingColumns].isna().sum().compute())

        # Optional: drop the bin column if you don't need it
        df = df.drop(columns="charttime_bin")

        return df