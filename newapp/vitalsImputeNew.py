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

import xgBoostFill as xgb
import InputData

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
    
    def cleanVitals(self):
        """
        Basic cleaning: time handling, dropping unused cols,
        label fixing, and dtype optimization.
        """

        self.vitals = InputData.clearEmpties_ddf (self.vitals, self.checkingColumns, "charttime", 3)
        self.checkingColumns.append('temperature')
        # Create 15-min bins
        self.vitals["charttime"] = dd.to_datetime(self.vitals["charttime"], errors="coerce")
        self.vitals["time_bin"] =self.vitals["charttime"].dt.floor("15min")

        # Drop unnecessary columns
        cols_to_del = ["race", "hadm_id", "gcs", "dod", "gcs_unable", "gcs_time", "gcs_calc"]
        self.vitals = self.vitals.drop(columns=cols_to_del, errors="ignore")

        #make floats have two decimals only
        decimal_cols = ['temperature','admission_age','los_hospital','los_icu','hours_before_sepsis']
        #self.vitals[decimal_cols] = transFloat32 (self.vitals, decimal_cols)

        # Define label columns
        label_cols = [
            "label_sepsis_within_6h",
            "label_sepsis_within_8h",
            "label_sepsis_within_12h",
            "label_sepsis_within_24h",
        ]

        # Fill NaNs in label cols
        self.vitals[label_cols] = self.vitals[label_cols].fillna(0)

        # Optimize dtypes
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

        print("✅ Vitals cleaned and dtypes optimized")
        
        

    def interpolate_and_fill(self):
        """
        Fill missing vitals per stay_id and time_bin.
        Saves intermediate parquet.
        """
        # Repartition so all rows of a stay are together
        self.vitals = self.vitals.set_index("stay_id", sorted=False, drop=False)
        self.vitals = self.vitals.repartition(npartitions=128)
        #add temperature in columns for interpolation
        print('Start partition filling')
        
        # Apply imputation respecting stay_id + time_bin
        self.vitals = self.vitals.map_partitions(
            vitalsImputeNew.fillVitals_partition,
            self.checkingColumns,
            meta=self.vitals._meta
        ).persist()

        # Save after interpolation
        print('Start saving filled data to parquets')
        self.vitals.to_parquet("/root/scripts/newapp/secondrun/vitals_filled.parquet/", write_index=False)
        print("💾 Saved interpolated vitals → filled/vitals_filled.parquet")
        # Check missing values only in checkingColumns
        missing_summary = self.vitals[self.checkingColumns].isna().sum().compute()
        print("🧐 Missing values per vital column after interpolation:")
        print(missing_summary[missing_summary > 0])
        
    

    @staticmethod
    def fillVitals_partition(df, vital_cols, edge_limit=2):
    # Make sure charttime is datetime
        if "stay_id" in df.index.names:
            df = df.reset_index(level="stay_id", drop=True)
    
        df = df.copy()
   
    # Make sure charttime is datetime
        df['charttime'] = pd.to_datetime(df['charttime'])

        # Sort by stay_id and charttime
        df = df.sort_values(['stay_id', 'charttime'])

        # Group by stay_id AND existing 15-min time_bin
        def _fill_group(g):

            # Interpolate only interior gaps
            g[vital_cols] = g[vital_cols].interpolate(method='linear', limit_area='inside')

            g['temperature'] = g['temperature'].ffill(limit=8).bfill(limit=8).interpolate(method='linear', limit_area='inside')
            # Optionally fill small edge gaps
            if edge_limit is not None and edge_limit > 0:
                g[vital_cols] = g[vital_cols].ffill(limit=edge_limit).bfill(limit=edge_limit)
            return g
        

        return df.groupby(['stay_id', 'time_bin'],  group_keys=False).apply(_fill_group)
    

    def prepareVitals(self, run_xgb=True, train_frac=1.0):
        """
        Full pipeline: clean → interpolate → hybrid temperature refinement.
        Returns: Dask DataFrame with fully processed vitals.
        """
        import time
        import dask.dataframe as dd

        start_time = time.time()

        # 1. Basic cleaning
        self.cleanVitals()

        # 2. Interpolation + edge filling (per stay_id + time_bin)
        self.interpolate_and_fill()

        # Reload filled dataset (as Dask for consistency)

        df_filled = dd.read_parquet("/root/scripts/newapp/secondrun/vitals_filled.parquet/")

        # #run it for filling temperature more aggressively
        # df_filled = df_filled.map_partitions(
        #     vitalsImputeNew.fill_temperature_continuous,
        #     meta=df_filled._meta
        # ).persist()
        
        # # 3. XGBoost refinement for temperature (optional)
        # if run_xgb:
        #     print('start LightGBM for temperature')
        #     feature_cols = ["heart_rate", "resp_rate", "sbp", "dbp", "mbp", "pulse_pressure","spo2", "fio2",
        #     "glucose", "wbc", "creatinine"]
        #     # self.xgboost_refine(frac=train_frac)
        #     dd_temperature = xgb.xgBoostFill(['temperature'],feature_cols)
        #     dd_temperature.fit(df_filled)
            #self, target_columns, features, random_state=42,feature_map=None

        elapsed = time.time() - start_time
        print(f"⏱️ Pipeline finished in {elapsed:.1f} seconds")

        return df_filled


    # def prepareVitals(self, run_xgb=True, train_frac=1.0):
    #     """
    #     Full pipeline: clean → interpolate → (optional) refine with XGBoost.
    #     Does NOT run evaluation.
    #     """
    #     import time
    #     import dask.dataframe as dd

    #     start_time = time.time()

    #     # 1. Basic cleaning
    #     self.cleanVitals()

    #     # 2. Interpolation + edge filling (per stay_id + time_bin)
    #     self.interpolate_and_fill()


    #     df_filled = self.xgboost_temperature_refine(df_filled).persist()

    #     # # 3. XGBoost refinement (optional)
    #     # if run_xgb:
    #     #     self.xgboost_refine(frac=train_frac)

    #     elapsed = time.time() - start_time
    #     print(f"⏱️ Pipeline finished in {elapsed:.1f} seconds")

    #     return df_filled
    
    def transform(self, df):
        """
        Fill missing vitals in the given DataFrame using the same logic as self.vitals.
        Works with both pandas and Dask DataFrames.
        """
        import dask.dataframe as dd

        # Detect if input is pandas
        is_pandas = isinstance(df, pd.DataFrame)

        if is_pandas:
            df_copy = dd.from_pandas(df, npartitions=8)
        else:
            df_copy = df.copy()

        # Ensure charttime is datetime
        df_copy['charttime'] = dd.to_datetime(df_copy['charttime'], errors='coerce')

        # Create time_bin if not exists
        if 'time_bin' not in df_copy.columns:
            df_copy['time_bin'] = df_copy['charttime'].dt.floor('15min')

        # Fill using the static method
        df_filled = df_copy.map_partitions(
            vitalsImputeNew.fillVitals_partition,
            self.checkingColumns,
            meta=df_copy._meta
        )

        # Return pandas if original was pandas
        if is_pandas:
            return df_filled.compute()
        return df_filled
    
    
    @staticmethod
    def fill_temperature_continuous(df):
        """
        Forward-fill and backward-fill temperature per stay_id to ensure continuity.
        Dask-compatible via transform to avoid index mismatch.
        """
        print('start filling temperature again')
        df = df.copy()
        df['charttime'] = pd.to_datetime(df['charttime'])
        df = df.sort_values(['stay_id', 'charttime'])

        # Dask-safe transform (keeps index)
        df['temperature'] = df.groupby('stay_id')['temperature'].transform(lambda g: g.ffill().bfill())

        return df

    # @staticmethod
    # def xgboost_temperature_refine(df):
    #     """
    #     Refine temperature using XGBoost, only predicting missing (originally NaN) entries.
    #     Fully Dask-compatible using map_partitions, no leakage across stay_ids.
    #     """
    #     import xgboost as xgb

    #     feature_cols = ["spo2", "sbp", "dbp", "pulse_pressure",
    #                     "heart_rate", "resp_rate", "mbp", "admission_age"]

    #     def refine_partition(pdf):
    #         pdf = pdf.copy()
    #         mask = pdf['temperature'].isna()
    #         if mask.any():
    #             df_obs = pdf[~mask]
    #             if len(df_obs) > 0:
    #                 X_train = df_obs[feature_cols].astype(float)
    #                 y_train = df_obs['temperature'].astype(float)

    #                 model = xgb.XGBRegressor(
    #                     n_estimators=200,
    #                     learning_rate=0.05,
    #                     max_depth=4,
    #                     random_state=42,
    #                     tree_method='hist',
    #                     n_jobs=1
    #                 )
    #                 model.fit(X_train, y_train)

    #                 X_pred = pdf.loc[mask, feature_cols].astype(float)
    #                 pdf.loc[mask, 'temperature'] = model.predict(X_pred)
    #         return pdf

    #     return df.map_partitions(refine_partition, meta=df._meta)
    # def prepareVitals(self, run_xgb=True):
    #     """
    #     Full pipeline: clean → interpolate → (optional) refine with XGBoost.
    #     """
    #     import time
    #     start_time = time.time()

    #     # 1. Basic cleaning
    #     self.cleanVitals()

    #     # 2. Interpolation + edge filling (per stay_id + time_bin)
    #     self.interpolate_and_fill()

    #     # 3. XGBoost refinement (optional)
    #     eval_df = None
    #     if run_xgb:
    #         eval_df = self.xgboost_refine()

    #     elapsed = time.time() - start_time
    #     print(f"⏱️ Pipeline finished in {elapsed:.1f} seconds")

    #     return eval_df if run_xgb else self.vitals
    

    # def prepareVitals(self):
    #     start_time = time.time()

    #     # Convert charttime to datetime
    #     self.vitals["charttime"] = dd.to_datetime(self.vitals["charttime"], errors="coerce")

    #     # Create 15-min bins
    #     self.vitals["time_bin"] = self.vitals["charttime"].dt.floor("15min")

    #     # Drop unnecessary columns
    #     cols_to_del = ["race", "hadm_id", "gcs", "dod", "gcs_unable", "gcs_time", "gcs_calc"]
    #     self.vitals = self.vitals.drop(columns=cols_to_del, errors='ignore')

    #     # Define the label columns
    #     label_cols = [
    #         "label_sepsis_within_6h",
    #         "label_sepsis_within_8h",
    #         "label_sepsis_within_12h",
    #         "label_sepsis_within_24h",
    #     ]

    #     # Replace NaNs with 0 in label columns
    #     self.vitals[label_cols] = self.vitals[label_cols].fillna(0)

    #     # Optimize datatypes (same as before)
    #     self.vitals = self.vitals.astype({
    #         "subject_id": pd.Int32Dtype(),
    #         "stay_id": pd.Int32Dtype(),
    #         "gender": pd.Int8Dtype(),
    #         "hospstay_seq": pd.Int8Dtype(),
    #         "icustay_seq": pd.Int8Dtype(),
    #         "hospital_expire_flag": pd.Int8Dtype(),
    #         "label_sepsis_within_6h": pd.Int8Dtype(),
    #         "label_sepsis_within_8h": pd.Int8Dtype(),
    #         "label_sepsis_within_12h": pd.Int8Dtype(),
    #         "label_sepsis_within_24h": pd.Int8Dtype(),
    #         "sepsis_label": pd.Int8Dtype(),
    #         "heart_rate": pd.Float32Dtype(),
    #         "resp_rate": pd.Float32Dtype(),
    #         "sbp": pd.Float32Dtype(),
    #         "dbp": pd.Float32Dtype(),
    #         "mbp": pd.Float32Dtype(),
    #         "spo2": pd.Float32Dtype(),
    #         "pulse_pressure": pd.Float32Dtype(),
    #         "temperature": pd.Float32Dtype(),
    #         "admission_age": pd.Float32Dtype(),
    #         "los_hospital": pd.Float32Dtype(),
    #         "los_icu": pd.Float32Dtype(),
    #         "hours_before_sepsis": pd.Float32Dtype(),
    #     })



    #     # Repartition by stay_id to avoid bleeding (much faster than shuffle)
    #     # Each partition will contain full stay_ids
    #     self.vitals = self.vitals.set_index("stay_id", sorted=False, drop=False)
    #     # Optional: repartition to control partition size
    #     self.vitals = self.vitals.repartition(npartitions=128)

    #        # Fill missing values per partition
    #     self.vitals = self.vitals.map_partitions(
    #         vitalsImputeNew.fillVitals_partition,
    #         self.checkingColumns,
    #         meta=self.vitals._meta
    #     )

    #     # After imputation: replace NaNs in categorical/label columns with 0 again
    #     label_cols = [
    #         "label_sepsis_within_6h",
    #         "label_sepsis_within_8h",
    #         "label_sepsis_within_12h",
    #         "label_sepsis_within_24h",
    #         "sepsis_label",
    #         "hospital_expire_flag",
    #         "gender",
    #         "hospstay_seq",
    #         "icustay_seq"
    #     ]
    #     self.vitals[label_cols] = self.vitals[label_cols].fillna(0)

    #     # Final cast to compact dtypes
    #     self.vitals = self.vitals.astype({
    #         "hospital_expire_flag": pd.Int8Dtype(),
    #         "label_sepsis_within_6h": pd.Int8Dtype(),
    #         "label_sepsis_within_8h": pd.Int8Dtype(),
    #         "label_sepsis_within_12h": pd.Int8Dtype(),
    #         "label_sepsis_within_24h": pd.Int8Dtype(),
    #         "sepsis_label": pd.Int8Dtype(),
    #         "gender": pd.Int8Dtype(),
    #         "hospstay_seq": pd.Int8Dtype(),
    #         "icustay_seq": pd.Int8Dtype(),
    #     })

    #     # Persist to memory (triggers computation)
    #     self.vitals = self.vitals.persist()

    #     # Count NaNs after fill
    #     empties_after = self.vitals[self.checkingColumns].isna().sum().compute()
    #     print("Empties after fill:")
    #     print(empties_after)

    #     # Save to Parquet
    #     self.vitals.to_parquet("filled/vitals_filled.parquet", write_index=False)

    #     elapsed = time.time() - start_time
    #     print(f"⏱️ Total preprocessing time for interpolation: {elapsed:.2f} seconds")


    #     eval_df = self.xgboost_refine(frac=0.2, mask_rate=0.3, n_runs=3)
    #     print(eval_df)


    #     return self.vitals
    
