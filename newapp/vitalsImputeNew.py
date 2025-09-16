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
    
    def cleanVitals(self):
        """
        Basic cleaning: time handling, dropping unused cols,
        label fixing, and dtype optimization.
        """
        import pandas as pd

        # Convert charttime
        self.vitals["charttime"] = dd.to_datetime(self.vitals["charttime"], errors="coerce")

        # Create 15-min bins
        self.vitals["time_bin"] = self.vitals["charttime"].dt.floor("15min")

        # Drop unnecessary columns
        cols_to_del = ["race", "hadm_id", "gcs", "dod", "gcs_unable", "gcs_time", "gcs_calc"]
        self.vitals = self.vitals.drop(columns=cols_to_del, errors="ignore")

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

        # Apply imputation respecting stay_id + time_bin
        self.vitals = self.vitals.map_partitions(
            vitalsImputeNew.fillVitals_partition,
            self.checkingColumns,
            meta=self.vitals._meta
        ).persist()

        # Save after interpolation
        self.vitals.to_parquet("filled/vitals_filled.parquet", write_index=False)
        print("💾 Saved interpolated vitals → filled/vitals_filled.parquet")
    
    
    

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
            # Optionally fill small edge gaps
            if edge_limit is not None and edge_limit > 0:
                g[vital_cols] = g[vital_cols].ffill(limit=edge_limit).bfill(limit=edge_limit)
            return g

        return df.groupby(['stay_id', 'time_bin'], group_keys=False).apply(_fill_group)
    

    def prepareVitals(self, run_xgb=True, train_frac=1.0):
        """
        Full pipeline: clean → interpolate → (optional) refine with XGBoost.
        Does NOT run evaluation.
        """
        import time
        import dask.dataframe as dd

        start_time = time.time()

        # 1. Basic cleaning
        self.cleanVitals()

        # 2. Interpolation + edge filling (per stay_id + time_bin)
        self.interpolate_and_fill()

        # Reload filled dataset (as Dask for consistency)
        df_filled = dd.read_parquet("filled/vitals_filled.parquet")

        # # 3. XGBoost refinement (optional)
        # if run_xgb:
        #     self.xgboost_refine(frac=train_frac)

        elapsed = time.time() - start_time
        print(f"⏱️ Pipeline finished in {elapsed:.1f} seconds")

        return df_filled
    
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
    

    def xgboost_refine(self, frac=0.2):
        """
        Train global XGBoost models for each vital sign 
        and impute missing values in the full dataset.
        """
        import pandas as pd, numpy as np, os
        from xgboost import XGBRegressor

        print("🔧 Training XGBoost models...")

        # Load filled dataset (after interpolation)
        df_full = pd.read_parquet("filled/vitals_filled.parquet")
        df_sample = df_full.sample(frac=frac, random_state=42)

        self.models = {}   # store trained models

        for target in self.checkingColumns:
            print(f"\n🚀 Training XGBoost for {target}")

            feature_cols = [c for c in df_sample.columns 
                            if c not in ["stay_id", "charttime","icu_intime","icu_outtime","time_bin"]]

            X = df_sample[feature_cols]
            y = df_sample[target]

            mask = ~y.isna()
            X_train, y_train = X[mask], y[mask]
            valid_mask = ~X_train.isna().any(axis=1)
            X_train, y_train = X_train[valid_mask], y_train[valid_mask]

            if len(y_train) < 100:
                print(f"⚠️ Skipping {target}, too few samples")
                continue

            model = XGBRegressor(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                tree_method="hist",
                n_jobs=-1,
                random_state=42
            )
            model.fit(X_train, y_train)

            # Save trained model
            self.models[target] = (model, feature_cols)

            # Fill missing values in full dataset
            missing_mask = df_full[target].isna()
            if missing_mask.any():
                X_missing = df_full.loc[missing_mask, feature_cols]
                valid_missing = ~X_missing.isna().any(axis=1)
                df_full.loc[missing_mask & valid_missing, target] = \
                    model.predict(X_missing[valid_missing])

        # Save refined dataset
        os.makedirs("imputed_xgb", exist_ok=True)
        out_path = "imputed_xgb/vitals_hybrid.parquet"
        df_full.to_parquet(out_path, index=False)
        print(f"\n💾 Hybrid-imputed dataset saved to {out_path}")