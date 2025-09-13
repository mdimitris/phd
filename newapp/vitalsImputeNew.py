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
    # def fillVitals_partition(df, vital_cols):
    #     # Sort charttime **within each stay_id**

    
    #     df = df.groupby("stay_id", group_keys=False).apply(lambda g: g.sort_values("charttime"))

    #     # Fill missing values for numeric vital columns
    #     df[vital_cols] = (
    #         df[vital_cols]
    #         .ffill()
    #         .bfill()
    #         .interpolate(method="linear", limit_direction="both")
    #     )

    #     return df

    # def prepareVitals(self):
    #     start_time = time.time()

    #     # Convert to datetime
    #     self.vitals["charttime"] = dd.to_datetime(self.vitals["charttime"], errors="coerce")

    #     # Create 15-min bins
    #     self.vitals["time_bin"] = self.vitals["charttime"].dt.floor("15min")

    #     # Set index (not sorted to avoid expensive operation)
    #     #self.vitals = self.vitals.set_index("charttime", sorted=False)

    #     # Drop unnecessary columns
    #     cols_to_del = ["race", "hadm_id", "gcs", "dod", "gcs_unable", "gcs_time", "gcs_calc"]
    #     self.vitals = self.vitals.drop(columns=cols_to_del, errors='ignore')

    #     # Optimize datatypes
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

    #     # Repartition to a reasonable number of partitions for 4 cores
    #     self.vitals = self.vitals.repartition(npartitions=128)

    #     # Count NaNs before filling
    #     # empties_before = self.vitals[self.checkingColumns].isna().sum().compute()
    #     # print("Empties before fill:")
    #     # print(empties_before)

    #     self.vitals = self.vitals.shuffle("stay_id")   # or set_index("stay_id")

    #     # Fill missing values using map_partitions
    #     self.vitals = self.vitals.map_partitions(
    #         vitalsImputeNew.fillVitals_partition,
    #         self.checkingColumns,
    #         meta=self.vitals._meta
    #     )

    #     # Persist to memory, triggers computation
    #     start_persist = time.time()
    #     print("start persist")
    #     self.vitals = self.vitals.persist()
    #     print(f"Persist took {time.time() - start_persist:.2f} seconds")

    #     # Count NaNs after filling
    #     empties_after = self.vitals[self.checkingColumns].isna().sum().compute()
    #     print("Empties after fill:")
    #     print(empties_after)

    #     # Total elapsed time
    #     elapsed = time.time() - start_time
    #     print(f"⏱️ Total preprocessing time: {elapsed:.2f} seconds")
    #     #self.vitals.to_csv("filled/vitals_filled-*.csv", single_file=False, index=False)
    #     self.vitals = self.vitals.reset_index()
    #     self.vitals.to_parquet("filled/vitals_filled.parquet", write_index=False)
    #     return self.vitals
        
    #     # print("sorted_groups")
    #     # sorted_groups = sorted_groups.reset_index("charttime")
    #     # print(sorted_groups.head(200, npartitions=1))

    #     # print('check for index in preparevitals...')
    #     # print(self.vitals.dtypes)

    #     # return self.find_duplicate_indices()

    def prepareVitals(self):
        start_time = time.time()

        # Convert charttime to datetime
        self.vitals["charttime"] = dd.to_datetime(self.vitals["charttime"], errors="coerce")

        # Create 15-min bins
        self.vitals["time_bin"] = self.vitals["charttime"].dt.floor("15min")

        # Drop unnecessary columns
        cols_to_del = ["race", "hadm_id", "gcs", "dod", "gcs_unable", "gcs_time", "gcs_calc"]
        self.vitals = self.vitals.drop(columns=cols_to_del, errors='ignore')

        # Define the label columns
        label_cols = [
            "label_sepsis_within_6h",
            "label_sepsis_within_8h",
            "label_sepsis_within_12h",
            "label_sepsis_within_24h",
        ]

        # Replace NaNs with 0 in label columns
        self.vitals[label_cols] = self.vitals[label_cols].fillna(0)

        # Optimize datatypes (same as before)
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



        # Repartition by stay_id to avoid bleeding (much faster than shuffle)
        # Each partition will contain full stay_ids
        self.vitals = self.vitals.set_index("stay_id", sorted=False, drop=False)
        # Optional: repartition to control partition size
        self.vitals = self.vitals.repartition(npartitions=128)

           # Fill missing values per partition
        self.vitals = self.vitals.map_partitions(
            vitalsImputeNew.fillVitals_partition,
            self.checkingColumns,
            meta=self.vitals._meta
        )

        # After imputation: replace NaNs in categorical/label columns with 0 again
        label_cols = [
            "label_sepsis_within_6h",
            "label_sepsis_within_8h",
            "label_sepsis_within_12h",
            "label_sepsis_within_24h",
            "sepsis_label",
            "hospital_expire_flag",
            "gender",
            "hospstay_seq",
            "icustay_seq"
        ]
        self.vitals[label_cols] = self.vitals[label_cols].fillna(0)

        # Final cast to compact dtypes
        self.vitals = self.vitals.astype({
            "hospital_expire_flag": pd.Int8Dtype(),
            "label_sepsis_within_6h": pd.Int8Dtype(),
            "label_sepsis_within_8h": pd.Int8Dtype(),
            "label_sepsis_within_12h": pd.Int8Dtype(),
            "label_sepsis_within_24h": pd.Int8Dtype(),
            "sepsis_label": pd.Int8Dtype(),
            "gender": pd.Int8Dtype(),
            "hospstay_seq": pd.Int8Dtype(),
            "icustay_seq": pd.Int8Dtype(),
        })

        # Persist to memory (triggers computation)
        self.vitals = self.vitals.persist()

        # Count NaNs after fill
        empties_after = self.vitals[self.checkingColumns].isna().sum().compute()
        print("Empties after fill:")
        print(empties_after)

        # Save to Parquet
        self.vitals.to_parquet("filled/vitals_filled.parquet", write_index=False)

        elapsed = time.time() - start_time
        print(f"⏱️ Total preprocessing time for interpolation: {elapsed:.2f} seconds")


        eval_df = self.xgboost_refine(frac=0.2, mask_rate=0.3, n_runs=3)
        print(eval_df)


        return self.vitals
    


    def xgboost_refine(self, frac=0.2, mask_rate=0.3, n_runs=3):
        """
        Refine missing values with global XGBoost models after interpolation.
        Trains once on a sample, applies to all stays, and evaluates.
        """
        import pandas as pd
        import numpy as np
        from xgboost import XGBRegressor
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        import random, os

        print("🔧 Starting XGBoost refinement...")

        # Convert dask -> pandas sample for training
        df_sample = self.vitals.sample(frac=frac).compute()
        df_full = self.vitals.compute()   # careful: loads all into memory

        results = []

        for target in self.checkingColumns:
            print(f"\n🚀 Training XGBoost for {target}")

            # Features = all other vitals (exclude id/time cols)
            feature_cols = [c for c in df_sample.columns 
                            if c not in ["stay_id", "charttime", "icu_intime", "icu_outtime", "time_bin"]]

            X = df_sample[feature_cols]
            y = df_sample[target]

            # Drop rows with missing target
            mask = ~y.isna()
            X_train, y_train = X[mask], y[mask]

            # Drop rows where features still NaN
            valid_mask = ~X_train.isna().any(axis=1)
            X_train, y_train = X_train[valid_mask], y_train[valid_mask]

            if len(y_train) < 100:
                print(f"⚠️ Skipping {target}, too few samples after cleaning")
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

            # --- Evaluation with masking ---
            scores = []
            for run in range(n_runs):
                df_eval = df_sample.copy()
                known_idx = df_eval[target].dropna().index
                if len(known_idx) < 10:
                    continue

                masked_idx = random.sample(list(known_idx), int(mask_rate * len(known_idx)))
                true_vals = df_eval.loc[masked_idx, target]
                df_eval.loc[masked_idx, target] = np.nan

                X_eval = df_eval[feature_cols].loc[masked_idx]

                # Drop rows with NaNs in features or target
                eval_mask = (~X_eval.isna().any(axis=1)) & (~true_vals.isna())
                X_eval, true_vals = X_eval[eval_mask], true_vals[eval_mask]

                if len(true_vals) == 0:
                    continue

                y_pred = model.predict(X_eval)
                mse = mean_squared_error(true_vals, y_pred)
                mae = mean_absolute_error(true_vals, y_pred)
                scores.append((mse, mae))

            if scores:
                mse_mean = np.mean([s[0] for s in scores])
                mae_mean = np.mean([s[1] for s in scores])
                results.append({"target": target, "MSE": mse_mean, "MAE": mae_mean})
                print(f"✅ {target}: MSE={mse_mean:.4f}, MAE={mae_mean:.4f}")
            else:
                print(f"⚠️ No valid evaluation scores for {target}")

            # --- Apply model to fill NaNs in full dataset ---
            missing_mask = df_full[target].isna()
            if missing_mask.any():
                X_missing = df_full.loc[missing_mask, feature_cols]
                valid_missing = ~X_missing.isna().any(axis=1)
                if valid_missing.sum() > 0:
                    df_full.loc[missing_mask & valid_missing, target] = \
                        model.predict(X_missing[valid_missing])

        # Save back into parquet
        out_path = "imputed_xgb/vitals_hybrid.parquet"
        os.makedirs("imputed_xgb", exist_ok=True)
        df_full.to_parquet(out_path, index=False)
        print(f"\n💾 Hybrid-imputed dataset saved to {out_path}")

        # Return metrics
        eval_df = pd.DataFrame(results)
        print("\n📊 Hybrid XGBoost Evaluation:")
        print(eval_df)
        return eval_df
        


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
