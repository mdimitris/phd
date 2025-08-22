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



class vitalsImpute:
    
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
        gc.collect()
        print('Optimizing dtypes and preprocessing...')

        # Optimize types
        self.vitals[["subject_id", "stay_id"]] = self.vitals[["subject_id", "stay_id"]].astype(pd.Int32Dtype())
        self.vitals["charttime"] = pd.to_datetime(self.vitals["charttime"], errors="coerce")

        categorical_cols = [
            "gender", "hospstay_seq", "icustay_seq", "gcs_unable", "hospital_expire_flag",
            "label_sepsis_within_6h", "label_sepsis_within_8h", "label_sepsis_within_12h",
            "label_sepsis_within_24h", "sepsis_label"
        ]
        self.vitals[categorical_cols] = self.vitals[categorical_cols].astype(pd.Int8Dtype())

        # Drop unnecessary columns
        self.vitals.drop(columns=["race", "hadm_id", "gcs", "dod", "gcs_unable", "gcs_time", "gcs_calc"], errors='ignore', inplace=True)

        # Remove empty rows
        self.vitals = InputData.clearEmpties(self.vitals, self.checkingColumns, "charttime", 4)

        # Float optimization
        float_cols = self.checkingColumns + ["temperature", "admission_age", "los_hospital", "los_icu", "hours_before_sepsis"]
        self.vitals[float_cols] = self.vitals[float_cols].round(2).astype(pd.Float32Dtype())

        # Time-based processing
        self.vitals.sort_values(by=["subject_id", "stay_id", "charttime"], inplace=True)
        self.vitals["group"] = self.vitals.groupby(["subject_id", "stay_id"])["charttime"].transform(
            lambda x: (x.diff() > pd.Timedelta(minutes=self.interval)).cumsum()
        )
        self.vitals["relative_time_min"] = self.vitals.groupby(["subject_id", "stay_id"])["charttime"].transform(
            lambda x: (x - x.min()).dt.total_seconds() / 60
        )
        self.vitals["time_gap_min"] = self.vitals.groupby(["subject_id", "stay_id"])["charttime"].transform(
            lambda x: x.diff().dt.total_seconds() / 60).fillna(0)

        print('Preprocessing done. Saving to disk...')
        save_dir = "./imputed_chunks"
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)  # Deletes the entire folder and contents
            os.makedirs(save_dir, exist_ok=True)  # Recreates the empty folder
        self.save_to_parquet_chunks(self.vitals, "./imputed_chunks", max_workers=4)

        print('Reading with Dask and applying interpolation...')
        ddf_interpolated = self.interpolate_with_dask("./imputed_chunks", self.checkingColumns)
        start = time.time()
        vital_cols = ["spo2", "sbp", "dbp", "mbp", "heart_rate", "pulse_pressure", "resp_rate", "temperature"]
        df_final = self.xgboost_impute(ddf_interpolated, vital_cols) 
        
        print("Interpolation complete.")
        print(f"Execution time: {time.time() - start:.2f} seconds")
        print(f"Memory used: {psutil.Process().memory_info().rss / 1024 ** 2:.2f} MB")
        print('Final outcome with dask and parquets:')
        print(df_final.info(memory_usage='deep'))  # See memory usage
        return df_final


    def save_to_parquet_chunks(self, df, save_dir, max_workers=4):
        
        print('start saving to parquet chuncks')
        os.makedirs(save_dir, exist_ok=True)
        grouped = list(df.groupby(["subject_id", "stay_id", "group"]))
        args = [(group_key, group_df.copy(), self.checkingColumns, save_dir) for group_key, group_df in grouped]

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for _ in executor.map(self._save_group_to_parquet, args):
                pass

    @staticmethod
    def _save_group_to_parquet(args):
        group_key, group_df, cols, save_dir = args
        group_df = group_df.sort_values("charttime")
        filename = f"{save_dir}/imputed_{'_'.join(map(str, group_key))}.parquet"
        group_df.to_parquet(filename, index=False)

    def interpolate_with_dask(self, parquet_dir, cols):
        ddf = dd.read_parquet(parquet_dir, assume_missing=True)

        def interpolate_partition(df):
            df = df.sort_values("charttime")
            print("filling parquet ")
            print(df.info)
            df[cols] = df[cols].ffill().bfill().interpolate(method="linear", limit_direction="both")
            return df

        ddf_interpolated = ddf.map_partitions(interpolate_partition)
        return ddf_interpolated#.compute() #later if needed
    
    # def dask_interpolate_parquet(self, parquet_dir, cols):
    #     print('return parquet')
    #     ddf = dd.read_parquet(parquet_dir)

    #     # # Ensure it's sorted before groupby
    #     # ddf = ddf.sort_values(["subject_id", "stay_id", "group", "charttime"])

    #     # # Repartition on group columns to ensure consistent divisions
    #     # ddf = ddf.repartition(partition_size="100MB")

    #     # # Groupby + interpolation
    #     # ddf = ddf.groupby(["subject_id", "stay_id", "group"]).apply(
    #     #     lambda group: group.assign(**{
    #     #         col: group[col].ffill().bfill().interpolate("linear") for col in cols
    #     #     }),
    #     #     meta=ddf._meta  # use ddf._meta to infer schema
    #     # )

    #     return ddf
    
    # def imputeValues_to_disk(self, df, cols, save_dir="./imputed_chunks", max_workers=4):
    #     os.makedirs(save_dir, exist_ok=True)
    #     print('grouped started')
    #     grouped = list(df.groupby(["subject_id", "stay_id", "group"]))
    #     args = [(group_key, group_df.copy(), cols, save_dir) for group_key, group_df in grouped]

    #     saved_files = []
    #     print('grouped finished')
    #     with ProcessPoolExecutor(max_workers=max_workers) as executor:
    #         for filename in executor.map(self.process_group_to_disk, args):
    #             saved_files.append(filename)

    #     return saved_files  # Return list of file paths
    
    
    # def process_group_to_disk(self,args):
    #     group_key, group_df, cols, save_dir = args

    #     group_df = group_df.sort_values("charttime")

    #     group_df[cols] = (
    #         group_df[cols]
    #         .ffill()
    #         .bfill()
    #         .interpolate(method="linear")
    #     )

    #     # Save to Parquet
    #     filename = f"{save_dir}/imputed_{'_'.join(map(str, group_key))}.parquet"
    #     print('saved to parquet:'+filename)
    #     group_df.reset_index(drop=True).to_parquet(filename, index=False)

    #     return filename  # return the path for later tracking
    

    
    # def prepareVitals(self):

    #     import gc
    #     gc.collect()
    #     print('checkingColumns and normalize vitals:')
    #     print(self.checkingColumns)
        
    #     # optimize df_vitals
    #     self.vitals[["subject_id", "stay_id"]] = self.vitals[["subject_id", "stay_id"]].astype(pd.Int32Dtype())
    #     self.vitals["charttime"]=pd.to_datetime(self.vitals['charttime'], format="%Y-%m-%d %H:%M:%S.%f", errors="coerce")
        
    #     self.vitals[["gender","hospstay_seq","icustay_seq","gcs_unable","hospital_expire_flag","label_sepsis_within_6h","label_sepsis_within_8h","label_sepsis_within_12h","label_sepsis_within_24h","sepsis_label"]] = self.vitals[["gender","hospstay_seq","icustay_seq","gcs_unable","hospital_expire_flag","label_sepsis_within_6h","label_sepsis_within_8h","label_sepsis_within_12h","label_sepsis_within_24h","sepsis_label"]].astype(pd.Int8Dtype())
    #     changecolumns=self.checkingColumns + ["temperature","admission_age","los_hospital","los_icu","hours_before_sepsis"]
    #     self.vitals[changecolumns] = self.vitals[changecolumns].round(2).astype(pd.Float32Dtype())
    #     #data[["gcs","gcs_calc"]] = data[["gcs","gcs_calc"]].astype(pd.Int16Dtype())

    #     # delete the empty rows and rows that more than four vitals are missing
    #     self.vitals = InputData.clearEmpties(self.vitals, self.checkingColumns, "charttime", 4)

    #     # transform race to categorical data (numerical)
    #     self.vitals["race"] = self.vitals["race"].astype("category")
    #     self.vitals["race_codes"] = self.vitals["race"].cat.codes
    #     # drop the race column since I have it in categorical data and gcs related columns
    #     self.vitals.drop(columns=['race', 'hadm_id' ,'gcs','dod','gcs_unable','gcs_time','gcs_calc'], inplace=True, errors='ignore')

        
    #      # Step 1: Preprocess
    #     self.vitals['charttime'] = pd.to_datetime(self.vitals['charttime'], format="%Y-%m-%d %H:%M:%S.%f", errors="coerce")
    #     self.vitals.sort_values(by=["subject_id", "stay_id", "charttime"], inplace=True)

    #     self.vitals["group"] = self.vitals.groupby(["subject_id", "stay_id"])['charttime'].transform(
    #         lambda x: (x.diff() > pd.Timedelta(minutes=self.interval)).cumsum()
    #     )

    #     self.vitals["relative_time_min"] = self.vitals.groupby(["subject_id", "stay_id"])["charttime"] \
    #         .transform(lambda x: (x - x.min()).dt.total_seconds() / 60)

    #     self.vitals["time_gap_min"] = self.vitals.groupby(["subject_id", "stay_id"])["charttime"] \
    #         .transform(lambda x: x.diff().dt.total_seconds() / 60).fillna(0)
        
    #     print('Normalize vitals end, this is the new size:')
    #     print (self.vitals.info(memory_usage='deep'))
        
    #     self.checkingColumns.append('temperature')
    #     self.imputeValues_to_disk(self.vitals, self.checkingColumns, save_dir="./imputed_chunks", max_workers=6)
    #     df = self.dask_interpolate_parquet("./imputed_chunks", self.checkingColumns).compute()
    #     print ('vitals df from dask to pandas')
    #     print(df.info(memory_usage='deep'))
    #     return df

    
    # def divideDataframe(self, data, interval):
    #     import gc
    #     gc.collect()
    #     print("Starting vitals filling and interpolation process for each group....")    

    #     # Step 3: Group
    #     grouped_chunks = [group for _, group in data.groupby(["subject_id", "stay_id", "group"])]
    #     print(f"Total groups to process: {len(grouped_chunks)}")

    #     # Step 4: Parallel processing
    #     result_chunks = []
    #     with ProcessPoolExecutor(max_workers=4) as executor:
    #         futures = [
    #             executor.submit(self.imputeValues, group.copy(), self.checkingColumns, interval)
    #             for group in grouped_chunks
    #         ]

    #         # ✅ Free the grouped_chunks memory early
    #         del grouped_chunks
    #         gc.collect()
    #         process = psutil.Process(os.getpid())
    #         print(f"[After submitting futures] Memory usage: {process.memory_info().rss / 1024**2:.2f} MB")

    #         for future in as_completed(futures):
    #             result_chunks.append(future.result())

    #     print(f"[During collection] Memory usage: {process.memory_info().rss / 1024**2:.2f} MB")
    #     # Step 5: Concatenate results
    #     result_df = pd.concat(result_chunks, ignore_index=False)
    #     result_df.reset_index(inplace=True)

    #     print("Vitals interpolation finished for all groups.")
    #     vital_cols = ["spo2", "sbp", "dbp", "mbp", "heart_rate", "pulse_pressure", "resp_rate", "temperature"]
    #     return self.xgboost_impute(result_df, vital_cols)

    # def divideDataframe(self, data,interval):
    #     print("Starting vitals filling and interpolation process for each group....")
    #     data.set_index("charttime", inplace=True)
        
    #      # Ensure proper datetime
    #     data['charttime'] = pd.to_datetime(df['charttime'], format="%Y-%m-%d %H:%M:%S.%f", errors="coerce")
    #     data.sort_values(by=["subject_id", "stay_id", 'charttime'], inplace=True)

    #     # Grouping for interpolation based on time gaps
    #     data["group"] = data.groupby(["subject_id", "stay_id"])['charttime'].transform(
    #         lambda x: (x.diff() > pd.Timedelta(minutes=interval)).cumsum())

    #     # 🕒 Add relative time (in minutes since first observation per stay)
    #     data["relative_time_min"] = data.groupby(["subject_id", "stay_id"])["charttime"] \
    #                                 .transform(lambda x: (x - x.min()).dt.total_seconds() / 60)

    #     # 🕒 Optionally add time gap (between measurements)
    #     data["time_gap_min"] = data.groupby(["subject_id", "stay_id"])["charttime"] \
    #                         .transform(lambda x: x.diff().dt.total_seconds() / 60).fillna(0)
        
    #     grouped_chunks = [group for _, group in data.groupby(["subject_id", "stay_id","group"])]
    #     result_chunks = []

    #     for chunk in grouped_chunks:
    #         imputed_chunk = self.imputeValues(chunk.copy(), self.checkingColumns, self.interval)
    #         result_chunks.append(imputed_chunk)

    #     result_df = pd.concat(result_chunks, ignore_index=False)
    #     result_df.reset_index(inplace=True)
    #     print("Vitals interpolation finished for all groups.")
    #     vital_cols = ["spo2", "sbp", "dbp", "mbp", "heart_rate", "pulse_pressure", "resp_rate", "temperature"]
    #     print("finished vitals filling and interpolation process for each group!")
    #     return self.xgboost_impute(result_df, vital_cols) 
    
    
    # def imputeValues(self,df,cols,interval):
        
     
    #     #df.replace([-1, "missing", "NA"], np.nan, inplace=True)
    #     df.infer_objects(copy=False)

       

    #     # Interpolation within each group
    #     df.set_index("charttime", inplace=True)
    #     df[cols] = (
    #         df[cols]
    #         .ffill()
    #         .bfill()
    #         .interpolate(method="linear")
    #     )
    #     df.reset_index(inplace=True)
    #     df[cols] = df[cols].infer_objects(copy=False)
    #     # Define which vitals to impute
        
        
    #     # 💡 Call imputation method that now can use time
    #     return df
    
    
    def imputeValues(self, df, cols, interval):
        # Ensure correct dtypes
        df.infer_objects(copy=False)

        # Create a copy to avoid modifying the original DataFrame in-place (if needed)
        imputed_dfs = []

        # Iterate through each group individually
        grouped = df.groupby(["subject_id", "stay_id", "group"])
        for group_key, group_df in grouped:
            # Sort by charttime to ensure interpolation is time-aware
            group_df = group_df.sort_values("charttime")

            # Forward fill, backward fill, then interpolate
            group_df[cols] = (
                group_df[cols]
                    .ffill()
                    .bfill()
                    .interpolate(method="linear")
            )
            imputed_dfs.append(group_df)

        # Concatenate all the processed groups back into one DataFrame
        result_df = pd.concat(imputed_dfs).sort_index()

        # Optional: infer dtypes again
        result_df[cols] = result_df[cols].infer_objects(copy=False)

        return result_df
        #self.xgboost_impute(df, vital_cols)
        
      
    #---START of ffill bfill evaluation-------#      
    def simulate_missing(self,df, col, frac=0.2, seed=42):
        import numpy as np

        df = df.copy()
        np.random.seed(seed)

        # Ensure valid (non-null) index for sampling
        valid_idx = df[df[col].notna()].index.to_numpy()

        if len(valid_idx) == 0:
            raise ValueError(f"No non-NaN values found in column '{col}' to simulate missing.")

        # Safe integer cast and cap sample size
        n_missing = int(np.floor(frac * len(valid_idx)))
        n_missing = max(1, n_missing)  # Ensure at least 1 for testing

        sample_idx = np.random.choice(valid_idx, n_missing, replace=False)

        df[f"{col}_true"] = df[col]
        df.loc[sample_idx, col] = np.nan

        return df, sample_idx

    def evaluate_imputation(self, df, col, sample_idx):
        # Ensure ground truth exists
        if f"{col}_true" not in df.columns:
            raise ValueError(f"{col}_true column missing. You need to simulate missing values with ground truth first.")

        y_true = df.loc[sample_idx, f"{col}_true"]
        y_pred = df.loc[sample_idx, col]

        # Filter out rows where y_pred is still NaN
        mask = ~y_pred.isna()
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]

        if len(y_true_clean) == 0:
            raise ValueError(f"No valid predictions to evaluate for {col} — all are NaN.")

        mae = mean_absolute_error(y_true_clean, y_pred_clean)
        rmse = mean_squared_error(y_true_clean, y_pred_clean) ** 0.5

        return {"MAE": mae, "RMSE": rmse, "n_eval": len(y_true_clean)}
    


    def plot_imputation_accuracy(self, eval_results, metrics=["MAE", "RMSE"],save_path="imputation_accuracy.png"):

        for metric in metrics:
            fig, ax = plt.subplots(figsize=(10, 5))
            vals = {k: v[metric] for k, v in eval_results.items() if metric in v}
            if not vals:
                print(f"No results to plot for {metric}.")
                continue

            ax.bar(vals.keys(), vals.values(), color='skyblue', edgecolor='black')
            ax.set_title(f"{metric} per Vital (lower is better)", fontsize=14)
            ax.set_ylabel(metric)
            ax.set_xlabel("Vital Sign")
            ax.set_xticks(range(len(vals)))
            ax.set_xticklabels(vals.keys(), rotation=45)
            ax.grid(True, axis='y', linestyle='--', alpha=0.7)

            plt.tight_layout()
            plt.show()
            
            # Save figure
        save_path='evaluation-1.png'
        plt.savefig(save_path, format='png', dpi=300)
        print(f"Plot saved to {save_path}")
        plt.savefig(save_path)
        plt.close()
    
    
    #---END of ffill bfill evaluation-------#
    
    def xgboost_impute(self, df, cols): 
        print("xgboost process for filling started.....")
        print(df.info())
        time_features = ["relative_time_min", "time_gap_min"]
        df_imputed = df.copy()
        del df
        for target_col in cols:
            # Exclude the target from predictors
            predictors = [col for col in cols if col != target_col] + time_features
            available_features = [col for col in predictors if col in df_imputed.columns]

            # Skip if no features available
            if not available_features:
                continue

            imputer = IterativeImputer(
                estimator=XGBRegressor(n_estimators=50, max_depth=4, tree_method='hist'),
                max_iter=20,
                random_state=42
            )

            try:
                model_df = df_imputed[[target_col] + available_features].copy()
                imputed = imputer.fit_transform(model_df)
                #df_imputed[target_col] = imputed[:, 0]  # First column is target for pandas
                df_imputed[target_col] = pd.Series(imputed[:, 0], index=df_imputed.index, dtype=np.float32)
            except Exception as e:
                print(f"Skipping {target_col}: {e}")
        df_imputed = df_imputed.astype({col: "float32" for col in df_imputed.select_dtypes(include="float64").columns})
        print("vitals imputation and filling process finished")
        return df_imputed


    def evaluate_all_vitals(self, df, cols, frac=0.2):
        results = {}

        for col in cols:
            try:
                print(f"Evaluating: {col}")
                df_with_missing, sample_idx = self.simulate_missing(df, col, frac=frac)
                
                # Impute values (you can switch to any method you prefer here)
                imputed_df = self.imputeValues(df_with_missing.copy(), cols, interval=15)

                metrics = self.evaluate_imputation(imputed_df, col, sample_idx)
                results[col] = metrics
            except Exception as e:
                print(f"Skipping {col}: {e}")
                results[col] = {"error": str(e)}
        
        return results