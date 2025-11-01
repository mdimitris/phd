import numpy as np
import vitalsImputeNew
import gasesImpute as gi
import dask.dataframe as dd
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, roc_auc_score
import InputData
import xgBoostFill as xgbFill
import random

class Evaluation:
    # --------------------------------------------------------
    # 1. Setup
    # --------------------------------------------------------

    # Your columns (Vitals and Labs separately)
    vitals = [
        "spo2", "sbp", "dbp", "pulse_pressure", "heart_rate",
        "resp_rate", "temperature", "mbp", "gcs"
    ]

    labs = [
        "wbc", "platelet", "hematocrit", "hemoglobin", "mch",
        "mchc", "mcv", "rbc", "rdw", "glucose", "creatinine"
    ]

    def __init__(self, imputer, data, columns_to_fill, mask_rate,n_runs):
        self.imputer = imputer
        self.data = data
        self.columns_to_fill = columns_to_fill
        self.mask_rate = mask_rate       
        self.n_runs=n_runs


    # Apply threshold cleaning
    def apply_medical_thresholds(self, df, thresholds):
        df_clean = df.copy()
        for col, (lower, upper) in thresholds.items():
            if col in df_clean.columns:
                outliers = (df_clean[col] < lower) | (df_clean[col] > upper)
                print(f"{col}: {outliers.sum()} outliers detected.")
                df_clean.loc[outliers, col] = np.nan  # Set outliers to NaN
        return df_clean


    # Assuming:
    # clean_df = original clean dataset (NO missing values)
    # df_with_nans = your real ICU dataset (WITH missing values)

    # def simulate_and_evaluate_dask_filling(self):
    #     """
    #     Simulate missingness and evaluate your Dask fill pipeline.
    #     Uses vitalsImputeNew.fillVitals_partition to impute.
    #     """
    
    #     results = []

    #     # Work on a small sample (so we can compute in memory)
    #     df_sample =  self.data.sample(frac=0.4).compute()  # 1% sample → Pandas
    #     df_sample = df_sample.reset_index()
    #     print('Evaluation Sample dataframe for vitals:')
    #     print(df_sample.info())
 
    #     df_sample["charttime"] = pd.to_datetime(df_sample["charttime"], errors="coerce")
    #     df_sample = df_sample.dropna(subset=["stay_id", "charttime"])
    #     df_sample = df_sample.sort_values(["stay_id", "charttime"])

    #     for col in self.columns_to_fill:
    #         maes, mses, r2s = [], [], []
    #         for _ in range(self.n_runs):
    #             df_copy = df_sample.copy()

    #             # Create a mask column
    #             df_copy["mask_flag"] = False
    #             mask = np.random.rand(len(df_copy)) < self.mask_rate
    #             df_copy.loc[mask, "mask_flag"] = True

    #             true_vals = df_copy.loc[df_copy["mask_flag"], col]
    #             df_copy.loc[df_copy["mask_flag"], col] = np.nan

    #             # Convert back to Dask
    #             ddf_masked = dd.from_pandas(df_copy, npartitions=4)

    #             if 'paco2' in df_copy.columns:
                        
    #                     df_gases=gases_imputer.prepareGases()
    #                     ddf_filled = ddf_masked.map_partitions(
    #                     gi.gasesImpute.imputeGases
    #                 )

    #             else:
    #                 ddf_filled = ddf_masked.map_partitions(
    #                     vitalsImputeNew.vitalsImputeNew.fillVitals_partition,
    #                     self.columns_to_fill,
    #                     meta=ddf_masked._meta
    #                 )

    #             df_filled = ddf_filled.compute()
    #             if 'paco2' in df_filled.columns:
    #                 df_filled.dropna(subset = ['paco2', 'fio2', 'pao2'],inplace=True)
    #             else:
    #                 df_filled.dropna(subset = ['spo2', 'sbp', 'dbp','pulse_pressure','heart_rate','resp_rate','mbp','temperature'],inplace=True)
               

    #             # Collect imputed values based on mask_flag
    #             imputed_vals = df_filled.loc[df_filled["mask_flag"], col]

    #             # Ensure alignment (drop NAs in true_vals too)
    #             true_vals = true_vals.loc[imputed_vals.index]
                
    #           # 1. Restrict to common index only
    #             common_idx = true_vals.index.intersection(imputed_vals.index)

    #             true_vals = true_vals.loc[common_idx]
    #             imputed_vals = imputed_vals.loc[common_idx]

    #             # 2. Drop NaNs together
    #             mask = true_vals.notna() & imputed_vals.notna()
    #             true_vals = true_vals[mask]
    #             imputed_vals = imputed_vals[mask]
                
    #             # Metrics
    #             maes.append(mean_absolute_error(true_vals, imputed_vals))
    #             #mses.append(root_mean_squared_error(true_vals, imputed_vals))
    #             mses.append(np.sqrt(mean_squared_error(true_vals, imputed_vals)))
    #             r2s.append(r2_score(true_vals, imputed_vals))

    #         print("start evaluation append")
    #         results.append({
    #             "Feature": col,
    #             "MAE": np.mean(maes),
    #             #"MSE": np.mean(mses),
    #             "RMSE": np.sqrt(np.mean(mses)),
    #             "R2": np.mean(r2s),
    #         })

    #     return pd.DataFrame(results)
    


    # def evaluate_xgboost_filling(self, frac=0.2, mask_rate=0.3, n_runs=3):
    #         """
    #         Example: evaluate XGBoost filling (requires trained self.models)
    #         """
    #         results = []
    #         df_full = pd.read_parquet("filled/vitals_filled.parquet") 
    #         df_sample = df_full.sample(frac=frac, random_state=42) 

    #         for target, (model, feature_cols) in self.models.items():
    #             print(f"\n📊 Evaluating {target}...")
    #             maes, rmses, r2s = [], [], []
    #             for _ in range(n_runs):
    #                 known_idx = df_sample[target].dropna().index
    #                 if len(known_idx) < 10:
    #                     continue

    #                 masked_idx = random.sample(list(known_idx), int(mask_rate * len(known_idx)))
    #                 true_vals = df_sample.loc[masked_idx, target]
    #                 df_sample.loc[masked_idx, target] = np.nan

    #                 X_eval = df_sample.loc[masked_idx, feature_cols]
    #                 valid_mask = (~X_eval.isna().any(axis=1)) & (~true_vals.isna())
    #                 X_eval, true_vals = X_eval[valid_mask], true_vals[valid_mask]

    #                 if len(true_vals) == 0:
    #                     continue
                    
    #                 evalImputer = xgbFill.xgBoostFill(
    #                         target_columns=[],
    #                         features=feature_cols,
    #                         random_state=42
    #                 )
                    
    #                 X_eval = evalImputer.clean_dtypes(X_eval)
    #                 y_pred = model.predict(X_eval)

    #                 maes.append(mean_absolute_error(true_vals, y_pred))
    #                 #rmses.append(np.sqrt(root_mean_squared_error(true_vals, y_pred)))
    #                 rmses.append(np.sqrt(mean_squared_error(true_vals, y_pred)))
    #                 r2s.append(r2_score(true_vals, y_pred))

    #             if maes:
    #                 results.append({
    #                     "Feature": target,
    #                     "MAE": np.mean(maes),
    #                     "RMSE": np.mean(rmses),
    #                     "R2": np.mean(r2s)
    #                 })

    #         eval_df = pd.DataFrame(results)
    #         # print("\n📊 XGBoost Evaluation Results:")
    #         # print(eval_df)
    #         return eval_df
        
    def evaluate(self, df, col, mask_frac=0.2, n_runs=3):
        results = []
        for _ in range(n_runs):
            res = self.evaluate_masking(df, col, mask_frac)
            results.append(res)
        
        avg = pd.DataFrame(results).mean(numeric_only=True).to_dict()
        avg["Feature"] = col   # ✅ Keep the feature name
        return avg


    def missing_report(self, ddf):
        """
        Report the count and percentage of missing values for each checking column.
        
        Args:
            ddf (dask.DataFrame): The Dask DataFrame to analyze.

        Returns:
            pd.DataFrame: A summary of missing counts and percentages.
        """
        # counts
        missing_counts = ddf[self.columns_to_fill].isna().sum().compute()
        # percentages
        missing_pct = (ddf[self.columns_to_fill].isna().mean().compute() * 100).round(2)

        summary = pd.DataFrame({
            "MissingCount": missing_counts,
            "MissingPct": missing_pct
        })

        print("\n📊 Missing Values Report:")
        print(summary)

        return summary
    
    
    def evaluate_masking(self, df, col, mask_frac=0.2, random_state=42):
        """
        Mask a fraction of observed values in a column, run the imputer,
        and compute MAE, RMSE, R2 for the masked values.
        Handles NaNs and misalignments safely.
        """
        rng = np.random.default_rng(random_state)

        # Copy + reset index for safe masking
        df_copy = df.copy().reset_index(drop=True)

        masked_indices = []
        true_values = []

        # --- Mask fraction per stay_id
        for stay_id, group in df_copy.groupby("stay_id"):
            observed_idx = group[group[col].notna()].index
            if len(observed_idx) == 0:
                continue

            mask_size = max(1, int(len(observed_idx) * mask_frac))
            mask_idx = rng.choice(observed_idx, size=mask_size, replace=False)

            masked_indices.extend(mask_idx)
            true_values.extend(df_copy.loc[mask_idx, col].values)

            df_copy.loc[mask_idx, col] = np.nan

        # --- Run imputer
        df_filled = self.imputer.transform(df_copy)

        # --- Extract predictions for masked rows
        preds = df_filled.loc[masked_indices, col].values
        true_vals = np.array(true_values)

        # --- Align + clean NaNs
        preds = np.array(preds)
        mask = ~np.isnan(true_vals) & ~np.isnan(preds)
        true_vals_clean = true_vals[mask]
        preds_clean = preds[mask]

        if len(true_vals_clean) == 0:
            print(f"⚠️ No valid pairs to evaluate for {col}")
            return {"Feature": col, "MAE": np.nan, "RMSE": np.nan, "R2": np.nan}

        # --- Compute metrics
        mae = mean_absolute_error(true_vals_clean, preds_clean)
        rmse = np.sqrt(mean_squared_error(true_vals_clean, preds_clean))
        #rmse = root_mean_squared_error(true_vals_clean, preds_clean)
        r2 = r2_score(true_vals_clean, preds_clean)

        return {"Feature": col, "MAE": mae, "RMSE": rmse, "R2": r2}
       
       
       
        # # Run the full imputer (pipeline)
        # df_filled = self.imputer.transform(df_copy)

        # # Collect predictions for masked values
        # preds = df_filled.loc[mask_idx, col]

        # # Align and drop any remaining NaNs
        # mask = true_vals.notna() & preds.notna()
        # true_vals = true_vals[mask]
        # preds = preds[mask]

        # # Compute metrics
        # mae = mean_absolute_error(true_vals, preds)
        # rmse = root_mean_squared_error(true_vals, preds)
        # r2 = r2_score(true_vals, preds)

        # return {"Feature": col, "MAE": mae, "RMSE": rmse, "R2": r2}
    
    
    def evaluate_filling_performance(self, original_df, filled_df, columns=None):
        """
        Evaluate the quality of imputation (MAE, RMSE, R²) between original and filled DataFrames.
        
        Parameters:
            original_df (pd.DataFrame or dd.DataFrame): Ground truth data (before masking).
            filled_df (pd.DataFrame or dd.DataFrame): Data after imputation.
            columns (list): Columns to evaluate. Defaults to self.columns_to_fill.

        Returns:
            pd.DataFrame: Metrics (MAE, RMSE, R²) per column + overall mean.
        """
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        import numpy as np
        import pandas as pd
        import dask.dataframe as dd

        if columns is None:
            columns = self.columns_to_fill

        # Convert to pandas if needed
        if isinstance(original_df, dd.DataFrame):
            print("Computing Dask DataFrames to Pandas for evaluation...")
            original_df = original_df.compute()
        if isinstance(filled_df, dd.DataFrame):
            filled_df = filled_df.compute()

        results = []
        for col in columns:
            mask = original_df[col].notna() & filled_df[col].notna()
            if mask.sum() == 0:
                print(f"⚠️ Skipping {col} — no overlapping non-null values.")
                continue

            y_true = original_df.loc[mask, col]
            y_pred = filled_df.loc[mask, col]

            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)

            results.append({
                "Feature": col,
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2
            })

        results_df = pd.DataFrame(results)
        if not results_df.empty:
            summary = results_df[["MAE", "RMSE", "R2"]].mean().to_dict()
            print("\n📊 Column-wise metrics:")
            print(results_df)
            print("\n📈 Overall Summary:")
            print({k: round(v, 4) for k, v in summary.items()})
        else:
            summary = {}

        return results_df, summary