import numpy as np
import vitalsImputeNew
import gasesImpute as gi
import dask.dataframe as dd
import dask.array as da
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, roc_auc_score,median_absolute_error

import xgBoostFill as xgbFill




class Evaluation:

    # columns (Vitals and Labs separately)
    vitals = [
        "spo2", "sbp", "dbp", "pulse_pressure", "heart_rate",
        "resp_rate", "temperature", "mbp", "gcs"
    ]

    labs = [
        "wbc", "platelet", "hematocrit", "hemoglobin", "mch",
        "mchc", "mcv", "rbc", "rdw", "glucose", "creatinine"
    ]

    def __init__(self, imputer, data, columns_to_fill, mask_rate=0.2, n_runs=3):
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
    




    def evaluate_sparse_with_ml(self, imputer, mask_frac=0.05, n_runs=3):

        from sklearn.metrics import mean_absolute_error
        import numpy as np
        import pandas as pd

        results = []

        for col in self.columns_to_fill:
            maes, medians, counts = [], [], []

            for _ in range(n_runs):
                df_eval = self.data.copy()
                mask = df_eval[col].notna()
                if mask.sum() == 0:
                    continue
                mask_idx = df_eval[mask].sample(frac=mask_frac, random_state=42).index
                y_true = df_eval.loc[mask_idx, col].values
                df_eval.loc[mask_idx, col] = np.nan

                # Run ML imputer
                if imputer is not None:
                    df_filled = imputer.transform(df_eval)
                else:
                    df_filled = df_eval.copy()
                    df_filled[col] = df_filled[col].ffill().bfill().interpolate()

                y_pred = df_filled.loc[mask_idx, col].values

                valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
                if valid_mask.sum() == 0:
                    continue

                maes.append(mean_absolute_error(y_true[valid_mask], y_pred[valid_mask]))
                medians.append(np.median(np.abs(y_true[valid_mask] - y_pred[valid_mask])))
                counts.append(valid_mask.sum())

            if maes:
                results.append({
                    "Feature": col,
                    "MAE": np.mean(maes),
                    "MedianAE": np.mean(medians),
                    "EvaluatedPoints": np.sum(counts)
                })

        return pd.DataFrame(results)
    
    
    
    def evaluate_filling_performance(self, original_df, filled_df, columns=None):


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