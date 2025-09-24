import numpy as np
import vitalsImputeNew
import dask.dataframe as dd
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score,roc_auc_score
import InputData
import xgboost as xgb

class evaluation:
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

    def __init__(self, data, columns_to_fill, mask_rate,n_runs):
      
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

    def simulate_and_evaluate_dask_filling(self):
        """
        Simulate missingness and evaluate your Dask fill pipeline.
        Uses vitalsImputeNew.fillVitals_partition to impute.
        """
    
        results = []

        # Work on a small sample (so we can compute in memory)
        df_sample =  self.data.sample(frac=0.15).compute()  # 1% sample → Pandas
        df_sample = df_sample.reset_index()
        print('Evaluation Sample dataframe for vitals:')
        print(df_sample.info())
 
        df_sample["charttime"] = pd.to_datetime(df_sample["charttime"], errors="coerce")
        df_sample = df_sample.dropna(subset=["stay_id", "charttime"])
        df_sample = df_sample.sort_values(["stay_id", "charttime"])

        for col in self.columns_to_fill:
            maes, mses, r2s = [], [], []
            for _ in range(self.n_runs):
                df_copy = df_sample.copy()

                # Create a mask column
                df_copy["mask_flag"] = False
                mask = np.random.rand(len(df_copy)) < self.mask_rate
                df_copy.loc[mask, "mask_flag"] = True

                true_vals = df_copy.loc[df_copy["mask_flag"], col]
                df_copy.loc[df_copy["mask_flag"], col] = np.nan

                # Convert back to Dask
                ddf_masked = dd.from_pandas(df_copy, npartitions=4)
                ddf_filled = ddf_masked.map_partitions(
                    vitalsImputeNew.vitalsImputeNew.fillVitals_partition,
                    self.columns_to_fill,
                    meta=ddf_masked._meta
                )

                df_filled = ddf_filled.compute()
                df_filled.dropna(subset = ['spo2', 'sbp', 'dbp','pulse_pressure','heart_rate','resp_rate','mbp','temperature'],inplace=True)
               

                # Collect imputed values based on mask_flag
                imputed_vals = df_filled.loc[df_filled["mask_flag"], col]

                # Ensure alignment (drop NAs in true_vals too)
                true_vals = true_vals.loc[imputed_vals.index]
                
              # 1. Restrict to common index only
                common_idx = true_vals.index.intersection(imputed_vals.index)

                true_vals = true_vals.loc[common_idx]
                imputed_vals = imputed_vals.loc[common_idx]

                # 2. Drop NaNs together
                mask = true_vals.notna() & imputed_vals.notna()
                true_vals = true_vals[mask]
                imputed_vals = imputed_vals[mask]
                
                # Metrics
                maes.append(mean_absolute_error(true_vals, imputed_vals))
                mses.append(mean_squared_error(true_vals, imputed_vals))
                r2s.append(r2_score(true_vals, imputed_vals))

            print("start evaluation append")
            results.append({
                "Feature": col,
                "MAE": np.mean(maes),
                # "MSE": np.mean(mses),
                "RMSE": np.sqrt(np.mean(mses)),
                "R2": np.mean(r2s),
            })

        return pd.DataFrame(results)
    


    def evaluate_xgboost_filling(self, frac=0.2, mask_rate=0.3, n_runs=3):
        """
        Evaluate XGBoost imputation by masking known values and checking accuracy.
        Requires self.models trained with xgboost_refine().
        """
        import pandas as pd, numpy as np, random
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        df_full = pd.read_parquet("filled/vitals_filled.parquet")
        df_sample = df_full.sample(frac=frac, random_state=42)

        results = []
        for target, (model, feature_cols) in self.models.items():
            print(f"\n📊 Evaluating {target}...")

            maes, mses, rmses, r2s = [], [], [], []
            for _ in range(n_runs):
                df_eval = df_sample.copy()
                known_idx = df_eval[target].dropna().index
                if len(known_idx) < 10:
                    continue

                masked_idx = random.sample(list(known_idx), int(mask_rate * len(known_idx)))
                true_vals = df_eval.loc[masked_idx, target]
                df_eval.loc[masked_idx, target] = np.nan

                X_eval = df_eval.loc[masked_idx, feature_cols]
                eval_mask = (~X_eval.isna().any(axis=1)) & (~true_vals.isna())
                X_eval, true_vals = X_eval[eval_mask], true_vals[eval_mask]

                if len(true_vals) == 0:
                    continue

                y_pred = model.predict(X_eval)

                maes.append(mean_absolute_error(true_vals, y_pred))
                mses.append(mean_squared_error(true_vals, y_pred))
                rmses.append(np.sqrt(mean_squared_error(true_vals, y_pred)))
                r2s.append(r2_score(true_vals, y_pred))

            if maes:
                results.append({
                    "Feature": target,
                    "MAE": np.mean(maes),
                    "MSE": np.mean(mses),
                    "RMSE": np.mean(rmses),
                    "R2": np.mean(r2s),
                    "Runs": len(maes)
                })

        eval_df = pd.DataFrame(results)
        print("\n📊 XGBoost Evaluation Results:")
        print(eval_df)
        return eval_df
