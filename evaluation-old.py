import numpy as np
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

    def __init__(self, data, columns_to_fill, columns_to_fill_2):
       
        self.data = data
        self.columns_to_fill = columns_to_fill
        self.columns_to_fill_2 = columns_to_fill_2
        
        


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

    def simulate_and_evaluate_interpolation(self,data, features, mask_rate, n_runs):
        """Simulate missingness and evaluate interpolation + bfill/ffill."""
        results = []
        for col in features:
            maes, mses,r2,auroc = [], [],[],[]
            for _ in range(n_runs):
                df_copy = data.copy()
                mask = np.random.rand(len(df_copy)) < mask_rate
                true_vals = df_copy.loc[mask, col]
                df_copy.loc[mask, col] = np.nan
                
                # Apply interpolation and filling
                df_copy[col] = df_copy[col].interpolate(method='time').bfill().ffill()

                imputed_vals = df_copy.loc[mask, col]
                maes.append(mean_absolute_error(true_vals, imputed_vals))
                mses.append(mean_squared_error(true_vals, imputed_vals))
                r2.append(r2_score(true_vals, imputed_vals))
                # auroc.append(roc_auc_score(true_vals, imputed_vals))
            results.append({
                "Feature": col,
                "MAE": np.mean(maes),
                "MSE": np.mean(mses),
                "r2": np.mean(r2),
                # "auroc": np.mean(auroc)
            })
        return pd.DataFrame(results)
