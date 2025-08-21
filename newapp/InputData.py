import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error


def clearEmpties(df, columns, time_field, thresh_num):
        # delete the empty rows
        df.replace("NULL", pd.NA, inplace=True)
        df.dropna(subset=columns, how="all", inplace=True)
        # delete rows that more than four vitals are missing
        df.dropna(subset=columns, thresh=thresh_num, inplace=True)
        df[time_field] = pd.to_datetime(
            df[time_field], format="%Y-%m-%d %H:%M:%S.%f", errors="coerce"
        )
        df.sort_values(by=["stay_id", time_field], inplace=True)
        return df

def transFloat32 (df, columns):

    df[columns] = df[columns].astype(pd.Float32Dtype()).round(2)

    return df


# ---START of ffill bfill evaluation-------#
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


# new Addition, merge all dataframes into a single one
def mergeDataframes(df_blood, lab_columns, df_glucoCreat, gluc_columns, df_vitals,df_gases,gases_columns):
    # first order dataframes by stay_id and time
    print('merging dataframes:')

    df_vitals.rename(columns={"vital_time": "charttime"}, inplace=True)   
    # make sure indexes are reset and they have date as data type
    df_vitals = df_vitals.reset_index()
    df_blood = df_blood.reset_index()

    df_glucoCreat[["subject_id", "stay_id"]] = df_glucoCreat[
            ["subject_id", "stay_id"]
        ].astype(pd.Int32Dtype())

    df_blood.sort_values(by=["charttime", "stay_id"], inplace=True)
    df_glucoCreat.sort_values(by=["charttime", "stay_id"], inplace=True)
    df_vitals.sort_values(by=["charttime", "stay_id"], inplace=True)
    df_gases.sort_values(by=["charttime", "stay_id"], inplace=True)

    df_vitalsBlood = pd.merge_asof(
            df_vitals,
            df_blood,
            on="charttime",  # Merging on charttime
            by=["stay_id"],  # Ensure same subject and stay
            suffixes=("_vitals", "_blood"),
            tolerance=pd.Timedelta("24h"),  # Adjust this to your tolerance level
        )

    print('after df_blood merging')
    print(df_vitalsBlood.info())     
    print(df_vitalsBlood.head()) 

    df_vitalsBloodGases = pd.merge_asof(
            df_vitalsBlood,
            df_gases,
            on="charttime",  # Merging on charttime
            by=["stay_id"],  # Ensure same subject and stay
            suffixes=("_vb", "_gases"),
            tolerance=pd.Timedelta("24h"),  # Adjust this to your tolerance level
        )

    print('merge glucose and creatinine new')
    all = pd.merge_asof(
            df_vitalsBloodGases,
            df_glucoCreat,
            on="charttime",  # Merging on charttime
            by=["stay_id"],  # Ensure same subject and stay
            suffixes=("_vbg", "_gluc"),
            tolerance=pd.Timedelta("24h"),  
        )

    print("after glucose merging")


    # all.set_index("charttime",inplace=True)
    all.rename(columns={"subject_id_vitals": "subject_id","hadm_id_vitals":"hadm_id"}, inplace=True)
    
    all.drop(
        [
            "hadm_id_blood",
            "subject_id_blood",
            "level_0",
            "index_vitals",
            "subject_id_vbg",
            "subject_id_gluc",
            "relative_time_min_gases",
            "time_gap_min_gases",
            "index_blood",
            "hadm_id",
            "group_gases"
        ],
        axis=1,
        inplace=True,
    )

    return all


def populateLabsResults(df,lab_columns,gluc_columns):
        print('fill blood results after they are joined with rest of the data:')
        print(df.info())
    
        
        lab_columns.extend(gluc_columns)
        "glucose", "creatinine"

        df[lab_columns] = (
            df.groupby(["subject_id", "stay_id", "group"], group_keys=False)
            .apply(lambda group: (
                group.set_index("charttime")[lab_columns]
                    .ffill()
                    .bfill()
                    .interpolate(method="linear")
                    #.reset_index(drop=True)
            ))
            .reset_index(drop=True)
        )
    
        return df
