import pandas as pd
import dask.dataframe as dd
import duckdb
from sklearn.metrics import mean_squared_error, mean_absolute_error


def clean_dtypes(df):
        df = df.copy()

        # ✅ force numeric columns to float and round to 2 decimals
        numeric_cols = [
            "admission_age", "los_hospital", "los_icu",
            "sbp", "dbp", "pulse_pressure",
            "heart_rate", "resp_rate", "mbp", "temperature", "spo2"
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32").round(2)

        # ✅ keep seq columns as integers
        if "hospstay_seq" in df.columns:
            df["hospstay_seq"] = pd.to_numeric(df["hospstay_seq"], errors="coerce").astype("int8")
        if "icustay_seq" in df.columns:
            df["icustay_seq"] = pd.to_numeric(df["icustay_seq"], errors="coerce").astype("int8")
        
        if "gender" in df.columns:
            df["gender"] = df["gender"].astype("float32")      # convert from object/str
            df["gender"] = df["gender"].fillna(0).astype("int8")  # ensure int8

                #df["gender"] = df["gender"].cat.codes.replace(-1, 0).astype("int8")
        return df

#this is for vitals cleaning
def clearEmpties_ddf(ddf, columns, time_field, thresh_num):
    # Replace "NULL" with NaN
    ddf = ddf.replace("NULL", None)  # Dask supports None as missing
    print("Unique patients before cleaning the dask dataframe:", ddf['subject_id'].nunique().compute())
    print('Rows before droping empties or missing data rows',ddf.shape[0].compute())

    # Drop rows that have fewer than thresh_num non-NA values in specified columns
    ddf = ddf.dropna(subset=columns, thresh=thresh_num)
    print("Unique patients after cleaning the dask dataframe:",  ddf['subject_id'].nunique().compute())
    print('Rows after droping empties or missing data rows',ddf.shape[0].compute())
     
    # Convert time field to datetime
    ddf[time_field] = dd.to_datetime(ddf[time_field], format="%Y-%m-%d %H:%M:%S.%f", errors="coerce")
    
    # Sort within partitions (cheaper than full sort)
    ddf = ddf.map_partitions(lambda df: df.sort_values(by=["stay_id", time_field]))
    
    # Optional: repartition if you plan global operations
    # ddf = ddf.repartition(npartitions=10)
    
    return ddf


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



def mergeDataframes():

    print('start merging the parquet files')

    # Connect (in-memory or persistent)
    con = duckdb.connect()

    # Optional: control parallelism (number of threads)
    con.execute("SET threads = 16;")

    # 1️⃣ Run the chained ASOF joins
    con.execute("""
    CREATE OR REPLACE TABLE all_merged AS
    WITH vitals_blood AS (
        SELECT *
        FROM read_parquet('filled/vitals_filled.parquet') AS vitals
        ASOF LEFT JOIN read_parquet('unfilled/blood.parquet') AS blood
        ON vitals.stay_id = blood.stay_id
        AND CAST(vitals.charttime AS TIMESTAMP) >= CAST(blood.charttime AS TIMESTAMP)
        AND CAST(vitals.charttime AS TIMESTAMP) - CAST(blood.charttime AS TIMESTAMP) <= INTERVAL '24 hour'
    ),

    vitals_blood_gases AS (
        SELECT *
        FROM vitals_blood AS vb
        ASOF LEFT JOIN read_parquet('unfilled/gases.parquet') AS gases
        ON vb.stay_id = gases.stay_id
        AND CAST(vb.charttime AS TIMESTAMP) >= CAST(gases.charttime AS TIMESTAMP)
        AND CAST(vb.charttime AS TIMESTAMP) - CAST(gases.charttime AS TIMESTAMP) <= INTERVAL '24 hour'
    ),

    final_merge AS (
        SELECT *
        FROM vitals_blood_gases AS vbg
        ASOF LEFT JOIN read_parquet('unfilled/glucCreat.parquet') AS gluc
        ON vbg.stay_id = gluc.stay_id
        AND CAST(vbg.charttime AS TIMESTAMP) >= CAST(gluc.charttime AS TIMESTAMP)
        AND CAST(vbg.charttime AS TIMESTAMP) - CAST(gluc.charttime AS TIMESTAMP) <= INTERVAL '24 hour'
    )
    SELECT * FROM final_merge;
    """)

    print("✅ ASOF merges complete: table 'all_merged' created.")

    # 2️⃣ Dynamically find unwanted columns
    cols_to_drop = con.execute("""
        SELECT string_agg(name, ', ')
        FROM pragma_table_info('all_merged')
        WHERE name LIKE 'subject_id_%'
        OR name LIKE 'stay_id_%'
        OR name LIKE 'hadm_id_%';
    """).fetchone()[0]

    print("Columns to exclude:", cols_to_drop)

    # 3️⃣ Export clean bucketed Parquet files
    df_merged_data = con.execute(f"""
        SELECT * EXCLUDE ({cols_to_drop}),
               (abs(hash(stay_id)) % 128) AS bucket
        FROM all_merged
    """).df()

    print("✅ Data fetched into pandas DataFrame, ready for cleaning.")

    # 4️⃣ Clean data types using your custom method
    df_merged_data = clean_dtypes(df_merged_data)
    print("✅ Data types cleaned using InputData.clean_dtypes().")

    # 5️⃣ Write cleaned and bucketed Parquet dataset
    con.register("cleaned_data", df_merged_data)
    con.execute("""
        COPY cleaned_data
        TO 'unfilled/all_merged.parquet'
        (FORMAT PARQUET, PARTITION_BY (bucket), OVERWRITE TRUE);
    """)

    print("✅ Export complete: Parquet dataset written to 'unfilled/all_merged.parquet/'")
    print("Each stay_id is fully contained within a single bucket and unwanted columns are removed.")

    # 6️⃣ Return as Dask DataFrame for further processing
    all_merged = dd.read_parquet("unfilled/all_merged.parquet")
    return all_merged



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
