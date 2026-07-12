import pandas as pd
import dask.dataframe as dd  
from pathlib import Path
import vitalsImputeNew as vi

vitals_columns = ["spo2", "sbp","dbp","pulse_pressure", "heart_rate","resp_rate", "mbp"]
dtypes = {
        "label_sepsis_within_6h": "Int8",
        "label_sepsis_within_8h": "Int8",
        "label_sepsis_within_12h": "Int8",
        "gender": "Int8",
        "sepsis_label": "Int8",
        "label_sepsis_within_24h": "Int8",
        "hospstay_seq" : "Int8", 
        "hour_index_rev" : "Int8",
        "hospital_expire_flag":"Int8",
        "icustay_seq": "Int8",
        "sepsis_label" : "Int8",
        "charttime":"object",
        "gcs_time": "object"
}


data_dir = Path("C:/phd-final/phd/new_data")
begin_dir = Path("C:/phd-final/phd/newapp")
time_interval = 120
vitals_dir = begin_dir/'secondrun/vitals_filled'

#vitals missingness
ddf_vitals = dd.read_csv(
        #data_dir/'24hours/vitalsDemo.csv', #use for testing purposes
        data_dir/'24hours/vitals_24_hours_final.csv',
        dtype=dtypes,
        sep="|",
    )



    
    # 2. Create the imputer object
imputer = vi.vitalsImputeNew(ddf_vitals, vitals_columns, time_interval,vitals_dir)

thresholds = {
        "heart_rate": (20, 250),
        "resp_rate": (2, 80),
        "temperature": (30, 43),
        "spo2": (50, 100),
        "sbp": (40, 300),
        "dbp": (20, 200),
        "mbp": (20, 250),
        "pulse_pressure": (0, 250),
        "gcs": (3, 15)
        }

vitals = imputer.apply_medical_thresholds(
            ddf_vitals,
            thresholds
        )



ddf_for_stats=imputer.cleanVitals()


missing_pct_vitals = (
    ddf_for_stats.isna().mean().compute() * 100
).sort_values(ascending=False)



print(missing_pct_vitals)
missing_pct_vitals.rename("Missing (%)").to_csv("vitals-missingness.csv")


output_path=str(begin_dir/"secondrun/unfilled/all_merged.parquet/")
ddf_blood = dd.read_parquet(output_path)  
#blood missingness


missing_pct_bld = (
    ddf_blood.isna().mean().compute() * 100
).sort_values(ascending=False)

print(missing_pct_bld)
missing_pct_bld.rename("Missing (%)").to_csv("blood-missingness.csv")