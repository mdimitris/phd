import pandas as pd
import InputData
import vitalsImputeNew as vi
import glucoseImpute as gl 
import gasesImpute as ga 
import numpy as np
import InputData
import Evaluation as ev
import dask.dataframe as dd
import pyarrow.parquet as pa
import os
import xgBoostFill as xg
from LSTMImputer import LSTMImputer
import SepsisTrainer  as lstm
import helpers as help
import bloodImpute as bloodImp
from pathlib import Path
from sklearn.model_selection import GroupKFold
import time

features_columns = ['gender', 'hospstay_seq', 'icustay_seq', 'admission_age', 'los_hospital', 'los_icu', "spo2", "sbp","dbp","pulse_pressure", "heart_rate","resp_rate", "mbp","temperature"]
blood_columns = ['hematocrit', 'hemoglobin', 'mch', 'mchc', 'mcv', 'rbc']
gases_columns = ['paco2', 'fio2', 'pao2']
glucCreat_columns = ["creatinine","glucose"]
# ------------24 hours-------------#
# vitals_dir="filled/vitals_filled.parquet/"
#vitals_dir="/root/scripts/newapp/secondrun/vitals_filled.parquet/"

while True:
    environment = input("Please choose environment, 1: Windows  2: Linux: ")

    if environment == "1":
        print("Windows platform chosen as environment")
        begin_dir = Path("C:/phd-final/phd/newapp")
        data_dir = Path("C:/phd-final/phd/new_data")
        break
    elif environment == "2":
        print("Linux platform chosen as environment")
        begin_dir =Path('/root/scripts/newapp')
        data_dir = Path("/root/scripts/new_data")
        break
    else:
        print("Wrong value for environment, please try again.")


vitals_dir = begin_dir/'secondrun/vitals_filled.parquet'

time_interval = 15
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

if os.listdir(vitals_dir) == []:

    rows = 100000
    ddf_vitals = dd.read_csv(
        #"/root/scripts/newapp/vitalsDemo.csv", #use for testing purposes
        data_dir/'24hours/vitals_24_hours_final.csv',
        dtype=dtypes,
        sep="|",
    )
    
    # 2. Create the imputer object
    imputer = vi.vitalsImputeNew(ddf_vitals, vitals_columns, time_interval)
    # 3. Prepare and impute the data
    imputer.prepareVitals()
    
    print('read filled parquets for vitals evaluation (no temperature)')    
    print ('read filled vitals from:',vitals_dir)
    ddf_vitals_filled = dd.read_parquet(vitals_dir)        
    cleaned_ddf = InputData.clean_dtypes(ddf_vitals_filled)
    df_sample = cleaned_ddf.sample(frac=0.3).compute() 

    
    # Step 3: Run evaluation
    imputer = vi.vitalsImputeNew(df_sample, vitals_columns, time_interval)

    vitals_evaluator = ev.Evaluation(
        imputer, df_sample, columns_to_fill=vitals_columns, mask_rate=0.2, n_runs=3
    )

    results = []

        
    for col in vitals_columns:
        print(f"Evaluating {col}...") 
        res = vitals_evaluator.evaluate_masking(df_sample, col, mask_frac=0.2)
        results.append(res)

    df_results = pd.DataFrame(results)
    print("\n📊 Vitals Evaluation Results:")
    print(df_results)


merged_dir = begin_dir/'secondrun/unfilled/all_merged.parquet'

if os.listdir(merged_dir) == []:    
    print('create parquets from csv files')
        # #ddf_bloodResults = dd.read_csv(r"C:\phd-final\phd\new_data\24hours\blood_24_hours.csv", sep='|')
        # ddf_bloodResults = dd.read_csv("/root/scripts/new_data/24hours/blood_24_hours.csv", sep='|')
        # help.prepareDataset(ddf_bloodResults,blood_columns,["rdwsd","admittime"],'blood')

        # ddf_gases = dd.read_csv("/root/scripts/new_data/24hours/gases_24_hours_final.csv", dtype={"charttime": "object"}, sep='|')
        # help.prepareDataset(ddf_gases,gases_columns,["hadm_id","sofa_time"],'gases')    

        # ddf_glucoCreat = dd.read_csv("/root/scripts/new_data/24hours/glucose_creatine_24_hours.csv", dtype={"charttime": "object"}, sep='|')
        # help.prepareDataset(ddf_glucoCreat,glucCreat_columns,["hadm_id"],'glucCreat')

    ddf_vitals = dd.read_parquet(begin_dir/"secondrun/vitals_filled.parquet/")
    print(ddf_vitals.info())
    InputData.mergeDataframes(begin_dir)



# print('read merged parquet (still with temperature not filled)')
merged_ddf = dd.read_parquet(merged_dir)


temperature_folder = begin_dir/'secondrun/filled/temperature_filled.parquet/'

if os.listdir(temperature_folder) == []:   
    # Diagnostics
    df_vitals = dd.read_csv(data_dir/'24hours/vitals_24_hours_final.csv', sep='|', dtype=dtypes)
    help.diagnostics(merged_ddf,df_vitals)

    
    temperature_imputer = vi.vitalsImputeNew(merged_ddf,['temperature'], 7)

    # Fill temperature and save result
    filled_ddf = temperature_imputer.fill_temperature_continuous(
        parquet_path=merged_dir,
        output_path=temperature_folder
    )

    # Sample for evaluation
    df_sample_eval = filled_ddf.sample(frac=0.2, random_state=42).compute()

    # Create the evaluator
    evaluator = ev.Evaluation(
        imputer=temperature_imputer,                 # our ffill/bfill imputer
        data=df_sample_eval,
        columns_to_fill=["temperature"], # what we evaluate
        mask_rate=0.2,                   # % of values to mask artificially
        n_runs=3                         # repeat 3 times for robustness
    )

    # Run evaluation for temperature
    results = []
    for col in ["temperature"]:
        print(f"Evaluating {col}...")
        res = evaluator.evaluate_masking(df_sample_eval, col, mask_frac=0.2)
        results.append(res)

    df_results = pd.DataFrame(results)
    print(df_results)


#since I finished with vitals now read final parquets  in temperature filled for blood results filling
blood_dir = begin_dir/'secondrun/filled/blood_filled.parquet'


merged_filled_blood = dd.read_parquet(temperature_folder)

if os.listdir(blood_dir) == []:   

    print('start Blood filling in the vitals filled dataset')

#Train for Blood MICE impute
    blood_imputer = bloodImp.bloodImpute(
        blood_ddf=merged_filled_blood,
        blood_columns=blood_columns,
        sample_size=250000,  # for MICE training sample
        output_folder=blood_dir,  # folder
        n_output_files=128  # save in 128 Parquets
    )

    blood_imputer.run()


    # -----------------------------
    # 4. Load a sample for evaluation
    # -----------------------------
    # Pick one or a few Parquet batches for evaluation
    merged_filled_blood = dd.read_parquet(blood_dir)
    print("Calculate missing values after blood filling")
    help.calculateMissing(merged_ddf)
    cleaned_ddf = InputData.clean_dtypes(merged_filled_blood)
    df_sample = cleaned_ddf.sample(frac=0.4).compute() 
    # -----------------------------
    # 5. Create Evaluation instance
    # -----------------------------
    evaluator = ev.Evaluation(
        imputer=blood_imputer,      # the filling object
        data=df_sample,             # sample for evaluation
        columns_to_fill=blood_columns,
        mask_rate=0.2,              # fraction of observed values to mask
        n_runs=3
    )

    # -----------------------------
    # 6. Run evaluation for each blood column
    # -----------------------------
    results = []
    for col in blood_columns:
        print(f"Evaluating {col}...") 
        res = evaluator.evaluate_masking(df_sample, col, mask_frac=0.2)
        results.append(res)

    df_results = pd.DataFrame(results)
    print("\n📊 Blood Imputation Evaluation Results:")
    print(df_results)


# Fill blood gases
gases_dir = begin_dir/'secondrun/filled/blood_filled_2nd.parquet'

if os.listdir(gases_dir) == []:   

    print('begin fill platelet process')

    #fill blood gases 
    feature_map_blood = {
        "platelet": [
            "rbc", "hemoglobin", "hematocrit", "mcv", "mch", "mchc", "rdw",
            "wbc", "creatinine",
            "heart_rate", "resp_rate", "sbp", "dbp", "mbp", "spo2",
            "glucose", 
            "admission_age", "gender"
        ],
        "wbc": [
            "rbc", "hemoglobin", "hematocrit", "mcv", "mch", "mchc",
            "platelet",
            "temperature", "spo2", "heart_rate", "resp_rate",
            "creatinine", "glucose",
            "admission_age", "gender"
        ],
        "rbc": [
            "hemoglobin", "hematocrit", "mcv", "mch", "mchc",
            "platelet", "wbc",
            "heart_rate", "resp_rate", "sbp", "dbp", "mbp",
            "temperature", 
            "glucose", 
            "admission_age", "gender"
        ]
    }

    merged_filled_gases = dd.read_parquet(blood_dir)

    sparse_features=['platelet','wbc', 'rdw','glucose','creatinine','paco2', 'fio2', 'pao2']

    xgbImputer = xg.xgBoostFill(
        target_columns=sparse_features,
        features=features_columns,
        feature_map=[],
        short_gap_targets=sparse_features,
        random_state=42
    )

    
    cleaned_ddf = InputData.clean_dtypes(merged_filled_gases)
    df_sample = cleaned_ddf.sample(frac=0.6).compute()  # small representative sample
    
    xgbImputer.fit(df_sample)
    #meta = InputData.clean_dtypes(merged_filled_gases._meta)
    ddf_filled = xgbImputer.transform(merged_filled_gases)  # Dask DF
    ddf_filled = ddf_filled.persist()
    ddf_filled.to_parquet(gases_dir, write_index=False)
    # # 3. Evaluate XGBoost
    # 7. Evaluate on a pandas sample using your evaluation class

    xgboost_evaluator = ev.Evaluation(
        imputer=xgbImputer,
        data=df_sample,
        columns_to_fill=sparse_features,
        mask_rate=0.3,
        n_runs=5
    )

    results_df_ml = xgboost_evaluator.evaluate_sparse_with_ml(
        imputer=xgbImputer,
        mask_frac=0.05,
        n_runs=5
    )
    print(results_df_ml)


####SEPSIS PREDICTION######

import xgboost as xgb
print('xgboost_version:',xgb.__version__)


#first read data and delete unecesairy columns
print ('begin sepsis prediction with all four methods')
merged_filled_all = dd.read_parquet(gases_dir)

cols_to_drop = ["fio2", "pao2", "paco2","pf_ratio","charttime_3","charttime_4","charttime_5"]
merged_filled_all = merged_filled_all.drop(columns=cols_to_drop, errors="ignore")
final_df = merged_filled_all.compute()
print("Total rows:",  len(final_df))
print("Unique subject_id:", final_df["subject_id"].nunique())
print("Unique stay_id:", final_df["stay_id"].nunique())

missing_counts = final_df.isna().sum()

print(missing_counts)

cols_required = [
    "heart_rate",
    "resp_rate",
    "temperature",
    "sbp",
    "dbp",
    "mbp",
    "spo2",
    "pulse_pressure",
    "wbc",
    "platelet",
    "rdw",
    "glucose", 
    "creatinine",
]

clean_df = final_df.dropna(subset=cols_required)
clean_df["charttime"] = pd.to_datetime(clean_df["charttime"], errors="coerce")
clean_df["icu_intime"] = pd.to_datetime(clean_df["icu_intime"], errors="coerce")


clean_df["hours_since_icu_intime"] = (
    (clean_df["charttime"] - clean_df["icu_intime"]).dt.total_seconds() / 3600.0
)

clean_df = clean_df.sort_values(["stay_id", "charttime"])
group = clean_df.groupby("stay_id")


# total_rows = len(clean_df)
# unique_stays = clean_df["stay_id"].nunique()
# unique_subjects = clean_df["subject_id"].nunique()

# print("Total rows:", total_rows)
# print("Unique stay_id:", unique_stays)
# print("Unique subject_id:", unique_subjects)
# print(clean_df["label_sepsis_within_6h"].value_counts())
# print(clean_df["label_sepsis_within_24h"].value_counts())



# feature columns for lstm
features = [
    "gender", "admission_age", "hours_since_icu_intime", 
    "icustay_seq", 
    "hospstay_seq",  # static
    "spo2", "sbp", "dbp", "pulse_pressure",
    "heart_rate", "resp_rate", "temperature",
    "mbp", "wbc", "platelet", "hematocrit", "hemoglobin",
    "mch", "mchc", "mcv", "rbc", "rdw",
    "glucose", "creatinine"
]


l_col_val="label_sepsis_within_6h"

##LSTM##



trainer = lstm.SepsisTrainer(
    features=features,
    label_col=l_col_val,  # or your chosen label
    seq_len=30,
    hidden_size=32,
    num_layers=2,
    batch_size=32,
    lr=1e-3,
    epochs=4,
)

trainer.prepare_data(clean_df, time_col="charttime")  # or "hour_index_rev" etc.
trainer.train()

# You can also re-evaluate later:

val_loss, val_auc, val_acc, val_auprc, val_mcc, thr = trainer.evaluate(trainer.val_loader)
print("Final validation -> Loss:", val_loss, "AUC:", val_auc, "AUPRC:", val_auprc, "MCC:", val_mcc, "thr:", thr, "Acc:", val_acc)


def run_groupkfold_cv(
    df,
    trainer_kwargs,
    time_col="charttime",
    group_col="stay_id",
    n_splits=5,
):
    gkf = GroupKFold(n_splits=n_splits)
    groups = df[group_col].values

    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(
        gkf.split(df, y=df[trainer_kwargs["label_col"]], groups=groups),
        start=1
    ):
        print(f"\n===== Fold {fold}/{n_splits} =====")

        start_time = time.time()

        df_train = df.iloc[train_idx].copy()
        df_val = df.iloc[val_idx].copy()

        trainer = lstm.SepsisTrainer(**trainer_kwargs)

        # ✅ Branch by model type
        if trainer.model_type in ("lstm", "gru"):
            trainer.prepare_data_from_splits(df_train, df_val, time_col=time_col)
            trainer.train()
            elapsed_min = (time.time() - start_time) / 60.0

            val_loss, val_auc, val_acc, val_auprc, val_mcc, thr = trainer.evaluate(trainer.val_loader)

        elif trainer.model_type in ("lgbm", "xgb"):
            # this method should train the tree model and return metrics dict
            metrics = trainer.train_tree_from_splits(df_train, df_val)
            elapsed_min = (time.time() - start_time) / 60.0  # or metrics["time_min"] if you store it there

            val_auc   = metrics["auc"]
            val_auprc = metrics["auprc"]
            val_mcc   = metrics["mcc"]
            val_acc   = metrics["acc"]
            thr       = metrics["thr"]
            val_loss  = float("nan")  # not applicable for trees (unless you compute logloss)

        else:
            raise ValueError(f"Unknown model_type: {trainer.model_type}")

        

        fold_metrics.append({
            "fold": fold,
            "auc": val_auc,
            "auprc": val_auprc,
            "mcc": val_mcc,
            "acc": val_acc,
            "thr": thr,
            "time_min": elapsed_min,

        })

        print(
            f"Fold {fold} -> "
            f"AUC={val_auc:.4f}, "
            f"AUPRC={val_auprc:.4f}, "
            f"MCC={val_mcc:.4f}, "
            f"thr={thr:.2f}, "
            f"Acc={val_acc:.4f}",
            f"Time={elapsed_min:.2f} min"
            
        )
   


    results = pd.DataFrame(fold_metrics)

    print("\n===== CV Summary (mean ± std) =====")
    for col in ["auc", "auprc", "mcc", "acc"]:
        print(f"{col.upper()}: {results[col].mean():.4f} ± {results[col].std():.4f}")

    print("\n===== Timing Summary =====")
    print(
        f"TIME (min): "
        f"{results['time_min'].mean():.2f} ± "
        f"{results['time_min'].std():.2f}"
    )

    return results

cv_results = run_groupkfold_cv(
    df=clean_df,
    trainer_kwargs=dict(
        features=features,
        label_col="label_sepsis_within_24h",
        seq_len=30,
        hidden_size=32,
        num_layers=2,
        batch_size=32,
        lr=1e-3,
        epochs=4,
        early_stop_patience=2,
        early_stop_metric="auprc",
    ),
    time_col="charttime",
    group_col="subject_id",  # or stay_id
    n_splits=5,
)


##GRU##
methods=['lstm','gru']

cv_results_gru = run_groupkfold_cv(
    df=clean_df,
    trainer_kwargs=dict(
        features=features,
        label_col=l_col_val,
        seq_len=30,
        hidden_size=32,
        num_layers=2,
        batch_size=32,
        lr=1e-3,
        epochs=4,
        early_stop_patience=2,
        early_stop_metric="auprc",
        model_type="gru",     # 👈 switch here
        dropout=0.2,
    ),
    time_col="charttime",
    group_col="subject_id",
    n_splits=5,
)



trainer_kwargs=dict(
    features=features,
    label_col=l_col_val,
    model_type="gru",
    dropout=0.2,
    seq_len=30,
    hidden_size=32,
    num_layers=2,
    batch_size=32,
    lr=1e-3,
    epochs=4,
    early_stop_patience=2,
    early_stop_metric="auprc",

)

print("\n===== GRU CV Summary =====")
print(cv_results_gru.mean(numeric_only=True))


# ####XGBOOST####

trainer_kwargs = dict(
    features=features,
    label_col=l_col_val,
    model_type="xgb",        # 👈 switch here
    early_stopping_rounds=50,
    tree_params={},          # optional overrides
)

cv_results_xgb = run_groupkfold_cv(
    df=clean_df,
    trainer_kwargs=trainer_kwargs,
    group_col="subject_id",
    n_splits=5,
)

print("\n===== XGBOOST CV Summary =====")
print(cv_results_xgb.mean(numeric_only=True))

####'lightGBM'####

trainer_kwargs = dict(
    features=features,
    label_col=l_col_val,
    model_type="lgbm",        # 👈 switch here
    early_stopping_rounds=50,
    tree_params={},          # optional overrides
)

cv_results_gbm = run_groupkfold_cv(
    df=clean_df,
    trainer_kwargs=trainer_kwargs,
    group_col="subject_id",
    n_splits=5,
)


print("\n===== LightGBM CV Summary =====")
print(cv_results_gbm.mean(numeric_only=True))