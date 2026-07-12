import sys
print("PYTHON:", sys.executable)
import pandas as pd
import InputData
import vitalsImputeNew as vi
import InputData
import Evaluation as ev
import dask.dataframe as dd
import pyarrow.parquet as pa
import os
import torch
from torch.utils.data import DataLoader, TensorDataset
import xgBoostFill as xg
#from LSTMImputer import LSTMImputer
import SepsisTrainer  as lstm
import helpers as help
import bloodImpute as bloodImp
from pathlib import Path
from sklearn.model_selection import GroupKFold
import time
import xgboost as xgb
import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
)


os.system('cls' if os.name == 'nt' else 'clear')


features_columns = ['gender', 'hospstay_seq', 'icustay_seq', 'admission_age', 'los_hospital', 'los_icu', "spo2", "sbp","dbp","pulse_pressure", "heart_rate","resp_rate", "mbp","temperature"]
blood_columns = ['hematocrit', 'hemoglobin', 'mch', 'mchc', 'mcv', 'rbc']
gases_columns = ['paco2', 'fio2', 'pao2']
glucCreat_columns = ["creatinine","glucose"]
# ------------24 hours-------------#
# eicu_dir="filled/vitals_filled.parquet/"
#eicu_dir="/root/scripts/newapp/secondrun/vitals_filled.parquet/"

while True:
    environment = input("Please choose environment, 1: Windows  2: Linux: ")

    if environment == "1":
        print("Windows platform chosen as environment")
        begin_dir = Path("C:/phd-final/phd/newapp")
        data_dir = Path("C:/phd-final/phd/new_data")
        output_file = begin_dir / "secondrun" / "eicu_final_imputed_single.parquet"
        break
    elif environment == "2":
        print("Linux platform chosen as environment")
        begin_dir =Path('/root/scripts/newapp')
        data_dir = Path("/root/scripts/new_data")
        break
    else:
        print("Wrong value for environment, please try again.")


eicu_dir = begin_dir/'secondrun/eicu_filled'


time_interval = 5
vitals_columns = ["spo2", "sbp","dbp","pulse_pressure", "heart_rate","resp_rate", "mbp"]

dtypes = {
        "label_sepsis_within_6h": "Int8",
        "label_sepsis_within_8h": "Int8",
        "label_sepsis_within_12h": "Int8",
        "label_sepsis_within_24h": "Int8",
        "gender": "Int8",
        "hospstay_seq" : "Int8", 
        "admission_age": "Int64",
        "sepsis_onset_hour" :"Int64",
        "admission_age": pd.Float32Dtype(),
        "los_hospital": pd.Float32Dtype(),
        "los_icu": pd.Float32Dtype(),
        "hours_before_sepsis": pd.Float32Dtype(),
#         "hour_index_rev" : "Int8",
#         "icustay_seq": "Int8",
#         "charttime":"object",
#         "gcs_time": "object"
}

if output_file.exists():
    print("Final eICU imputed dataset already exists. Going imediately to models evaluation")
else:
    if os.listdir(eicu_dir) == []:

        ddf_vitals = dd.read_csv(
            #data_dir/'eicu/final_dataset_ml_debug.csv', #use for testing purposes
            data_dir/'eicu/final_dataset_ml_50k.csv',
            dtype=dtypes,
            sep="|",
        )
        
        # 2. Create the imputer object
        imputer = vi.vitalsImputeNew(ddf_vitals, vitals_columns, time_interval,eicu_dir)
        # 3. Prepare and impute the data
        imputer.prepareVitals()
        
        print('read filled parquets for vitals evaluation (no temperature)')    
        print ('read filled vitals from:',eicu_dir)
    # import os
    # import pyarrow.parquet as pq

    # for f in sorted(os.listdir(eicu_dir)):
    #     path = eicu_dir / f

    #     try:
    #         pq.read_table(path)
    #         print(f"{f}  OK")

    #     except Exception as e:
    #         print(f"{f}  FAILED")
    #         print(e)
    # exit()
        ddf_vitals_filled = dd.read_parquet(eicu_dir)     

        cleaned_ddf = InputData.clean_dtypes(ddf_vitals_filled)
        df_sample = cleaned_ddf.sample(frac=0.3).compute()
        

        # Step 3: Run evaluation
        imputer = vi.vitalsImputeNew(df_sample, vitals_columns, time_interval,eicu_dir)

        vitals_evaluator = ev.Evaluation(
            imputer, df_sample, columns_to_fill=vitals_columns, mask_rate=0.2, n_runs=3
        )

        results = []

            
        for col in vitals_columns:
            print(f"Evaluating {col}...") 
            res = vitals_evaluator.evaluate_masking(df_sample, col, mask_frac=0.2)
            results.append(res)
            print(res)

        df_results = pd.DataFrame(results)
        print("\n📊 Vitals Evaluation Results for eICU:")
        print(df_results)


    # print('read merged parquet (still with temperature not filled)')
    merged_ddf = dd.read_parquet(eicu_dir)


    temperature_folder = begin_dir/'secondrun/eicu_filled_temperature/'


    if os.listdir(temperature_folder) == []:   
        
        temperature_imputer = vi.vitalsImputeNew(merged_ddf,['temperature'], 7,temperature_folder)

        # Fill temperature and save result
        filled_ddf = temperature_imputer.fill_temperature_continuous(
            parquet_path=eicu_dir,
            output_path=temperature_folder
        )

        # Sample for evaluation
        df_sample_eval = filled_ddf.sample(frac=0.3, random_state=42).compute()

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

    print ("Call blood imputer")
    eicu_blood_dir = begin_dir/'secondrun/eicu_filled_blood'

    ######### Prepare Blood Data ############
    print("Read vitals-filled data")
    vitals_empty_temp = dd.read_parquet(temperature_folder)


    blood_imputer = bloodImp.bloodImpute(
            blood_ddf=vitals_empty_temp,
            blood_columns=blood_columns,
            sample_size=250000,  # for MICE training sample
            output_folder=eicu_blood_dir,  # folder
            dataset_name='eicu',
            n_output_files=128  # save in 128 Parquets
        )

    blood_imputer.run()


    print("Running blood imputation on eICU...")

    # blood_imputer = bloodImp.bloodImpute(
    #     blood_ddf=dd.read_parquet(temperature_folder),
    #     blood_columns=blood_columns,
    #     sample_size=250000,          # ignored for eICU
    #     output_folder=eicu_blood_dir,
    #     dataset_name="eicu",
    #     n_output_files=128,
    # )

    blood_imputer.run()


    merged_filled_blood = dd.read_parquet(eicu_blood_dir)

    cleaned_ddf = InputData.clean_dtypes(merged_filled_blood)

    df_sample = cleaned_ddf.sample(frac=0.4).compute()

    evaluator = ev.Evaluation(
        imputer=blood_imputer,
        data=df_sample,
        columns_to_fill=blood_columns,
        mask_rate=0.2,
        n_runs=3,
    )

    results = []

    for col in blood_columns:
        print(f"Evaluating {col}...")
        res = evaluator.evaluate_masking(df_sample, col, mask_frac=0.2)
        results.append(res)

    print(pd.DataFrame(results))

    merged_filled = dd.read_parquet(eicu_blood_dir)

    xgbImputer = joblib.load(
        begin_dir / "models" / "xgb_sparse_imputer.pkl"
    )

    print("xgb imputer for sparse data is loaded")
    eicu_filled = xgbImputer.transform(merged_filled)


    ddf_filled = eicu_filled.compute()



    cleaned_ddf = InputData.clean_dtypes(ddf_filled)

    if isinstance(cleaned_ddf, dd.DataFrame):
        df_sample = cleaned_ddf.sample(frac=0.4).compute()
    else:
        df_sample = cleaned_ddf.sample(frac=0.4, random_state=42)

    print("evaluate xgb imputer for sparse data")
    xgboost_evaluator = ev.Evaluation(
        imputer=xgbImputer,
        data=df_sample,
        columns_to_fill=[
            "platelet", 
            "wbc",
            "rdw",
            "glucose",
            "creatinine",
            "paco2",
            "fio2",
            "pao2",
        ],
        mask_rate=0.3,
        n_runs=5,
    )

    results_df_ml = xgboost_evaluator.evaluate_sparse_with_ml(
        imputer=xgbImputer,
        mask_frac=0.05,
        n_runs=5,
    )

    print(results_df_ml)


    if isinstance(ddf_filled, dd.DataFrame):
        final_df = ddf_filled.compute()
    else:
        final_df = ddf_filled

    final_df = final_df.sort_values(
        ["stay_id", "hour_index"]
    )



    final_df.to_parquet(
        output_file,
        index=False
    )



features = [
    "gender",
    "admission_age",
    "hours_since_icu_intime",
    "icustay_seq",
    "hospstay_seq",
    "spo2",
    "sbp",
    "dbp",
    "pulse_pressure",
    "heart_rate",
    "resp_rate",
    "temperature",
    "mbp",
    "wbc",
    "platelet",
    "hematocrit",
    "hemoglobin",
    "mch",
    "mchc",
    "mcv",
    "rbc",
    "rdw",
    "glucose",
    "creatinine",
]


print("Read the final parquet file")
final_dd = pd.read_parquet(output_file)
print ('sepsis prevalence before delete:')
print(final_dd["label_sepsis_within_24h"].value_counts(normalize=True))

print("Drop rows with missing values")
final_dd = final_dd.dropna(subset=features)
print ('sepsis prevalence after delete:')
print(final_dd.shape)

print(final_dd["label_sepsis_within_24h"].value_counts())

print(final_dd["label_sepsis_within_24h"].value_counts(normalize=True))


no_empty = begin_dir / "secondrun" / "eicu_final_imputed_single_2.parquet"
final_dd.to_parquet(no_empty)

print(final_dd[features].describe().T)
eicu_stats = final_dd[features].describe().T
eicu_stats.to_csv(begin_dir / "eicu_feature_stats.csv")
print("Saved eICU statistics.")

final_df = pd.read_parquet(no_empty)

features = joblib.load(
    begin_dir / "models" / "features.pkl"
)

print("Features.plk loaded")


import torch
from torch.utils.data import DataLoader, TensorDataset

def find_best_threshold(y_true, probs):
    thresholds = np.linspace(0.01, 0.99, 99)

    best_thr = 0.5
    best_mcc = -1

    for t in thresholds:
        preds = (probs >= t).astype(int)
        mcc = matthews_corrcoef(y_true, preds)

        if mcc > best_mcc:
            best_mcc = mcc
            best_thr = t

    return best_thr, best_mcc

def evaluate_rnn_model(
    model_type,
    horizon,
    final_df,
    begin_dir,
    features,
    threshold,
):

    seq_len = 12
    hidden_size = 32
    num_layers = 2

    label_col = f"label_sepsis_within_{horizon}"

    scaler = joblib.load(
        begin_dir / "models" / horizon / f"final_{model_type}_scaler.pkl"
    )

    df = final_df.copy()

    if "gender" in df.columns and df["gender"].dtype == object:
        df["gender"] = (
            df["gender"]
            .map({"M": 1, "F": 0})
            .astype("float32")
        )

    df["charttime"] = pd.to_datetime(df["charttime"])

    df = df.sort_values(
        ["stay_id", "charttime"]
    )

    df[features] = scaler.transform(
        df[features]
    )

    X_seq = []
    y_seq = []

    for _, g in df.groupby("stay_id"):

        values_X = g[features].values
        values_y = g[label_col].values

        if len(g) <= seq_len:
            continue

        for i in range(len(g) - seq_len):

            X_seq.append(
                values_X[i:i + seq_len]
            )

            y_seq.append(
                values_y[i + seq_len]
            )

    X_seq = np.asarray(X_seq, dtype=np.float32)
    y_seq = np.asarray(y_seq)

    X_tensor = torch.from_numpy(X_seq)

    loader = DataLoader(
        TensorDataset(X_tensor),
        batch_size=512,
        shuffle=False,
    )

    if model_type == "lstm":

        model = lstm.LSTMModel(
            input_size=len(features),
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=1,
        )

    else:

        model = lstm.GRUModel(
            input_size=len(features),
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=1,
        )

    state_dict = torch.load(
        begin_dir / "models" / horizon / f"final_{model_type}.pt",
        map_location="cpu",
    )

    model.load_state_dict(state_dict)

    model.eval()

    probs = []

    with torch.no_grad():

        for (batch,) in loader:

            logits = model(batch)

            probs.extend(
                torch.sigmoid(logits)
                .numpy()
                .ravel()
            )

    probs = np.asarray(probs)

    preds = (probs >= threshold).astype(int)

    return {
        "Model": model_type.upper(),
        "Window": horizon,
        "Accuracy": accuracy_score(y_seq, preds),
        "AUROC": roc_auc_score(y_seq, probs),
        "AUPRC": average_precision_score(y_seq, probs),
        "MCC": matthews_corrcoef(y_seq, preds),
    }


def get_threshold(thresholds, horizon, model):

    return thresholds.loc[
        (thresholds["Window"] == horizon) &
        (thresholds["Model"] == model),
        "Threshold"
    ].iloc[0]


horizons = ["24h", "12h", "8h", "6h"]
results = []

thresholds = pd.read_csv(
    begin_dir / "models" / "thresholds.csv",
    sep=";"
)

for horizon in horizons:

    print(f"\n========== {horizon} ==========")

    label_col = f"label_sepsis_within_{horizon}"

    X = final_df[features]
    y = final_df[label_col]

    print(final_df["label_sepsis_within_24h"].value_counts())
    print(final_df["label_sepsis_within_24h"].value_counts(normalize=True))
    

    #################################################
    # XGBOOST
    #################################################
    print('evalute xgboost eICU')


    xgb_model = xgb.Booster()
    xgb_model.load_model(
        str(begin_dir / "models" / f"{horizon}" / f"final_xgb.json")
    )



    dtest = xgb.DMatrix(X)

    probs = xgb_model.predict(dtest)

    threshold = get_threshold(
        thresholds,
        horizon,
        "XGBoost"
    )

    preds = (probs >= threshold).astype(int)

    tmp = pd.DataFrame({
        "prob": probs,
        "label": y.values
    })

    # print(tmp.groupby("label")["prob"].describe())


    tmp["decile"] = pd.qcut(tmp["prob"], 10, labels=False)

    print(
        tmp.groupby("decile")
        .agg(
            mean_prob=("prob", "mean"),
            prevalence=("label", "mean"),
            n=("label", "size")
        )
    )
    


    # best_thr, best_mcc = find_best_threshold(y, probs)

    # print(f"Best threshold: {best_thr:.2f}")
    # preds = (probs >= best_thr).astype(int)
    results.append({
        "Model": "XGBoost",
        "Window": horizon,
        "Accuracy": accuracy_score(y, preds),
        "AUROC": roc_auc_score(y, probs),
        "AUPRC": average_precision_score(y, probs),
        "MCC": matthews_corrcoef(y, preds),
    })

    print(results)

    #################################################
    # LIGHTGBM
    #################################################
    print('evalute lightgbm eICU')
    lgb_model = joblib.load(
        begin_dir / "models" / f"{horizon}" / f"final_lgbm.pkl"
    )

    probs = lgb_model.predict_proba(X)[:,1]

    threshold = get_threshold(
        thresholds,
        horizon,
        "LightGBM"
    )

    preds = (probs >= threshold).astype(int)

    # print(f"Best threshold: {best_thr:.2f}")

    results.append({
        "Model": "LightGBM",
        "Window": horizon,
        "Accuracy": accuracy_score(y, preds),
        "AUROC": roc_auc_score(y, probs),
        "AUPRC": average_precision_score(y, probs),
        "MCC": matthews_corrcoef(y, preds),
    })


    #################################################
    # LSTM
    #################################################


    print('evalute lstm eICU')
    threshold = get_threshold(
        thresholds,
        horizon,
        "LSTM"
    )

    results.append(
        evaluate_rnn_model(
            model_type="lstm",
            horizon=horizon,
            final_df=final_df,
            begin_dir=begin_dir,
            features=features,
            threshold=threshold
        )
    )

    #################################################
    # GRU
    #################################################
    threshold = get_threshold(
        thresholds,
        horizon,
        "GRU"
    )


    print('evalute gru eICU')
    results.append(
        evaluate_rnn_model(
            model_type="gru",
            horizon=horizon,
            final_df=final_df,
            begin_dir=begin_dir,
            features=features,
            threshold=threshold
        )
    )




results = pd.DataFrame(results)

print(results)

results.to_csv(
    "external_validation_eicu_results.csv",
    index=False
)

print("\nSaved results to external_validation_results.csv")



