import pandas as pd
import InputData
import vitalsImputeNew as vi
import glucoseImpute as gl 
import gasesImpute as ga 
import numpy as np
import InputData
import Evaluation as ev
import dask.dataframe as dd
from dask.distributed import Client
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

        break
    elif environment == "2":
        print("Linux platform chosen as environment")
        begin_dir =Path('/root/scripts/newapp/')
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

    # df_vitals = dd.read_csv(r"C:\phd-final\phd\new_data\24hours\vitals_24_hours_final.csv", sep='|', dtype=dtypes)
    ddf_vitals = dd.read_csv(
        #"/root/scripts/newapp/vitalsDemo.csv", #use for testing purposes
        "/root/scripts/new_data/24hours/vitals_24_hours_final.csv",
        dtype=dtypes,
        sep="|",
    )
    
    # 2. Create the imputer object
    imputer = vi.vitalsImputeNew(ddf_vitals, vitals_columns, time_interval)
    # 3. Prepare and impute the data
    imputer.prepareVitals()
    
    print('read filled parquets for vitals evaluation (no temperature)')    
    print ('read filled vitals from:',vitals_dir)
    #ddf_vitals_filled = dd.read_parquet('/root/scripts/newapp/secondrun/vitals_filled.parquet/')
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



    from PlotEvaluation import PlotEvaluation  # if you save it as evaluator.py
    evaluator = PlotEvaluation(df_results)
    evaluator.run_all()

    
#merged_dir='/root/scripts/newapp/secondrun/unfilled/all_merged.parquet/'
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

# #temperature_feature_cols = ["heart_rate", "resp_rate", "sbp", "dbp", "mbp", "pulse_pressure","spo2", "fio2", "glucose", "wbc", "creatinine",'hematocrit', 'hemoglobin', 'mch', 'mchc', 'mcv', 'wbc', 'platelet', 'rbc', 'rdw']
# temperature_feature_cols = ["heart_rate", "resp_rate", "sbp", "dbp", "mbp", "pulse_pressure","spo2"]
# # 2️⃣ Initialize and fit the imputer
# temperature_imputer = xg.xgBoostFill(
#     target_columns=["temperature"],
#     features=temperature_feature_cols,
#     short_gap_targets=["temperature"]
# )

temperature_folder = begin_dir/'secondrun/filled/temperature_filled.parquet/'

if os.listdir(temperature_folder) == []:   
    # Diagnostics
    df_vitals = dd.read_csv(r"C:\phd-final\phd\new_data\24hours\vitals_24_hours_final.csv", sep='|', dtype=dtypes)
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

   

#     print('start LightGBM for temperature')

#     # Fit on a clean, representative sample
#     sample_df = merged_ddf.dropna(subset=temperature_feature_cols).sample(frac=0.8, random_state=42).compute()
#     sample_df = temperature_imputer.short_gap_fill(sample_df, "temperature", limit=4)
#     temperature_imputer.transform(sample_df)

#     # Transform the full dataset
#     filled_ddf = temperature_imputer.transform(merged_ddf)
#     filled_ddf.to_parquet(temperature_folder)
#     print(f"Saved filled data to: {temperature_folder}")

# #read parquets with temperature filled and apply Evaluation
# print('begin temperature evaluation by reading the saved parquets')
# #ddf_vitals_filled = dd.read_parquet('/root/scripts/newapp/secondrun/filled/temperature_parquet/')
# ddf_vitals_filled = dd.read_parquet(temperature_folder)

# # 4️⃣ Evaluation
# df_sample_eval = ddf_vitals_filled.sample(frac=0.8,random_state=42).compute()  # pandas for evaluation

# evaluator = ev.Evaluation(
#     imputer=temperature_imputer,
#     data=df_sample_eval,
#     columns_to_fill=["temperature"],
#     mask_rate=0.2,
#     n_runs=3
# )

# results = []
# for col in ["temperature"]:
#     print(f"Evaluating {col}...") 
#     res = evaluator.evaluate_masking(df_sample_eval, col, mask_frac=0.2)
#     results.append(res)

# df_results = pd.DataFrame(results)
# print("\n📊 Temperature & SpO₂ Imputation Evaluation Results:")
# print(df_results)

# desc = df_sample_eval['temperature'].describe()
# print(desc)
# # temperature_evaluator = ev.Evaluation(
# #         temperature_imputer, merged_ddf, columns_to_fill=['temperature'], mask_rate=0.5, n_runs=3
# #     )
# # cleaned_ddf = InputData.clean_dtypes(merged_ddf)
# # df_sample = cleaned_ddf.sample(frac=0.4).compute() 
# # results = []
# # for col in ['temperature']:
# #     print(f"Evaluating {col}...") 
# #     res = temperature_imputer.evaluate_masking(df_sample, col, mask_frac=0.2)
# #     results.append(res)


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




#first read data and delete unecesairy columns
print ('begin sepsis prediction with all four methods')
merged_filled_all = dd.read_parquet(gases_dir)

cols_to_drop = ["fio2", "pao2", "paco2","pf_ratio","charttime_3","charttime_4","charttime_5"]
merged_filled_all = merged_filled_all.drop(columns=cols_to_drop, errors="ignore")
final_df = merged_filled_all.compute()
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

##LSTM##

l_col_val="label_sepsis_within_24h"

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

        trainer.prepare_data_from_splits(df_train, df_val, time_col=time_col)
        trainer.train()
        elapsed_sec = time.time() - start_time
        elapsed_min = elapsed_sec / 60.0

        val_loss, val_auc, val_acc, val_auprc, val_mcc, thr = trainer.evaluate(trainer.val_loader)

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

# cv_results = run_groupkfold_cv(
#     df=clean_df,
#     trainer_kwargs=dict(
#         features=features,
#         label_col="label_sepsis_within_24h",
#         seq_len=30,
#         hidden_size=32,
#         num_layers=2,
#         batch_size=32,
#         lr=1e-3,
#         epochs=4,
#         early_stop_patience=2,
#         early_stop_metric="auprc",
#     ),
#     time_col="charttime",
#     group_col="subject_id",  # or stay_id
#     n_splits=5,
# )


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




exit()

##########SO FAR SO GOOD######




#features = ['gender','admission_age','hospstay_seq','los_hospital','hospstay_seq','los_icu','icustay_seq','spo2', 'sbp', 'dbp', 'pulse_pressure', 'heart_rate', 'resp_rate','temperature','gcs',"wbc","platelet","hematocrit","hemoglobin",
           # "mbp","mch","mchc","mcv","rbc","rdw","glucose","creatinine"]
# features_2 = ['spo2', 'sbp', 'dbp', 'pulse_pressure', 'heart_rate', 'resp_rate','temperature']
# features_3 = ['gender','admission_age','hospstay_seq','los_hospital','hospstay_seq','los_icu','icustay_seq','spo2', 'sbp', 'dbp', 'pulse_pressure', 'heart_rate', 'resp_rate','temperature','gcs','hemoglobin']

# missing_rows = merged_data[merged_data['spo2'].isna()]
# print(missing_rows)



input_size= len(features)
hidden_size=64
num_layers=3
# create the LSTM model
lstm = LSTMClass(input_size,hidden_size,num_layers)

# Set sequence length and create sequences
sequence_length = 30
sequences = lstm.create_sequences(final_df, sequence_length,features)
tensors = lstm.createTensors(sequences)

X=tensors[0]
y=tensors[1]
# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=42)


batch_size = 32
train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = optim.Adam(lstm.parameters(), lr=0.001)

# Train the model
lstm.trainLSTM(train_loader, criterion, optimizer, epochs=10)

# Evaluate the model
lstm.evaluateLSTM(test_loader, criterion)

lstm.createPlotPredictions(test_loader)



# 2. Run XGBoost refinement
# read the parquet files from the interpolation
# print('start xgboost filling process for vitals')
# ddf = dd.read_parquet("filled/temperature_filled.parquet")

# checking_columns.append('temperature')
# features_columns = ['gender', 'hospstay_seq', 'icustay_seq', 'admission_age', 'los_hospital', 'los_icu', "spo2", "sbp","dbp","pulse_pressure", "heart_rate","resp_rate", "mbp","temperature"]

# FEATURE_MAP = {
#         "sbp": ["dbp", "mbp", "pulse_pressure", "heart_rate", "spo2"],
#         "dbp": ["sbp", "mbp", "pulse_pressure", "heart_rate", "spo2"],
#         "mbp": ["sbp", "dbp", "pulse_pressure", "heart_rate"],
#         "pulse_pressure": ["sbp", "dbp", "mbp"],

#         "heart_rate": ["resp_rate", "spo2", "temperature", "sbp", "dbp"],
#         "resp_rate": ["spo2", "heart_rate", "temperature"],
#         "spo2": ["resp_rate", "heart_rate", "temperature"],
#         "temperature": ["heart_rate", "resp_rate", "spo2"],

#         "gcs": ["spo2", "temperature", "heart_rate"],
#         }

# xgbImputer = xg.xgBoostFill(
#         target_columns=checking_columns,
#         features=features_columns,
#         feature_map=FEATURE_MAP,
#         random_state=42
# )
# cleaned_ddf = InputData.clean_dtypes(ddf)
# df_sample = cleaned_ddf.sample(frac=0.6).compute()  # small representative sample
# xgbImputer.fit(df_sample)
# meta = InputData.clean_dtypes(ddf._meta)
# ddf_filled = ddf.map_partitions(xgbImputer.transform, meta=meta)
# ddf_filled = ddf_filled.persist()
# ddf_filled.to_parquet("filled/vitals_xgb_filled.parquet", write_index=False)
# # # 3. Evaluate XGBoost
# # 7. Evaluate on a pandas sample using your evaluation class
# xgboost_evaluator = ev.Evaluation(
# imputer=xgbImputer,
# data = ddf_filled,
# columns_to_fill=checking_columns,
# mask_rate=0.3,
# n_runs=3
# )

# results = []
# for col in checking_columns:
#         res = xgboost_evaluator.evaluate(df_sample, col, mask_frac=0.3, n_runs=3)
#         results.append(res)

# results_df = pd.DataFrame(results)
# print(results_df)
# # xgboost_evaluator = ev.Evaluation(ddf_filled, checking_columns, mask_rate=0.3, n_runs=3)
# # xgboost_evaluator.models = {col: (model, [f for f in features_columns if f != col])
# #                 for col, model in xgbImputer.models.items()}

# # results = xgboost_evaluator.evaluate_xgboost_filling(frac=0.8, mask_rate=0.3, n_runs=3)
# # print(results)
# # Check missing values only in checkingColumns
# missing_summary_xgboost = xgboost_evaluator.missing_report(ddf_filled)
# # print("🧐 Missing values per vital column after XGBoost:")
# # print(missing_summary_xgboost[missing_summary_xgboost > 0])

# # xgb_evaluator.models = imputer.models  # reuse trained models
# # xgb_results = xgb_evaluator.evaluate_xgboost_filling(frac=0.2, mask_rate=0.3, n_runs=3)
# # print("XGBoost results:\n", xgb_results)

# Blood gases data
gases_columns = ['paco2', 'fio2', 'pao2']
gases_dir = 'filled/gases_filled.parquet'

if os.listdir(gases_dir) == []:

                df_bloodGases = dd.read_csv(r"C:\phd-final\phd\new_data\24hours\gases_24_hours_final.csv", dtype={"charttime": "object"}, sep='|')
                #df_bloodGases = dd.read_csv('/root/scripts/new_data/24hours/gases_24_hours_final.csv', dtype={"charttime": "object"}, sep='|')
                gases_columns = ['paco2', 'fio2', 'pao2']
                gases_imputer = ga.gasesImpute(df_bloodGases,gases_columns,24)
                df_gases=gases_imputer.prepareGases()

                df_gases.to_parquet("filled/gases_filled.parquet", write_index=False)

                evaluator = ev.Evaluation(imputer=gases_imputer, data=df_bloodGases,
                                columns_to_fill=gases_columns,
                                mask_rate=0.2, n_runs=3)

                results, summary = evaluator.evaluate_filling_performance(df_bloodGases, df_gases)
                del df_bloodGases

# Glucose and creatinine data

glucCreat_dir = 'filled/glucCreat_filled.parquet'
glucCreat_columns = ["creatinine","glucose"]

if os.listdir(glucCreat_dir) == []:

                print("read and impute glucose and creatine:")        
                #df_glucoCreat = dd.read_csv('/root/scripts/new_data/24hours/glucose_creatine_24_hours.csv',  sep='|')
                df_glucoCreat = dd.read_csv(r"C:\phd-final\phd\new_data\24hours\glucose_creatine_24_hours.csv", dtype={"charttime": "object"}, sep='|')

                glucCreat_columns = ["creatinine","glucose"]
                glucCreat_imputer = gl.glucoseImpute(df_glucoCreat,glucCreat_columns,3600)
                glucCreat_df = glucCreat_imputer.prepareGlucose()
                print("Glucose and creatine save in parquet and start evaluation:")
                print(glucCreat_df.info())
                glucCreat_df.to_parquet("filled/glucCreat_filled.parquet", index=False)

                glucCreat_evaluator = ev.Evaluation(imputer=glucCreat_imputer, data=df_glucoCreat,
                                columns_to_fill=glucCreat_columns,
                                mask_rate=0.4, n_runs=3)

                results, summary = glucCreat_evaluator.evaluate_filling_performance(df_glucoCreat, glucCreat_df)
                del df_glucoCreat

# Delete initial dataframes to gain memory

# del df_bloodGases
# del df_vitals

# print(clean_df.compute().info())


# merge vitals and blood
# 08/10 i have to create the merging
# df_vitals_blood = InputData.mergeDataframes(bloodResults, lab_columns, glucCreat_df, glucCreat_columns, clean_df,df_gases,gases_columns)
# Now we merge the previous created parquet files with dask


xgboost_dir = 'filled/vitals_xgb_filled.parquet'
features_columns = ['gender', 'hospstay_seq', 'icustay_seq', 'admission_age', 'los_hospital', 'los_icu', "spo2", "sbp","dbp","pulse_pressure", "heart_rate","resp_rate", "mbp","temperature"]
if os.listdir(blood_dir) == []:
#Fill the mbp,sbp,dbp with xgboost

        
        #columns_for_xgboost=['sbp','dbp','mbp','pulse_pressure']        
        columns_for_xgboost=['sbp','dbp','mbp','pulse_pressure']    
        FEATURE_MAP = {
                        "sbp": ["dbp", "mbp", "pulse_pressure", "heart_rate", "spo2"],
                        "dbp": ["sbp", "mbp", "pulse_pressure", "heart_rate", "spo2"],
                        "mbp": ["sbp", "dbp", "pulse_pressure", "heart_rate"],
                        "pulse_pressure": ["sbp", "dbp", "mbp"],

                        "heart_rate": ["resp_rate", "spo2", "temperature", "sbp", "dbp"],
                        "resp_rate": ["spo2", "heart_rate", "temperature"],
                        "spo2": ["resp_rate", "heart_rate", "pao2","paco2","fio2",'hematocrit', 'hemoglobin',"creatinine","glucose"],
                        "temperature": ["admission_age","los_icu","heart_rate", "resp_rate", "spo2", "sbp", "dbp", "mbp","wbc", "hematocrit", "glucose", "paco2","fio2", "pao2","wbc"],

                        "gcs": ["spo2", "temperature", "heart_rate"],
                        }
                
        xgbImputer = xg.xgBoostFill(
        target_columns=columns_for_xgboost,
        features=features_columns,
        feature_map=FEATURE_MAP,
        random_state=42
        )



        df_sample = cleaned_ddf.sample(frac=0.7).compute()  # small representative sample
        xgbImputer.fit(cleaned_ddf)
        meta = InputData.clean_dtypes(df_merged_data._meta)
        ddf_filled = df_merged_data.map_partitions(xgbImputer.transform, meta=meta)
        ddf_filled = ddf_filled.persist()
        ddf_filled.to_parquet("filled/vitals_xgb_filled.parquet", write_index=False)
        # # 3. Evaluate XGBoost
        # 7. Evaluate on a pandas sample using your evaluation class
        xgboost_evaluator = ev.Evaluation(
        imputer=xgbImputer,
        data = df_sample,
        columns_to_fill=columns_for_xgboost, 
        mask_rate=0.3, 
        n_runs=3
        )

        results = []
        for col in columns_for_xgboost:
                res = xgboost_evaluator.evaluate(df_sample, col, mask_frac=0.3, n_runs=3)
                results.append(res)

        results_df = pd.DataFrame(results)
        print(results_df)
# I need to calculate the temperature taking in consideration other parameters from the dataset

# Load data
ddf = dd.read_parquet("filled/vitals_xgb_filled.parquet")

# Define target + features
target_col = ['temperature']
feature_cols = [
    "heart_rate", "resp_rate", "spo2", "pao2", "paco2",
    "fio2", "hematocrit", "hemoglobin", "glucose", "admission_age","wbc","los_icu"
]

# Initialize imputer
lstm_imputer = LSTMImputer(
    target_col=target_col,
    feature_cols=feature_cols,
    epochs=50,  # you can tune this
    seq_len=12
)

# Fit + transform + save
lstm_imputer.fit(ddf)
ddf_filled = lstm_imputer.transform(ddf)
lstm_imputer.save(ddf_filled, "filled/temperature_filled.parquet")

# Evaluate lstm
cleaned_ddf = InputData.clean_dtypes(ddf_filled)
df_sample = cleaned_ddf.sample(frac=0.7).compute() 
lstm_evaluator = ev.Evaluation(
imputer=lstm_imputer,
data = df_sample,
columns_to_fill=target_col, 
mask_rate=0.3, 
n_runs=3
)

results = []
for col in target_col:
        res = lstm_evaluator.evaluate(df_sample, col, mask_frac=0.3, n_runs=3)
        results.append(res)

results_df = pd.DataFrame(results)
print(results_df)

exit()

# delete blood result object in order to free memory


# 4. Evaluate imputation quality across all vitals
# eval_results = imputer.evaluate_all_vitals(clean_df, [
#     "spo2", "sbp", "dbp", "mbp", "heart_rate", "pulse_pressure", "resp_rate", "temperature"
# ])

# # 5. Display results
# print(eval_results)

# Plot both MAE and RMSE
# imputer.plot_imputation_accuracy(eval_results, metrics=["MAE", "RMSE"])


print('final dataset  to be used in lstm etc')
print(df_final_dataset.info())
print(df_final_dataset.head())
df_final_dataset.to_csv("final_for_lstm_20250715.csv", sep="|", index=False)
print('final dataset  to be used in lstm etc')
exit()


# df_bloodGases.drop(['admittime','rdwsd','hadm_id'], axis=1, inplace=True)

# ------------12 hours-------------#

# read the patient vitals nrows=50000,
# df_vitals = pd.read_csv('/root/scripts/new_data/12hours/vitals_12_hours.csv',     sep='|')

# df_vitals.drop(['icu_intime','icu_outtime','gcs_time'], axis=1, inplace=True)

# #read the patient glucose and creatine
# df_glucoCreat = pd.read_csv('/root/scripts/new_data/12hours/glucose_creatine_12_hours.csv', sep='|')
# df_glucoCreat.drop(['hadm_id'], axis=1, inplace=True)
# df_examsBg = pd.read_csv('/root/scripts/new_data/12hours/blood_12_hours.csv', sep='|')
# df_examsBg.drop(['admittime','rdwsd','hadm_id'], axis=1, inplace=True)


# read output-new.csv (This is the ouput data)
df_output = pd.read_csv('/root/scripts/new_data/patients_sepsis3.csv', sep='|' )


clean_df = normalized.dropna(subset=columns_to_fill + columns_to_fill_2).copy()

####### Evaluation section #######

eval_results = evaluation.evaluation (clean_df, columns_to_fill,columns_to_fill_2)

final_clean_df = eval_results.apply_medical_thresholds(clean_df, medical_thresholds)

final_clean_df2=final_clean_df.dropna()
# print(final_df.info())

# vital_interpolation_results = eval_results.simulate_and_evaluate_interpolation(final_clean_df2, columns_to_fill, 0.2, 5)
# print("\n🩺 Vitals - Interpolation Performance")
# print(vital_interpolation_results.sort_values(by="MAE"))

n_groups = final_clean_df.groupby(["subject_id", "stay_id", "group"]).ngroups
print(f"Number of groups: {n_groups}")
# print(filled[['stay_id','gender','charttime','spo2', 'sbp', 'pulse_pressure', 'mbp', 'heart_rate', 'resp_rate','temperature','gcs','gcs_calc','dbp']])
final_clean_df.to_csv("final_for_lstm_withsepsis.csv", sep="|", index=False)

exit()

