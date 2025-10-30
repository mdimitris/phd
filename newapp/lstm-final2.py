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
import helpers as help
import bloodImpute as bloodImp


blood_columns = ['hematocrit', 'hemoglobin', 'mch', 'mchc', 'mcv', 'wbc', 'platelet', 'rbc', 'rdw']
gases_columns = ['paco2', 'fio2', 'pao2']
glucCreat_columns = ["creatinine","glucose"]
# ------------24 hours-------------#
# vitals_dir="filled/vitals_filled.parquet/"
vitals_dir="/root/scripts/newapp/secondrun/vitals_filled.parquet/"

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

        df_vitals = dd.read_csv(r"C:\phd-final\phd\new_data\24hours\vitals_24_hours_final.csv", sep='|', dtype=dtypes)

        time_interval=15 
        # 2. Create the imputer object 
        imputer = vi.vitalsImputeNew(df_vitals,checking_columns,time_interval) 
        # 3. Prepare and impute the data
        clean_df = imputer.prepareVitals()

        # Step 2: Reload from saved CSVs (so evaluation runs on same data you persisted)
        df_filled = dd.read_parquet("filled/vitals_filled.parquet")

        # Step 3: Run evaluation
        vitals_evaluator = ev.evaluation(df_filled,imputer.get_checkingColumns(), mask_rate=0.5,n_runs=3)
        evaluation_results = vitals_evaluator.simulate_and_evaluate_dask_filling()
        print(evaluation_results)

merged_dir='/root/scripts/newapp/secondrun/unfilled/all_merged.parquet/'
# merged_dir='unfilled/all_merged.parquet/'
if os.listdir(merged_dir) == []:    

        #ddf_bloodResults = dd.read_csv(r"C:\phd-final\phd\new_data\24hours\blood_24_hours.csv", sep='|')
        ddf_bloodResults = dd.read_csv("/root/scripts/new_data/24hours/blood_24_hours.csv", sep='|')
        help.prepareDataset(ddf_bloodResults,blood_columns,["rdwsd","admittime"],'blood')

        ddf_gases = dd.read_csv("/root/scripts/new_data/24hours/gases_24_hours_final.csv", dtype={"charttime": "object"}, sep='|')
        help.prepareDataset(ddf_gases,gases_columns,["hadm_id","sofa_time"],'gases')    

        ddf_glucoCreat = dd.read_csv("/root/scripts/new_data/24hours/glucose_creatine_24_hours.csv", dtype={"charttime": "object"}, sep='|')
        help.prepareDataset(ddf_glucoCreat,glucCreat_columns,["hadm_id"],'glucCreat')

        ddf_vitals = dd.read_parquet("/root/scripts/newapp/secondrun/vitals_filled.parquet/")
        print(ddf_vitals.info())
        df_merged_data = InputData.mergeDataframes()

print('read merged parquet (still not completely filled)')
merged_ddf = dd.read_parquet(merged_dir)
print('dtype is:',merged_ddf.dtypes["stay_id"])


# Diagnostics
#df_vitals = dd.read_csv(r"C:\phd-final\phd\new_data\24hours\vitals_24_hours_final.csv", sep='|', dtype=dtypes)
df_vitals = dd.read_csv("/root/scripts/new_data/24hours/vitals_24_hours_final.csv",sep='|', dtype=dtypes)
print ("Patients before starting procedures:",len(pd.unique(df_vitals['subject_id'])))
print ("Rows before starting procedures:",df_vitals.shape[0].compute())
print("Unique patients before merging but after vitalsimputeNew:", df_vitals['subject_id'].nunique().compute())
print("Unique patients after merging all the sources:", merged_ddf['subject_id'].nunique().compute())
print('Rows after merging:',merged_ddf.shape[0].compute())

rows_with_missing=merged_ddf.isnull().any(axis=1).sum().compute()
print(f"Number of rows with empty cells: {rows_with_missing}")
print("Calculate missing values per column")
help.calculateMissing(merged_ddf)

print('start LightGBM for temperature')

feature_cols = ["heart_rate", "resp_rate", "sbp", "dbp", "mbp", "pulse_pressure",
                "spo2", "fio2", "glucose", "wbc", "creatinine"]


# 1️⃣ Sample small fraction for training
sample_df = merged_ddf.sample(frac=0.8).compute()  

# 2️⃣ Initialize and fit the imputer
temperature_imputer = xg.xgBoostFill(
    target_columns=["temperature", "spo2"],
    features=feature_cols,
    short_gap_targets=["temperature"]
)

temperature_imputer.fit(sample_df)  # ⚡ fit on pandas

# 3️⃣ Apply to full Dask DataFrame
filled_ddf = temperature_imputer.transform(merged_ddf)
filled_ddf.to_parquet("/root/scripts/newapp/secondrun/filled/temp_parquet/")

# 4️⃣ Evaluation
df_sample_eval = filled_ddf.sample(frac=0.8).compute()  # pandas for evaluation

evaluator = ev.Evaluation(
    imputer=temperature_imputer,
    data=df_sample_eval,
    columns_to_fill=["temperature","spo2"],
    mask_rate=0.2,
    n_runs=3
)

results = []
for col in ["temperature","spo2"]:
    print(f"Evaluating {col}...") 
    res = evaluator.evaluate_masking(df_sample_eval, col, mask_frac=0.2)
    results.append(res)

df_results = pd.DataFrame(results)
print("\n📊 Temperature & SpO₂ Imputation Evaluation Results:")
print(df_results)


# temperature_evaluator = ev.Evaluation(
#         temperature_imputer, merged_ddf, columns_to_fill=['temperature'], mask_rate=0.5, n_runs=3
#     )
# cleaned_ddf = InputData.clean_dtypes(merged_ddf)
# df_sample = cleaned_ddf.sample(frac=0.4).compute() 
# results = []
# for col in ['temperature']:
#     print(f"Evaluating {col}...") 
#     res = temperature_imputer.evaluate_masking(df_sample, col, mask_frac=0.2)
#     results.append(res)
exit()
df_results = pd.DataFrame(results)
print("\n📊 Blood Imputation Evaluation Results:")
print(df_results)

exit()
blood_imputer = bloodImp.bloodImpute(
    blood_ddf=merged_ddf,
    blood_columns=blood_columns,
    sample_size=250_000,  # for MICE training sample
    output_folder="/root/scripts/newapp/filled/all_merged/blood.parquet/",  # folder
    n_output_files=128  # save in 128 Parquets
)

blood_imputer.run()


# -----------------------------
# 4. Load a sample for evaluation
# -----------------------------
# Pick one or a few Parquet batches for evaluation
merged_filled_blood = dd.read_parquet("/root/scripts/newapp/filled/all_merged/blood.parquet/")
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


exit()
# df_vitals = dd.read_csv(r"C:\phd-final\phd\new_data\24hours\vitals_24_hours_final.csv", sep='|', dtype=dtypes)
# print ("Patients before starting procedures:",len(pd.unique(df_vitals['subject_id'])))
# print("Unique patients before merging but after vitalsimputeNew:", df_vitals['subject_id'].nunique().compute())
# print("Unique patients after merging all the sources:", merged_ddf['subject_id'].nunique().compute())
# print('Rows after merging:',merged_ddf.shape[0].compute())
# rows_with_missing=merged_ddf.isnull().any(axis=1).sum().compute()

# print(f"Number of rows with empty cells: {rows_with_missing}")

# Fill blood gases
gases_columns = ['paco2', 'fio2', 'pao2']
gases_imputer = ga.gasesImpute(merged_ddf,gases_columns,24)
merged_gasesFilled_ddf=gases_imputer.imputeGases()

merged_gasesFilled_ddf.to_parquet("filled/merged_gases.parquet", write_index=False)


df_sample = merged_gasesFilled_ddf.sample(frac=0.3).compute()  # small representative sample

meta = InputData.clean_dtypes(df_merged_data._meta)

# # 3. Evaluate XGBoost
# 7. Evaluate on a pandas sample using your evaluation class
xgboost_evaluator = ev.Evaluation(
imputer=gases_imputer,
data = df_sample,
columns_to_fill=gases_columns, 
mask_rate=0.3, 
n_runs=3
)

results, summary = evaluator.evaluate_filling_performance(merged_ddf, merged_gasesFilled_ddf)
print(results,summary)
exit()

# df_filled = dd.read_parquet("filled/vitals_filled.parquet")

# imputer = vi.vitalsImputeNew(df_filled,checking_columns,15)

# vitals_evaluator = ev.Evaluation(df_filled,imputer.get_checkingColumns(), mask_rate=0.5,n_runs=3)

# evaluation_results = vitals_evaluator.simulate_and_evaluate_dask_filling()
# print('evaluation_results for ffill bfill:')
# print(evaluation_results)

# def fill_temperature(g, col='temperature', edge_limit=3):

#         # Forward fill
#         g[col] = g[col].ffill(limit=edge_limit)
#         # Backward fill
#         g[col] = g[col].bfill(limit=edge_limit)
#         return g

# reRun temperature filling because of sparcity
# df_temp = pd.read_parquet("filled/vitals_filled.parquet")
# df_temp = df_temp.sort_values(['stay_id', 'charttime'])
# print('start refilling temperature')
# df_temp = df_temp.groupby('stay_id').apply(fill_temperature)

# df_temp.to_parquet("filled/temperature_filled.parquet", index=False)


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

# -------Blood lab results preparation-------#
blood_dir = 'filled/blood.parquet'
blood_columns = ['hematocrit', 'hemoglobin', 'mch', 'mchc', 'mcv', 'wbc', 'platelet', 'rbc', 'rdw']

if os.listdir(blood_dir) == []:
        print("read and prepare blood tests..")
        df_bloodResults = dd.read_csv(r"C:\phd-final\phd\new_data\24hours\blood_24_hours.csv", sep='|')
        #df_bloodResults = dd.read_csv('/root/scripts/new_data/24hours/blood_24_hours.csv', sep='|')

        time_interval = 3600
        glucCreat_df=[]
        print('initial labs info before normalization:')
        print(df_bloodResults.info())
        # Initialize imputer
        labResult = lb.bloodImpute(
                blood=df_bloodResults,
                glucoseCreatinine=glucCreat_df,
                blood_columns=blood_columns,
                glucCreat_columns=glucCreat_columns,
                interval=time_interval
        )

        #Delete initial blood dataframe to gain memory
        del df_bloodResults
        #labResult = lb.bloodImpute(df_bloodResults,glucCreat_df,blood_columns,glucCreat_columns,time_interval)
        blood_df = labResult.prepareblood().compute()  # Convert from Dask to Pandas for imputation
        print(blood_df.head(100))

        bloodResults = labResult.prepareblood()
        print(bloodResults.head(200))

        df_sample = bloodResults.sample(frac=0.4).compute() 

        columns_tofill = blood_columns+glucCreat_columns+gases_columns
        
        # Initialize evaluator
        # Initialize imputer (for evaluation)
        blood_imputer = lb.bloodImpute(
        blood=blood_df,
        glucoseCreatinine=None,
        blood_columns=blood_columns,
        glucCreat_columns=glucCreat_columns,
        interval=2
        )

        # Initialize evaluator
        blood_evaluator = ev.Evaluation(
        imputer=blood_imputer,
        data=blood_df,
        columns_to_fill=blood_columns,
        mask_rate=0.3,
        n_runs=3
        )

        # Run evaluation
        results = []
        for col in blood_columns:
                res = blood_evaluator.evaluate_masking(blood_df, col, mask_frac=0.3)
                results.append(res)

        results_bdf = pd.DataFrame(results)
        ('print results:')
        print(results_bdf)

exit()

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

# Create another object with df_vitals_blood and fill blood features
labImputer = lb.labsImpute(df_vitals_blood,glucCreat_df,lab_columns,gl_columns,time_interval)
df_final_dataset = labImputer.populateLabResults(gases_columns)


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
# feature columns for lstm
features = ['gender','admission_age','hospstay_seq','los_hospital','hospstay_seq','los_icu','icustay_seq','spo2', 'sbp', 'dbp', 'pulse_pressure', 'heart_rate', 'resp_rate','temperature','gcs',"wbc","platelet","hematocrit","hemoglobin",
            "mbp","mch","mchc","mcv","rbc","rdw","glucose","creatinine"]
features_2 = ['spo2', 'sbp', 'dbp', 'pulse_pressure', 'heart_rate', 'resp_rate','temperature']
features_3 = ['gender','admission_age','hospstay_seq','los_hospital','hospstay_seq','los_icu','icustay_seq','spo2', 'sbp', 'dbp', 'pulse_pressure', 'heart_rate', 'resp_rate','temperature','gcs','hemoglobin']

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

exit()
