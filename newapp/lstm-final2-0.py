import pandas as pd
import vitalsImpute as vi
import bloodImpute_OLD as lb
import numpy as np
import InputData as ind

#------------24 hours-------------#

df_glucoCreat = pd.read_csv('/root/scripts/new_data/24hours/glucose_creatine_24_hours.csv', nrows=50000,sep='|')

#-------Blood lab results preparation-------#
df_examsBg = pd.read_csv('/root/scripts/new_data/24hours/blood_24_hours.csv', sep='|')
checking_columns_labs = ["glucose", "creatinine"]
time_interval=3600
labImputer = lb.labsImpute(df_examsBg,df_glucoCreat,checking_columns_labs,time_interval)
final_labs=labImputer.prepareLabs()
print(final_labs.info())
exit()
# #read the patient vitals nrows=50000,
# 1. Load your vitals data

df_vitals = pd.read_csv('/root/scripts/new_data/24hours/vitals_24_hours_final.csv', nrows=500000, sep='|')
checking_columns = ["spo2", "sbp","dbp","pulse_pressure", "heart_rate","resp_rate", "mbp"]
time_interval=15
# 2. Create the imputer object
imputer = vi.vitalsImpute(df_vitals.copy(),checking_columns,time_interval)  # pass a copy to keep df_vitals unchanged

# 3. Prepare and impute the data
clean_df = imputer.prepareVitals()

# 4. Evaluate imputation quality across all vitals
eval_results = imputer.evaluate_all_vitals(clean_df, [
    "spo2", "sbp", "dbp", "mbp", "heart_rate", "pulse_pressure", "resp_rate", "temperature"
])

# 5. Display results
print(eval_results)

# Plot both MAE and RMSE
imputer.plot_imputation_accuracy(eval_results, metrics=["MAE", "RMSE"])
# df_clean = df_vitals.copy()

# vitals = vi.vitalsImpute(df_clean)

# df_missing, sample_idx = vitals.simulate_missing(df_vitals.copy(), col="spo2", frac=0.1)
# imputed_df = vitals.imputeValues(df_missing.copy(), cols=["spo2"], interval=15)
# eval_results = vitals.evaluate_imputation(imputed_df, col="spo2", sample_idx=sample_idx)
# print("Evaluation results:", eval_results)

exit()

df_vitals.drop(['icu_intime','icu_outtime','gcs_time'], axis=1, inplace=True)
df_vitals.rename(columns={'charttime': 'vital_time'},inplace=True)
#read the patient glucose and creatine
df_glucoCreat = pd.read_csv('/root/scripts/new_data/24hours/glucose_creatine_24_hours.csv', nrows=50000,sep='|')
df_glucoCreat.drop(['hadm_id'], axis=1, inplace=True)

df_examsBg = pd.read_csv('/root/scripts/new_data/24hours/blood_24_hours.csv', nrows=50000,sep='|')
df_examsBg.drop(['admittime','rdwsd','hadm_id'], axis=1, inplace=True)

df_bloodGases = pd.read_csv('/root/scripts/new_data/24hours/gases_24_hours_final.csv', nrows=50000,sep='|')
#df_bloodGases.drop(['admittime','rdwsd','hadm_id'], axis=1, inplace=True)

#------------12 hours-------------#

#read the patient vitals nrows=50000,
# df_vitals = pd.read_csv('/root/scripts/new_data/12hours/vitals_12_hours.csv',     sep='|')

# df_vitals.drop(['icu_intime','icu_outtime','gcs_time'], axis=1, inplace=True)

# #read the patient glucose and creatine
# df_glucoCreat = pd.read_csv('/root/scripts/new_data/12hours/glucose_creatine_12_hours.csv', sep='|')
# df_glucoCreat.drop(['hadm_id'], axis=1, inplace=True)
# df_examsBg = pd.read_csv('/root/scripts/new_data/12hours/blood_12_hours.csv', sep='|')
# df_examsBg.drop(['admittime','rdwsd','hadm_id'], axis=1, inplace=True)


#read output-new.csv (This is the ouput data)
df_output = pd.read_csv('/root/scripts/new_data/patients_sepsis3.csv', sep='|' )

columns_to_fill = [
                "spo2",
                "sbp",
                "dbp",
                "pulse_pressure",
                "mbp",
                "heart_rate",
                "resp_rate",
                "temperature",
                "gcs",
                "gcs_calc"         
            ]

interpolate_cols=["temperature", "heart_rate", "resp_rate", "spo2","gcs","gcs_calc","pao2","paco2","fio2"]
xgboost_cols = ["mbp", "sbp", "dbp","spo2","temperature","gcs","pao2","paco2","fio2"]

columns_to_fill_2 = [
                "hematocrit",
                "hemoglobin",
                "mch",
                "mchc",
                "mcv",
                "wbc",
                "platelet",                               
                "rbc",
                "rdw",        
                "glucose", 
                "creatinine",
                  
            ]    

data = InputData.InputData(df_vitals,df_glucoCreat, df_examsBg,df_output,df_bloodGases,columns_to_fill,columns_to_fill_2,interpolate_cols,xgboost_cols)

vital_columns = ['heart_rate', 'sbp', 'dbp', 'spo2', 'resp_rate']
# Add relevant non-vital features like time or context
context_features = ['vital_time', 'subject_id', 'stay_id', 'group']
vitals = data.prepareVitals()

vitals["group"] = vitals.groupby(["subject_id", "stay_id"])['vital_time'].transform(
            lambda x: (x.diff() > pd.Timedelta(minutes=15)).cumsum()
        )

vitals = vitals.groupby(["subject_id", "stay_id", "group"])

for col in vital_columns:
    other_vitals = [c for c in vital_columns if c != col]
    feature_cols = other_vitals + context_features
    
    # Make sure all feature columns exist and are numeric
    feature_cols = [c for c in feature_cols if c in vitals.columns and pd.api.types.is_numeric_dtype(vitals[c])]
    
    df = data.xgb_impute_column(vitals, col, feature_cols)
    print(df.info())

exit()
def evaluate_imputation(df, col_to_test, feature_cols, frac=0.1, seed=42):
        # Step 1: Copy the dataframe
        df_copy = df.copy()

        # Step 2: Select non-missing values in target column
        valid_idx = df_copy[df_copy[col_to_test].notnull()].index
        test_idx = valid_idx.to_series().sample(frac=frac, random_state=seed)

        # Step 3: Store the true values
        true_values = df_copy.loc[test_idx, col_to_test].copy()

        # Step 4: Mask them (simulate missingness)
        df_copy.loc[test_idx, col_to_test] = np.nan

        # Step 5: Impute using XGBoost
        df_imputed = data.xgb_impute_column(df_copy, col_to_test, feature_cols)

        # Step 6: Evaluate predictions
        pred_values = df_imputed.loc[test_idx, col_to_test]

        mae = mean_absolute_error(true_values, pred_values)
        rmse = mean_squared_error(true_values, pred_values, squared=False)
        r2 = r2_score(true_values, pred_values)

        print(f"Evaluation for '{col_to_test}':")
        print(f"  MAE  = {mae:.4f}")
        print(f"  RMSE = {rmse:.4f}")
        print(f"  R²   = {r2:.4f}")


vital = 'spo2'
feature_cols = ['heart_rate', 'sbp', 'dbp', 'resp_rate', 'charttime', 'subject_id', 'group']
evaluate_imputation(df_vitals, vital, feature_cols)
exit()
#2. prepare the data for the algorithms
vitals = data.prepareVitals()
df_glucoCreat = data.prepareGlucose()
df_blood = data.prepareExamsBg()
df_gases = data.prepareGases()

#3. step merge all the dataframes in one big dataframe
merged=data.mergeDataframes(df_blood,df_glucoCreat,vitals,df_gases)
print(merged)


#4. Normalize merged df columns
normalized = data.normalizeColumns(merged)
# print(normalized.info())
filled_vitals = data.divideDataframe(normalized,'vitals')
filled_vitals_exams = data.divideDataframe(filled_vitals,'exams')
filled_all = data.divideDataframe(filled_vitals_exams,'gases')
print(filled_vitals_exams.info())


filled_vitals_exams.reset_index(inplace=True)
# print("last but not least")
# print(filled.info())

final_df = data.mergeSepsis3(filled_vitals_exams,df_output)
print(final_df.info())


del filled_vitals_exams
exit()
# print(final_df.info())
# --------------------------------------------------------
# 3. Step 1: Evaluate on Clean Data (Simulated Missingness)
# --------------------------------------------------------
medical_thresholds = {
                "gcs":           (3,15),
                "gcs_calc":      (3,15),
                "spo2":          (70,100), 
                "sbp":           (60,250),
                "pulse_pressure": (10,150),
                "mbp":           (40,180),
                "heart_rate":    (30,220),
                "resp_rate":     (5,60),
                "temperature":   (33,43)
    }

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
#print(filled[['stay_id','gender','charttime','spo2', 'sbp', 'pulse_pressure', 'mbp', 'heart_rate', 'resp_rate','temperature','gcs','gcs_calc','dbp']])
final_clean_df.to_csv("final_for_lstm_withsepsis.csv", sep="|", index=False)

exit()
#feature columns for lstm
features = ['gender','admission_age','hospstay_seq','los_hospital','hospstay_seq','los_icu','icustay_seq','spo2', 'sbp', 'dbp', 'pulse_pressure', 'heart_rate', 'resp_rate','temperature','gcs',"wbc","platelet","hematocrit","hemoglobin",
            "mbp","mch","mchc","mcv","rbc","rdw","glucose","creatinine"]
features_2 = ['spo2', 'sbp', 'dbp', 'pulse_pressure', 'heart_rate', 'resp_rate','temperature']
features_3 = ['gender','admission_age','hospstay_seq','los_hospital','hospstay_seq','los_icu','icustay_seq','spo2', 'sbp', 'dbp', 'pulse_pressure', 'heart_rate', 'resp_rate','temperature','gcs','hemoglobin']

# missing_rows = merged_data[merged_data['spo2'].isna()]
# print(missing_rows)


input_size= len(features)
hidden_size=64
num_layers=3
#create the LSTM model
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