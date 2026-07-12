# sepsis_models/config.py

FEATURES = [
    "gender", "admission_age",
    "spo2", "sbp", "dbp", "pulse_pressure",
    "heart_rate", "resp_rate", "temperature",
    "mbp", "wbc", "platelet", "hematocrit", "hemoglobin",
    "mch", "mchc", "mcv", "rbc", "rdw",
    "glucose", "creatinine",
]

LABEL_COL = "label_sepsis_within_6h"   
SEQ_LEN = 30
TIME_COL = "charttime"            
