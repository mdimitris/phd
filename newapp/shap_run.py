import os
import joblib
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================
# Configuration
# ==========================
MODEL_PATH = "models/xgb_24h_final.pkl"
VAL_DATA_PATH = "data/val_df_24h.parquet"
OUTPUT_DIR = "figures"
LABEL_COL = "label_sepsis_within_24h"

FEATURES = [
    # "gender", "admission_age", "hours_since_icu_intime",
    # "icustay_seq", "hospstay_seq",
    "spo2", "sbp", "dbp", "pulse_pressure",
    "heart_rate", "resp_rate", "temperature",
    "mbp", "wbc", "platelet", "hematocrit", "hemoglobin",
    "mch", "mchc", "mcv", "rbc", "rdw",
    "glucose", "creatinine"
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================
# Load model and data
# ==========================
print("Loading model and validation data...")

model = joblib.load(MODEL_PATH)
val_df = pd.read_parquet(VAL_DATA_PATH)

X_val = val_df[FEATURES]

# ==========================
# Compute SHAP values 
# ==========================
print("Computing SHAP values...")

model = joblib.load("models/xgb_24h_final.pkl")
print("Loaded model type:", type(model))

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_val)

# ==========================
# 1️⃣ Global Importance (Bar Plot)
# ==========================
plt.figure(figsize=(8,6))
shap.summary_plot(
    shap_values,
    X_val,
    plot_type="bar",
    max_display=25,
    show=False
)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/shap_bar_plot_24h.pdf", dpi=300)
plt.close()

# ==========================
# 2️⃣ Detailed SHAP Summary Plot
# ==========================
plt.figure(figsize=(8,6))
shap.summary_plot(
    shap_values,
    X_val,
    max_display=25,
    show=False
)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/shap_summary_plot_24h.pdf", dpi=300)
plt.close()

# ==========================
# 3️⃣ SHAP Feature Importance Table
# ==========================
mean_abs_shap = np.abs(shap_values).mean(axis=0)

importance_df = pd.DataFrame({
    "Feature": FEATURES,
    "MeanAbsSHAP": mean_abs_shap
}).sort_values(by="MeanAbsSHAP", ascending=False)

importance_df.to_csv(f"{OUTPUT_DIR}/shap_importance_24h.csv", index=False)

print("SHAP analysis completed successfully.")
