import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = [
    # Model, Horizon, AUPRC, AUPRC_std, MCC, MCC_std, ACC
    ("LSTM", 6, 0.365, 0.030, 0.372, 0.016, 0.798),
    ("LSTM", 8, 0.440, 0.035, 0.410, 0.033, 0.810),
    ("LSTM", 12, 0.580, 0.037, 0.490, 0.019, 0.810),
    ("LSTM", 24, 0.680, 0.039, 0.540, 0.041, 0.830),

    ("GRU", 6, 0.416, 0.030, 0.404, 0.037, 0.805),
    ("GRU", 8, 0.497, 0.023, 0.456, 0.028, 0.795),
    ("GRU", 12, 0.613, 0.039, 0.511, 0.037, 0.808),
    ("GRU", 24, 0.685, 0.041, 0.540, 0.035, 0.827),

    ("XGBoost", 6, 0.486, 0.021, 0.457, 0.015, 0.820),
    ("XGBoost", 8, 0.519, 0.017, 0.476, 0.013, 0.821),
    ("XGBoost", 12, 0.590, 0.012, 0.508, 0.011, 0.820),
    ("XGBoost", 24, 0.699, 0.016, 0.550, 0.016, 0.825),

    ("LightGBM", 6, 0.441, 0.029, 0.378, 0.037, 0.892),
    ("LightGBM", 8, 0.490, 0.021, 0.459, 0.011, 0.876),
    ("LightGBM", 12, 0.566, 0.017, 0.494, 0.007, 0.854),
    ("LightGBM", 24, 0.686, 0.017, 0.542, 0.013, 0.884),
]

df = pd.DataFrame(data, columns=[
    "Model", "Horizon", "AUPRC", "AUPRC_std", "MCC", "MCC_std", "Accuracy"
])



plt.figure()
for model in df["Model"].unique():
    subset = df[df["Model"] == model]
    plt.plot(subset["Horizon"], subset["AUPRC"], marker="o", label=model)

plt.xlabel("Prediction Horizon (hours)")
plt.ylabel("AUPRC")
plt.legend()
#plt.title("AUPRC trend in prediction horizons")
plt.show()

# Pivot data for heatmaps
auprc_data = df.pivot(index="Model", columns="Horizon", values="AUPRC")
mcc_data   = df.pivot(index="Model", columns="Horizon", values="MCC")

# Sort horizons
auprc_data = auprc_data.sort_index(axis=1)
mcc_data   = mcc_data.sort_index(axis=1)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Heatmap for AUPRC
sns.heatmap(auprc_data, annot=True, fmt=".3f", cmap="viridis", linewidths=0.6, ax=axes[0])
#axes[0].set_title("AUPRC across Horizons")
axes[0].set_xlabel("Horizon (hours)")
axes[0].set_ylabel("Model")

# Heatmap for MCC
sns.heatmap(mcc_data, annot=True, fmt=".3f", cmap="magma", linewidths=0.6, ax=axes[1])
#axes[1].set_title("MCC across Horizons")
axes[1].set_xlabel("Horizon (hours)")
axes[1].set_ylabel("")

#plt.suptitle("Figure 2: Performance Heatmaps across Horizons", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

plt.figure()
for model in df["Model"].unique():
    subset = df[df["Model"] == model]
    plt.scatter(subset["Accuracy"], subset["AUPRC"], label=model)

plt.xlabel("Accuracy")
plt.ylabel("AUPRC")
plt.legend()
#plt.title("Accuracy vs AUPRC trade-off")
plt.show()