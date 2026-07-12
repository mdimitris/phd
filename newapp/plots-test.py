import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text

data = [
    # Model, Horizon, AUPRC, AUPRC_std, MCC, MCC_std, ACC

    ("LSTM", 6, 0.381, 0.030, 0.389, 0.016, 0.813),
    ("LSTM", 8, 0.463, 0.035, 0.428, 0.033, 0.800),
    ("LSTM", 12, 0.581, 0.037, 0.485, 0.019, 0.806),
    ("LSTM", 24, 0.659, 0.039, 0.527, 0.041, 0.820),

    ("GRU", 6, 0.410, 0.030, 0.415, 0.037, 0.797),
    ("GRU", 8, 0.489, 0.023, 0.457, 0.028, 0.804),
    ("GRU", 12, 0.603, 0.039, 0.506, 0.037, 0.812),
    ("GRU", 24, 0.679, 0.041, 0.555, 0.035, 0.829),

    ("XGBoost", 6, 0.486, 0.021, 0.456, 0.015, 0.818),
    ("XGBoost", 8, 0.522, 0.017, 0.476, 0.013, 0.819),
    ("XGBoost", 12, 0.587, 0.012, 0.506, 0.011, 0.816),
    ("XGBoost", 24, 0.700, 0.016, 0.550, 0.016, 0.824),

    ("LightGBM", 6, 0.438, 0.029, 0.383, 0.037, 0.891),
    ("LightGBM", 8, 0.490, 0.021, 0.459, 0.011, 0.876),
    ("LightGBM", 12, 0.569, 0.017, 0.495, 0.007, 0.854),
    ("LightGBM", 24, 0.684, 0.017, 0.542, 0.013, 0.884),
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

#AUPRC vs MCC

plt.figure(figsize=(8, 6))

markers = {
    "LSTM": "o",
    "GRU": "s",
    "XGBoost": "D",
    "LightGBM": "^"
}

texts = []

for model in df["Model"].unique():
    subset = df[df["Model"] == model]

    plt.scatter(
        subset["MCC"],
        subset["AUPRC"],
        marker=markers[model],
        s=80,
        label=model
    )

    for _, row in subset.iterrows():
        texts.append(
            plt.text(
                row["MCC"],
                row["AUPRC"],
                f'{row["Horizon"]}h',
                fontsize=9
            )
        )

adjust_text(texts, arrowprops=dict(arrowstyle="-", color="gray", lw=0.5))

plt.xlabel("Matthews Correlation Coefficient (MCC)")
plt.ylabel("Area Under the Precision-Recall Curve (AUPRC)")
plt.grid(alpha=0.3)
plt.legend(title="Model")
plt.tight_layout()
plt.show()