# =========================================
# ModelEvaluator Class for Regression Models
# =========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class PlotEvaluation:
    def __init__(self, metrics_df: pd.DataFrame, output_dir: str = "evaluation_charts"):
        """
        Initialize the evaluator.
        
        Args:
            metrics_df (pd.DataFrame): DataFrame with columns ["Feature", "MAE", "RMSE", "R2"]
            output_dir (str): Directory where charts will be saved.
        """
        required_cols = {"Feature", "MAE", "RMSE", "R2"}
        if not required_cols.issubset(metrics_df.columns):
            raise ValueError(f"metrics_df must contain columns: {required_cols}")
        
        self.df = metrics_df.copy()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    # ---------- Plot 1: Bar chart (R²) ----------
    def plot_r2_bar(self):
        plt.figure(figsize=(10,5))
        plt.bar(self.df["Feature"], self.df["R2"], color="skyblue", edgecolor="black")
        plt.title("Model Performance (R² by Feature)")
        plt.ylabel("R² Score")
        plt.ylim(0, 1)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig(self.output_dir / "r2_by_feature.png", dpi=300)
        plt.close()

    # ---------- Plot 2: Combined MAE + R² ----------
    def plot_mae_r2_combined(self):
        fig, ax1 = plt.subplots(figsize=(10,6))
        ax1.bar(self.df["Feature"], self.df["MAE"], color="lightcoral", label="MAE")
        ax2 = ax1.twinx()
        ax2.plot(self.df["Feature"], self.df["R2"], color="blue", marker="o", label="R²")

        ax1.set_ylabel("MAE")
        ax2.set_ylabel("R² Score")
        ax1.set_title("MAE and R² by Feature")
        ax1.grid(axis="y", linestyle="--", alpha=0.6)
        fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
        plt.tight_layout()
        plt.savefig(self.output_dir / "mae_r2_combined.png", dpi=300)
        plt.close()

    # ---------- Plot 3: Parity plot ----------
    def plot_parity(self, y_true, y_pred, feature_name="example_feature"):
        plt.figure(figsize=(6,6))
        plt.scatter(y_true, y_pred, alpha=0.6)
        plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
        plt.xlabel("True Values")
        plt.ylabel("Predicted Values")
        plt.title(f"Parity Plot ({feature_name})")
        plt.grid(alpha=0.4)
        plt.tight_layout()
        plt.savefig(self.output_dir / f"parity_{feature_name}.png", dpi=300)
        plt.close()

    # ---------- Plot 4: Residual distribution ----------
    def plot_residuals(self, y_true, y_pred, feature_name="example_feature"):
        residuals = np.array(y_true) - np.array(y_pred)
        plt.figure(figsize=(7,5))
        plt.hist(residuals, bins=20, color="gray", edgecolor="black")
        plt.title(f"Residual Distribution ({feature_name})")
        plt.xlabel("Prediction Error")
        plt.ylabel("Frequency")
        plt.grid(axis="y", alpha=0.6)
        plt.tight_layout()
        plt.savefig(self.output_dir / f"residuals_{feature_name}.png", dpi=300)
        plt.close()

    # ---------- Plot 5: Heatmap ----------
    def plot_heatmap(self):
        plt.figure(figsize=(8,4))
        sns.heatmap(self.df.set_index("Feature")[["MAE", "RMSE", "R2"]], annot=True, cmap="coolwarm", fmt=".3f")
        plt.title("Model Evaluation Metrics by Feature")
        plt.tight_layout()
        plt.savefig(self.output_dir / "metrics_heatmap.png", dpi=300)
        plt.close()

    # ---------- Run all ----------
    def run_all(self):
        self.plot_r2_bar()
        self.plot_mae_r2_combined()
        self.plot_heatmap()
        print(f"✅ All summary charts saved to: {self.output_dir.resolve()}")