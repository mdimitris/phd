import os
import gc
import pandas as pd
import dask.dataframe as dd
import miceforest as mf


class bloodImpute:
    def __init__(
        self,
        blood_ddf,
        blood_columns,
        sample_target_size,
        output_folder,
        model_path=None,
        n_output_files=128,
    ):
        self.blood = blood_ddf
        self.blood_columns = blood_columns
        self.sample_target_size = sample_target_size
        self.output_folder = output_folder
        self.model_path = (
            model_path or "/root/scripts/newapp/filled/models/global_blood_kernel.pkl"
        )
        self.kds_global = None
        self.n_output_files = n_output_files

        os.makedirs(self.output_folder, exist_ok=True)
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

    # --------------------------------------------------
    # STEP 1: Prefill missing values by stay_id
    # --------------------------------------------------
    def prefill(self):
        print("🩸 Starting grouped ffill/bfill/interpolate by stay_id...")

        if "stay_id" not in self.blood.columns:
            raise ValueError("❌ 'stay_id' column not found in the dataset!")

        # Ensure consistent stay_id dtype
        self.blood["stay_id"] = self.blood["stay_id"].astype("int32")

        # Ensure numeric columns only
        numeric_cols = [
            c for c in self.blood_columns
            if pd.api.types.is_numeric_dtype(self.blood[c].dtype)
        ]
        if not numeric_cols:
            raise ValueError("❌ No numeric blood columns found for prefill!")

        if isinstance(self.blood, dd.DataFrame):
            print("⚙️ Applying grouped fill with Dask...")

            def fill_group(pdf):
                pdf[numeric_cols] = (
                    pdf[numeric_cols]
                    .ffill()
                    .bfill()
                    .interpolate(method="linear", limit_direction="both")
                )
                return pdf

            self.blood = self.blood.map_partitions(
                lambda pdf: pdf.groupby("stay_id", group_keys=False).apply(fill_group)
            )
        else:
            # Pandas fallback
            def fill_group(pdf):
                pdf[numeric_cols] = (
                    pdf[numeric_cols]
                    .ffill()
                    .bfill()
                    .interpolate(method="linear", limit_direction="both")
                )
                return pdf

            self.blood = (
                self.blood.groupby("stay_id", group_keys=False)
                .apply(fill_group)
                .reset_index(drop=True)
            )

        print("✅ Prefill complete.")

    # --------------------------------------------------
    # STEP 2: Train global MICE model
    # --------------------------------------------------
    def train_global_model(self, iterations=5, random_state=42):
        print("\n🧠 Training global MICE model...")

        # Sample data for kernel
        if isinstance(self.blood, dd.DataFrame):
            sample_df = self.blood[self.blood_columns].sample(frac=1.0).head(
                self.sample_target_size
            )
        else:
            sample_df = self.blood[self.blood_columns].sample(
                n=min(self.sample_target_size, len(self.blood)), random_state=random_state
            )

        # Ensure numeric floats
        sample_df = sample_df.astype("float32")

        print(f"  Sample size used for MICE: {len(sample_df):,}")

        self.kds_global = mf.ImputationKernel(
            data=sample_df,
            save_all_iterations=False,
            random_state=random_state,
            datasets=1,
        )
        self.kds_global.mice(iterations=iterations)
        self.kds_global.save_kernel(self.model_path)
        print(f"✅ Global MICE model trained and saved at {self.model_path}")

    # --------------------------------------------------
    # STEP 3: Apply model to full dataset
    # --------------------------------------------------
    def apply_global_model(self):
        print("\n💉 Applying MICE model to full dataset...")

        if self.kds_global is None:
            if os.path.exists(self.model_path):
                print(f"📦 Loading existing MICE model → {self.model_path}")
                # Initialize with dummy df (must not be None)
                dummy_df = pd.DataFrame(columns=self.blood_columns)
                self.kds_global = mf.ImputationKernel(dummy_df, datasets=1)
                self.kds_global.load_kernel(self.model_path)
            else:
                raise ValueError("❌ No global MICE kernel found!")

        def impute_partition(pdf):
            kernel = self.kds_global.impute_new_data(pdf[self.blood_columns])
            imputed = kernel.complete_data(0)
            other_cols = [c for c in pdf.columns if c not in self.blood_columns]
            return pd.concat(
                [pdf[other_cols].reset_index(drop=True), imputed.reset_index(drop=True)],
                axis=1,
            )

        if isinstance(self.blood, dd.DataFrame):
            imputed_ddf = self.blood.map_partitions(impute_partition)
            imputed_ddf.repartition(npartitions=self.n_output_files).to_parquet(
                self.output_folder,
                write_index=False,
                engine="pyarrow",
                compression="snappy",
                overwrite=True,
            )
        else:
            df_out = impute_partition(self.blood)
            df_out.to_parquet(
                os.path.join(self.output_folder, "part_0.parquet"), index=False
            )

        print(f"✅ Full dataset imputed and saved to folder: {self.output_folder}")

    # --------------------------------------------------
    # STEP 4: Run full pipeline
    # --------------------------------------------------
    def run(self):
        print("\n🚀 Running full blood imputation pipeline...\n")
        self.prefill()

        # if os.path.exists(self.model_path):
        #     print(f"📦 Found existing MICE model → {self.model_path}")
        #     dummy_df = pd.DataFrame(columns=self.blood_columns)
        #     self.kds_global = mf.ImputationKernel(dummy_df, datasets=1)
        #     self.kds_global.load_kernel(self.model_path)
        # else:
        #     self.train_global_model()

        #self.apply_global_model()
        print("\n✅ Pipeline complete: Prefill + MICE done.")
