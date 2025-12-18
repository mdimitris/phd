import os
import gc
import pandas as pd
import dask.dataframe as dd
import miceforest as mf
#from miceforest import builtin_mean_match_schemes as schemes

class bloodImpute:
    def __init__(
        self,
        blood_ddf,
        blood_columns,
        sample_size,
        output_folder,
        model_path=None,
        n_output_files=64,
    ):
        self.blood = blood_ddf
        self.blood_columns = blood_columns
        self.sample_size = sample_size
        self.output_folder = output_folder
        self.model_path = model_path or "models/global_blood_kernel.pkl"
        self.kds_global = None
        self.n_output_files = n_output_files

        os.makedirs(self.output_folder, exist_ok=True)
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

    # --------------------------------------------------
    # STEP 1: Prefill missing values by stay_id
    # --------------------------------------------------
    def prefill(self):
        print("🩸 Prefill missing values by stay_id...")

        if "stay_id" not in self.blood.columns:
            raise ValueError("'stay_id' column not found!")

        self.blood["stay_id"] = self.blood["stay_id"].astype("int32")

        numeric_cols = [
            c for c in self.blood_columns
            if pd.api.types.is_numeric_dtype(self.blood[c].dtype)
        ]
        if not numeric_cols:
            raise ValueError("No numeric blood columns found for prefill!")

        def fill_group(pdf):
            pdf[numeric_cols] = (
                pdf[numeric_cols]
                .ffill()
                .bfill()
                .interpolate(method="linear", limit_direction="both")
            )
            return pdf

        if isinstance(self.blood, dd.DataFrame):
            self.blood = self.blood.map_partitions(
                lambda pdf: pdf.groupby("stay_id", group_keys=False).apply(fill_group)
            )
        else:
            self.blood = self.blood.groupby("stay_id", group_keys=False).apply(fill_group).reset_index(drop=True)

        print("✅ Prefill complete.")

    # --------------------------------------------------
    # STEP 2: Train global MICE model
    # --------------------------------------------------
    def train_global_model(self, iterations=4):
        print("\n🧠 Training global MICE model...")
        
        # print(mf.__version__)
        # print(mf.ImputationKernel.__doc__)
        # print(mf.__file__)

        # Only use rows with missing values
        if isinstance(self.blood, dd.DataFrame):
            sample_df = self.blood[self.blood_columns].sample(frac=1.0).head(self.sample_size)
        else:
            sample_df = self.blood[self.blood_columns].sample(
                n=min(self.sample_size, len(self.blood)), random_state=42
            )

        sample_df = sample_df.astype("float32")

        if sample_df.isnull().sum().sum() == 0:
            print("⚠️ No missing values found — skipping MICE training.")
            return False
        #mean_match = schemes.mean_match_default
        self.kds_global = mf.ImputationKernel(
            data=sample_df,
            save_all_iterations=False,
            random_state=42,
            data_subset=0.7,
            #mean_match_scheme=mean_match,
            # mean_match_candidates=5,
            datasets=1
        )
        self.kds_global.mice(iterations=iterations,n_jobs=os.cpu_count())
        self.kds_global.save_kernel(self.model_path)
        print(f"✅ Global MICE model trained and saved at {self.model_path}")
        return True

    # --------------------------------------------------
    # STEP 3: Apply global MICE kernel batch-wise
    # --------------------------------------------------
    def apply_global_model(self):
        print("\n💉 Applying global MICE model to full dataset...")

        if self.kds_global is None:
            if os.path.exists(self.model_path):
                # Initialize dummy row (needed for miceforest 5.6.2)
                dummy_df = pd.DataFrame([[0.0]*len(self.blood_columns)], columns=self.blood_columns)
                self.kds_global = mf.ImputationKernel(dummy_df, datasets=1)
                self.kds_global.load_kernel(self.model_path)
            else:
                print("⚠️ No MICE model found — will save Parquets without imputation.")

        # Unique stay_ids
        unique_stays = self.blood["stay_id"].drop_duplicates().compute().to_numpy()
        unique_stays = [s for s in unique_stays if pd.notna(s)]
        batch_size = max(1, len(unique_stays) // self.n_output_files)
        batch_num = 0

        for i in range(0, len(unique_stays), batch_size):
            batch_stays = unique_stays[i:i + batch_size]
            batch_ddf = self.blood[self.blood["stay_id"].isin(batch_stays)]
            batch_ddf[self.blood_columns] = batch_ddf[self.blood_columns].map_partitions(
                lambda df: df.astype("float32")
            )

            batch_df = batch_ddf.compute()
            if batch_df.empty:
                batch_num += 1
                continue

            # Only run MICE if missing values exist and global kernel exists
            if self.kds_global and batch_df[self.blood_columns].isnull().sum().sum() > 0:
                kds = self.kds_global.impute_new_data(batch_df[self.blood_columns])
                df_imputed = kds.complete_data(0)
            else:
                df_imputed = batch_df[self.blood_columns].copy()

            other_cols = [c for c in batch_df.columns if c not in self.blood_columns]
            batch_final = pd.concat(
                [batch_df[other_cols].reset_index(drop=True),
                 df_imputed.reset_index(drop=True)],
                axis=1
            )

            batch_file = os.path.join(self.output_folder, f"filledBlood_{batch_num:03d}.parquet")
            batch_final.to_parquet(batch_file, index=False)
            print(f"💾 Saved batch {batch_num + 1} → {batch_file}")

            del batch_df, df_imputed, batch_final
            gc.collect()
            batch_num += 1

        print("✅ Global MICE applied (or skipped) and all Parquets saved.")

    # --------------------------------------------------
    # STEP 4: Run full pipeline
    # --------------------------------------------------
    def run(self):
        print("\n🚀 Running full blood imputation pipeline...\n")
        self.prefill()
        print('empty after prefill blodd columns:')
        print(self.blood[self.blood_columns].isna().sum())
        self.train_global_model()
        self.apply_global_model()
        print("\n✅ Pipeline complete: Prefill + MICE (or Parquets only) done.")

    def transform(self, df):
        # If using global kernel
        df_copy = df.copy().reset_index(drop=True)

        # Only fill blood columns
        if self.kds_global and df_copy[self.blood_columns].isnull().sum().sum() > 0:
            kds = self.kds_global.impute_new_data(df_copy[self.blood_columns])
            df_copy[self.blood_columns] = kds.complete_data(0)

        return df_copy