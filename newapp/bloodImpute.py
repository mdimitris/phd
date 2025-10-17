import dask.dataframe as dd
import pandas as pd
import numpy as np
import os
import gc
import miceforest as mf

class bloodImpute:
    def __init__(self, blood, blood_columns, batch_size, output_dir, max_batches=128):
        """
        Parameters
        ----------
        blood : dask.DataFrame
            The merged dataframe containing blood data.
        blood_columns : list[str]
            Columns to apply MICE imputation on.
        batch_size : int
            Number of stay_ids per batch.
        output_dir : str
            Directory where imputed batches will be saved.
        """
        self.blood = blood
        self.blood_columns = blood_columns
        self.batch_size = batch_size
        self.output_dir = output_dir
        self.max_batches = max_batches

        os.makedirs(self.output_dir, exist_ok=True)

    # --------------------------------------------------------------------------
    def prefill_dask(self):
        """
        Forward-fill, backward-fill, and interpolate numeric blood columns
        grouped by stay_id.
        """
        print("🩸 Starting ffill/bfill/interpolation in blood columns...")

        # ✅ Ensure stay_id exists and is numeric/string consistently
        if "stay_id" not in self.blood.columns:
            raise ValueError("❌ 'stay_id' column not found in blood dataframe!")
        self.blood["stay_id"] = self.blood["stay_id"].astype(str)

        # ✅ Explicit Dask meta
        meta = {col: "f4" for col in self.blood_columns}

        def fill_group(df):
            return df.ffill().bfill().interpolate(method="linear")

        try:
            filled = (
                self.blood.groupby("stay_id")[self.blood_columns]
                .apply(fill_group, meta=meta)
            )
            # ✅ Properly merge ffilled data
            for col in self.blood_columns:
                self.blood[col] = filled[col]
        except Exception as e:
            raise RuntimeError(f"❌ Error during interpolation step: {e}")

        print("✅ Prefill (ffill/bfill/interpolate) done.")

    # --------------------------------------------------------------------------


    def batch_mice_imputation(self):
        """
        Runs MICE imputation in batches by stay_id and saves each batch to Parquet.
        Fully Dask-safe to avoid KeyErrors related to dtype mappings.
        """
        print("🧩 Starting MICE imputation for blood columns...")

        if self.blood is None:
            raise ValueError("❌ self.blood is not initialized.")
        if self.batch_size <= 0:
            raise ValueError(f"❌ Invalid batch_size: {self.batch_size}")

        # ----------------------------
        # 1️⃣ Ensure stay_id is int32 and persist
        # ----------------------------
        if "stay_id" not in self.blood.columns:
            raise ValueError("❌ 'stay_id' column not found in self.blood!")
        self.blood["stay_id"] = self.blood["stay_id"].astype("int32")
        self.blood = self.blood.persist()  # stabilize _meta across partitions

        # ----------------------------
        # 2️⃣ Get unique stay_ids
        # ----------------------------
        unique_stays = self.blood["stay_id"].drop_duplicates().compute().to_numpy()
        unique_stays = [s for s in unique_stays if pd.notna(s)]
        n_stays = len(unique_stays)
        print(f"Total unique stay_id: {n_stays}")

        if n_stays == 0:
            raise ValueError("❌ No valid stay_id values found in self.blood — cannot batch impute.")

        batch_num = 0

        # ----------------------------
        # 3️⃣ Loop over batches
        # ----------------------------
        for i in range(0, n_stays, self.batch_size):
            if batch_num >= self.max_batches:
                print(f"Reached limit of {self.max_batches} Parquet files. Stopping early.")
                break

            batch_stays = unique_stays[i:i + self.batch_size]
            if not batch_stays:
                print(f"⚠️ Empty batch {batch_num}, skipping.")
                continue

            print(f"🧠 Processing batch {batch_num + 1}/{self.max_batches} "
                f"({len(batch_stays)} patients)")

            # ----------------------------
            # 4️⃣ Filter Dask dataframe safely
            # ----------------------------
            try:
                batch_ddf = self.blood[self.blood["stay_id"].isin(batch_stays)]

                # ----------------------------
                # 5️⃣ Cast numeric blood columns to float32 for MICE
                # ----------------------------
                batch_ddf[self.blood_columns] = batch_ddf[self.blood_columns].map_partitions(
                    lambda df: df.astype("float32")
                )

                # ----------------------------
                # 6️⃣ Compute to Pandas safely
                # ----------------------------
                batch_df = batch_ddf.compute()
            except Exception as e:
                print(f"❌ Failed to prepare batch {batch_num}: {e}")
                continue

            if batch_df.empty:
                print(f"⚠️ Batch {batch_num} is empty, skipping.")
                continue

            # ----------------------------
            # 7️⃣ Run MICE imputation
            # ----------------------------
            try:
                kds = mf.ImputationKernel(
                    batch_df[self.blood_columns],
                    save_all_iterations=False,
                    random_state=42
                )
                kds.mice(iterations=3)
                df_imputed = kds.complete_data(dataset=0)
            except Exception as e:
                print(f"❌ Error running MICE on batch {batch_num}: {e}")
                continue

            # ----------------------------
            # 8️⃣ Reattach non-lab columns
            # ----------------------------
            columns_excluded = [c for c in batch_df.columns if c not in self.blood_columns]
            batch_final = pd.concat(
                [
                    batch_df[columns_excluded].reset_index(drop=True),
                    df_imputed.reset_index(drop=True),
                ],
                axis=1,
            )

            # ----------------------------
            # 9️⃣ Save batch to Parquet
            # ----------------------------
            batch_file = os.path.join(self.output_dir, f"batch_{batch_num:03d}.parquet")
            try:
                batch_final.to_parquet(batch_file, index=False)
                print(f"💾 Saved batch {batch_num + 1} → {batch_file}")
            except Exception as e:
                print(f"❌ Failed to save batch {batch_num}: {e}")

            # ----------------------------
            # 10️⃣ Cleanup
            # ----------------------------
            batch_num += 1
            del batch_df, df_imputed, batch_final
            gc.collect()

        print("✅ MICE imputation completed for all batches.")


    # --------------------------------------------------------------------------
    def run(self):
        print("***** Starting Lab Imputation Pipeline ******")
        self.prefill_dask()
        self.batch_mice_imputation()
        print("✅ Lab imputation finished. All batches saved to:", self.output_dir)