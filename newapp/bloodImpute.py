import dask.dataframe as dd
import pandas as pd
import numpy as np
import os
import gc
import miceforest as mf

class bloodImpute:
    def __init__(self, blood, blood_columns, batch_size, output_dir):
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

        os.makedirs(self.output_dir, exist_ok=True)

    # --------------------------------------------------------------------------
    def prefill_dask(self):
        """
        Forward-fill, backward-fill, and interpolate numeric blood columns
        grouped by stay_id.
        """
        print("🩸 Starting ffill/bfill/interpolation in blood columns...")

        # Ensure stay_id exists
        if "stay_id" not in self.blood.columns:
            raise ValueError("❌ 'stay_id' column not found in blood dataframe!")

        # Explicit Dask meta (required for apply)
        meta = {col: "f4" for col in self.blood_columns}

        # Apply ffill/bfill/interpolation grouped by stay_id
        def fill_group(df):
            return df.ffill().bfill().interpolate(method="linear")

        try:
            filled = (
                self.blood.groupby("stay_id")[self.blood_columns]
                .apply(fill_group, meta=meta)
            )
            # Merge filled data back with original Dask DataFrame
            self.blood = self.blood.assign(**{col: filled[col] for col in self.blood_columns})
        except Exception as e:
            raise RuntimeError(f"❌ Error during interpolation step: {e}")

        print("✅ Prefill with ffill/bfill/interpolation done.")

    # --------------------------------------------------------------------------
    def batch_mice_imputation(self):
        """
        Runs MICE imputation in batches by stay_id and saves each batch to Parquet.
        """
        print("🧩 Starting MICE imputation for blood columns...")

        # Validate inputs
        if self.blood is None:
            raise ValueError("❌ self.blood is not initialized.")
        if self.batch_size <= 0:
            raise ValueError(f"❌ Invalid batch_size: {self.batch_size}")

        # Compute unique stay_ids safely
        unique_stays = self.blood["stay_id"].drop_duplicates().compute().to_numpy()
        unique_stays = unique_stays[~pd.isna(unique_stays)]
        n_stays = len(unique_stays)
        print(f"Total unique stay_id: {n_stays}")

        if n_stays == 0:
            raise ValueError("❌ No valid stay_id values found in self.blood — cannot batch impute.")

        batch_num = 0
        max_batches = 128  # limit total parquet files

        for i in range(0, n_stays, self.batch_size):
            if batch_num >= max_batches:
                print(f"Reached limit of {max_batches} Parquet files. Stopping early.")
                break

            batch_stays = list(unique_stays[i:i + self.batch_size])
            if not batch_stays:
                print(f"⚠️ Empty batch {batch_num}, skipping.")
                continue

            print(f"🧠 Processing batch {batch_num + 1}/{max_batches} "
                  f"({len(batch_stays)} patients)")

            # --- Filter Dask DataFrame safely using .isin() ---
            try:
                batch_ddf = self.blood[self.blood["stay_id"].isin(batch_stays)].compute()
            except Exception as e:
                print(f"❌ Failed to filter batch {batch_num}: {e}")
                continue

            if batch_ddf.empty:
                print(f"⚠️ Batch {batch_num} is empty, skipping.")
                continue

            # --- Run MICE Imputation ---
            try:
                kds = mf.ImputationKernel(
                    batch_ddf[self.blood_columns],
                    save_all_iterations=False,
                    random_state=42
                )
                kds.mice(iterations=3)
                df_imputed = kds.complete_data(dataset=0)
            except Exception as e:
                print(f"❌ Error running MICE on batch {batch_num}: {e}")
                continue

            # --- Reattach non-lab columns ---
            columns_excluded = [c for c in batch_ddf.columns if c not in self.blood_columns]
            batch_final = pd.concat(
                [
                    batch_ddf[columns_excluded].reset_index(drop=True),
                    df_imputed.reset_index(drop=True),
                ],
                axis=1,
            )

            # --- Save batch ---
            batch_file = os.path.join(self.output_dir, f"batch_{batch_num:03d}.parquet")
            try:
                batch_final.to_parquet(batch_file, index=False)
                print(f"💾 Saved batch {batch_num + 1} → {batch_file}")
            except Exception as e:
                print(f"❌ Failed to save batch {batch_num}: {e}")

            # Cleanup
            batch_num += 1
            del batch_ddf, df_imputed, batch_final
            gc.collect()

        print("✅ MICE imputation completed for all batches.")

    # --------------------------------------------------------------------------
    def run(self):
        print("***** Starting Lab Imputation Pipeline ******")
        self.prefill_dask()
        self.batch_mice_imputation()
        print("✅ Lab imputation finished. All batches saved to:", self.output_dir)
