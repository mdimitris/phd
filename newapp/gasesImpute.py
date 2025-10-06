import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
import numpy as np
import gc
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import dask.dataframe as dd

class gasesImpute:
    
    def __init__(self, df_gases, columns, interval):
        
        self.gases = df_gases
        self.columns = columns
        self.interval = interval       
        
        
    def set_gases(self, gases):
        self.gases = gases

    def get_gases(self):
        return self.gases

    def set_interval(self, interval):
        self.interval = interval

    def get_interval(self):
        return self.interval
    
    def set_columns(self, columns):
        self.columns = columns

    def get_columns(self):
        return self.columns
    
    def prepareGases(self):
        import gc
        gc.collect()
        print('checking gases Columns')
        print(self.columns)
        # optimize df_vitals
        self.gases[["subject_id", "stay_id","hadm_id"]] = self.gases[["subject_id", "stay_id","hadm_id"]].astype(pd.Int32Dtype())
        self.gases["charttime"]=dd.to_datetime(self.gases['charttime'], format="%Y-%m-%d %H:%M:%S.%f", errors="coerce")
        self.gases[self.columns]=self.gases[self.columns].astype('float').astype('float32').round(2)
        self.gases.drop(columns=['hadm_id', 'sofa_time','pf_ratio'],  errors='ignore')
        return self.imputeGases()
    
        
    def imputeGases(self):
        print("Starting gases imputation and filling process....")
        gc.collect()

        # Replace invalid values
        dd_gases = self.gases.map_partitions(
            lambda pdf: pdf.replace([-1, "missing", "NA"], np.nan)
        )
        # Ensure proper datetime
        dd_gases["charttime"] = dd.to_datetime(dd_gases["charttime"], errors="coerce")

        # Sort by stay/time per partition
        dd_gases = dd_gases.map_partitions(
            lambda d: d.sort_values(by=["subject_id", "stay_id", "charttime"])
        )

        # Grouping for interpolation based on time gaps
        dd_gases["group"] = dd_gases.groupby(["subject_id", "stay_id"])["charttime"].transform(
            lambda x: (x.diff() > pd.Timedelta(hours=self.interval)).cumsum(),
            meta=("group", "int32")
        )

        # Add relative time (minutes since first observation)
        dd_gases["relative_time_min"] = dd_gases.groupby(["subject_id", "stay_id"])["charttime"].transform(
            lambda x: (x - x.min()).dt.total_seconds() / 60,
            meta=("relative_time_min", "f4")
        )

        # Add time gaps (between measurements)
        dd_gases["time_gap_min"] = dd_gases.groupby(["subject_id", "stay_id"])["charttime"].transform(
            lambda x: x.diff().dt.total_seconds() / 60,
            meta=("time_gap_min", "f4")
        ).fillna(0)

        gc.collect()

        # Explicit meta for interpolation
        meta_dict = {col: np.float32 for col in self.columns}

        # Dask-safe filling: no set_index inside apply
        dd_gases[self.columns] = dd_gases.groupby(["subject_id", "stay_id", "group"], group_keys=False).apply(
            lambda group: group[self.columns].ffill().bfill().interpolate(method="linear"),
            meta=meta_dict
        )

        gc.collect()
        return dd_gases

        
   
