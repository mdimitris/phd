import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
import numpy as np
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
        return self.imputeGases(self.gases)
    
        
    def imputeGases(self,df): 
        print("Starting gases imputation and filling process....")
        import gc
        gc.collect()
        
        df.reset_index()
        df.replace([-1, "missing", "NA"], np.nan )

        # Ensure proper datetime
        df['charttime'] = dd.to_datetime(df['charttime'], format="%Y-%m-%d %H:%M:%S.%f", errors="coerce")
        df.sort_values(by=["subject_id", "stay_id", 'charttime'] )

        # Grouping for interpolation based on time gaps
        df["group"] = df.groupby(["subject_id", "stay_id"])["charttime"].transform(
            lambda x: (x.diff() > pd.Timedelta(hours=self.interval)).cumsum(),
            meta=("group", "int32"))
        
        # 🕒 Add relative time (in minutes since first observation per stay)
        df["relative_time_min"] = df.groupby(["subject_id", "stay_id"])["charttime"].transform(
            lambda x: (x - x.min()).dt.total_seconds() / 60,
            meta=("relative_time_min", np.float32)   # ✅ float32
        )

        # 🕒 Optionally add time gap (between measurements)
        df["time_gap_min"] = df.groupby(["subject_id", "stay_id"])["charttime"].transform(
            lambda x: x.diff().dt.total_seconds() / 60,
            meta=("time_gap_min", "f4")
        ).fillna(0)
        
        gc.collect()
        meta_dict = {col: np.float32 for col in self.columns}
        
        df[self.columns] = (
            df.groupby(["subject_id", "stay_id", "group"], group_keys=False)
            .apply(lambda group: (
                group.set_index("charttime")[self.columns]
                    .ffill()
                    .bfill()
                    .interpolate(method="linear")
                    #.reset_index(drop=True)
            ),
          meta=meta_dict  # ✅ explicit dtypes
         )
         .reset_index(drop=True)
        )
        gc.collect()
        
        return df 

        
   
