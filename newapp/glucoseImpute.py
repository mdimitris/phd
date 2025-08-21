import InputData
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import miceforest as mf
import dask.dataframe as dd
import os

class glucoseImpute:
    
    def __init__(self, glucoseCreatinine, columns, interval):
        
        self.glucoseCreatinine = glucoseCreatinine
        self.columns = columns
        self.interval = interval    
                   
    
    def set_glucoseCreatinine(self, glucoseCreatinine):
        self.glucoseCreatinine = glucoseCreatinine

    def get_glucoseCreatinine(self):
        return self.glucoseCreatinine

    def set_interval(self, interval):
        self.interval = interval

    def get_interval(self):
        return self.interval

    def prepareGlucose (self):    
        self.glucoseCreatinine.drop(['hadm_id'], axis=1)
        self.glucoseCreatinine['charttime'] = dd.to_datetime(self.glucoseCreatinine['charttime'])
        self.glucoseCreatinine[["subject_id","stay_id"]] = self.glucoseCreatinine[["subject_id","stay_id"]].astype(pd.Int32Dtype())
        self.glucoseCreatinine[self.columns] =self.glucoseCreatinine[self.columns].astype('float').astype('float32')
        return self.glucoseCreatinine
    
    def imputeGlucose (self,df,interval,cols):
        print("Starting glucose and creatinine imputation and filling process....")
        import gc
        gc.collect()
        
        df.reset_index(inplace=True)
        df.replace([-1, "missing", "NA"], np.nan)

        # Ensure proper datetime
        df['charttime'] = pd.to_datetime(df['charttime'], format="%Y-%m-%d %H:%M:%S.%f", errors="coerce")
        df.sort_values(by=["subject_id", "stay_id", 'charttime'])

        # Grouping for interpolation based on time gaps
        df["group"] = df.groupby(["subject_id", "stay_id"])['charttime'].transform(
            lambda x: (x.diff() > pd.Timedelta(minutes=interval)).cumsum())

        # 🕒 Add relative time (in minutes since first observation per stay)
        df["relative_time_min"] = df.groupby(["subject_id", "stay_id"])["charttime"] \
                                    .transform(lambda x: (x - x.min()).dt.total_seconds() / 60)

        # 🕒 Optionally add time gap (between measurements)
        df["time_gap_min"] = df.groupby(["subject_id", "stay_id"])["charttime"] \
                            .transform(lambda x: x.diff().dt.total_seconds() / 60).fillna(0)
        gc.collect()
        # Interpolation within each group
        df[cols] = (
            df.groupby(["subject_id", "stay_id", "group"], group_keys=False)
            .apply(lambda group: (
                group.set_index("charttime")[cols]
                    .ffill()
                    .bfill()
                    .interpolate(method="linear")
                    #.reset_index(drop=True)
            ))
            .reset_index(drop=True)
        )
        
        return df