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
        return self.imputeGlucose (self.interval,self.columns)
    
    def imputeGlucose (self,interval,cols):
        print("Starting glucose and creatinine imputation and filling process....")
        import gc
        gc.collect()
        
        #self.glucoseCreatinine.reset_index(inplace=True)
        self.glucoseCreatinine.replace([-1, "missing", "NA"], np.nan)

        # Ensure proper datetime
        self.glucoseCreatinine['charttime'] = dd.to_datetime(self.glucoseCreatinine['charttime'], format="%Y-%m-%d %H:%M:%S.%f", errors="coerce")
        self.glucoseCreatinine.sort_values(by=["subject_id", "stay_id", 'charttime'])

        # Grouping for interpolation based on time gaps
        self.glucoseCreatinine = self.glucoseCreatinine.compute()
        self.glucoseCreatinine["group"] = self.glucoseCreatinine.groupby(["subject_id", "stay_id"])['charttime'].transform(
            lambda x: (x.diff() > pd.Timedelta(minutes=interval)).cumsum())

        # 🕒 Add relative time (in minutes since first observation per stay)
        self.glucoseCreatinine["relative_time_min"] = self.glucoseCreatinine.groupby(["subject_id", "stay_id"])["charttime"] \
                                    .transform(lambda x: (x - x.min()).dt.total_seconds() / 60)

        # 🕒 Optionally add time gap (between measurements)
        self.glucoseCreatinine["time_gap_min"] = self.glucoseCreatinine.groupby(["subject_id", "stay_id"])["charttime"] \
                            .transform(lambda x: x.diff().dt.total_seconds() / 60).fillna(0)
        gc.collect()
        # Interpolation within each group
        self.glucoseCreatinine[cols] = (
            self.glucoseCreatinine.groupby(["subject_id", "stay_id", "group"], group_keys=False)
            .apply(lambda group: (
                group.set_index("charttime")[cols]
                    .ffill()
                    .bfill()
                    .interpolate(method="linear")
                    #.reset_index(drop=True)
            ))
            .reset_index(drop=True)
        )
        
        return self.glucoseCreatinine