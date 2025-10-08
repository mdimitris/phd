import pandas as pd
import dask.dataframe as dd
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import miceforest as mf

class bloodImpute:
    
    def __init__(self, blood, glucoseCreatinine, blood_columns, glucCreat_columns, interval):
        self.blood = blood
        self.glucoseCreatinine = glucoseCreatinine
        self.blood_columns = blood_columns
        self.glucCreat_columns = glucCreat_columns
        self.interval = interval    
                   
    def set_blood(self, blood):
        self.blood = blood

    def get_blood(self):
        return self.blood
    
    def set_glucoseCreatinine(self, glucoseCreatinine):
        self.glucoseCreatinine = glucoseCreatinine

    def get_glucoseCreatinine(self):
        return self.glucoseCreatinine
    
    
    
    
    
    def prepareblood(self):
        
        #first delete rows that all data are missing
        #df_glucoseCreatinine = InputData.clearEmpties(self.glucoseCreatinine, self.lab_columns, "charttime", 2)
        self.blood['admittime'] = dd.to_datetime(self.blood['admittime'])
        self.blood['charttime'] = dd.to_datetime(self.blood['charttime'])
        self.blood[self.blood_columns] = self.blood[self.blood_columns].astype("float32").round(2)
        self.blood[["subject_id","stay_id","hadm_id"]] = self.blood[["subject_id","stay_id","hadm_id"]].astype(pd.Int32Dtype())
        self.blood=self.blood.drop('rdwsd', axis=1)
        #self.blood.drop(columns=['rdwsd'], inplace=True, errors='ignore')
        print('blood info after normalization:')
        print(self.blood.info())
        #save to parquet in order to use it later
        self.blood.to_parquet("filled/blood.parquet", write_index=False)
        return self.blood
      
        

    def prepareExamsBg(self):
        exam_columns=self.get_columns_to_fill_2()

        exam_columns.remove("creatinine")
        exam_columns.remove("glucose")
        df_examsBg = self.clearEmpties(
            self.get_examsBg(), exam_columns, "charttime",3
        )    
    
    def populateLabResults(self,gas_columns):
        print("Lab results imputation started")
        
        
        columns_tofill=self.lab_columns+self.glucCreat_columns+gas_columns
        df_for_mice = self.blood[columns_tofill].copy()
        kds = mf.ImputationKernel(
            df_for_mice,
            save_all_iterations=False,
            random_state=100
        )

        kds.mice(iterations=3)
        df_imputed = kds.complete_data(dataset=0)

        # Reattach ID/time columns
        df_imputed.reset_index(drop=True, inplace=True)
        self.blood.reset_index(drop=True, inplace=True)
        columns_excluded = [col for col in self.blood.columns if col not in columns_tofill]
        blood_imputed = pd.concat([self.blood[columns_excluded], df_imputed], axis=1)

        # Optionally drop glucose/creatinine
        #blood_imputed.drop(columns=['glucose', 'creatinine'], inplace=True, errors='ignore')
        return blood_imputed

        