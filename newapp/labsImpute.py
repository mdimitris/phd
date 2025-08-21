import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import miceforest as mf

class labsImpute:
    
    def __init__(self, labs, glucoseCreatinine, lab_columns, glucCreat_columns, interval):
        self.labs = labs
        self.glucoseCreatinine = glucoseCreatinine
        self.lab_columns = lab_columns
        self.gl_columns = glucCreat_columns
        self.interval = interval    
                   
    def set_labs(self, labs):
        self.labs = labs

    def get_labs(self):
        return self.labs
    
    def set_glucoseCreatinine(self, glucoseCreatinine):
        self.glucoseCreatinine = glucoseCreatinine

    def get_glucoseCreatinine(self):
        return self.glucoseCreatinine
    
    
    
    
    
    def prepareLabs(self):
        
        #first delete rows that all data are missing
        #df_glucoseCreatinine = InputData.clearEmpties(self.glucoseCreatinine, self.lab_columns, "charttime", 2)
        self.labs['admittime'] = pd.to_datetime(self.labs['admittime'])
        self.labs['charttime'] = pd.to_datetime(self.labs['charttime'])
        self.labs[self.lab_columns] = self.labs[self.lab_columns].astype("float32").round(2)
        self.labs[["subject_id","stay_id","hadm_id"]] = self.labs[["subject_id","stay_id","hadm_id"]].astype(pd.Int32Dtype())
        self.labs.drop(columns=['rdwsd'], inplace=True, errors='ignore')
        print('labs info after normalization:')
        print(self.labs.info())
        return self.labs
      
        

    def prepareExamsBg(self):
        exam_columns=self.get_columns_to_fill_2()

        exam_columns.remove("creatinine")
        exam_columns.remove("glucose")
        df_examsBg = self.clearEmpties(
            self.get_examsBg(), exam_columns, "charttime",3
        )    
    
    def populateLabResults(self,gas_columns):
        print("Lab results imputation started")
        
        
        columns_tofill=self.lab_columns+self.gl_columns+gas_columns
        df_for_mice = self.labs[columns_tofill].copy()
        kds = mf.ImputationKernel(
            df_for_mice,
            save_all_iterations=False,
            random_state=100
        )

        kds.mice(iterations=3)
        df_imputed = kds.complete_data(dataset=0)

        # Reattach ID/time columns
        df_imputed.reset_index(drop=True, inplace=True)
        self.labs.reset_index(drop=True, inplace=True)
        columns_excluded = [col for col in self.labs.columns if col not in columns_tofill]
        labs_imputed = pd.concat([self.labs[columns_excluded], df_imputed], axis=1)

        # Optionally drop glucose/creatinine
        #labs_imputed.drop(columns=['glucose', 'creatinine'], inplace=True, errors='ignore')
        return labs_imputed

        