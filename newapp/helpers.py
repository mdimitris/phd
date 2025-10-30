import pandas as pd
import dask.dataframe as dd

def prepareDataset(ddf, blood_cols,drop_cols,dataset):
    #first delete rows that all data are missing
        #ddf = ddf.drop(columns=["rdwsd","admittime"])
        ddf = ddf.drop(columns=drop_cols)
        ddf['charttime'] = dd.to_datetime(ddf['charttime'], errors="coerce")
        ddf[blood_cols] = ddf[blood_cols].astype("float32").round(2)
        ddf[["subject_id","stay_id","hadm_id"]] = ddf[["subject_id","stay_id"]].astype(pd.Int32Dtype())
        #ddf=ddf.drop('rdwsd', axis=1)
        #ddf.drop(columns=['rdwsd'], inplace=True, errors='ignore')
        #save to parquet in order to use it later
        ddf.to_parquet(f"/root/scripts/newapp/secondrun/unfilled/{dataset}.parquet", write_index=False)
        print(type,'dataset saved to parquet and it is ready for later usage.')
        # return ddf


# def prepareGases(self):

#     #     self.gases[["subject_id", "stay_id","hadm_id"]] = self.gases[["subject_id", "stay_id","hadm_id"]].astype(pd.Int32Dtype())
#     #     self.gases["charttime"]=dd.to_datetime(self.gases['charttime'], format="%Y-%m-%d %H:%M:%S.%f", errors="coerce")
#     #     self.gases[self.columns]=self.gases[self.columns].astype('float').astype('float32').round(2)
#     #     self.gases.drop(columns=['hadm_id', 'sofa_time','pf_ratio'],  errors='ignore')
#     #     return self.imputeGases()



def calculateMissing(ddf):
    # Compute percentage of missing values for each column
    missing_percentage = (ddf.isnull().sum() / len(ddf) * 100).compute()
    # Display sorted results (highest missing first)
    missing_percentage = missing_percentage.sort_values(ascending=False)
    print(missing_percentage)