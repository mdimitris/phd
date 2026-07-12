import dask.dataframe as dd

df = dd.read_parquet(
    r"C:\phd-final\phd\newapp\secondrun\filled\blood_filled_2nd.parquet"
)

print("########Stats after merging#######")
print("Rows:", len(df))
print("Patients:", df["subject_id"].nunique().compute())
print("Admissions:", df["hadm_id"].nunique().compute())
print("ICU stays:", df["stay_id"].nunique().compute())


import dask.dataframe as dd


df.to_csv(
    r"C:\phd-final\phd\newapp\secondrun\filled\blood_filled_2nd.csv",
    single_file=True,
    index=False
)