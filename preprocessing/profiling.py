import pandas as pd
import pandas_profiling

df = pd.read_csv("../data/trainSet.csv")
df_static = df.iloc[:,0:32]
report = df_static.profile_report()
report.to_file('./static_EDA.html')