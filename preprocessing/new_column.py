import pandas as pd
import numpy as np

def avg_columns(target_column_list:list, column_label:str, source_df:pd.DataFrame)->pd.DataFrame:
    source_df[column_label] = (source_df[target_column_list].sum(axis=1))/len(target_column_list)
    print(source_df.tail())
    return source_df

def avg_columns_single_row(target_column_list:list, column_label:str, source_df:pd.Series)->pd.Series:
    sum_col = 0
    for c in target_column_list:
        sum_col = sum_col + source_df[c]
    source_df[column_label] = sum_col
    return source_df