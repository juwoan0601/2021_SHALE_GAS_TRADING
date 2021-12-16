import pandas as pd
import numpy as np

def avg_columns(target_column_list:list, column_label:str, source_df:pd.DataFrame)->pd.DataFrame:
    try:
        source_df[column_label] = (source_df[target_column_list].sum(axis=1))/len(target_column_list)
    except ValueError:
        sum_c = 0
        for c in target_column_list: sum_c = sum_c + source_df[c]
        source_df[column_label] = sum_c/len(target_column_list)
    return source_df