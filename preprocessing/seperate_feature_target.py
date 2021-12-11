import numpy as np
import pandas as pd

def all_columns(target_column:str, df:pd.DataFrame):
    df = df.dropna(axis='index',subset=[target_column])
    feature = df.loc[:,:]
    target = df.loc[:,target_column]

    # Print trained columns
    print("Train Column List({0}): {1}".format(len(feature.columns), feature.columns))
    print("Target Column: {0}".format(target_column))
    return feature, target

def collective_columns(feature_columns_list:list, target_column:str, df:pd.DataFrame):
    df = df.dropna(axis='index',subset=[target_column])
    feature = df.loc[:,feature_columns_list]
    target = df.loc[:,target_column]

    # Print trained columns
    print("Train Column List({0}): {1}".format(len(feature.columns), feature.columns))
    print("Target Column: {0}".format(target_column))
    return feature, target

def except_columns(target:list, exception:list)-> list:
    for e in exception:
        target.remove(e)
    return target
