
import pandas as pd
import copy
from config import TRAIN_DATASET_PATH, STATIC_COLUMNS_WITHOUT_NAN, SERIES_COLUMNS_GAS, STATIC_COLUMNS_WITHOUT_NAN_ID
from config import TARGET_COLUMN_LAST_6_NAME, TARGET_COLUMN_FIRST_6_NAME
from preprocessing.seperate_feature_target import collective_columns, except_columns
from train.gradient_boost import gradient_boost_first_6
from train.ada_boost import ada_boost_last_6
import config

from preprocessing.new_column import avg_columns

from validation_with_train import compare_two_csv_files

# # Preprocessing
# df = pd.read_csv(TRAIN_DATASET_PATH)
# new_df = copy.deepcopy(avg_columns(SERIES_COLUMNS_GAS[:30],"First 30 mo. Avg. GAS (Mcf)",df))

# # Train
# features, target = collective_columns(
#     STATIC_COLUMNS_WITHOUT_NAN_ID+["First 30 mo. Avg. GAS (Mcf)"],
#     TARGET_COLUMN_FIRST_6_NAME,
#     new_df)
# ada_boost_last_6(features, target, result=True)

# # Preprocessing
# df = pd.read_csv(TRAIN_DATASET_PATH)
# df = df.dropna(axis='index',subset=[TARGET_COLUMN_LAST_6_NAME])
# print(df.head())

# # Train
# features, target = collective_columns(
#     STATIC_COLUMNS_WITHOUT_NAN_ID + config.SERIES_COLUMNS_GAS[:30] + config.SERIES_COLUMNS_CND[:30] + config.SERIES_COLUMNS_HRS[:30],
#     TARGET_COLUMN_LAST_6_NAME,
#     df)
# from train.ada_boost import ada_boost_simple, ada_boost_last_6
# ada_boost_simple(features, target, result=True)

F1 = r"D:\POSTECH\대외활동\2021 제1회 데이터사이언스경진대회\2021_SHALE_GAS_TRADING\submission_train_decision_20211219234534.csv"
F2 = r"D:\POSTECH\대외활동\2021 제1회 데이터사이언스경진대회\2021_SHALE_GAS_TRADING\submission_train_decision_20211219234617.csv"
compare_two_csv_files(F1,F2,sep=15)

# from decision.simple import profit_top, top, brute_force, dynamic_programming, dynamic_programming_back_tracking
# df = pd.read_csv(r"D:\POSTECH\대외활동\2021 제1회 데이터사이언스경진대회\2021_SHALE_GAS_TRADING\submission_18_gas_20211217214351.csv")
# dynamic_programming(df,15000000)
# dynamic_programming_back_tracking(df,15000000)
