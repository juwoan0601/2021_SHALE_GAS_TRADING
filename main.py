
import pandas as pd
import copy
from config import TRAIN_DATASET_PATH, STATIC_COLUMNS_WITHOUT_NAN, SERIES_COLUMNS_GAS, STATIC_COLUMNS_WITHOUT_NAN_ID
from config import TARGET_COLUMN_LAST_6_NAME, TARGET_COLUMN_FIRST_6_NAME
from preprocessing.seperate_feature_target import collective_columns, except_columns
from train.gradient_boost import gradient_boost_first_6
from train.ada_boost import ada_boost_last_6

from preprocessing.new_column import avg_columns

from validation import compare_two_csv_files

# Preprocessing
df = pd.read_csv(TRAIN_DATASET_PATH)
new_df = copy.deepcopy(avg_columns(SERIES_COLUMNS_GAS[:30],"First 30 mo. Avg. GAS (Mcf)",df))

# Train
features, target = collective_columns(
    STATIC_COLUMNS_WITHOUT_NAN_ID+["First 30 mo. Avg. GAS (Mcf)"],
    TARGET_COLUMN_FIRST_6_NAME,
    new_df)
ada_boost_last_6(features, target, result=True)
