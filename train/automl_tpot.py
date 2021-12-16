from sklearn.model_selection import RepeatedKFold
from tpot import TPOTRegressor
import pandas as pd
import numpy as np
from datetime import datetime

from config import TRAIN_DATASET_PATH, NAN_COLUMNS

TARGET_COLUMN = 'Last 6 mo. Avg. GAS (Mcf)'
#RESULT_PATH = False
RESULT_PATH = './tpot_gas_product_best_model_{0}.py'.format(datetime.now().strftime("%Y%m%d%H%M%S"))

# Load dataset
df = pd.read_csv(TRAIN_DATASET_PATH)
df = df.dropna(axis='index',subset=[TARGET_COLUMN])
feature = df.loc[:,'Reference (KB) Elev. (ft)':'Total Sand Proppant Placed (tonne)']
feature = feature.drop(NAN_COLUMNS,'columns')
target = df.loc[:,TARGET_COLUMN]

model = TPOTRegressor(verbosity=2)
model.fit(feature, target)
model.export(RESULT_PATH)