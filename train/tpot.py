from sklearn.model_selection import RepeatedKFold
from tpot import TPOTRegressor
import pandas as pd
import numpy as np
from datetime import datetime

DATA_SOURCE = "D:/POSTECH/대외활동/2021 제1회 데이터사이언스경진대회/data/trainSet.csv"
NAN_COLUMNS = [
    'On Prod YYYY/MM/DD', 
    'First Prod YYYY/MM',
    'Last Prod. YYYY/MM',
    'Stimulation Fluid',
    'Proppant Composition',
    'Proppant Name 1',
    'Proppant Size 1'
]
TARGET_COLUMN = 'Last 6 mo. Avg. GAS (Mcf)'
#RESULT_PATH = False
RESULT_PATH = './tpot_gas_product_best_model_{0}.py'.format(datetime.now().strftime("%Y%m%d%H%M%S"))

# Load dataset
df = pd.read_csv(DATA_SOURCE)
df = df.dropna(axis='index',subset=[TARGET_COLUMN])
feature = df.loc[:,'Reference (KB) Elev. (ft)':'Total Sand Proppant Placed (tonne)']
feature = feature.drop(NAN_COLUMNS,'columns')
target = df.loc[:,TARGET_COLUMN]

model = TPOTRegressor(verbosity=2)
model.fit(feature, target)
model.export(RESULT_PATH)