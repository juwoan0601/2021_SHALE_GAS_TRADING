import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import GradientBoostingRegressor
import pickle
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
RESULT_PATH = './gradient_boost_last6_{0}.pkl'.format(datetime.now().strftime("%Y%m%d%H%M%S"))

# Load dataset
df = pd.read_csv(DATA_SOURCE)
df = df.dropna(axis='index',subset=[TARGET_COLUMN])
feature = df.loc[:,'Reference (KB) Elev. (ft)':'Total Sand Proppant Placed (tonne)']
feature = feature.drop(NAN_COLUMNS,'columns')
target = df.loc[:,TARGET_COLUMN]

# Print trained columns
print("Train Column List({0}): {1}".format(len(feature.columns), feature.columns))
print("Target Column: {0}".format(TARGET_COLUMN))

# Average CV score on the training set was: -679592002.095346
exported_pipeline = make_pipeline(
    RobustScaler(),
    VarianceThreshold(threshold=0.01),
    GradientBoostingRegressor(alpha=0.9, learning_rate=0.5, loss="huber", max_depth=1, max_features=0.05, min_samples_leaf=2, min_samples_split=3, n_estimators=100, subsample=1.0)
)

exported_pipeline.fit(feature,target) #training_features, training_target
results = exported_pipeline.predict(feature)#testing_features

def smape(a, f):
    return 1/len(a) * np.sum(2 * np.abs(f-a) / (np.abs(a) + np.abs(f))*100)

err = smape(target, results)
print("sMAPE: {0} %".format(err))

# Save Model
if RESULT_PATH:
    pickle.dump(exported_pipeline, open(RESULT_PATH, 'wb')) #dump해야 모델 전체가 저장됨