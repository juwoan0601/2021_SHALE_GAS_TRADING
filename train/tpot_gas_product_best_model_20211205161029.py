import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.pipeline import make_pipeline
from tpot.builtins import StackingEstimator
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

'''
# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv(DATA_SOURCE, sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)
'''

# Average CV score on the training set was: -361942049.346251
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=GradientBoostingRegressor(alpha=0.75, learning_rate=0.01, loss="lad", max_depth=6, max_features=1.0, min_samples_leaf=8, min_samples_split=9, n_estimators=100, subsample=0.6500000000000001)),
    StackingEstimator(estimator=RidgeCV()),
    StackingEstimator(estimator=ElasticNetCV(l1_ratio=0.8, tol=0.001)),
    AdaBoostRegressor(learning_rate=0.1, loss="linear", n_estimators=100)
)

exported_pipeline.fit(feature, target)
results = exported_pipeline.predict(feature)

def smape(a, f):
    return 1/len(a) * np.sum(2 * np.abs(f-a) / (np.abs(a) + np.abs(f))*100)

err = smape(target, results)
print("sMAPE: {0} %".format(err))

# Save Model
if RESULT_PATH:
    pickle.dump(exported_pipeline, open(RESULT_PATH, 'wb')) #dump해야 모델 전체가 저장됨