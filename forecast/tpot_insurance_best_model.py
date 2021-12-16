import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: -679592002.095346
exported_pipeline = make_pipeline(
    RobustScaler(),
    VarianceThreshold(threshold=0.01),
    GradientBoostingRegressor(alpha=0.9, learning_rate=0.5, loss="huber", max_depth=1, max_features=0.05, min_samples_leaf=2, min_samples_split=3, n_estimators=100, subsample=1.0)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
