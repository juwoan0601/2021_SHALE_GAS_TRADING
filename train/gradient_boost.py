import numpy as np
import pandas as pd
import pickle
from datetime import datetime
from sklearn.pipeline import make_pipeline

def gradient_boost_first_6(feature,target,result=False):
    from sklearn.preprocessing import RobustScaler
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.ensemble import GradientBoostingRegressor
    # Average CV score on the training set was: -679592002.095346
    exported_pipeline = make_pipeline(
        RobustScaler(),
        VarianceThreshold(threshold=0.01),
        GradientBoostingRegressor(alpha=0.9, learning_rate=0.5, loss="huber", max_depth=1, max_features=0.05, min_samples_leaf=2, min_samples_split=3, n_estimators=100, subsample=1.0, random_state=1)
    )

    exported_pipeline.fit(feature,target) #training_features, training_target
    results = exported_pipeline.predict(feature)#testing_features

    def smape(a, f):
        return 1/len(a) * np.sum(2 * np.abs(f-a) / (np.abs(a) + np.abs(f))*100)

    err = smape(target, results)
    print("sMAPE: {0} %".format(err))

    # Save Model
    if result:
        pickle.dump(exported_pipeline, open("./gradient_boost_first_6_{0}.pkl".format(round(err,2)), 'wb')) #dump해야 모델 전체가 저장됨