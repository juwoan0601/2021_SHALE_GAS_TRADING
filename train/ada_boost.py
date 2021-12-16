import numpy as np
import pandas as pd
import pickle
from datetime import datetime
from sklearn.pipeline import make_pipeline

def ada_boost_last_6(feature,target,result=False):
    from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor
    from sklearn.linear_model import ElasticNetCV, RidgeCV
    from tpot.builtins import StackingEstimator
    # Average CV score on the training set was: -361942049.346251
    exported_pipeline = make_pipeline(
        StackingEstimator(estimator=GradientBoostingRegressor(alpha=0.75, learning_rate=0.01, loss="lad", max_depth=6, max_features=1.0, min_samples_leaf=8, min_samples_split=9, n_estimators=100, subsample=0.6500000000000001)),
        StackingEstimator(estimator=RidgeCV()),
        StackingEstimator(estimator=ElasticNetCV(l1_ratio=0.8, tol=0.001)),
        AdaBoostRegressor(learning_rate=0.1, loss="linear", n_estimators=100, random_state=1)
    )

    exported_pipeline.fit(feature, target)
    results = exported_pipeline.predict(feature)

    def smape(a, f):
        return 1/len(a) * np.sum(2 * np.abs(f-a) / (np.abs(a) + np.abs(f))*100)

    err = smape(target, results)
    print("sMAPE: {0} %".format(err))

    # Save Model
    if result:
        pickle.dump(exported_pipeline, open("./gradient_boost_last_6_{0}.pkl".format(round(err,2)), 'wb')) #dump해야 모델 전체가 저장됨
        