import numpy as np
import pandas as pd
import copy
from sklearn.pipeline import make_pipeline
from preprocessing.seperate_feature_target import collective_columns
from preprocessing.new_column import avg_columns

import config
from config import TRAIN_DATASET_PATH, STATIC_COLUMNS_WITHOUT_NAN_ID, TARGET_COLUMN_LAST_6_NAME, TARGET_COLUMN_FIRST_6_NAME, SERIES_COLUMNS_GAS

def smape(a, f):
    return 1/len(a) * np.sum(2 * np.abs(f-a) / (np.abs(a) + np.abs(f))*100)

def gradeint_boost_first6_train(test_df):
    from sklearn.preprocessing import RobustScaler
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.ensemble import GradientBoostingRegressor
    # Average CV score on the training set was: -679592002.095346
    exported_pipeline = make_pipeline(
        RobustScaler(),
        VarianceThreshold(threshold=0.01),
        GradientBoostingRegressor(alpha=0.9, learning_rate=0.5, loss="huber", max_depth=1, max_features=0.05, min_samples_leaf=2, min_samples_split=3, n_estimators=100, subsample=1.0, random_state=1)
    )

    train_df = pd.read_csv(TRAIN_DATASET_PATH)

    feature, target = collective_columns(
        STATIC_COLUMNS_WITHOUT_NAN_ID, 
        TARGET_COLUMN_FIRST_6_NAME,
        train_df)

    exported_pipeline.fit(feature,target) #training_features, training_target
    results = exported_pipeline.predict(feature)#testing_features

    err = smape(target, results)
    print("[TRAIN SET] sMAPE: {0} %".format(err))

    test_feature, test_target = collective_columns(
        STATIC_COLUMNS_WITHOUT_NAN_ID,
        "No",
        test_df)
    test_results = exported_pipeline.predict(test_feature)
    return test_results

def ada_boost_last6_train(test_df):
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

    train_df = pd.read_csv(TRAIN_DATASET_PATH)
    feature, target = collective_columns(
        STATIC_COLUMNS_WITHOUT_NAN_ID, 
        TARGET_COLUMN_LAST_6_NAME,
        train_df)

    exported_pipeline.fit(feature, target)
    results = exported_pipeline.predict(feature)

    err = smape(target, results)
    print("[TRAIN SET] sMAPE: {0} %".format(err))

    test_feature, test_target = collective_columns(
        STATIC_COLUMNS_WITHOUT_NAN_ID, 
        "No",
        test_df)

    test_results = exported_pipeline.predict(test_feature)
    return test_results

def ada_boost_last6_train_exp(test_df):
    from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor
    from sklearn.linear_model import ElasticNetCV, RidgeCV
    from tpot.builtins import StackingEstimator
    # Average CV score on the training set was: -361942049.346251
    exported_pipeline = make_pipeline(
        StackingEstimator(estimator=GradientBoostingRegressor(alpha=0.75, learning_rate=0.01, loss="lad", max_depth=6, max_features=1.0, min_samples_leaf=8, min_samples_split=9, n_estimators=100, subsample=0.6500000000000001, random_state=1)),
        StackingEstimator(estimator=RidgeCV()),
        StackingEstimator(estimator=ElasticNetCV(l1_ratio=0.8, tol=0.001)),
        AdaBoostRegressor(learning_rate=0.1, loss="linear", n_estimators=100, random_state=1)
    )

    train_df = pd.read_csv(TRAIN_DATASET_PATH)
    new_train_df = copy.deepcopy(avg_columns(SERIES_COLUMNS_GAS[:30],"First 30 mo. Avg. GAS (Mcf)",train_df))

    # Train
    feature, target = collective_columns(
        STATIC_COLUMNS_WITHOUT_NAN_ID+["First 30 mo. Avg. GAS (Mcf)"],
        TARGET_COLUMN_LAST_6_NAME,
        new_train_df)

    exported_pipeline.fit(feature, target)
    results = exported_pipeline.predict(feature)

    err = smape(target, results)
    print("[TRAIN SET] sMAPE: {0} %".format(err))

    new_test_df = copy.deepcopy(avg_columns(SERIES_COLUMNS_GAS[:30],"First 30 mo. Avg. GAS (Mcf)",test_df))
    test_feature, test_target = collective_columns(
        STATIC_COLUMNS_WITHOUT_NAN_ID+["First 30 mo. Avg. GAS (Mcf)"],
        "No",
        new_test_df)

    test_results = exported_pipeline.predict(test_feature)
    return test_results

def random_forest_first6_train(test_df):
    from sklearn.ensemble import RandomForestRegressor
    
    train_df = pd.read_csv(TRAIN_DATASET_PATH)

    feature, target = collective_columns(
        STATIC_COLUMNS_WITHOUT_NAN_ID, 
        TARGET_COLUMN_FIRST_6_NAME,
        train_df)

    clf = RandomForestRegressor(max_depth=2, random_state=1)
    clf.fit(feature, target)
    results = clf.predict(feature)#testing_features

    err = smape(target, results)
    print("[TRAIN SET] sMAPE: {0} %".format(err))

    test_feature, test_target = collective_columns(
        STATIC_COLUMNS_WITHOUT_NAN_ID,
        "No",
        test_df)
    test_results = clf.predict(test_feature)
    return test_results

def random_forest_last6_train(test_df):
    from sklearn.ensemble import RandomForestRegressor
    
    train_df = pd.read_csv(TRAIN_DATASET_PATH)

    feature, target = collective_columns(
        STATIC_COLUMNS_WITHOUT_NAN_ID, 
        TARGET_COLUMN_LAST_6_NAME,
        train_df)

    clf = RandomForestRegressor(max_depth=2, random_state=1)
    clf.fit(feature, target)
    results = clf.predict(feature)#testing_features

    err = smape(target, results)
    print("[TRAIN SET] sMAPE: {0} %".format(err))

    test_feature, test_target = collective_columns(
        STATIC_COLUMNS_WITHOUT_NAN_ID,
        "No",
        test_df)
    test_results = clf.predict(test_feature)
    return test_results

def random_forest_last6_train_exp(test_df):
    from sklearn.ensemble import RandomForestRegressor
    
    train_df = pd.read_csv(TRAIN_DATASET_PATH)
    new_train_df = copy.deepcopy(avg_columns(SERIES_COLUMNS_GAS[:30],"First 30 mo. Avg. GAS (Mcf)",train_df))

    # Train
    feature, target = collective_columns(
        STATIC_COLUMNS_WITHOUT_NAN_ID+["First 30 mo. Avg. GAS (Mcf)"],
        TARGET_COLUMN_LAST_6_NAME,
        new_train_df)

    clf = RandomForestRegressor(max_depth=2, random_state=1)
    clf.fit(feature, target)
    results = clf.predict(feature)#testing_features

    err = smape(target, results)
    print("[TRAIN SET] sMAPE: {0} %".format(err))

    new_test_df = copy.deepcopy(avg_columns(SERIES_COLUMNS_GAS[:30],"First 30 mo. Avg. GAS (Mcf)",test_df))
    test_feature, test_target = collective_columns(
        STATIC_COLUMNS_WITHOUT_NAN_ID+["First 30 mo. Avg. GAS (Mcf)"],
        "No",
        new_test_df)
    test_results = clf.predict(test_feature)
    return test_results

def ada_boost_last6_train_all(test_df):
    from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor
    from sklearn.linear_model import ElasticNetCV, RidgeCV
    from tpot.builtins import StackingEstimator
    # Average CV score on the training set was: -361942049.346251
    exported_pipeline = make_pipeline(
        StackingEstimator(estimator=GradientBoostingRegressor(alpha=0.75, learning_rate=0.01, loss="lad", max_depth=6, max_features=1.0, min_samples_leaf=8, min_samples_split=9, n_estimators=100, subsample=0.6500000000000001, random_state=1)),
        StackingEstimator(estimator=RidgeCV()),
        StackingEstimator(estimator=ElasticNetCV(l1_ratio=0.8, tol=0.001)),
        AdaBoostRegressor(learning_rate=0.1, loss="linear", n_estimators=100, random_state=1)
    )

    train_df = pd.read_csv(TRAIN_DATASET_PATH)
    train_df = train_df.dropna(axis='index',subset=[TARGET_COLUMN_LAST_6_NAME])

    SELECTED_COLUMNS = config.STATIC_COLUMNS_WITHOUT_NAN_ID+config.SERIES_COLUMNS_GAS[:30] + config.SERIES_COLUMNS_CND[:30] + config.SERIES_COLUMNS_HRS[:30]

    # Train
    feature, target = collective_columns(
        SELECTED_COLUMNS,
        TARGET_COLUMN_LAST_6_NAME,
        train_df)

    exported_pipeline.fit(feature, target)
    results = exported_pipeline.predict(feature)

    err = smape(target, results)
    print("[TRAIN SET] sMAPE: {0} %".format(err))

    test_feature, test_target = collective_columns(
        SELECTED_COLUMNS,
        "No",
        test_df)

    test_results = exported_pipeline.predict(test_feature)
    return test_results

def ada_boost_simple_last6_train_all(test_df):
    from sklearn.ensemble import AdaBoostRegressor
    train_df = pd.read_csv(TRAIN_DATASET_PATH)
    train_df = train_df.dropna(axis='index',subset=[TARGET_COLUMN_LAST_6_NAME])

    SELECTED_COLUMNS = config.STATIC_COLUMNS_WITHOUT_NAN_ID+config.SERIES_COLUMNS_GAS[:30] + config.SERIES_COLUMNS_CND[:30] + config.SERIES_COLUMNS_HRS[:30]

    # Train
    feature, target = collective_columns(
        SELECTED_COLUMNS,
        TARGET_COLUMN_LAST_6_NAME,
        train_df)

    abr_model = AdaBoostRegressor(n_estimators=50, learning_rate=0.1)

    abr_model.fit(feature, target)
    results = abr_model.predict(feature)

    err = smape(target, results)
    print("[TRAIN SET] sMAPE: {0} %".format(err))

    test_feature, test_target = collective_columns(
        SELECTED_COLUMNS,
        "No",
        test_df)

    test_results = abr_model.predict(test_feature)
    return test_results

def hist_gradient_boost_simple_first6_train(test_df):
    from sklearn.ensemble import HistGradientBoostingRegressor
    
    train_df = pd.read_csv(TRAIN_DATASET_PATH)
    train_df = train_df.dropna(axis='index',subset=[TARGET_COLUMN_FIRST_6_NAME])

    SELECTED_COLUMNS = config.STATIC_COLUMNS_WITHOUT_NAN_ID

    # Train
    feature, target = collective_columns(
        SELECTED_COLUMNS,
        TARGET_COLUMN_FIRST_6_NAME,
        train_df)

    model = HistGradientBoostingRegressor(random_state = 1)

    model.fit(feature, target)
    results = model.predict(feature)

    err = smape(target, results)
    print("[TRAIN SET] sMAPE: {0} %".format(err))

    test_feature, test_target = collective_columns(
        SELECTED_COLUMNS,
        "No",
        test_df)

    test_results = model.predict(test_feature)
    return test_results

def XGBoost_simple_last6_train(test_df):
    from xgboost import XGBRegressor

    train_df = pd.read_csv(TRAIN_DATASET_PATH)
    train_df = train_df.dropna(axis='index',subset=[TARGET_COLUMN_LAST_6_NAME])

    SELECTED_COLUMNS = config.STATIC_COLUMNS_WITHOUT_NAN_ID+config.SERIES_COLUMNS_GAS[:30] + config.SERIES_COLUMNS_CND[:30] + config.SERIES_COLUMNS_HRS[:30]

    # Train
    feature, target = collective_columns(
        SELECTED_COLUMNS,
        TARGET_COLUMN_LAST_6_NAME,
        train_df)

    #model = XGBRegressor(n_estimators=1000, max_depth=7, eta=0.1, subsample=0.7, colsample_bytree=0.8)
    #model = XGBRegressor(n_estimators=1000, min_child_weight=100, gamma=100, max_depth=3, eta=0.1, subsample=0.5, colsample_bytree=0.5)
    model = XGBRegressor(n_estimators=1000, min_child_weight=10, gamma=1, max_depth=3, eta=0.1, subsample=0.5, colsample_bytree=0.5)
    model.fit(feature, target)
    results = model.predict(feature)

    err = smape(target, results)
    print("[TRAIN SET] sMAPE: {0} %".format(err))

    test_feature, test_target = collective_columns(
        SELECTED_COLUMNS,
        "No",
        test_df)

    test_results = model.predict(test_feature)
    return test_results

def XGBoost_simple_first6_train(test_df):
    from xgboost import XGBRegressor

    train_df = pd.read_csv(TRAIN_DATASET_PATH)
    train_df = train_df.dropna(axis='index',subset=[TARGET_COLUMN_FIRST_6_NAME])

    SELECTED_COLUMNS = config.STATIC_COLUMNS_WITHOUT_NAN_ID+config.SERIES_COLUMNS_GAS[:30] + config.SERIES_COLUMNS_CND[:30] + config.SERIES_COLUMNS_HRS[:30]

    # Train
    feature, target = collective_columns(
        SELECTED_COLUMNS,
        TARGET_COLUMN_FIRST_6_NAME,
        train_df)

    #model = XGBRegressor(n_estimators=1000, max_depth=7, eta=0.1, subsample=0.7, colsample_bytree=0.8)
    #model = XGBRegressor(n_estimators=1000, min_child_weight=100, gamma=100, max_depth=3, eta=0.1, subsample=0.5, colsample_bytree=0.5)
    model = XGBRegressor(n_estimators=1000, min_child_weight=10, gamma=1, max_depth=3, eta=0.1, subsample=0.5, colsample_bytree=0.5)
    model.fit(feature, target)
    results = model.predict(feature)

    err = smape(target, results)
    print("[TRAIN SET] sMAPE: {0} %".format(err))

    test_feature, test_target = collective_columns(
        SELECTED_COLUMNS,
        "No",
        test_df)

    test_results = model.predict(test_feature)
    return test_results