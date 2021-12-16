import pickle
import numpy as np
import copy

from pandas.io.parsers import read_csv
from preprocessing.new_column import avg_columns
from preprocessing.seperate_feature_target import collective_columns, except_columns
from config import ID_COLUMNS, SERIES_COLUMNS_GAS, STATIC_COLUMNS_WITHOUT_NAN, STATIC_COLUMNS_WITHOUT_NAN_ID

def random_forest(info)->float: #dahyeon
    MODEL_PATH = './saved_model/random_forest_211202.sav'
    loaded_model = pickle.load(open(MODEL_PATH, 'rb'))

    x = np.zeros(2)
    y = np.zeros(1)
    density = info['Avg Proppant Placed / Meter (tonne)'] / info['Avg Fluid Pumped / Meter (m3)']
    mean_norm_proppant = info['Avg Proppant Placed / Meter (tonne)'] / 1.8627083333333339 #mean of 'Avg Proppant Placed / Meter (tonne)' in trainSet
    mean_norm_density = density  / 0.16664898999685573 #mean of density in tainSet

    x[0] = info['Avg Proppant Placed / Meter (tonne)'] ** 0.7 + density  ** 0.3
    x[1] = mean_norm_proppant**0.7 + mean_norm_density**0.3

    #y[0] = info['First 6 mo. Avg. GAS (Mcf)'] /64247.25694229165
    x = x.reshape(-1, 2)

    #loaded_model.fit(x, y)
    value = loaded_model.predict(x)
    return value

def gradeint_boost(info)-> float:
    # Average CV score on the training set was: -679592002.095346
    MODEL_PATH = './saved_model/Gradient_boost_211205.pkl'
    loaded_model = pickle.load(open(MODEL_PATH, 'rb'))

    x = np.zeros(22)
    for i in range(22):
        x[i] = info[STATIC_COLUMNS_WITHOUT_NAN_ID[i]]

    x = x.reshape(-1, 22)
    result = loaded_model.predict(x)

    return float(result)

def gradeint_boost_last6(info)-> float:
    # Average CV score on the training set was: -679592002.095346

    MODEL_PATH = './saved_model/gradient_boost_last6_20211207002137.pkl'
    loaded_model = pickle.load(open(MODEL_PATH, 'rb'))

    x = np.zeros(22)
    for i in range(22):
        x[i] = info[STATIC_COLUMNS_WITHOUT_NAN_ID[i]]

    x = x.reshape(-1, 22)
    result = loaded_model.predict(x)

    return float(result)

def load_model_C22(info)->float:
    MODEL_PATH = r"D:\POSTECH\대외활동\2021 제1회 데이터사이언스경진대회\2021_SHALE_GAS_TRADING\gradient_boost_first_6_24.37.pkl"
    loaded_model = pickle.load(open(MODEL_PATH, 'rb'))

    x = np.zeros(22)
    for i in range(22):
        x[i] = info[STATIC_COLUMNS_WITHOUT_NAN_ID[i]]
    x = x.reshape(-1, 22)
    result = loaded_model.predict(x)

    return float(result)

def load_molel_C23(info)->float:
    MODEL_PATH = r"D:\POSTECH\대외활동\2021 제1회 데이터사이언스경진대회\2021_SHALE_GAS_TRADING\gradient_boost_last_6_17.61.pkl"
    loaded_model = pickle.load(open(MODEL_PATH, 'rb'))

    # preprocessing
    pros_info =  copy.deepcopy(avg_columns(SERIES_COLUMNS_GAS[:30],"First 30 mo. Avg. GAS (Mcf)",info))
    used_columns = STATIC_COLUMNS_WITHOUT_NAN_ID + ["First 30 mo. Avg. GAS (Mcf)"]

    x = np.zeros(23)
    for i in range(23):
        x[i] = pros_info[used_columns[i]]
    x = x.reshape(-1, 23)
    result = loaded_model.predict(x)

    return float(result)

def gradeint_boost_first6_bulk(test_df):
    # Average CV score on the training set was: -679592002.095346
    MODEL_PATH = './saved_model/Gradient_boost_211205.pkl'
    loaded_model = pickle.load(open(MODEL_PATH, 'rb'))

    feature, target = collective_columns(
        STATIC_COLUMNS_WITHOUT_NAN_ID, 
        "No",
        test_df)

    result = loaded_model.predict(feature)

    return result

def gradeint_boost_last6_bulk(test_df):
    # Average CV score on the training set was: -679592002.095346

    MODEL_PATH = './saved_model/gradient_boost_last6_20211207002137.pkl'
    loaded_model = pickle.load(open(MODEL_PATH, 'rb'))

    feature, target = collective_columns(
        STATIC_COLUMNS_WITHOUT_NAN_ID, 
        "No",
        test_df)

    result = loaded_model.predict(feature)

    return result