import pickle
import numpy as np
import copy
from preprocessing.new_column import avg_columns, avg_columns_single_row
from config import SERIES_COLUMNS_GAS

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

    x[0] = info['Reference (KB) Elev. (ft)']
    x[1] = info['Ground Elevation (ft)']
    x[2] = info['MD (All Wells) (ft)']
    x[3] = info['TVD (ft)']
    x[4] = info['Bot-Hole direction (N/S)/(E/W)']
    x[5] = info['Bot-Hole Easting (NAD83)']
    x[6] = info['Bot-Hole Northing (NAD83)']
    x[7] = info['Total Proppant Placed (tonne)']
    x[8] = info['Avg Proppant Placed per Stage (tonne)']
    x[9] = info['Total Fluid Pumped (m3)']
    x[10] = info['Avg Fluid Pumped per Stage (m3)']
    x[11] = info['Stages Actual']
    x[12] = info['Completed Length (m)']
    x[13] = info['Avg Frac Spacing (m)']
    x[14] = info['Load Fluid Rec (m3)']
    x[15] = info['Load Fluid (m3)']
    x[16] = info['Avg Fluid Pumped / Meter (m3)']
    x[17] = info['Avg Proppant Placed / Meter (tonne)']
    x[18] = info['Avg Proppant 1 Placed (tonne)']
    x[19] = info['Total Proppant 1 Placed (tonne)']
    x[20] = info['Total Ceramic Proppant Placed (tonne)']
    x[21] = info['Total Sand Proppant Placed (tonne)']

    x = x.reshape(-1, 22)
    result = loaded_model.predict(x)

    return float(result)

def gradeint_boost_last6(info)-> float:
    # Average CV score on the training set was: -679592002.095346

    MODEL_PATH = './saved_model/gradient_boost_last6_20211207002137.pkl'
    loaded_model = pickle.load(open(MODEL_PATH, 'rb'))

    x = np.zeros(22)

    x[0] = info['Reference (KB) Elev. (ft)']
    x[1] = info['Ground Elevation (ft)']
    x[2] = info['MD (All Wells) (ft)']
    x[3] = info['TVD (ft)']
    x[4] = info['Bot-Hole direction (N/S)/(E/W)']
    x[5] = info['Bot-Hole Easting (NAD83)']
    x[6] = info['Bot-Hole Northing (NAD83)']
    x[7] = info['Total Proppant Placed (tonne)']
    x[8] = info['Avg Proppant Placed per Stage (tonne)']
    x[9] = info['Total Fluid Pumped (m3)']
    x[10] = info['Avg Fluid Pumped per Stage (m3)']
    x[11] = info['Stages Actual']
    x[12] = info['Completed Length (m)']
    x[13] = info['Avg Frac Spacing (m)']
    x[14] = info['Load Fluid Rec (m3)']
    x[15] = info['Load Fluid (m3)']
    x[16] = info['Avg Fluid Pumped / Meter (m3)']
    x[17] = info['Avg Proppant Placed / Meter (tonne)']
    x[18] = info['Avg Proppant 1 Placed (tonne)']
    x[19] = info['Total Proppant 1 Placed (tonne)']
    x[20] = info['Total Ceramic Proppant Placed (tonne)']
    x[21] = info['Total Sand Proppant Placed (tonne)']

    x = x.reshape(-1, 22)
    result = loaded_model.predict(x)

    return float(result)

def gradeint_boost_last6_C23(info)-> float:
    # Average CV score on the training set was: -679592002.095346

    MODEL_PATH = r'D:\POSTECH\대외활동\2021 제1회 데이터사이언스경진대회\2021_SHALE_GAS_TRADING\gradient_boost_last_6_17.03.pkl'
    loaded_model = pickle.load(open(MODEL_PATH, 'rb'))
    info = copy.deepcopy(avg_columns_single_row(SERIES_COLUMNS_GAS[:30],"First 30 mo. Avg. GAS (Mcf)",info))
    
    x = np.zeros(23)

    x[0] = info['Reference (KB) Elev. (ft)']
    x[1] = info['Ground Elevation (ft)']
    x[2] = info['MD (All Wells) (ft)']
    x[3] = info['TVD (ft)']
    x[4] = info['Bot-Hole direction (N/S)/(E/W)']
    x[5] = info['Bot-Hole Easting (NAD83)']
    x[6] = info['Bot-Hole Northing (NAD83)']
    x[7] = info['Total Proppant Placed (tonne)']
    x[8] = info['Avg Proppant Placed per Stage (tonne)']
    x[9] = info['Total Fluid Pumped (m3)']
    x[10] = info['Avg Fluid Pumped per Stage (m3)']
    x[11] = info['Stages Actual']
    x[12] = info['Completed Length (m)']
    x[13] = info['Avg Frac Spacing (m)']
    x[14] = info['Load Fluid Rec (m3)']
    x[15] = info['Load Fluid (m3)']
    x[16] = info['Avg Fluid Pumped / Meter (m3)']
    x[17] = info['Avg Proppant Placed / Meter (tonne)']
    x[18] = info['Avg Proppant 1 Placed (tonne)']
    x[19] = info['Total Proppant 1 Placed (tonne)']
    x[20] = info['Total Ceramic Proppant Placed (tonne)']
    x[21] = info['Total Sand Proppant Placed (tonne)']
    x[22] = info['First 30 mo. Avg. GAS (Mcf)']

    x = x.reshape(-1, 23)
    result = loaded_model.predict(x)

    return float(result)
