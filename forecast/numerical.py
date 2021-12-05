import pickle
import numpy as np

def predict_36month_from_first_6month(info)->float:
    import numpy as np
    import math
    avg_first_6m = (info["GAS_MONTH_1"]+info["GAS_MONTH_2"]+info["GAS_MONTH_3"]+info["GAS_MONTH_4"]+info["GAS_MONTH_5"]+info["GAS_MONTH_6"])/6
    serial = np.zeros(36)
    C1 = 1.6519
    C2 = -0.045
    for m in range(1,36):
        serial[m] = avg_first_6m*C1*pow(math.e,C2*m)
    return np.mean(serial[30:])

def multi_regression(info)->float:
    """ Multi regression Model with minitab
    """
    value = -6120704+(1073*info['TVD (ft)'])+(2762194*info['Bot-Hole Northing (NAD83)'])+(29.4*info['Total Proppant Placed (tonne)'])-(17.6*info['Total Sand Proppant Placed (tonne)'])-(0.0465*info['TVD (ft)']*info['TVD (ft)'])-(147336*info['Bot-Hole Northing (NAD83)']*info['Bot-Hole Northing (NAD83)'])+(0.001349*info['Total Proppant Placed (tonne)']*info['Total Proppant Placed (tonne)'])-(248.3*info['TVD (ft)']*info['Bot-Hole Northing (NAD83)'])-(36.02*info['Bot-Hole Northing (NAD83)']*info['Total Sand Proppant Placed (tonne)'])
    return value

def random_forest(info)->float: #dahyeon
    filename = 'C:/Users/백다현/PycharmProjects/pythonProject/2021_SHALE_GAS_TRADING/forecast/finalized_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))

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

def Gradeint_Boost(info): #-> float:
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import RobustScaler
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.ensemble import GradientBoostingRegressor
    # Average CV score on the training set was: -679592002.095346

    x = np.zeros(22)
    y = np.zeros(1) #target first 6 month

    #for num in range(len(x)+1):
     #   x[num - 1] = info.iloc[:, num] #error: too many indexer
    x[0] = info['Reference (KB) Elev. (ft)']
    x[1] = info['Ground Elevation (ft)']
    x[2] = info['MD (All Wells) (ft)']
    x[3] = info['TVD (ft)']
    x[4] = info['Bot-Hole direction (N/S)/(E/W)']
    x[5] = info['Bot-Hole Easting (NAD83)']
    x[6] = info['Bot-Hole Northing (NAD83)']
    #x[7] = info['On Prod YYYY/MM/DD']
    #x[8] = info['First Prod YYYY/MM']
    #x[9] = info['Last Prod. YYYY/MM']
    #x[7] = info['Stimulation Fluid']
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
    #x[18] = info['Proppant Composition']
    #x[23] = info['Proppant Name 1']
    #x[24] = info['Proppant Size 1']
    x[18] = info['Avg Proppant 1 Placed (tonne)']
    x[19] = info['Total Proppant 1 Placed (tonne)']
    x[20] = info['Total Ceramic Proppant Placed (tonne)']
    x[21] = info['Total Sand Proppant Placed (tonne)']

    y[0] = info['First 6 mo. Avg. GAS (Mcf)']

    x = x.reshape(-1, 22)
    #y = y.reshape(-1, 1)

    exported_pipeline = make_pipeline(
        RobustScaler(),
        VarianceThreshold(threshold=0.01),
        GradientBoostingRegressor(alpha=0.9, learning_rate=0.5, loss="huber", max_depth=1, max_features=0.05,
                                  min_samples_leaf=2, min_samples_split=3, n_estimators=100, subsample=1.0)
    )

    exported_pipeline.fit(x, y)
    result = exported_pipeline.predict(x)

    return result