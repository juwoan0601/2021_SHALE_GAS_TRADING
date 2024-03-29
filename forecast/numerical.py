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

def multi_regression_first6(info)->float: ## jeongwoo new
    """ Multi regression Model with minitab
    """
    value = -369503+(43.9*info['MD (All Wells) (ft)'])+(130.5*info['Avg Frac Spacing (m)'])+(20.32*info['Completed Length (m)'])-(781*info['Bot-Hole direction (N/S)/(E/W)'])-(0.001363*info['MD (All Wells) (ft)']*info['MD (All Wells) (ft)'])+(13.95*info['Avg Frac Spacing (m)']*info['Bot-Hole direction (N/S)/(E/W)'])
    return value

def multi_regression_last6(info)->float: # JUWAN
    """ Multi regression Model with minitab (Last 6 month)
    """
    value =-369503+(43.9*info['MD (All Wells) (ft)'])+(130.5*info['Avg Frac Spacing (m)'])+(20.32*info['Completed Length (m)'])-(781*info['Bot-Hole direction (N/S)/(E/W)'])-(0.001363*info['MD (All Wells) (ft)']*info['MD (All Wells) (ft)'])+(13.95*info['Avg Frac Spacing (m)']*info['Bot-Hole direction (N/S)/(E/W)'])
    return value
