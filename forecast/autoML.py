import pickle
import numpy as np

def random_forest(info)->float:
    """ random forest model with ensemble method
    """
    load_model = pickle.load(open("./saved_model/random_forest_211202.sav",'rb'))
    x_value = np.zeros(2)
    x_value[0] = info['Avg Proppant Placed / Meter (tonne)'] / info['Avg Fluid Pumped / Meter (m3)'] #일단 denstiy 함
    x_value[1] = info['Avg Proppant Placed / Meter (tonne)'] ** 0.7 + x_value[0] ** 0.3
    x_value = x_value.reshape(-1, 2)
    value = load_model.predict(x_value)
    return value