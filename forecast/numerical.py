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


'''
#hyunji

def adaboost(info)->int:
    x = df.drop(['First 6 mo. Avg. GAS (Mcf)'], axis=1)
    y = df['First 6 mo. Avg. GAS (Mcf)']
 
    # Split dataset into training set and test set
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3) # 70% training and 30% test
 
    # Create adaboost classifer object
    #base_model = DecisionTreeClassifier(max_depth = 5)
    abr_model = AdaBoostRegressor(n_estimators=50,
                                learning_rate=0.1)
    # Train Adaboost Classifer
    abr_model.fit(x_train, y_train)

    # 모델 검증 (예측의 결정계수 반환)
    print(abr_model.score(x_train, y_train))
    print(abr_model.score(x_test, y_test))

    return  
'''