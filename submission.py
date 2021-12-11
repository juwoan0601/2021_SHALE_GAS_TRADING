### IMPORT YOUR FORCAST FUNCTION
from forecast.simple import test_serial, test_static
from forecast.numerical import predict_36month_from_first_6month
from forecast.numerical import multi_regression
from forecast.autoML import gradeint_boost
from forecast.numerical import random_forest
### IMPORT YOUR DECISION FUNCTION
from decision.simple import top, random, profit_top
### SET SUBMISSION START
EXAM_FILE_PATH      = "C:/Users/백다현/examSet.csv"
TEST_FILE_PATH      = "C:/Users/백다현/trainSet.csv"
RESULT_FILE_NAME    = "Gradeint_Boost"
STATIC_FUNCTION     = gradeint_boost #model 이거 바꿔야함
SERIAL_FUNCTION     = gradeint_boost #predict_36month_from_first_6month
SKIP_DECISION       = True
DECISION_FUNCTION   = top # if you dont use decision function, set DECISION_FUNCTION = any
COST_MAX            = 15000000
### SET SUBMISSION END

import numpy as np
import pandas as pd
from datetime import datetime
df_exam = pd.read_csv(EXAM_FILE_PATH, index_col=0)

def submission(exam_path:str, func_static, func_serial, func_decision, product_result_path="./gas.csv", decision_result_path="./decision.csv", skip_decision=False)->bool:
    """ function for make submission file (*.csv)
    """
    df_exam = pd.read_csv(exam_path, index_col=0)
    n_exam = len(df_exam)
    result_data = np.zeros((n_exam,2)) # Column: [avg gas production of 6 month, with or without selection]
    # Forecasting Gas production
    for num in range(n_exam):
        if pd.isna(df_exam.iloc[num]["GAS_MONTH_1"]):   # Use Static function
            result_data[num][0] = func_static(df_exam.iloc[num])
            #print(df_exam.iloc[:,29])
        else:                                           # Use Serial function
            result_data[num][0] = func_serial(df_exam.iloc[num])
    df_exam["Pred 6 mo. Avg. GAS (Mcf)"] = result_data[:,0]
    df_exam.to_csv(product_result_path)
    # Make decision 
    if skip_decision:
        for num in range(n_exam): result_data[num][1] = 0
    else:
        decision_data = func_decision(df_exam,COST_MAX)
        for num in range(n_exam):
            result_data[num][1] = decision_data[num]
    # Save files
    np.savetxt(decision_result_path,result_data,delimiter=',')
    return True

if __name__ == "__main__":
    production_file_path = "./{0}_gas_{1}.csv".format(
                                            RESULT_FILE_NAME,
                                            datetime.now().strftime("%Y%m%d%H%M%S"))
    decision_file_path = "./{0}_decision_{1}.csv".format(
                                            RESULT_FILE_NAME,
                                            datetime.now().strftime("%Y%m%d%H%M%S"))
    submission(
            EXAM_FILE_PATH,
            STATIC_FUNCTION,
            SERIAL_FUNCTION,
            DECISION_FUNCTION,
            product_result_path=production_file_path,
            decision_result_path=decision_file_path,
            skip_decision=SKIP_DECISION)
    print("[SUBMISSION] Result file for Gas      : {0}".format(production_file_path))
    print("[SUBMISSION] Result file for Decision : {0}".format(decision_file_path))