## NOW BEST: first-gradeint_boost_first6_bulk(9.2%) / last-ada_boost_last6_train_all(16.8%)

### IMPORT YOUR FORCAST FUNCTION
import forecast.autoML as ML
import forecast.autoML_train as MLT
### IMPORT YOUR DECISION FUNCTION
from decision.simple import top, random, profit_top
from decision.dynamic_programming import dynamic_programming_back_tracking, dynamic_programming
from config import TEST_DATASET_PATH
### SET SUBMISSION START
EXAM_FILE_PATH      = TEST_DATASET_PATH
RESULT_FILE_NAME    = "submission_train"
STATIC_FUNCTION     = ML.gradeint_boost_first6_bulk
SERIAL_FUNCTION     = MLT.ada_boost_last6_train_all
SKIP_DECISION       = False
DECISION_FUNCTION   = dynamic_programming_back_tracking # if you dont use decision function, set DECISION_FUNCTION = any
COST_MAX            = 15000000
### SET SUBMISSION END

import numpy as np
import pandas as pd
from datetime import datetime

def submission(exam_path:str, func_static, func_serial, func_decision, product_result_path="./gas.csv", decision_result_path="./decision.csv", skip_decision=False,is_exam=False)->bool:
    """ function for make submission file (*.csv)
    """
    df_exam = pd.read_csv(exam_path)
    n_exam = len(df_exam)
    result_data = np.zeros((n_exam,2)) # Column: [avg gas production of 6 month, with or without selection]
    if is_exam:
        n_static = 15
    else:
        n_static = 48
    n_serial = n_exam - n_static

    # Forecasting Gas production
    product_static = func_static(df_exam.iloc[:n_static,:])
    product_serial = func_serial(df_exam.iloc[n_static:,:])
    for num in range(n_static):
        result_data[num][0] = product_static[num]
    for num in range(n_serial):
        result_data[num+n_static][0] = product_serial[num]
    df_exam["Pred 6 mo. Avg. GAS (Mcf)"] = result_data[:,0]
    df_exam.to_csv(product_result_path)
    # Make decision
    if skip_decision:
        for num in range(n_exam): result_data[num][1] = 0
    else:
        dynamic_programming(df_exam,COST_MAX)
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
            skip_decision=SKIP_DECISION,
            is_exam=True)
    print("[SUBMISSION] Result file for Gas      : {0}".format(production_file_path))
    print("[SUBMISSION] Result file for Decision : {0}".format(decision_file_path))