### IMPORT YOUR FORCAST FUNCTION
from forecast.simple import test_serial, test_static
from forecast.numerical import predict_36month_from_first_6month
### IMPORT YOUR DECISION FUNCTION
from decision.simple import top, random
### SET SUBMISSION START
TEST_FILE_PATH = "D:/POSTECH/대외활동/2021 제1회 데이터사이언스경진대회/data/examSet.csv"
RESULT_FILE_NAME = "submission_exam"
STATIC_FUNCTION = test_static
SERIAL_FUNCTION = predict_36month_from_first_6month
DECISION_FUNCTION = top # if you dont use decision function, set DECISION_FUNCTION = any
### SET SUBMISSION END

import numpy as np
import pandas as pd
from datetime import datetime
df_exam = pd.read_csv("D:/POSTECH/대외활동/2021 제1회 데이터사이언스경진대회/data/examSet.csv", index_col=0)
COST = df_exam["PRICE ($)"].to_numpy()
COST_MAX = 15000000

def submission(exam_path:str, func_static, func_serial, func_decision, result_path:str, skip_decision=False)->bool:
    """ function for make submission file (*.csv)
    """
    df_exam = pd.read_csv(exam_path, index_col=0)
    n_exam = len(df_exam)
    result_data = np.zeros((n_exam,2)) # Column: avg gas production of 6 month, with or without selection
    for num in range(n_exam):
        if pd.isna(df_exam.iloc[num]["GAS_MONTH_1"]):   # Use Static function
            result_data[num][0] = func_static(df_exam.iloc[num])
        else:                                           # Use Serial function
            result_data[num][0] = func_serial(df_exam.iloc[num])
    if skip_decision:
        for num in range(n_exam): result_data[num][1] = 0
    else:
        decision_data = func_decision(COST,result_data[:,0],COST_MAX)
        for num in range(n_exam):
            result_data[num][1] = decision_data[num]
    np.savetxt(result_path,result_data,delimiter=',')
    return True

if __name__ == "__main__":
    result_file_path = "./{0}_{1}.csv".format(
                                            RESULT_FILE_NAME,
                                            datetime.now().strftime("%Y%m%d%H%M%S"))
    submission(
            TEST_FILE_PATH,
            STATIC_FUNCTION,
            SERIAL_FUNCTION,
            DECISION_FUNCTION,
            result_file_path,
            skip_decision=False)
    print("Result file : {0}".format(result_file_path))