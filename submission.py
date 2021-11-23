### IMPORT YOUR FORCAST FUNCTION
from forecast.simple import test_serial, test_static
from forecast.numerical import predict_36month_from_first_6month
### SET SUBMISSION START
TEST_FILE_PATH = "D:/POSTECH/대외활동/2021 제1회 데이터사이언스경진대회/data/trainSet.csv"
RESULT_FILE_NAME = "submission"
STATIC_FUNCTION = test_static
SERIAL_FUNCTION = predict_36month_from_first_6month
### SET SUBMISSION END

import numpy as np
import pandas as pd
from datetime import datetime

def submission(exam_path:str, func_static, func_serial, result_path:str)->bool:
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
        result_data[num][1] = 1 #TODO
    np.savetxt(result_path,result_data,delimiter=',')
    return True

if __name__ == "__main__":
    result_file_path = "./{0}_{1}.csv".format(RESULT_FILE_NAME,datetime.now().strftime("%Y%m%d%H%M%S"))
    submission(
            TEST_FILE_PATH,
            STATIC_FUNCTION,
            SERIAL_FUNCTION,
            result_file_path)
    print("Result file : {0}".format(result_file_path))