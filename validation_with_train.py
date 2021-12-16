### SET FILE PATH END
from submission_with_train import submission
### IMPORT YOUR FORCAST FUNCTION
from forecast.autoML_train import ada_boost_last6_train, gradeint_boost_first6_train, ada_boost_last6_train_exp, random_forest_first6_train, random_forest_last6_train, random_forest_last6_train_exp
from forecast.autoML import gradeint_boost_first6_bulk, gradeint_boost_last6_bulk
### IMPORT YOUR DECISION FUNCTION
from decision.simple import top, random, profit_top
from config import TRAIN_DATASET_PATH, TEST_DATASET_PATH, TRUE_PRODUCT_FILE_PATH
### SET SUBMISSION START
EXAM_FILE_PATH      = TEST_DATASET_PATH
TEST_FILE_PATH      = TRAIN_DATASET_PATH
TRUE_FILE_PATH      = TRUE_PRODUCT_FILE_PATH
RESULT_FILE_NAME    = "submission_with_train"
STATIC_FUNCTION     = gradeint_boost_first6_bulk
SERIAL_FUNCTION     = ada_boost_last6_train_exp
### SET SUBMISSION END

from datetime import datetime
import numpy as np
def sMAPE(A, F)->float:
    return 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))

def sMAPE_static(A, F, sep)->float:
    return sMAPE(A[:sep], F[:sep])

def sMAPE_serial(A, F,sep)->float:
    return sMAPE(A[sep:], F[sep:])

def compare_two_csv_files(file1:str, file2:str, sep=48):
    date_time = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    actual_value = np.loadtxt(file1,delimiter=',')[:,0]
    fake_value = np.loadtxt(file2,delimiter=',')[:,0]
    print("***** Calculate sMAPE *****")
    print("- Time Stamp  : {0}".format(date_time))
    print("- Actual file : {0}".format(file1))
    print("- Fake   file : {0}".format(file2))
    print("- sMAPE       : {0} %".format(sMAPE(actual_value,fake_value)))
    print("--    Static  : {0} %".format(sMAPE_static(actual_value,fake_value,sep)))
    print("--    Serial  : {0} %".format(sMAPE_serial(actual_value,fake_value,sep)))
    print("***** Calculate sMAPE END *****")

if __name__ == "__main__":
    production_file_path = "./{0}_gas_{1}.csv".format(
                                            RESULT_FILE_NAME,
                                            datetime.now().strftime("%Y%m%d%H%M%S"))
    decision_file_path = "./{0}_decision_{1}.csv".format(
                                            RESULT_FILE_NAME,
                                            datetime.now().strftime("%Y%m%d%H%M%S"))
    submission(
            TEST_FILE_PATH,
            STATIC_FUNCTION,
            SERIAL_FUNCTION,
            any,
            product_result_path=production_file_path,
            decision_result_path=decision_file_path,
            skip_decision=True)
    compare_two_csv_files(TRUE_FILE_PATH,decision_file_path)
