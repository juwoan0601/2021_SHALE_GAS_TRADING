### SET FILE PATH

TRUE_FILE_PATH = "C:/Users/백다현\OneDrive - postech.ac.kr/문서/2021_SHALE_GAS_TRADING/forecast/answer.csv"
FAKE_FILE_PATH = "C:/Users/백다현\OneDrive - postech.ac.kr/문서/2021_SHALE_GAS_TRADING/Gradeint_Boost_decision_20211205135921.csv"

### SET FILE PATH END
from submission import submission
### IMPORT YOUR FORCAST FUNCTION
from forecast.autoML import gradeint_boost_last6, gradeint_boost, load_molel_C23
### IMPORT YOUR DECISION FUNCTION
from decision.simple import top, random, profit_top
from config import TRAIN_DATASET_PATH, TEST_DATASET_PATH
### SET SUBMISSION START
EXAM_FILE_PATH      = TEST_DATASET_PATH
TEST_FILE_PATH      = TRAIN_DATASET_PATH
RESULT_FILE_NAME    = "submission_train_gradientBoost"
STATIC_FUNCTION     = gradeint_boost
SERIAL_FUNCTION     = load_molel_C23
### SET SUBMISSION END

from datetime import datetime
import numpy as np
def sMAPE(A, F)->float:
    return 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))

def sMAPE_static(A, F)->float:
    return sMAPE(A[:48], F[:48])

def sMAPE_serial(A, F)->float:
    return sMAPE(A[48:], F[48:])

def compare_two_csv_files(file1:str, file2:str):
    date_time = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    actual_value = np.loadtxt(file1,delimiter=',')[:,0]
    fake_value = np.loadtxt(file2,delimiter=',')[:,0]
    print("***** Calculate sMAPE *****")
    print("- Time Stamp  : {0}".format(date_time))
    print("- Actual file : {0}".format(file1))
    print("- Fake   file : {0}".format(file2))
    print("- sMAPE       : {0} %".format(sMAPE(actual_value,fake_value)))
    print("--    Static  : {0} %".format(sMAPE_static(actual_value,fake_value)))
    print("--    Serial  : {0} %".format(sMAPE_serial(actual_value,fake_value)))
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
