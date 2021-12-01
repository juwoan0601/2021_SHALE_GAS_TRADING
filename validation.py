### SET FILE PATH
TRUE_FILE_PATH = "./forecast/answer.csv"
FAKE_FILE_PATH = "./submission_train_decision_20211201224341.csv"
### SET FILE PATH END

from datetime import datetime
import numpy as np
def sMAPE(A, F)->float:
    return 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))

def sMAPE_static(A, F)->float:
    return sMAPE(A[:48], F[:48])

def sMAPE_serial(A, F)->float:
    return sMAPE(A[48:], F[48:])

if __name__ == "__main__":
    date_time = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    actual_value = np.loadtxt(TRUE_FILE_PATH,delimiter=',')[:,0]
    fake_value = np.loadtxt(FAKE_FILE_PATH,delimiter=',')[:,0]
    print("***** Calculate sMAPE *****")
    print("- Time Stamp  : {0}".format(date_time))
    print("- Actual file : {0}".format(TRUE_FILE_PATH))
    print("- Fake   file : {0}".format(FAKE_FILE_PATH))
    print("- sMAPE       : {0} %".format(sMAPE(actual_value,fake_value)))
    print("--    Static  : {0} %".format(sMAPE_static(actual_value,fake_value)))
    print("--    Serial  : {0} %".format(sMAPE_serial(actual_value,fake_value)))
    print("***** Calculate sMAPE END *****")