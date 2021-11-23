def test_static(info)->float:
    return info["Ground Elevation (ft)"]

def submission(exam_path:str, func_static:function, func_serial:function, result_path:str)->bool:
    """ function for make submission file (*.csv)
    """
    import pandas as pd
    import os

    df_exam = pd.read_csv(exam_path, index_col=0)
    return True

