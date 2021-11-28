import numpy as np
import pandas as pd

def random(cost,product,cost_max:float)->list:
    if not len(cost) == len(product):
        print("[ERROR] length of cost({0}) and product({1}) is not match.".format(len(cost), len(product)))
        return False
    return np.random.randint(0,2,size=len(cost))

def top(info,cost_max:float)->list:
    cost = info["PRICE ($)"].to_numpy()
    product = info["Pred 6 mo. Avg. GAS (Mcf)"].to_numpy()
    if not len(cost) == len(product):
        print("[ERROR] length of cost({0}) and product({1}) is not match.".format(len(cost), len(product)))
        return False
    decision = np.zeros(len(cost))
    index_sorted_ascending  = product.argsort() # 오름차순 정렬
    index_sorted_descending = index_sorted_ascending[::-1] # 내림차순 정렬 (오름차순 뒤집기)
    sum_cost = 0
    sum_product = 0
    for rank in index_sorted_descending:
        if sum_cost + cost[rank] > cost_max: break
        else: 
            sum_cost = sum_cost + cost[rank]
            sum_product = sum_product + product[rank]
            decision[rank] = 1
    print("Total Cost   : {0}/{1} - {2} %".format(sum_cost,cost_max,round((sum_cost/cost_max)*100,2)))
    print("Total Product: {0}".format(sum_product))
    return decision

def profit_top(info,cost_max:float)->list:
    pass


'''
product = np.loadtxt(r"D:\POSTECH\대외활동\2021 제1회 데이터사이언스경진대회\2021_SHALE_GAS_TRADING\submission_exam_20211124032523.csv",delimiter=',')
df_exam = pd.read_csv("D:/POSTECH/대외활동/2021 제1회 데이터사이언스경진대회/data/examSet.csv", index_col=0)
cost = df_exam["PRICE ($)"].to_numpy()
print(random(cost,product[:,0],15000000))
'''