import numpy as np
import pandas as pd

def random(cost,product,cost_max:float)->list:
    if not len(cost) == len(product):
        print("[ERROR] length of cost({0}) and product({1}) is not match.".format(len(cost), len(product)))
        return False
    return np.random.randint(0,2,size=len(cost))

def top(info,price_max:float)->list:
    try:
        price = info["PRICE ($)"].to_numpy()
        product = info["Pred 6 mo. Avg. GAS (Mcf)"].to_numpy()
        cost = info["Per Month Operation Cost ($)"].to_numpy()
    except KeyError as e:
        print("[ERROR] Key {0} is not exist in recived information.".format(e))
        return False
    n_data = len(info)
    profit = np.zeros(n_data)
    # Calculate profit
    for n in range(n_data):
        profit[n] = (6*product[n]*7-price[n]-cost[n])
    # Make Decision    
    decision = np.zeros(n_data)
    index_sorted_ascending  = product.argsort() # 오름차순 정렬
    index_sorted_descending = index_sorted_ascending[::-1] # 내림차순 정렬 (오름차순 뒤집기)
    sum_price = 0
    sum_product = 0
    sum_profit = 0
    for rank in index_sorted_descending:
        if sum_price + price[rank] > price_max: break # Burst 
        else: 
            sum_price = sum_price + price[rank]
            sum_product = sum_product + product[rank]
            sum_profit = sum_profit + profit[rank]
            decision[rank] = 1
    print("Num of Prod  : {0}/{1}".format(sum(decision),n_data))
    print("Total Price  : {0}/{1} - {2} %".format(sum_price,price_max,round((sum_price/price_max)*100,2)))
    print("Total Product: {0}".format(sum_product))
    print("Total Profit : {0}".format(sum_profit))
    return decision

def profit_top(info,price_max:float)->list:
    try:
        price = info["PRICE ($)"].to_numpy()
        product = info["Pred 6 mo. Avg. GAS (Mcf)"].to_numpy()
        cost = info["Per Month Operation Cost ($)"].to_numpy()
    except KeyError as e:
        print("[ERROR] Key {0} is not exist in recived information.".format(e))
        return False
    n_data = len(info)
    profit = np.zeros(n_data)
    # Calculate profit
    for n in range(n_data):
        profit[n] = (6*product[n]*5-price[n]-cost[n])
    # Make Decision    
    decision = np.zeros(n_data)
    index_sorted_ascending  = profit.argsort() # 오름차순 정렬
    index_sorted_descending = index_sorted_ascending[::-1] # 내림차순 정렬 (오름차순 뒤집기)
    sum_price = 0
    sum_product = 0
    sum_profit = 0
    for rank in index_sorted_descending:
        if sum_price + price[rank] > price_max: break # Burst 
        else: 
            sum_price = sum_price + price[rank]
            sum_product = sum_product + product[rank]
            sum_profit = sum_profit + profit[rank]
            decision[rank] = 1
    print("Num of Prod  : {0}/{1}".format(sum(decision),n_data))
    print("Total Price  : {0}/{1} - {2} %".format(sum_price,price_max,round((sum_price/price_max)*100,2)))
    print("Total Product: {0}".format(sum_product))
    print("Total Profit : {0}".format(sum_profit))
    return decision