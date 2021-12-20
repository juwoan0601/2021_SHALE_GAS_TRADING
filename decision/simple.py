import numpy as np
import pandas as pd

from config import COST_COLUMN_NAME, PREDICT_COLUMN_NAME, PRICE_COLUMN_NAME

def random(cost,product,cost_max:float)->list:
    if not len(cost) == len(product):
        print("[ERROR] length of cost({0}) and product({1}) is not match.".format(len(cost), len(product)))
        return False
    return np.random.randint(0,2,size=len(cost))

def top(info,price_max:float)->list:
    try:
        price = info[PRICE_COLUMN_NAME].to_numpy()
        product = info[PREDICT_COLUMN_NAME].to_numpy()
        cost = info[COST_COLUMN_NAME].to_numpy()
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
        price = info[PRICE_COLUMN_NAME].to_numpy()
        product = info[PREDICT_COLUMN_NAME].to_numpy()
        cost = info[COST_COLUMN_NAME].to_numpy()
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

def brute_force(info,price_max:float,max_trial=100)->list:
    import random
    import copy
    from tqdm import tqdm
    try:
        price = info[PRICE_COLUMN_NAME].to_numpy()
        product = info[PREDICT_COLUMN_NAME].to_numpy()
        cost = info[COST_COLUMN_NAME].to_numpy()
    except KeyError as e:
        print("[ERROR] Key {0} is not exist in recived information.".format(e))
        return False
    n_data = len(info)
    profit = np.zeros(n_data)
    # Calculate profit
    for n in range(n_data):
        profit[n] = (6*product[n]*7-price[n]-cost[n])
    # Make Decision    
    decision_set = []
    for t in tqdm(range(max_trial),desc="Burte Forcing"):
        options = [i for i in range(n_data)]
        decision = np.zeros(n_data)
        sum_price = 0
        sum_product = 0
        sum_profit = 0
        for _ in range(n_data):
            selected_index = random.choice(options)
            if sum_price + price[selected_index] > price_max: 
                options.remove(selected_index)
                break # Burst
            else: 
                sum_price = sum_price + price[selected_index]
                sum_product = sum_product + product[selected_index]
                sum_profit = sum_profit + profit[selected_index]
                decision[selected_index] = 1
                options.remove(selected_index)
        decision_set.append({
            "trial":t, 
            "decision":copy.deepcopy(decision), 
            "price":sum_price, 
            "product":sum_product, 
            "profit":sum_profit})
    # Select Decision
    from operator import itemgetter
    sorted_decision = sorted(decision_set,key=itemgetter('profit'),reverse=True)
    # for i in range(5):
    #     print(sorted_decision[i])
    final_decision = sorted_decision[0]["decision"]
    final_profit = sorted_decision[0]["profit"]
    final_price = sorted_decision[0]["price"]
    final_product = sorted_decision[0]["product"]

    print("Num of Prod  : {0}/{1}".format(sum(final_decision),n_data))
    print("Total Price  : {0}/{1} - {2} %".format(final_price,price_max,round((final_price/price_max)*100,2)))
    print("Total Product: {0}".format(final_product))
    print("Total Profit : {0}".format(final_profit))
    return decision
