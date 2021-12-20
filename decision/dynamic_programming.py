import numpy as np
import pandas as pd

from config import COST_COLUMN_NAME, PREDICT_COLUMN_NAME, PRICE_COLUMN_NAME

def dynamic_programming(info,price_max:float,scale=1000)->list:
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
    #N, K = map(int, input().split())
    N = n_data
    K = int(price_max/scale)
    stuff = [[0,0]]
    knapsack = [[0 for _ in range(K + 1)] for _ in range(N + 1)]
    for i in range(N):
        stuff.append([int(price[i]/scale),profit[i]])

    # 0-1 Knapsack Problem
    for i in tqdm(range(1,N+1),desc="DP First Loop"):
        for j in tqdm(range(1,K+1),desc="DP Second Loop",leave=False):
            weight = stuff[i][0]
            value = stuff[i][1]
            if j < weight:
                knapsack[i][j] = knapsack[i - 1][j] #weight보다 작으면 위의 값을 그대로 가져온다
            else:
                knapsack[i][j] = max(value + knapsack[i - 1][j - weight], knapsack[i - 1][j])

    print("Max Profit - DP without Back-Tracking: ", knapsack[N][K])
    #np.savetxt("./DP.csv",np.array(knapsack[N]),delimiter=',')

class knapsack_back_tracking:
    def __init__(self,weight_limit,weights:list,profits:list) -> None:
        self.n = len(weights) # number of stuff
        self.W = weight_limit # Limit of weight
        self.w = np.insert(weights,0,0) # weight of each stuff
        self.p = np.insert(profits,0,0) # profit of each stuff

        self.max_profit = 0 # current max profit
        self.num_best = 0 # current best number of stuff 
        self.best_set = [] # current best subset of stuff
        self.include = [False]*(self.n+1) # current best subset of stuff

    def knapsack3 (self, i, profit, weight):
        if (weight <= self.W and profit > self.max_profit):
            self.max_profit = profit
            self.num_best = i
            self.best_set = self.include[:]
        if (self.promising(i, profit, weight)):
            self.include[i + 1] = True
            self.knapsack3(i + 1, profit + self.p[i+1], weight + self.w[i+1])
            self.include[i + 1] = False
            self.knapsack3(i + 1, profit, weight)

    def promising (self, i, profit, weight):
        if (weight > self.W):
            return False
        else:
            j = i + 1
            bound = profit
            totweight = weight
            while (j <= self.n and totweight + self.w[j] <= self.W):
                totweight += self.w[j]
                bound += self.p[j]
                j += 1
            k = j
            if (k <= self.n):
                bound += (self.W - totweight) * self.p[k] / self.w[k]
            return bound > self.max_profit

def dynamic_programming_back_tracking(info,price_max:float,scale=1000)->list:
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

    # Calculate profit and Filter Positive profits
    index_pos = []
    profit_positive = []
    price_positive = []
    for n in range(n_data):
        if profit[n] > 0:
            profit_positive.append(profit[n])
            price_positive.append(price[n])
            index_pos.append(n)

    DP = knapsack_back_tracking(
        int(price_max/scale),
        np.array(price_positive)/scale,
        np.array(profit_positive))
    DP.knapsack3(0,0,0)
    print("Max Profit - DP with Back-Tracking: ", DP.max_profit)

    decision = np.zeros(n_data)
    n_positive = len(index_pos)
    for idx in range(n_positive):
        if DP.best_set[idx+1]:
            decision[index_pos[idx]] = 1

    # Evaluate Numbers
    sum_price = 0
    sum_product = 0
    sum_profit = 0
    for idx in range(n_data):
        if decision[idx] == 1:
            sum_price += price[idx]
            sum_product += product[idx]
            sum_profit += profit[idx]
    print("Num of Prod  : {0}/{1}".format(sum(decision),n_data))
    print("Total Price  : {0}/{1} - {2} %".format(sum_price,price_max,round((sum_price/price_max)*100,2)))
    print("Total Product: {0}".format(sum_product))
    print("Total Profit : {0}".format(sum_profit))

    return decision