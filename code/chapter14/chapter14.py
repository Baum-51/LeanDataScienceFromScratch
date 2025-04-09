import os, sys
sys.path.append(os.getcwd())

def predict(alpha: float, beta: float, x_i: float) -> float:
    return beta * x_i + alpha

def error(alpha:float, beta: float, x_i: float, y_i: float) -> float:
    """
    実際の値がy_i出会った場合の
    予測値beta * x_i + alphaからの誤差
    """
    return predict(alpha, beta, x_i) - y_i

from scratch.linear_algebra import Vector

def sum_of_sqerrors(alpha: float, beta: float, x: Vector, y: Vector) -> float:
    return sum(error(alpha, beta, x_i, y_i) ** 2 for x_i, y_i in zip(x, y))

from scratch.statistics import correlation, standard_deviation, mean

def least_squares_fit(x: Vector, y: Vector) -> tuple[float, float]:
    """
    与えられた2つのベクトルxとyから
    alphaとbetaの最小二乗値を求める
    """
    beta = correlation(x, y) * standard_deviation(y) / standard_deviation(x)
    alpha = mean(y) - beta * mean(x)
    return alpha, beta

x = [i for i in range(-100, 100, 10)]
y = [3 * i - 5 for i in x]

# y = 3x - 5となるべき
assert least_squares_fit(x, y) == (-5, 3)

from scratch.statistics import num_friends_good, daily_minutes_good

alpha, beta = least_squares_fit(num_friends_good, daily_minutes_good)
assert 22.9 < alpha < 23.0
assert 0.9 < beta < 0.905

from scratch.statistics import de_mean

def total_sum_of_squares(y: Vector) -> float:
    """y_iと平均の差の二乗総和"""
    return sum(v ** 2 for v in de_mean(y))

def r_squared(alpha: float, beta: float, x: Vector, y: Vector) -> float:
    """
    モデルから得られたyの場合は、(1 - モデルから得られなかったyの場合)に等しい
    """
    return 1.0 - (sum_of_sqerrors(alpha, beta, x, y) / total_sum_of_squares(y))

rsq = r_squared(alpha, beta, num_friends_good, daily_minutes_good)
assert 0.328 < rsq < 0.330

import random
import tqdm
from scratch.gradient_descent import gradient_step

num_epochs = 10000
random.seed(0)

guess = [random.random(), random.random()] # 開始点をランダムに設定

learning_rate = 0.00001

with tqdm.trange(num_epochs) as t:
    for _ in t:
        alpha, beta = guess
        
        # alphaに関する損失の偏微分
        grad_a = sum(2 * error(alpha, beta, x_i, y_i) for x_i, y_i in zip(num_friends_good, daily_minutes_good))
        
        # betaに関する損失の偏微分
        grad_b = sum(2 * error(alpha, beta, x_i, y_i) * x_i for x_i, y_i in zip(num_friends_good, daily_minutes_good))
        
        # tqdmに表示する損失値の計算
        loss = sum_of_sqerrors(alpha, beta, num_friends_good, daily_minutes_good)
        t.set_description(f"loss: {loss: .3f}")
        
        # 最後に予測値を更新
        guess = gradient_step(guess, [grad_a, grad_b], -learning_rate)
        
# ほぼ同じ結果が得られるべき
alpha, beta = guess
assert 22.9 < alpha < 23.0
assert 0.9  < beta  < 0.905