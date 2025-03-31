from typing import Callable
import matplotlib.pyplot as plt
import random

import os
import sys
sys.path.append(os.getcwd())
from scratch.linear_algebra import Vector, dot, distance, add, scalar_multiply, vector_mean

def sum_of_squares(v: Vector) -> float:
    """vの各要素の二乗を合計する"""
    return dot(v, v)

def difference_quotient(f: Callable[[float], float],
                        x: float,
                        h: float) -> float:
    return (f(x + h) - f(x)) / h

def square(x: float) -> float:
    return x * x

def derivative(x: float) -> float:
    return 2 * x

xs = range(-10, 11)
actuals = [derivative(x) for x in xs]
estimates = [difference_quotient(square, x, h=0.001) for x in xs]

plt.figure()
plt.title("Actual Derivative vs. Estimates")
plt.plot(xs, actuals, 'rx', label='Actual')
plt.plot(xs, estimates, 'b+', label='Estimate')
plt.legend(loc=9)
plt.savefig('../picture/chap8-1.png')
plt.close()

def partial_difference_quotient(f: Callable[[Vector], float],
                                v: Vector,
                                i: int,
                                h: float) -> float:
    """関数fと変数ベクトルvに対するi番目の差分商を返す"""
    w = [v_j + (h if j == i else 0) for j, v_j in enumerate(v)]
    return (f(w) - f(v)) / h

def estimate_gradient(f: Callable[[Vector], float],
                      v: Vector,
                      h: float=0.0001) -> list[float]:
    return [partial_difference_quotient(f, v, i, h) for i in range(len(v))]

def gradient_step(v: Vector, gradient: Vector, step_size: float) -> Vector:
    """vからgradient方向にstep_size移動する"""
    assert len(v) == len(gradient)
    step = scalar_multiply(step_size, gradient)
    return add(v, step)

def sum_of_squares_gradient(v: Vector) -> Vector:
    return [2 * v_i for v_i in v]

# 開始点を無作為に選択する
v = [random.uniform(-10, 10) for i in range(3)]

for epoch in range(1000):
    grad = sum_of_squares_gradient(v) # vにおける勾配を計算する
    v = gradient_step(v, grad, -0.01) # 勾配の負数分移動する
    print(epoch, v)
    
assert distance(v, [0, 0, 0]) < 0.001 # vは0に近くなければならない

# xの範囲は-50から49。yは常に20 * x + 5
inputs = [(x, 20 * x + 5) for x in range(-50, 50)]

def linear_gradient(x: float, y: float, theta: Vector) -> Vector:
    slope, intercept = theta
    predicted = slope * x + intercept # モデルの予測
    error = (predicted - y)           # 誤差は（予測 - 実際の値）
    squared_error = error ** 2        # 勾配を使って
    grad = [2 * error * x, 2 * error] # 二乗誤差を最小にする
    return grad

# ランダムな傾きと切片でスタートする
theta = [random.uniform(-1, 1), random.uniform(-1, 1)]
learning_rate = 0.001

for epoch in range(5000):
    # 勾配の平均を求める
    grad = vector_mean([linear_gradient(x, y, theta) for x, y in inputs])
    # 勾配の方向に沿って進む
    theta = gradient_step(theta, grad, -learning_rate)
    print(epoch, theta)
    
slope, intercept = theta
assert 19.9 <  slope < 20.1
assert 4.9 < intercept <5.1