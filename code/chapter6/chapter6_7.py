import random
from collections import Counter
import matplotlib.pyplot as plt
import math
from chapter6_6 import normal_cdf

def bernoulli_trial(p: float) -> int:
    """確率pとして1、確率1-pとして0を返す"""
    return 1 if random.random() < p else 0
def binomial(n: int, p: float) -> int:
    """n回bernoulli(p)試行の合計を返す"""
    return sum(bernoulli_trial(p) for _ in range(n))

def binomial_histogram(p: float, n: int, num_points: int) -> None:
    """binomial(n, p)を収集し、それらのヒストグラムをプロットする"""
    data = [binomial(n, p) for _ in range(num_points)]
    
    # 二項分布を棒グラフでプロットする
    histogram = Counter(data)
    plt.figure()
    plt.bar([x - 0.4 for x in histogram.keys()],
            [v / num_points for v in histogram.values()],
            0.8,
            color='0.75')
    mu = p * n
    sigma = math.sqrt(n * p * (1-p))
    
    # 正規分布の近似を折れ線グラフでプロットする
    xs = range(min(data), max(data) + 1)
    ys = [normal_cdf(i + 0.5, mu, sigma) - normal_cdf(i - 0.5, mu, sigma) for i in xs]
    
    plt.plot(xs, ys)
    plt.title("Binomial Distribution vs. Normal Approximation")
    plt.savefig('../picture/chap6-3.png')
    
binomial_histogram(0.75, 100, 10000)