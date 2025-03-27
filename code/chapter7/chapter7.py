import math

import os
import sys
sys.path.append(os.getcwd())
from scratch.probability import normal_cdf, inverse_normal_cdf

def normal_approximation_to_binomial(n: int, p: float) -> tuple[float, float]:
    """Binomial(n, p)に相当するμとσを計算する"""
    mu = p * n
    sigma = math.sqrt(p * (1 - p) * n)
    return mu, sigma

# 変数が閾値を下回る確率はnormal_cdfで表せる
normal_probability_below = normal_cdf

# 閾値を下回っていなければ、閾値より上にある
def normal_probability_above(lo: float,
                             mu: float=0,
                             sigma: float=1) -> float:
    """N(mu, sigma)がloよりも大きい確率"""
    return 1 - normal_cdf(lo, mu, sigma)

# hiより小さく、loより大きければ、値はその間にある
def normal_probability_between(lo: float,
                               hi: float,
                               mu: float=0,
                               sigma: float=1) -> float:
    """N(mu, sigma)がloとhiの間にある確率"""
    return normal_cdf(hi, mu, sigma) - normal_cdf(lo, mu, sigma)

# 間になければ、範囲外にある
def normal_probability_outside(lo: float,
                               hi: float,
                               mu: float=0,
                               sigma: float=1) -> float:
    """N(mu, sigma)がloとhiの間にない確率"""
    return 1 - normal_probability_between(lo, hi, mu, sigma)

def normal_upper_bound(probability: float,
                       mu: float=0,
                       sigma: float=1) -> float:
    """確率P(Z<=z)となるzを返す"""
    return inverse_normal_cdf(probability, mu, sigma)

def normal_lower_bound(probability: float,
                       mu: float=0,
                       sigma: float=1) -> float:
    """確率P(Z>=z)となるzを返す"""
    return inverse_normal_cdf(1-probability, mu, sigma)

def normal_two_sided_bounds(probability: float,
                            mu: float=0,
                            sigma: float=1) -> tuple[float, float]:
    """1指定された確率を包含する（平均を中心に）対称な境界を返す"""
    tail_probability = (1 - probability) / 2
    # 上側の境界はテイル確率(tail_probability)分上に
    upper_bound = normal_lower_bound(tail_probability, mu, sigma)
    # 下側の境界はテイル確率(tail_probability)分下に
    lower_bound = normal_upper_bound(tail_probability, mu, sigma)
    
    return lower_bound, upper_bound

mu_0, sigma_0 = normal_approximation_to_binomial(1000, 0.5)
lower_bound, upper_bound = normal_two_sided_bounds(0.95, mu_0, sigma_0)
print(lower_bound, upper_bound)

# pが0.5であると想定のもとで、95%の境界を確認する
lo, hi = normal_two_sided_bounds(0.95, mu_0, sigma_0)

# p = 0.55出会った場合の、μとσを計算する
mu_1, sigma_1 = normal_approximation_to_binomial(1000, 0.55)

# 第二種の過誤とは、帰無仮説を棄却しないという誤りがあり
# Xが当初想定の領域に入っている場合に生じる
type_2_probability = normal_probability_between(lo, hi, mu_1, sigma_1)
power = 1 - type_2_probability
print(power)

hi = normal_upper_bound(0.95, mu_0, mu_0)
type_2_probability  = normal_probability_below(hi, mu_1, sigma_1)
power = 1 - type_2_probability
print(power)