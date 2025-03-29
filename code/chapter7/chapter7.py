import math
import random
import matplotlib.pyplot as plt

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

hi = normal_upper_bound(0.95, mu_0, sigma_0)
print(hi)
type_2_probability  = normal_probability_below(hi, mu_1, sigma_1)
power = 1 - type_2_probability
print(power)

def two_sided_p_value(x: float, mu: float=0, sigma: float=1) -> float:
    """
    値がN(mu, sigma)に従うとして、（上側でえも下側でも）
    極端なxが現れる可能性はどの程度か
    """
    if x >= mu:
        # xが平均より大きい場合、テイル確率はxより大きい分
        return 2 * normal_probability_above(x, mu, sigma)
    else:
        # xが平均より小さい場合、テイル確率はxより小さい分
        return 2 * normal_probability_below(x, mu, sigma)
    
print(two_sided_p_value(529.5, mu_0, sigma_0))


extreme_value_count = 0
for _ in range(1000):
    num_heads = sum(1 if random.random() < 0.5 else 0
                    for _ in range(1000))

    if num_heads >= 530 or num_heads <= 470:
        extreme_value_count += 1
# p値は0.062 => 1000回のうち62回は極端な値
# assert 59 < extreme_value_count < 65, f"{extreme_value_count}"

print(two_sided_p_value(531.5, mu_0, sigma_0))

p_hat = 525 / 1000
mu = p_hat
sigma = math.sqrt(p_hat*(1-p_hat) / 1000)
print(normal_two_sided_bounds(0.95, mu, sigma))

p_hat = 540 / 1000
mu = p_hat
sigma = math.sqrt(p_hat*(1-p_hat) / 1000)
print(normal_two_sided_bounds(0.95, mu, sigma))

def run_experiment() ->  list[bool]:
    """歪みのないコインを1000回投げて表が出たらTrue、裏はFalseとする"""
    return [random.random() < 0.5 for _ in range(1000)]

def reject_fairness(experiment: list[bool]) -> bool:
    """5%を有意水準を用いる"""
    num_heads = len([flip for flip in experiment if flip])
    return num_heads < 469 or num_heads > 531

random.seed(0)
experiments = [run_experiment() for _ in range(1000)]
num_rejections = len([experiment for experiment in experiments if reject_fairness(experiment)])

print(num_rejections)
assert num_rejections == 46

def estimated_parameters(N: int, n: int) -> tuple[float, float]:
    p = n / N
    sigma = math.sqrt(p*(1-p)/N)
    return p, sigma

def a_b_test_statistic(N_A: int, n_A: int, N_B: int, n_B: int) -> float:
    p_A, sigma_A = estimated_parameters(N_A, n_A)
    p_B, sigma_B = estimated_parameters(N_B, n_B)
    return (p_B - p_A) / math.sqrt(sigma_A**2 + sigma_B**2)

z = a_b_test_statistic(1000, 200, 1000, 180)
print(z)
print(two_sided_p_value(z))

z = a_b_test_statistic(1000, 200, 1000, 150)
print(z)
print(two_sided_p_value(z))

# コインの表裏で考えた場合、alphaが表の確率（割合）、betaが裏の確率（割合）といえる。
# 試行が増えるごとにalphaに表が出た回数を、betaに裏が出た回数を追加していく。
# コインの裏表がでる割合が試行が進むごとに真の割合に近似していく
def B(alpha: float, beta: float) -> float:
    """確率の総和が１となるように定数で正規化する"""
    return math.gamma(alpha) * math.gamma(beta) / math.gamma(alpha+beta)

def beta_pdf(x: float, alpha: float, beta: float) -> float:
    if x <= 0 or x >= 1:
        return 0
    return x ** (alpha-1) * (1-x) ** (beta-1) / B(alpha, beta)

xs = [i/100 for i in range(1, 100)]
y = [beta_pdf(x, alpha=1, beta=1) for x in xs]
plt.figure()
plt.plot(xs, [beta_pdf(x, alpha=1, beta=1) for x in xs], '-',  label='Beta(1, 1)')
plt.plot(xs, [beta_pdf(x, alpha=10, beta=10) for x in xs], '--', label='Beta(10, 10)')
plt.plot(xs, [beta_pdf(x, alpha=4, beta=16) for x in xs], ':',  label='Beta(4, 16)')
plt.plot(xs, [beta_pdf(x, alpha=16, beta=4) for x in xs], '-.', label='Beta(16, 4)')
plt.xlim(0, 1)
plt.ylim(0, 5)
plt.legend(loc=0)
plt.savefig('../picture/chap7.png')