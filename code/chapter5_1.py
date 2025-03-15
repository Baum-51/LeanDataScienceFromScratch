from collections import Counter
import matplotlib.pyplot as plt
from typing import List

import math
from chapter4.chapter4_1 import sum_of_squares

num_friends = [100.0,49,41,40,25,21,21,19,19,18,18,16,15,15,15,15,14,14,13,13,13,13,12,12,11,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,8,8,8,8,8,8,8,8,8,8,8,8,8,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]


friend_counts = Counter(num_friends)
xs = range(101)
ys = [friend_counts[x] for x in xs]
plt.bar(xs, ys)
plt.axis([0, 101, 0, 25])
plt.title("Histogram of Friends Counts")
plt.xlabel("# of friends")
plt.ylabel("# of people")
plt.savefig('/app/picture/chap5-1.png')

num_points = len(num_friends)
print('num_points    : ', num_points)
largest_value = max(num_friends)
print('largest_value : ', largest_value)
smallest_value = min(num_friends)
print('smallest_value: ', smallest_value)

sorted_values = sorted(num_friends)
smallest_value = sorted_values[0]
print('sorted smallest_value: ', smallest_value)

second_smallest_value = sorted_values[1]
print('second smallest_value: ', second_smallest_value)
second_largest_value = sorted_values[-2]
print('second largest value : ', second_largest_value)

def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs)

mean(num_friends)
print(mean(num_friends))

# 関数名のアンダースコアは、これが「プライベート」関数であることを示している
# これらは、median関数によって呼び出されることを目的としているが
# 我々の統計ライブラリを使用する人から呼び出されることをいとしていない
def _median_odd(xs: List[float]) -> float:
    """len(xs)が偶数の場合、中央値は中央の要素"""
    return sorted(xs)[len(xs) // 2]
def _median_even(xs: List[float]) -> float:
    """len(xs)が奇数の場合、中央値は中央の２つの要素の平均"""
    sorted_xs = sorted(xs)
    hi_midpoint = len(xs) // 2
    return (sorted_xs[hi_midpoint - 1] + sorted_xs[hi_midpoint]) / 2
def median(v: List[float]) -> float:
    """vの中央値を求める"""
    return _median_even(v) if len(v) % 2 == 0 else _median_odd(v)

assert median([1, 10, 2, 9, 5]) == 5
assert median([1, 9, 2, 10]) == (2 + 9) / 2

print(median(num_friends))

def quantile(xs: List[float], p: float) -> float:
    """x中のp百分位数を返す"""
    p_index = int(p * len(xs))
    return sorted(xs)[p_index]

assert quantile(num_friends, 0.10) == 1
assert quantile(num_friends, 0.25) == 3
assert quantile(num_friends, 0.75) == 9
assert quantile(num_friends, 0.90) == 13

def mode(x: List[float]) -> List[float]:
    """モードは１つとは限らないため、リストを返す"""
    counts = Counter(x)
    max_count = max(counts.values())
    return [x_i for x_i, count in counts.items() if count == max_count]

assert set(mode(num_friends)) == {1, 6}

def data_range(xs: List[float]) -> float:
    return max(xs) - min(xs)

assert data_range(num_friends) == 99

def de_mean(xs: List[float]) -> List[float]:
    """xを変換して、xとxの平均との差とする（結果の平均が0となるように）"""
    x_bar = mean(xs)
    return [x - x_bar for x in xs]

def variance(xs: List[float]) -> float:
    """平均との差のおおよその二乗平均"""
    assert len(xs) >= 2
    n = len(xs)
    deviations = de_mean(xs)
    return sum_of_squares(deviations) / (n -1)

assert 81.54 < variance(num_friends) < 81.55

def standard_deviations(xs: List[float]) -> float:
    """標準偏差は分散の平方根"""
    return math.sqrt(variance(xs))

assert 9.02 < standard_deviations(num_friends) < 9.04

def interquartile_range(xs: List[float]) -> float:
    """75パーセンタイルと25パーセンタイルの差を返す"""
    return quantile(xs, 0.75) - quantile(xs, 0.25)
assert interquartile_range(num_friends) == 6