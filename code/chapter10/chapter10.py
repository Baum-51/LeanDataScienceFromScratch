from collections import Counter
import math

import matplotlib.pyplot as plt

def bucketize(point: float, bucket_size: float) -> float:
    """pointの値を切り捨ててバケツの下限の値を揃える"""
    return bucket_size * math.floor(point / bucket_size)

def make_histogram(points: list[float], bucket_size: float) -> dict[float, int]:
    """pointをバケツに入れ、何個入ったか数える"""
    return Counter(bucketize(point, bucket_size) for point in points)

def plot_histogram(points: list[float], bucket_size: float, title: str=''):
    histogram = make_histogram(points, bucket_size)
    plt.bar(histogram.keys(), histogram.values(), width=bucket_size)
    plt.title(title)
    
import random
import os
import sys
sys.path.append(os.getcwd())
from scratch.probability import inverse_normal_cdf

random.seed(0)

# -100から100までの一様分布
uniform = [200 * random.random() - 100 for _ in range(10000)]

# 平均0、標準偏差57の正規分布
normal = [57 * inverse_normal_cdf(random.random()) for _ in range(10000)]

plt.figure()
plot_histogram(uniform, 10, "Uniform Histogram")
plt.savefig('../picture/chap10-1.png')
plt.close()

plt.figure()
plot_histogram(normal, 10, "Uniform Histogram")
plt.savefig('../picture/chap10-2.png')
plt.close()

def random_normal() -> float:
    """標準正規分布に従う無作為の数を返す"""
    return inverse_normal_cdf(random.random())

xs = [random_normal() for _ in range(1000)]
ys1 = [ x + random_normal() / 2 for x in xs]
ys2 = [-x + random_normal() / 2 for x in xs]

plt.figure()
plt.scatter(xs, ys1, marker='.', color='black', label='ys1')
plt.scatter(xs, ys2, marker='.', color='gray',  label='ys2')
plt.xlabel('xs')
plt.ylabel('ys')
plt.legend(loc=9)
plt.title('Very Different Joint Distributions')
plt.savefig('../picture/chap10-3.png')
plt.close()

from scratch.statistics import correlation

print(correlation(xs, ys1))
print(correlation(xs, ys2))

from scratch.linear_algebra import Matrix, Vector, make_matrix

def correlation_matrix(data: list[Vector]) -> Matrix:
    """
    (i, j)番目のエントリがdata[i]とdata[j]の相関となるような
    len(data) x len(data)行列を返す
    """
    def correlation_ij(i: int, j: int) -> float:
        return correlation(data[i], data[j])
    
    return make_matrix(len(data), len(data), correlation_ij)

# corr_dataは、4つの100次元ベクトルのリスト
from scratch.working_with_data import corr_data
num_vectors = len(corr_data)
fig, ax = plt.subplots(num_vectors, num_vectors)

for i in range(num_vectors):
    for j in range(num_vectors):
        # x軸のcolumn_j、y軸のcolumn_iの位置に散布図を描画する
        if i != j: ax[i][j].scatter(corr_data[j], corr_data[i])
        # i == jであれば、列名を表示する
        else: ax[i][j].annotate("series " + str(i), (0.5, 0.5),
                                xycoords="axes fraction",
                                ha="center", va="center")
        # 左端と一番下のサブプロット以外は、軸ラベルを表示しない
        if i < num_vectors - 1: ax[i][j].xaxis.set_visible(False)
        if j > 0: ax[i][j].yaxis.set_visible(False)
        
# 右下と左上のサブプロットは、テキストのみ表示しているため、
# 軸ラベルが誤っている。ここで正しく修正する
ax[-1][-1].set_xlim(ax[0][-1].get_xlim())
ax[0][0].set_ylim(ax[0][1].get_xlim())

fig.savefig('../picture/chap10-4.png')