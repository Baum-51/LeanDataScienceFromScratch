import os, sys
sys.path.append(os.getcwd())
from collections import Counter

def raw_majority_vote(labels: list[str]) -> str:
    votes = Counter(labels)
    winner, _ = votes.most_common(1)[0]
    return winner

assert raw_majority_vote(['a', 'b', 'c', 'b']) == 'b'

def majority_vote(labels: list[str]) -> str:
    """ラベルは近いものから遠いものへ整列していると想定する"""
    vote_counts = Counter(labels)
    winner, winner_count = vote_counts.most_common(1)[0]
    num_winner = len([count
                      for count in vote_counts.values()
                      if count == winner_count])
    if num_winner == 1:
        return winner                     # 唯一の多数が決まったため、結果とする
    else:
        return majority_vote(labels[:-1]) # 最も遠いものを除外して、再度試す

# 同点になるため４番目までを採用して'b'が選ばれる
assert majority_vote(['a', 'b', 'c', 'b', 'a']) == 'b'

from typing import NamedTuple
from scratch.linear_algebra import Vector, distance

class LabeledPoint(NamedTuple):
    point: Vector
    label: str
    
def knn_classify(k: int,
                 labeled_points: list[LabeledPoint],
                 new_point: Vector) -> str:
    # ラベル付きデータポイントを近いものから順に並べる
    by_distance = sorted(labeled_points, key=lambda lp: distance(lp.point, new_point))
    
    # 近い順にk個取り出す
    k_nearest_labels = [lp.label for lp in by_distance[:k]]
    
    # 多数決を行う
    return majority_vote(k_nearest_labels)

import requests
data = requests.get(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
)

with open('./other_text/iris.data', 'w') as f:
    f.write(data.text)
    
import csv
from collections import defaultdict

def parse_iris_row(row: list[str]) -> LabeledPoint:
    """
    sepal_length, sepal_width, petal_length, petal_width, class
    """
    measurements = [float(value) for value in row[:-1]]
    # 品種の、例えば"Iris-virginica"のうち、"virginica"の部分だけ使用する
    label = row[-1].split('-')[-1]
    return LabeledPoint(measurements, label)

with open('./other_text/iris.data') as f:
    reader = csv.reader(f)
    iris_data = [parse_iris_row(row) for row in reader if row]
    
# また、データポイントを品種（ラベル）別にグループ化してプロットできるようにする
points_by_species: dict[str, list[Vector]] = defaultdict(list)
for iris in iris_data:
    points_by_species[iris.label].append(iris.point)

import matplotlib.pyplot as plt
metrics = ['sepal length', 'sepal width', 'petal length', 'petal width']
pairs = [(i, j) for i in range(4) for j in range(4) if i < j]
marks = ['+', '.', 'x'] # 品種が３つのため、マーカーも３つ

fig, ax = plt.subplots(2, 3)
for row in range(2):
    for col in range(3):
        i, j = pairs[3 * row + col]
        ax[row][col].set_title(f'{metrics[i]} vs {metrics[j]}', fontsize=8)
        ax[row][col].set_xticks([])
        ax[row][col].set_yticks([])
        
        for mark, (species, points) in zip(marks, points_by_species.items()):
            xs = [point[i] for point in points]
            ys = [point[j] for point in points]
            ax[row][col].scatter(xs, ys, marker=mark, label=species)
            
ax[-1][-1].legend(loc='lower right', prop={'size': 6})
plt.savefig('../picture/chap12-1.png')

import random
from chapter11.chapter11 import split_data

random.seed(12)
iris_train, iris_test = split_data(iris_data, 0.70)
assert len(iris_train) == 0.7 * 150
assert len(iris_test)  == 0.3 * 150

# (予測, 実際)の回数を記録する
confusion_matrix: dict[tuple[str, str], int] = defaultdict(int)
num_correct = 0

for iris in iris_test:
    predicted = knn_classify(5, iris_train, iris.point)
    actual = iris.label
    
    if predicted == actual:
        num_correct += 1
    confusion_matrix[(predicted, actual)] += 1
    
pct_correct = num_correct / len(iris_test)
print(pct_correct, confusion_matrix)

def random_point(dim: int) -> Vector:
    return [random.random() for _ in range(dim)]

def random_distances(dim: int, num_pairs: int):
    return [distance(random_point(dim), random_point(dim))
            for _ in range(num_pairs)]

import tqdm
dimensions = range(1, 101)

avg_distances = []
min_distances = []

random.seed(0)
for dim in tqdm.tqdm(dimensions, desc='Curse of Dimensionality'):
    distances = random_distances(dim, 10000)     # 10,000個の無作為点間の距離
    avg_distances.append(sum(distances) / 10000) # 平均を記録する
    min_distances.append(min(distances))         # 最低値を記録する
    
plt.figure()
plt.plot(dimensions, avg_distances, label='average distance')
plt.plot(dimensions, min_distances, label='minimum distance')
plt.xlabel('# of dimenstions')
plt.grid()
plt.legend()
plt.savefig('../picture/chap12-2.png')
plt.close()

min_avg_ratio = [min_dist / avg_dist for min_dist, avg_dist in zip(min_distances, avg_distances)]
plt.figure()
plt.plot(dimensions, min_avg_ratio)
plt.xlabel('# of dimenstions')
plt.grid()
plt.savefig('../picture/chap12-3.png')
plt.close()