import os, sys
sys.path.append(os.getcwd())
from scratch.linear_algebra import Vector
import random

def num_differences(v1: Vector, v2: Vector) -> int:
    assert len(v1) == len(v2)
    return len([x1 for x1, x2 in zip(v1, v2) if x1 != x2])

assert num_differences([1, 2, 3], [2, 1, 3]) == 2
assert num_differences([1, 2], [1, 2]) == 0

from scratch.linear_algebra import vector_mean

def cluster_means(k: int,
                  inputs: list[Vector],
                  assignments: list[int]) -> list[Vector]:
    # cluster[i]にはクラスタiに割り当てられた入力が入る
    clusters = [[] for i in range(k)]
    for input, assignment in zip(inputs, assignments):
        clusters[assignment].append(input)
    # クラスタがからの場合には、ランダムなポイントを使用する
    return [vector_mean(cluster) if cluster else random.choice(inputs) for cluster in clusters]

import itertools
import tqdm
from scratch.linear_algebra import squared_distance

class KMeans:
    def __init__(self, k: int) -> None:
        self.k = k        # クラスの数
        self.means = None

    def classify(self, input: Vector) -> int:
        """入力に最も近いクラスタのインデックスを返す"""
        return min(range(self.k), key=lambda i: squared_distance(input, self.means[i]))
    
    def train(self, inputs: list[Vector]) -> None:
        # ランダムポイントから開始する
        assignments = [random.randrange(self.k) for _ in inputs]
        with tqdm.tqdm(itertools.count()) as t:
            for _ in t:
                # 中心を計算し、新しいクラスタの割り当てを行う
                self.means = cluster_means(self.k, inputs, assignments)
                new_assignments = [self.classify(input) for input in inputs]
                
                # いくつか割り当てが変更されたかを数え、完了したか確認する
                num_changed = num_differences(assignments, new_assignments)
                if num_changed == 0:
                    return 
                
                # そうでない場合、新しい割り当てを使って、中心を再計算する
                assignments = new_assignments
                self.means = cluster_means(self.k, inputs, assignments)
                t.set_description(f"changed: {num_changed} / {len(inputs)}")

from matplotlib import pyplot as plt
def squared_clustering_errors(inputs: list[Vector], k: int) -> float:
    """k平均法のクラスタの誤差二乗和を求める"""
    clusterer = KMeans(k)
    clusterer.train(inputs)
    means = clusterer.means
    assignments = [clusterer.classify(input) for input in inputs]
    
    return sum(squared_distance(input, means[cluster]) for input, cluster in zip(inputs, assignments))

def recolor(pixel: Vector) -> Vector:
    cluster = clusterer.classify(pixel) # ピクセルが分離されたクラスタのインデックス
    return clusterer.means[cluster]     # クラスタの平均値

def recolor_cluster():
    image_path = r"../picture/夕日とバンシー.jpg"
    import matplotlib.image as mpimg
    img = mpimg.imread(image_path) / 256
    
    top_row = img[0]
    top_left_pixel = top_row[0]
    red, green, blue = top_left_pixel
    
    # .tolist()はNumpy配列をPythonのリストに変換する
    pixels = [pixel.tolist() for row in img for pixel in row]
    
    clusterer = KMeans(5)
    clusterer.train(pixels) # 計算に時間がかかる
    
    new_img = [[recolor(pixel) for pixel in row] for row in img]
    
    plt.figure()
    plt.imshow(new_img)
    plt.axis('off')
    plt.savefig('recolor_夕日とバンシー.jpg')

from typing import NamedTuple, Union

class Leaf(NamedTuple):
    value: Vector

class Merged(NamedTuple):
    children: tuple
    order: int

leaf1 = Leaf([10, 20])
leaf2 = Leaf([30, -15])
    
merged = Merged((leaf1, leaf2), order=1)
Cluster = Union[Leaf, Merged]

def get_values(cluster: Cluster) -> list[Vector]:
    if isinstance(cluster, Leaf):
        return [cluster.value]
    else:
        return [value
                for child in cluster.children
                for value in get_values(child)]
        
assert get_values(merged) == [[10, 20], [30, -15]]

from typing import Callable
from scratch.linear_algebra import distance

def cluster_distance(cluster1: Cluster,
                     cluster2: Cluster,
                     distance_agg: Callable = min) -> float:
    """
    cluster1とcluster2の間の各要素間の距離を計算し、
    結果のリストに集計関数_distance_agg_を適用する
    """
    return distance_agg([distance(v1, v2)
                         for v1 in get_values(cluster1)
                         for v2 in get_values(cluster2)])
    
def get_merge_order(cluster: Cluster) -> float:
    if isinstance(cluster, Leaf):
        return float('inf') # 結合されたものではない
    else:
        return cluster.order

def get_children(cluster: Cluster):
    if isinstance(cluster, Leaf):
        raise TypeError("Leaf has no children")
    else:
        return cluster.children
    
def bottom_up_cluster(inputs: list[Vector],
                      distance_agg: Callable=min) -> Cluster:
    # すべて末端クラスタとして開始する
    clusters: list[Cluster] = [Leaf(input) for input in inputs]
    
    def pair_distance(pair: tuple[Cluster, Cluster]) -> float:
        return cluster_distance(pair[0], pair[1], distance_agg)
    
    # 1つ以上のクラスタが残っている限り続ける
    while len(clusters) > 1:
        # 2つの近傍クラスタを選択
        c1, c2 = min(((cluster1, cluster2)
                     for i, cluster1 in enumerate(clusters)
                     for cluster2 in clusters[:i]),
                     key=pair_distance)
        
        # それらをクラスタのリストから取り除く
        clusters = [c for c in clusters if c != c1 and c != c2]
        
        # 結合順をクラスタのリストから取り除く
        merged_cluster = Merged((c1, c2), order=len(clusters))
        
        # 結合したクラスタをリストに加える
        clusters.append(merged_cluster)
    
    # クラスタ数が1になったので、それを返す
    return clusters[0]

def generate_clusters(base_cluster: Cluster,
                      num_clusters: int) -> list[Cluster]:
    # 元のクラスタをリストにしてから開始する
    clusters = [base_cluster]
    
    # 指定した数のクラスタになるまで繰り返す
    while len(clusters) < num_clusters:
        # 最後に行われた結合を探す
        next_cluster = min(clusters, key=get_merge_order)
        # クラスタのリストから、取り除く
        clusters = [c for c in clusters if c != next_cluster]
        
        # 分離したクラスタをリストに追加する
        clusters.extend(get_children(next_cluster))
        
    # 指定したクラスタ数になったものを返す
    return clusters

if __name__ == '__main__':
    print('start chapter20')
    inputs: list[list[float]] = [[-14,-5],[13,13],[20,23],[-19,-11],[-9,-16],[21,27],[-49,15],[26,13],[-46,5],[-34,-1],[11,15],[-49,0],[-22,-16],[19,28],[-12,-8],[-13,-19],[-41,8],[-11,-6],[-25,-9],[-18,-3]]
    random.seed(12)
    clusterer = KMeans(k=3)
    clusterer.train(inputs)
    means = sorted(clusterer.means)
    
    assert len(means) == 3
    
    # クラスタの中心が期待する点に近いことを確認する
    assert squared_distance(means[0], [-44, 5]) < 1
    assert squared_distance(means[1], [-16, -10]) < 1
    assert squared_distance(means[2], [18, 20]) < 1
    
    random.seed(0)
    clusterer = KMeans(k=2)
    clusterer.train(inputs)
    means = sorted(clusterer.means)
    
    assert len(means) == 2
    assert squared_distance(means[0], [-26, -5]) < 1
    assert squared_distance(means[1], [18, 20]) < 1
    
    ks = range(1, len(inputs) + 1)
    errors = [squared_clustering_errors(inputs, k) for k in ks]
    
    plt.figure()
    plt.plot(ks, errors)
    plt.xticks(ks)
    plt.xlabel('k')
    plt.ylabel("total squared error")
    plt.title("Total Error vs. # of Clusters")
    plt.savefig('../picture/chap20-1.png')
    
    base_cluster = bottom_up_cluster(inputs)
    
    three_clusters = [get_values(cluster)
                     for cluster in generate_clusters(base_cluster, 3)]
    
    plt.figure()
    for i, cluster, marker, color in zip([1, 2, 3],
                                         three_clusters,
                                         ['D', 'o', '*'],
                                         ['r', 'g', 'b']):
        xs, ys = zip(*cluster) # 結合を分解するunzipトリック
        plt.scatter(xs, ys, color=color, marker=marker)
        
        # クラスタの平均を表示する
        x, y = vector_mean(cluster)
        plt.plot(x, y, marker='$' + str(i) + '$', color='black')
    plt.title("User Locations -- 3 Bottom-Up Clusters, Min")
    plt.xlabel("blocks east of city center")
    plt.ylabel("blocks north of city center")
    plt.savefig('../picture/chap20-2.png')