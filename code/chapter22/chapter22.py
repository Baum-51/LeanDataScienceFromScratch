import os, sys
sys.path.append(os.getcwd())

from typing import NamedTuple

class User(NamedTuple):
    id: int
    name: str

users = [User(0, "Hero"), User(1, "Dunn"), User(2, "Sue"), User(3, "Chi"),
         User(4, "Thor"), User(5, "Clive"), User(6, "Hichs"),
         User(7, "Devin"), User(8, "Kate"), User(9, "Klein")]

friend_pairs = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4),
                (4, 5), (5, 6), (5, 7), (6, 8), (7, 8), (8, 9)]

# 交友関係を保持するための型エイリアス
Friendships = dict[int, list[int]]

friendships: Friendships = {user.id: [] for user in users}
print('friendships', friendships)

for i, j in friend_pairs:
    friendships[i].append(j)
    friendships[j].append(i)

assert friendships[4] == [3, 5]
assert friendships[8] == [6, 7, 9]

from collections import deque

Path = list[int]

def shortest_paths_from(from_user_id: int,
                        friendships: Friendships) -> dict[int, list[Path]]:
    # 指定のユーザーに至るすべての最短経路を保持する辞書
    shortest_paths_to: dict[int, list[Path]] = {from_user_id: [[]]}
    
    # 確認すべきユーザを（前のユーザ、今のユーザ）形式でキューに入れる
    # 開始時点では、(from_user, from_userの友達)組のすべてを持たせる
    frontier = deque((from_user_id, friend_id)
                     for friend_id in friendships[from_user_id])
    
    # キューに値がある限り続ける
    while frontier:
        # キュー内にある次のペアを削除する
        prev_user_id, user_id = frontier.popleft()
        
        # キューへの値追加方法により、
        # 必然的に「前のユーザー」までの最短経路は既知である
        paths_to_prev_user = shortest_paths_to[prev_user_id]
        new_paths_to_user  = [path + [user_id] for path in paths_to_prev_user]
        
        # user_idへの最短経路をすでに知っている可能性
        old_paths_to_user = shortest_paths_to.get(user_id, [])
        
        # ここに至る最短経路をすでに知っている可能性
        if old_paths_to_user:
            min_path_length = len(old_paths_to_user[0])
        else:
            min_path_length = float('inf')
        
        # その経路長がこれまでのもの以下で、新しい経路であった場合のみリストに加える
        new_paths_to_user = [path
                             for path in new_paths_to_user
                             if len(path) <= min_path_length
                             and path not in old_paths_to_user]
        
        shortest_paths_to[user_id] = old_paths_to_user + new_paths_to_user
        
        # frontierキューに、新しく見つけた友達を加える
        frontier.extend((user_id, friend_id)
                        for friend_id in friendships[user_id]
                        if friend_id not in shortest_paths_to)
        
    return shortest_paths_to

# from_userごと、to_userごとの最短経路のリスト
shortest_paths = {user.id: shortest_paths_from(user.id, friendships)
                  for user in users}
print('shortest_paths', shortest_paths)

betweenness_centrality = {user.id: 0.0 for user in users}

for source in users:
    for target_id, paths in shortest_paths[source.id].items():
        if source.id < target_id:
            num_paths = len(paths)
            contrib = 1 / num_paths
            for path in paths:
                for between_id in path:
                    if between_id not in [source.id, target_id]:
                        betweenness_centrality[between_id] += contrib
print('betweenness_centrality', betweenness_centrality)

def farness(user_id: int) -> float:
    """他の全ユーザに対する最短経路長の合計"""
    return sum(len(paths[0])
               for paths in shortest_paths[user_id].values())

closeness_centrality = {user.id: 1 / farness(user.id) for user in users}
print('closeness_centrality', closeness_centrality)

from scratch.linear_algebra import Matrix, make_matrix, shape

def matrix_times_matrix(m1: Matrix, m2: Matrix) -> Matrix:
    nr1, nc1 = shape(m1)
    nr2, nc2 = shape(m2)
    assert nc1 == nr2, "（m1の列数） == （m2の行数）でなければならない"
    
    def entry_fn(i: int, j: int) -> float:
        """m1のi番目の行とm2のj番目のドット積"""
        return sum(m1[i][k] * m2[k][j] for k in range(nc1))
    
    return make_matrix(nr1, nc2, entry_fn)

from scratch.linear_algebra import Vector, dot

def matrix_times_vector(m: Matrix, v: Vector) -> Vector:
    nr, nc = shape(m)
    n = len(v)
    assert nc == n, "（mの列数）==（vの要素数）でなければならない"
    
    return [dot(row, v) for row in m] # 結果の長さはnr

import random
from scratch.linear_algebra import magnitude, distance

def find_eigenvector(m: Matrix,
                     tolerance: float=0.00001) -> tuple[Vector, float]:
    guess = [random.random() for _ in m]
    
    while True:
        result = matrix_times_vector(m, guess)  # 予測を変換
        norm = magnitude(result)                # ノルムの計算
        next_guess = [x / norm for x in result] # リスケール
        
        if distance(guess, next_guess) < tolerance:
            # 収束したら（固有ベクトル、固有値）を返す
            return next_guess, norm
        
        guess = next_guess
        
def entry_fn(i: int, j: int):
    return 1 if (i, j) in friend_pairs or (j, i) in friend_pairs else 0

n = len(users)
adjacency_matrix = make_matrix(n, n, entry_fn)
print('adjacency_matrix', adjacency_matrix)

eigenvector_centralities, _ = find_eigenvector(adjacency_matrix)
print('eigenvector_centralities', eigenvector_centralities)

matrix_times_vector(adjacency_matrix, eigenvector_centralities)