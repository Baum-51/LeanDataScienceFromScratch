import os, sys
sys.path.append(os.getcwd())
from pprint import pprint

users_interests = [
    ["Hadoop", "Big Data", "HBase", "Java", "Spark", "Storm", "Cassandra"],
    ["NoSQL", "MongoDB", "Cassandra", "HBase", "Postgres"],
    ["Python", "scikit-learn", "scipy", "numpy", "statsmodels", "pandas"],
    ["R", "Python", "statistics", "regression", "probability"],
    ["machine learning", "regression", "decision trees", "libsvm"],
    ["Python", "R", "Java", "C++", "Haskell", "programming languages"],
    ["statistics", "probability", "mathematics", "theory"],
    ["machine learning", "scikit-learn", "Mahout", "neural networks"],
    ["neural networks", "deep learning", "Big Data", "artificial intelligence"],
    ["Hadoop", "Java", "MapReduce", "Big Data"],
    ["statistics", "R", "statsmodels"],
    ["C++", "deep learning", "artificial intelligence", "probability"],
    ["pandas", "R", "Python"],
    ["databases", "HBase", "Postgres", "MySQL", "MongoDB"],
    ["libsvm", "regression", "support vector machines"]
]

from collections import Counter

popular_interests = Counter(interest
                            for user_interests in users_interests
                            for interest in user_interests)

print(popular_interests)
print("="*100)

def most_popular_new_interests(
    user_interests: list[str],
    max_results: int=5) -> list[tuple[str, int]]:
    suggestions = [(interest, frequency)
                    for interest, frequency in popular_interests.most_common()
                    if interest not in user_interests]
    return suggestions[:max_results]

print(most_popular_new_interests(users_interests[0]))
print('='*100)

unique_interests = sorted({interest
                          for user_interests in users_interests
                          for interest in user_interests})

assert unique_interests[:6] == [
    'Big Data',
    'C++',
    'Cassandra',
    'HBase',
    'Hadoop',
    'Haskell',
    # ...
]

def make_user_interest_vector(user_interests: list[str]) -> list[str]:
    """
    興味の一覧から、i番目のアイテムにユーザが興味を持っていれば1、
    そうでなければ0が設定されたベクトルを作る
    """
    return [1 if interest in user_interests else 0
            for interest in unique_interests]

user_interest_vectors = [make_user_interest_vector(user_interests)
                         for user_interests in users_interests]

from scratch.nlp import cosine_similarity

user_similarities = [[cosine_similarity(interest_vector_i, interest_vector_j)
                      for interest_vector_j in user_interest_vectors]
                     for interest_vector_i in user_interest_vectors]

assert 0.56 < user_similarities[0][9] < 0.58, "複数の興味を共有している"
# ユーザー0と8が共有する興味はビッグデータの1つだけ
assert 0.18 < user_similarities[0][8] < 0.20, "興味を1つしか共有していない"

def most_similar_users_to(user_id: int) -> list[tuple[int, float]]:
    pairs = [(other_user_id, similarity)                     # 類似度が
             for other_user_id, similarity in                # 0以外の
             enumerate(user_similarities[user_id])           # ユーザを
             if user_id != other_user_id and similarity > 0] # 検索する
    
    return sorted(pairs,
                  key=lambda pair: pair[-1],
                  reverse=True)

print(most_similar_users_to(0))
print('='*100)

from collections import defaultdict

def user_based_suggestions(user_id: int,
                           include_current_interests: bool=False):
    # 類似度を加算
    suggestions: dict[str, str] = defaultdict(float)
    for other_user_id, similarity in most_similar_users_to(user_id):
        for interest in users_interests[other_user_id]:
            suggestions[interest] += similarity
    
    # リストをソートする
    suggestions = sorted(suggestions.items(),
                         key=lambda pair: pair[-1], # 重み
                         reverse=True)
    
    # （おそらく）すでに興味として持っているものを取り除く
    if include_current_interests:
        return suggestions
    else:
        return [(suggestion, weight)
                for suggestion, weight in suggestions
                if suggestion not in users_interests[user_id]]

pprint(user_based_suggestions(0))
print("="*100)

# 行が興味のあるもの、列が対応するユーザ
interest_user_matrix = [[user_interest_vector[j]
                         for user_interest_vector in user_interest_vectors]
                        for j, _ in enumerate(unique_interests)]

pprint(interest_user_matrix)
print("="*100)

interest_similarities = [[cosine_similarity(user_vector_i, user_vector_j)
                          for user_vector_j in interest_user_matrix]
                         for user_vector_i in interest_user_matrix]

print(interest_similarities)
print("="*100)

def most_similar_interests_to(interest_id: int):
    similarities = interest_similarities[interest_id]
    pairs = [(unique_interests[other_interest_id], similarity)
             for other_interest_id, similarity in enumerate(similarities)
             if interest_id != other_interest_id and similarity > 0]
    return sorted(pairs,
                  key=lambda pair: pair[-1],
                  reverse=True)

print(most_similar_interests_to(0))
print("="*100)

def item_based_suggestions(user_id: int,
                           include_current_interests: bool=False):
    # 持っている興味に対する類似の分野の類似度を加算する
    suggestions = defaultdict(float)
    user_interest_vector = user_interest_vectors[user_id]
    for interest_id, is_interested in enumerate(user_interest_vector):
        if is_interested == 1:
            similar_interests = most_similar_interests_to(interest_id)
            for interest, similarity in similar_interests:
                suggestions[interest] += similarity
    
    # 類似度の合計でソートする
    suggestions = sorted(suggestions.items(),
                         key=lambda pair: pair[-1],
                         reverse=True)
    
    if include_current_interests:
        return suggestions
    else:
        return [(suggestion, weight)
                for suggestion, weight in suggestions
                if suggestion not in users_interests[user_id]]

print(item_based_suggestions(0))

MOVIES  = "../ml-100k/ml-100k/u.item"
RATINGS = "../ml-100k/ml-100k/u.data"

from typing import NamedTuple

class Rating(NamedTuple):
    user_id: str
    movie_id: str
    rating: float

import csv
# UnicodeDecodeErrorを回避するために、エンコーディングを指定する
# 詳しくはhttps://stackoverflow.com/a/53136168/1076346を参照
with open(MOVIES, encoding="iso-8859-1") as f:
    reader = csv.reader(f, delimiter="|")
    movies = {movie_id: title for movie_id, title, *_ in reader}

# Ratingのリストを作成する
with open(RATINGS, encoding="iso-8859-1") as f:
    reader = csv.reader(f, delimiter='\t')
    ratings = [Rating(user_id, movie_id, float(rating))
               for user_id, movie_id, rating, _ in reader]

# 943のユーザにより1682の映画が評価されている
assert len(movies) == 1682
assert len(list({rating.user_id for rating in ratings})) == 943

import re

# movie_idごとに評価を蓄積するためのデータ構造
star_wars_ratings = {movie_id: []
                     for movie_id, title in movies.items()
                     if re.search("Star Wars|Empire Strikes|Jedi", title)}

# スターウォーズ評価を蓄積する
for rating in ratings:
    if rating.movie_id in star_wars_ratings:
        star_wars_ratings[rating.movie_id].append(rating.rating)
        
# 作品ごとに評価の平均を求める
avg_ratings = [(sum(title_ratings) / len(title_ratings), movie_id)
               for movie_id, title_ratings in star_wars_ratings.items()]

# 順番に出力
for avg_rating, movie_id in sorted(avg_ratings, reverse=True):
    print(f"{avg_rating:.2f} {movies[movie_id]}")

import random
random.seed(0)
random.shuffle(ratings)

split1 = int(len(ratings) * 0.7)
split2 = int(len(ratings) * 0.85)

train = ratings[:split1]            # 学習用にデータの70％
validation = ratings[split1:split2] # 検証用にデータの15％
test = ratings[split2:]             # テスト用にデータの15%

avg_rating = sum(rating.rating for rating in train) / len(train)
baseline_error = sum((rating.rating - avg_rating) ** 2
                     for rating in test) / len(test)

# これより良くするのが目標
assert 1.26 < baseline_error < 1.27

from scratch.deep_learning import random_tensor

EMBEDDING_DIM = 2

# 一意のIDを検索する
user_ids = {rating.user_id for rating in ratings}
movie_ids = {rating.movie_id for rating in ratings}

# IDごとにランダムなベクトルを作成する
user_vectors = {user_id: random_tensor(EMBEDDING_DIM)
                for user_id in user_ids}
movie_vectors = {movie_id: random_tensor(EMBEDDING_DIM)
                 for movie_id in movie_ids}

import tqdm
from scratch.linear_algebra import dot

def loop(dataset: list[Rating],
         learning_rate: float=None) -> None:
    with tqdm.tqdm(dataset) as t:
        loss = 0.0
        for i, rating in enumerate(t):
            movie_vector = movie_vectors[rating.movie_id]
            user_vector = user_vectors[rating.user_id]
            predicted = dot(user_vector, movie_vector)
            error = predicted - rating.rating
            loss += error ** 2
            
            if learning_rate is not None:
                # 予測 = m_0 * u_0 + ... + m_k * u_kであるため
                # 各u_jは係数m_jと共に出力に加わり
                # 各m_jは係数u_jで共に出力に加わる
                user_gradient = [error * m_j for m_j in movie_vector]
                movie_gradient = [error * u_j for u_j in user_vector]
                
                # 勾配ステップを繰り返す
                for j in range(EMBEDDING_DIM):
                    user_vector[j] -= learning_rate * user_gradient[j]
                    movie_vector[j] -= learning_rate * movie_gradient[j]
            t.set_description(f"avg loss: {loss / (i + 1)}")

from scratch.working_with_data import pca, transform

original_vectors = [vector for vector in movie_vectors.values()]
components = pca(original_vectors, 2)

ratings_by_movie = defaultdict(list)
for rating in ratings:
    ratings_by_movie[rating.movie_id].append(rating.rating)

vectors = [
    (movie_id,
    sum(ratings_by_movie[movie_id]) / len(ratings_by_movie[movie_id]),
    movies[movie_id],
    vector)
    for movie_id, vector in zip(movie_vectors.keys(),
                                transform(original_vectors, components))
]

# 第1主成分の上位25と下位25を出力する
print(sorted(vectors, key=lambda v: v[-1][0])[:25])
print(sorted(vectors, key=lambda v: v[-1][0])[-25:])

if __name__ == '__main__':
    learning_rate = 0.05
    for epoch in range(20):
        learning_rate *= 0.9
        print(epoch, learning_rate)
        loop(train, learning_rate=learning_rate)
        loop(validation)
    loop(test)