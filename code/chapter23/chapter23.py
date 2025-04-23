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