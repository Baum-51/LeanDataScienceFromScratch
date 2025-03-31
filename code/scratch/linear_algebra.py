import math

Vector = list[float]

def add(v: Vector, w: Vector) -> Vector:
    """対応する要素の和"""
    assert len(v) == len(w), "vectors must be the same length"
    return [v_i + w_i for v_i, w_i in zip(v, w)]

assert add([1, 2, 3], [4, 5, 6]) == [5, 7, 9]

def subtract(v: Vector, w: Vector) -> Vector:
    """対応する要素の差"""
    assert len(v) == len(w), "vectors must be the same length"
    return [v_i - w_i for v_i, w_i in zip(v, w)]

assert subtract([5, 7, 9], [4, 5, 6]) == [1, 2, 3]

def vector_sum(vectors: list[Vector]) -> Vector:
    """対応する要素ごとの合計"""
    assert vectors, "no vectors provided !"
    
    num_elements = len(vectors[0])
    assert all(len(v) == num_elements for v in vectors), "different sizes !"
    
    return [sum(vector[i] for vector in vectors) for i in range(num_elements)]

assert vector_sum([[1, 2], [3, 4], [5, 6], [7, 8]]) == [16, 20]

def scalar_multiply(c: float, v: Vector) -> Vector:
    """各要素にcを乗ずる"""
    return [c * v_i for v_i in v]

assert scalar_multiply(2, [1, 2, 3]) == [2, 4, 6]

def vector_mean(vectors: list[Vector]) -> Vector:
    """要素ごとの平均を求める"""
    n = len(vectors)
    return scalar_multiply(1/n, vector_sum(vectors))

assert vector_mean([[1, 2], [3, 4], [5, 6]]) == [3, 4]

def dot(v: Vector, w: Vector) -> float:
    """v_1 * w_1 + ...  + v_n * w_nを求める"""
    assert len(v) == len(w), "vectors must be same length"
    return sum(v_i * w_i for v_i, w_i in zip(v, w))

assert dot([1, 2, 3], [4, 5, 6]) == 32

def sum_of_squares(v: Vector) -> float:
    """Return v_1 * v_1 + ... + v_n * v_n"""
    return dot(v, v)

assert sum_of_squares([1, 2, 3]) == 14

def magnitude(v: Vector) -> float:
    """vのマグニチュード（または大きさ）を求める"""
    return math.sqrt(sum_of_squares(v))

assert magnitude([3, 4]) == 5

def squared_distance(v: Vector, w: Vector) -> float:
    """(v_1 - w_1)**2 + ... + (v_n - w_n)**2を求める"""
    return sum_of_squares(subtract(v, w))

def distance(v: Vector, w: Vector) -> float:
    """ベクトルvとwの距離を求める"""
    return math.sqrt(squared_distance(v, w))
def distance(v: Vector, w: Vector) -> float:
    return magnitude(subtract(v, w))