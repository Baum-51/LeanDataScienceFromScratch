from typing import List, Tuple, Callable

Vector = List[float]
Matrix = List[List[float]]

A = [[1, 2, 3],
     [4, 5, 6]]
B = [[1, 2],
     [3, 4],
     [5, 6]]

def shape(A: Matrix) -> Tuple[int, int]:
    """Aの行数と列数を返す"""
    num_rows = len(A)
    num_cols = len(A[0] if A else 0)
    return num_rows, num_cols

assert shape([[1, 2, 3], [4, 5, 6]]) == (2, 3)

def get_row(A: Matrix, i: int) -> Vector:
    """Aのi番目の行を（ベクトルとして）返す"""
    return A[i]
def get_column(A: Matrix, j: int) -> Vector:
    """Aのj番目の列を（ベクトルとして）返す"""
    return [A_i[j] for A_i in A]

def make_matrix(num_rows: int,
                num_cols: int,
                entry_fn: Callable[[int, int], float]) -> Matrix:
    """
    num_rows x num_colsの行列を返す
    (i, j)の要素は、entry_fn(i, j)が与える
    """
    return [[entry_fn(i, j) for j in range(num_cols)] for i in range(num_rows)]

def identity_matrix(n: int) -> Matrix:
    """n×nの単位行列を返す"""
    return make_matrix(n, n, lambda i, j: 1 if i == j else 0)

assert identity_matrix(5) == [[1, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0],
                              [0, 0, 1, 0, 0],
                              [0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 1]]
