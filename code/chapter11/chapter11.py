import random
from typing import TypeVar

X = TypeVar('X') # データポイントを表す汎用型

def split_data(data: list[X], prob: float) -> tuple[list[X], list[X]]:
    """データを[prob, 1-prob]の割合に分割する"""
    data = data[:]                 # shuffleはリスト自体を変更するため
    random.shuffle(data)           # シャローコピーを作成する
    cut = int(len(data) * prob)    # probを
    return data[:cut], data[cut:]

data = [n for n in range(1000)]
train, test = split_data(data, 0.75)

# 割合が正しくなければならない
assert len(train) == 750
assert len(test)  == 250

assert sorted(train + test) == data

Y = TypeVar('Y') # 出力変数を表す汎用型

def train_test_split(xs: list[X],
                     ys: list[Y],
                     test_pct: float) -> tuple[list[X], list[X], list[Y], list[Y]]:
    # インデックスを生成し、リストを分割する
    idxs = [i for i in range(len(xs))]
    train_idxs, test_idxs = split_data(idxs, 1 - test_pct)
    
    return ([xs[i] for i in train_idxs], # x_train
            [xs[i] for i in test_idxs],  # x_test
            [ys[i] for i in train_idxs], # y_train
            [ys[i] for i in test_idxs])  # y_test
    
xs = [x for x in range(1000)]
ys = [2 * x for x in xs]
x_train, x_test, y_train, y_test = train_test_split(xs, ys, 0.25)

# 割合が正しいことを確認
assert len(x_train) == len(y_train) == 750
assert len(x_test) == len(y_test)   == 250
# 対応するデータポイントが正しくペアリングされていることを確認する
assert all(y == 2 * x for x, y in zip(x_train, y_train))
assert all(y == 2 * x for x, y in zip(x_test, y_test))

def accuracy(tp: int, fp: int, fn: int, tn: int) -> float:
    correct = tp + tn
    total = tp + fp + fn +tn
    return correct / total

assert accuracy(70, 4930, 13930, 981070) == 0.98114

def precision(tp: int, fp: int, fn: int, tn: int) -> float:
    return tp / (tp + fp)

assert precision(70, 4930, 13930, 981070) == 0.014

def recall(tp: int, fp: int, fn: int, tn: int) -> float:
    return tp / (tp + fn)

assert recall(70, 4930, 13930, 981070) == 0.005

