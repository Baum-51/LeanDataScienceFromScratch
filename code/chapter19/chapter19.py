import os, sys
sys.path.append(os.getcwd())

Tensor = list

def shape(tensor: Tensor) -> list[int]:
    sizes: list[int] = []
    while isinstance(tensor, list):
        sizes.append(len(tensor))
        tensor = tensor[0]
    return sizes

assert shape([1, 2, 3]) == [3]
assert shape([[1, 2], [3, 4], [5, 6]]) == [3, 2]

def is_1d(tensor: Tensor) -> bool:
    """
    tensor[0]がリストであれば高次元テンソルであるが、
    そうでなければ、テンソルは1次元（つまりベクトル）である
    """
    return not isinstance(tensor[0], list)

assert is_1d([1, 2, 3])
assert not is_1d([[1, 2], [3, 4]])

def tensor_sum(tensor: Tensor) -> float:
    """テンソル中の値を合計する"""
    if is_1d(tensor):
        return sum(tensor) # floatのリストなので、Pythonのsumを使う
    else:
        return sum(tensor_sum(tensor_i)    # 順番にtensor_sumを呼び出し
                   for tensor_i in tensor) # 結果を合計する

assert tensor_sum([1, 2, 3]) == 6
assert tensor_sum([[1, 2], [3, 4]]) == 10

from typing import Callable

def tensor_apply(f: Callable[[float], float], tensor: Tensor) -> Tensor:
    """要素ごとに関数fを適用する"""
    if is_1d(tensor):
        return [f(x) for x in tensor]
    else:
        return [tensor_apply(f, tensor_i) for tensor_i in tensor]
    
assert tensor_apply(lambda x: x + 1, [1, 2, 3]) == [2, 3, 4]
assert tensor_apply(lambda x: 2 * x, [[1, 2], [3, 4]]) == [[2, 4], [6, 8]]

def zeros_like(tensor: Tensor) -> Tensor:
    return tensor_apply(lambda _: 0.0, tensor)

def tensor_combine(f: Callable[[float, float], float],
                   t1: Tensor,
                   t2: Tensor) -> Tensor:
    if is_1d(t1):
        return [f(x, y) for x, y in zip(t1, t2)]
    else:
        return [tensor_combine(f, t1_i, t2_i)
                for t1_i, t2_i in zip(t1, t2)]
        
import operator
assert tensor_combine(operator.add, [1, 2, 3], [4, 5, 6]) == [5, 7, 9]
assert tensor_combine(operator.mul, [1, 2, 3], [4, 5, 6]) == [4, 10, 18]

from typing import Iterable

class Layer:
    """
    ニューラルネットワークは層で構成する。各層は、
    入力に対して順（forward）方向の計算を行い、
    逆（backward）方向に勾配を伝播する手段を持つ
    """
    def forward(self, input):
        """
        型を指定していない点に注意。
        層がどのような入力を受け付け、どのような出力を返すかに
        ついてはあらかじめ規定しない
        """
        raise NotImplementedError
    
    def backward(self, gradient):
        """
        同様に、勾配をどのように扱うかについてもあらかじめ規定しない。
        正しく動作するかは、利用者に委ねられる。
        """
        raise NotImplementedError
    
    def params(self) -> Iterable[Tensor]:
        """
        この層のパラメータを返す。デフォルトの実装は何も返さないため、
        層がパラメータを持たないなら実装する必要はない
        """
        return ()
    
    def grads(self) -> Iterable[Tensor]:
        """
        params()と同じように、この層の勾配を返す
        """
        return ()
    
from chapter18.chapter18 import sigmoid

class Sigmoid(Layer):
    def forward(self, input: Tensor) -> Tensor:
        """
        入力テンソルの各要素にシグモイドを適用し、
        backwardで使用するために結果を保存する。
        """
        self.sigmoids = tensor_apply(sigmoid, input)
        return self.sigmoids
    
    def backward(self, gradient: Tensor) -> Tensor:
        return tensor_combine(lambda sig, grad: sig * (1 - sig) * grad,
                              self.sigmoids,
                              gradient)
        
import random
from scratch.probability import inverse_normal_cdf

def random_uniform(*dims: int) -> Tensor:
    if len(dims) == 1:
        return [random.random() for _ in range(dims[0])]
    else:
        return [random_uniform(*dims[1:]) for _ in range(dims[0])]

def random_normal(*dims: int,
                  mean: float=0.0,
                  variance: float=1.0) -> Tensor:
    if len(dims) == 1:
        return [mean + variance * inverse_normal_cdf(random.random())
                for _ in range(dims[0])]
    else:
        return [random_normal(*dims[1:], mean=mean, variance=variance)
                for _ in range(dims[0])]

assert shape(random_uniform(2, 3, 4)) == [2, 3, 4]
assert shape(random_normal(5, 6, mean=10)) == [5, 6]

def random_tensor(*dims: int, init: str='normal') -> Tensor:
    if init == 'normal':
        return random_normal(*dims)
    elif init == 'uniform':
        return random_uniform(*dims)
    elif init == 'xavier':
        variance = len(dims) / sum(dims)
        return random_normal(*dims, variance=variance)
    else:
        raise ValueError(f"unknown init: {init}")
    
from scratch.linear_algebra import dot

class Linear(Layer):
    def __init__(self, input_dim: int, output_dim: int, init: str='xavier') -> None:
        """
        output_dim層のニューロンと、input_dimの重み（およびバイアス）
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # self.w[o]は、o番目ニューロンの重み
        self.w = random_tensor(output_dim, input_dim, init=init)
        
        # self.b[o]は、o番目ニューロンのバイアス
        self.b = random_tensor(output_dim, init=init)
        
    def forward(self, input: Tensor) -> Tensor:
        # backwardで使うためにinputを保存する
        self.input = input
        
        # 各ニューロンの出力をベクトルにして返す
        return [dot(input, self.w[o]) + self.b[o]
                for o in range(self.output_dim)]
    
    def backward(self, gradient: Tensor) -> Tensor:
        # b[o]はそれぞれoutput[o]を加えるため、
        # bの勾配はoutputの勾配と等しくなる
        self.b_grad = gradient
        
        # w[o][i]はそれぞれinput[i]を乗じ、output[o]を加算するため
        # 勾配はinput[i] * gradient[o]となる
        self.w_grad = [[self.input[i] * gradient[o]
                        for i in range(self.input_dim)]
                       for o in range(self.output_dim)]
        # input[i]にはw[o][i]をそれぞれ乗じ、それぞれoutput[o]を加算する。
        # したがって、その勾配は、すべてのoutputに対するw[o][i] * gradient[o]の合計となる
        return [sum(self.w[o][i] * gradient[o] for o in range(self.output_dim))
                for i in range(self.input_dim)]
        
    def params(self) -> Iterable[Tensor]:
        return [self.w, self.b]
    def grads(self) -> Iterable[Tensor]:
        return [self.w_grad, self.b_grad]
    
class Sequential(Layer):
    """
    層のシーケンスとして構成される要素を表す
    各層の出力が次の層の入力として意味がある否かは
    利用者自身に委ねられる
    """
    def __init__(self, layers: list[Layer]) -> None:
        self.layers = layers
    
    def forward(self, input):
        """inputを順に層にforwardする"""
        for layer in self.layers:
            input = layer.forward(input)
        return input
    
    def backward(self, gradient):
        """逆順に勾配を逆伝播させる"""
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)
        return gradient
    
    def params(self) -> Iterable[Tensor]:
        """各層のパラメータ（重みとバイアス）を返す"""
        return (param for layer in self.layers for param in layer.params())
    def grads(self) -> Iterable[Tensor]:
        """各層の勾配を返す"""
        return (grad for layer in self.layers for grad in layer.grads())
    
xor_net = Sequential([
    Linear(input_dim=2, output_dim=2),
    Sigmoid(),
    Linear(input_dim=2, output_dim=1),
    Sigmoid()
])

class Loss:
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        """予測はどの程度良いか（より大きな数字ほど悪い）"""
        raise NotImplementedError
    
    def gradient(self, predicted: Tensor, actual: Tensor) -> Tensor:
        """予測が変化すると、損失はどのように変化するか"""
        raise NotImplementedError
    
class SSE(Loss):
    """損失関数として誤差の二乗和を計算する"""
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        # テンソルの二乗誤差を計算
        squared_errors = tensor_combine(
            lambda predicted, actual: (predicted - actual) ** 2,
            predicted,
            actual
        )
        
        # 合計する
        return tensor_sum(squared_errors)
    
    def gradient(self, predicted: Tensor, actual: Tensor) -> Tensor:
        return tensor_combine(
            lambda predicted, actual: 2 * (predicted - actual),
            predicted,
            actual
        )

sse_loss = SSE()
assert sse_loss.loss([1, 2, 3], [10, 20, 30]) == 9 ** 2 + 18 ** 2 + 27 ** 2
assert sse_loss.gradient([1, 2, 3], [10, 20, 30]) == [-18, -36, -54]

class Optimizer:
    """
    オプティマイザは、層またはオプティマイザのいずれか（または両方）が
    持っている情報を使用して、層の重みを（その場で）更新する
    """
    def step(self, layer: Layer) -> None:
        raise NotImplementedError

class GradientDescent(Optimizer):
    def __init__(self, learning_rate: float=0.1) -> None:
        self.lr = learning_rate
    
    def step(self, layer: Layer) -> None:
        for param, grad in zip(layer.params(), layer.grads()):
            # 勾配ステップを使ってparamを更新する
            param[:] = tensor_combine(
                lambda param, grad: param - grad * self.lr,
                param,
                grad)

tensor = [[1, 2], [3, 4]]

for row in tensor:
    row = [0, 0]
assert tensor == [[1, 2], [3, 4]], "assignment doesn't update a list"

for row in tensor:
    row[:] = [0, 0]
assert tensor == [[0, 0], [0, 0]], "but slice assignment does"

class Momentum(Optimizer):
    def __init__(self,
                 learning_rate: float,
                 momentum: float=0.9) -> None:
        self.lr = learning_rate
        self.mo = momentum
        self.updates: list[Tensor] = [] # 移動平均
    
    def step(self, layer: Layer) -> None:
        # 更新の値が設定されていない場合は、すべて0から開始する
        if not self.updates:
            self.updates = [zeros_like(grad) for grad in layer.grads()]
        
        for update, param, grad in zip(self.updates,
                                       layer.params(),
                                       layer.grads()):
            # momentumを適用する
            update[:] = tensor_combine(
                lambda u, g: self.mo * u + (1 - self.mo) * g,
                update,
                grad)
        
            # 勾配ステップを適用する
            param[:] = tensor_combine(
                lambda p, u: p - self.lr * u,
                param,
                update)
        
# 学習用データ
xs = [[0., 0], [0., 1], [1., 0], [1., 1]]
ys = [[0.], [1.], [1.], [0.]]

random.seed(0)

net = Sequential([
    Linear(input_dim=2, output_dim=2),
    Sigmoid(),
    Linear(input_dim=2, output_dim=1)
])

import tqdm
optimizer = GradientDescent(learning_rate=0.1)
loss = SSE()

def train_xor():
    with tqdm.trange(3000) as t:
        for epoch in t:
            epoch_loss = 0.0
            
            for x, y in zip(xs, ys):
                predicted = net.forward(x)
                epoch_loss += loss.loss(predicted, y)
                gradient = loss.gradient(predicted, y)
                net.backward(gradient)
                
                optimizer.step(net)
                
            t.set_description(f"xor loss {epoch_loss:.3f}")

    for param in net.params():
        print(param)


import math

def tanh(x: float) -> float:
    # xが非常に大きいか小さい場合、tanhは（本質的に）1または-1となる。
    # 例えばmath.exp(1000)はエラーとなるため、値のチェックを行う
    if   x < -100: return -1
    elif x >  100: return  1
    
    em2x = math.exp(-2 * x)
    return (1 - em2x) / (1 + em2x)

class Tanh(Layer):
    def forward(self, input: Tensor) -> Tensor:
        # backwardで使用するため、tanhの値を保存する
        self.tanh = tensor_apply(tanh, input)
        return self.tanh
    
    def backward(self, gradient: Tensor) -> Tensor:
        return tensor_combine(
            lambda tanh, grad: (1 - tanh ** 2) * grad,
            self.tanh,
            gradient)
        
class ReLu(Layer):
    def forward(self, input: Tensor) -> Tensor:
        self.input = input
        return tensor_apply(lambda x: max(x, 0), input)
    
    def backward(self, gradient: Tensor) -> Tensor:
        return tensor_combine(lambda x, grad: grad if x > 0 else 0,
                              self.input,
                              gradient)
        
from chapter18.chapter18 import binary_encode, fizz_buzz_encode, argmax

xs = [binary_encode(n) for n in range(101, 1024)]
ys = [fizz_buzz_encode(n) for n in range(101, 1024)]

NUM_HIDDEN = 25
random.seed(0)

net = Sequential([
    Linear(input_dim=10, output_dim=NUM_HIDDEN, init='uniform'),
    Tanh(),
    Linear(input_dim=NUM_HIDDEN, output_dim=4, init='uniform'),
    Sigmoid()
])

def fizzbuzz_accuracy(low: int, hi: int, net: Layer) -> float:
    num_correct = 0
    for n in range(low, hi):
        x = binary_encode(n)
        predicted = argmax(net.forward(x))
        actual = argmax(fizz_buzz_encode(n))
        if predicted == actual:
            num_correct += 1
    
    return num_correct / (hi - low)

optimizer = Momentum(learning_rate=0.1, momentum=0.5)
loss = SSE()

def train_fizzbuzz():
    with tqdm.trange(1000) as t:
        for epoch in t:
            epoch_loss = 0.0
            
            for x, y in zip(xs, ys):
                predicted = net.forward(x)
                epoch_loss += loss.loss(predicted, y)
                gradient = loss.gradient(predicted, y)
                net.backward(gradient)
                
                optimizer.step(net)
                
            accuracy = fizzbuzz_accuracy(101, 1024, net)
            t.set_description(f"fb loss: {epoch_loss:.2f} acc: {accuracy:.2f}")
        print("test results", fizzbuzz_accuracy(1, 101, net))
    
def softmax(tensor: Tensor) -> Tensor:
    """次元に沿ったソフトマックス"""
    if is_1d(tensor):
        # エラーを抑止するため、最大値を減ずる
        largest = max(tensor)
        exps = [math.exp(x - largest) for x in tensor]
        
        sum_of_exps = sum(exps)
        return [exp_i / sum_of_exps
                for exp_i in exps]
    else:
        return [softmax(tensor_i) for tensor_i in tensor]
    

class SoftmaxCrossEntropy(Loss):
    """
    ニューラルネットワークモデルに与えられた観測データに対する負の対数尤度を求める。
    そのため、重みを選択して最小化することで、観測データの尤度が最大化される
    """
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        # 確率を得るためにソフトマックスを適用する
        probabilities = softmax(predicted)
        
        # この値は、クラスiではlog p_i、その他のクラスでは0となる。
        # log(0)の計算とならないように、pにわずかな量を追加する
        likelihoods = tensor_combine(lambda p, act: math.log(p + 1e-30) * act,
                                     probabilities,
                                     actual)
        # 最後に負数を合計する
        return -tensor_sum(likelihoods)
    
    def gradient(self, predicted: Tensor, actual: Tensor) -> Tensor:
        probabilities = softmax(predicted)
        
        # 非常に簡単になる
        return tensor_combine(lambda p, actual: p - actual,
                              probabilities,
                              actual)
        
random.seed(0)

net = Sequential([
    Linear(input_dim=10, output_dim=NUM_HIDDEN, init='uniform'),
    Tanh(),
    Linear(input_dim=NUM_HIDDEN, output_dim=4, init='uniform')
    # 最後にシグモイドは配置しない
])

optimizer = Momentum(learning_rate=0.1, momentum=0.9)
loss = SoftmaxCrossEntropy()

def train_fizzbuzz2():
    with tqdm.trange(100) as t:
        for epoch in t:
            epoch_loss = 0.0
            
            for x, y in zip(xs, ys):
                predicted = net.forward(x)
                epoch_loss += loss.loss(predicted, y)
                gradient = loss.gradient(predicted, y)
                
                net.backward(gradient)
                
                optimizer.step(net)
            
            accuracy = fizzbuzz_accuracy(101, 1024, net)
            t.set_description(f"fb loss: {epoch_loss:.3f} acc: {accuracy:.2f}")
        # テスト用データの結果を確認する
        print("test results", fizzbuzz_accuracy(101, 1024, net))
    
class Dropout(Layer):
    def __init__(self, p: float) -> None:
        self.p = p
        self.train = True
    
    def forward(self, input: Tensor) -> Tensor:
        if self.train:
            # 指定の確率に従い、
            # 入力の形に合わせた0と1のマスクを作成する
            self.mask = tensor_apply(
                lambda _: 0 if random.random() < self.p else 1,
                input)
            # マスクを乗じてドロップアウトを行う
            return tensor_combine(operator.mul, input, self.mask)
        else:
            # 評価時は、単に値を均一に縮小する
            return tensor_apply(lambda x: x * (1 - self.p), input)
        
    def backward(self, gradient: Tensor) -> Tensor:
        if self.train:
            # maskが1の箇所だけ伝播させる
            return tensor_combine(operator.mul, gradient, self.mask)
        else:
            raise RuntimeError("backwardは学習モード時のみcallされる")
        
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
digits = fetch_openml(name='mnist_784', version=1)

# データのダウンロード場所を好みの場所に変更する
# (ライブラリは引数が0個の関数が代入されることを期待している)
# (以前行わないようにアドバイスしたが、ここでは指定により敢えてlambdaを変数に代入する)

# 次の関数は、データをダウンロードしnumpy配列を返す
# テンソルは単なるリストであるため、tolist()を呼び出す

train_images, test_images = train_test_split(digits.data, test_size=10000, shuffle=False)
train_labels, test_labels = train_test_split(digits.target, test_size=10000, shuffle=False)

train_images = [image.reshape(28, 28).tolist() for image in train_images.to_numpy()]
train_labels = [int(label) for label in train_labels.tolist()]

print(shape(train_images))
assert shape(train_images) == [60000, 28, 28]
assert shape(train_labels) == [60000]

import matplotlib.pyplot as plt

fig, ax = plt.subplots(10, 10)

for i in range(10):
    for j in range(10):
        # 各画像を白黒でプロットし、軸を非表示にする
        ax[i][j].imshow(train_images[10 * i + j], cmap='Grays')
        ax[i][j].xaxis.set_visible(False)
        ax[i][j].yaxis.set_visible(False)
plt.savefig('../picture/chap19-2.png')

test_images = [image.reshape(28, 28).tolist() for image in test_images.to_numpy()]
test_labels = [int(label) for label in test_labels.tolist()]

assert shape(test_images) == [10000, 28, 28]
assert shape(test_labels) == [10000]

# ピクセル値の平均を計算する
avg = tensor_sum(train_images) / 60000 / 28/ 28

# 平均の除算、スケールの変更、フラット化
train_images = [[(pixel - avg) / 256 for row in image for pixel in row] for image in train_images]
test_images = [[(pixel - avg) / 256 for row in image for pixel in row] for image in test_images]

assert shape(train_images) == [60000, 784], "images should be flattened"
assert shape(test_images) == [10000, 784], "images should be flattened"

def one_hot_encode(i: int, num_labels: int = 10) -> list[float]:
    return [1.0 if j == i else 0.0 for j in range(num_labels)]

assert one_hot_encode(3) == [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
assert one_hot_encode(2, num_labels=5) == [0, 0, 1, 0, 0]

train_labels = [one_hot_encode(label) for label in train_labels]
test_labels  = [one_hot_encode(label) for label in test_labels]

assert shape(train_labels) == [60000, 10]
assert shape(test_labels) == [10000, 10]

def loop(model: Layer,
         images: list[Tensor],
         labels: list[Tensor],
         loss: Loss,
         optimizer: Optimizer=None) -> None:
    correct = 0      # 正しく予測できた数を記録する
    total_loss = 0.0 # 損失の総計を記録する
    
    with tqdm.trange(len(images)) as t:
        for i in t:
            predicted = model.forward(images[i])
            if argmax(predicted) == argmax(labels[i]):
                correct += 1
            total_loss += loss.loss(predicted, labels[i])
            
            # 学習を行っている場合は、勾配を逆伝播し、重みを更新する
            if optimizer is not None:
                gradient = loss.gradient(predicted, labels[i])
                model.backward(gradient)
                optimizer.step(model)
            
            # プログレスバーの状況表示を更新する
            avg_loss = total_loss / (i + 1)
            acc = correct / (i + 1)
            t.set_description(f"mnist loss: {avg_loss:.3f} acc: {acc:.3f}")

#train_fizzbuzz2()

random.seed(0)

# ロジスティック会期モデルは線形層とソフトマックスの組み合わせ
model = Linear(784, 10)
loss = SoftmaxCrossEntropy()

# このオプティマイザで試す
optimizer = Momentum(learning_rate=0.01, momentum=0.99)

# 学習用データを与えて学習を行う
loop(model, train_images, train_labels, loss, optimizer)

# テスト用データで結果の評価する（オプティマイザの指定を行わないため評価のみ行われる）
loop(model, test_images, test_labels, loss)

random.seed(0)

# 学習でon/offできるように名前を付ける
dropout1 = Dropout(0.1)
dropout2 = Dropout(0.1)

model = Sequential([
    Linear(784, 30),
    dropout1,
    Tanh(),
    Linear(30, 10),
    Tanh(),
    Linear(10, 10)
])

optimizer = Momentum(learning_rate=0.01, momentum=0.99)
loss = SoftmaxCrossEntropy()

# ドロップアウトを有効化して学習を行う
dropout1.train = dropout2.train = True
loop(model, train_images, train_labels, loss, optimizer)

# ドロップアウトを無効化して評価を行う
dropout1.train = dropout2.train = False
loop(model, test_images, test_labels, loss)