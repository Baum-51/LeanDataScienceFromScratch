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

class Liner(Layer):
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
    Liner(input_dim=2, output_dim=2),
    Sigmoid(),
    Liner(input_dim=2, output_dim=1),
    Sigmoid()
])