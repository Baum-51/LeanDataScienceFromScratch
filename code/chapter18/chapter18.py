import os, sys
sys.path.append(os.getcwd())

from scratch.linear_algebra import Vector, dot

def step_function(x: float) -> float:
    return 1.0 if x >= 0 else 0.0

def perceptron_output(weights: Vector, bias: float, x: Vector) -> float:
    """パーセプトロンが発火する1を返し、そうでなければ0を返す"""
    calculation = dot(weights, x) + bias
    return step_function(calculation)

and_weight = [2., 2]
and_bias = -3

assert perceptron_output(and_weight, and_bias, [1, 1]) == 1
assert perceptron_output(and_weight, and_bias, [0, 1]) == 0
assert perceptron_output(and_weight, and_bias, [1, 0]) == 0
assert perceptron_output(and_weight, and_bias, [0, 0]) == 0

or_weight = [2., 2]
or_bias = -1

assert perceptron_output(or_weight, or_bias, [1, 1]) == 1
assert perceptron_output(or_weight, or_bias, [0, 1]) == 1
assert perceptron_output(or_weight, or_bias, [1, 0]) == 1
assert perceptron_output(or_weight, or_bias, [0, 0]) == 0

not_weight = [-2.]
not_bias = 1.

assert perceptron_output(not_weight, not_bias, [0]) == 1
assert perceptron_output(not_weight, not_bias, [1]) == 0

and_gate = min
or_gate  = max
xor_gate = lambda x, y: 0 if x == y else 1

import math

def sigmoid(t: float) -> float:
    return 1 / (1 + math.exp(-t))

def neuron_output(weights: Vector, inputs: Vector) -> float:
    # weightsにはbiasが含まれ、inputには1が含まれる
    return sigmoid(dot(weights, inputs))

def feed_forward(neural_network: list[list[Vector]],
                 input_vector: Vector) -> list[Vector]:
    """
    ニューラルネットワークを介して入力ベクトルを送る。
    （最後のレイヤーだけでなく）すべてのレイヤーの出力を返す
    """
    outputs: list[Vector] = []
    
    for layer in neural_network:
        input_with_bias = input_vector + [1]
        output = [neuron_output(neuron, input_with_bias) for neuron in layer]
        outputs.append(output)
        
        # このニューロン層の出力を次のニューロン層の入力とする
        input_vector = output
    #print(outputs)
    return outputs

xor_network = [ # 隠れ層
               [[20., 20, -30],   # ANDニューロン
                [20., 20, -10]],  # ORニューロン
                # 出力層
               [[-60., 60, -30]]] # 1番目の入力のNOTと2番目の入力とANDするニューロン

# feed_forwardは全ニューロンの出力を保持しているため、
# [-1]は出力層ニューロンを、[0]はその出力を表す
assert 0.000 < feed_forward(xor_network, [0, 0])[-1][0] < 0.001
assert 0.999 < feed_forward(xor_network, [1, 0])[-1][0] < 1.000
assert 0.999 < feed_forward(xor_network, [0, 1])[-1][0] < 1.000
assert 0.000 < feed_forward(xor_network, [1, 1])[-1][0] < 0.001

def sqerror_gradients(network: list[list[Vector]],
                      input_vector: Vector,
                      target_vector: Vector) -> list[list[Vector]]:
    """
    与えられたニューラルネットワーク、入力ベクトル、および
    ターゲットベクトルを用いてニューロンの重みに対する
    二乗誤差損失の勾配を計算し、予測を行う
    """
    # 順伝播
    hidden_outputs, outputs = feed_forward(network, input_vector)
    # 出力ニューロンの活性化出力前の勾配
    output_deltas = [output * (1 - output) * (output - target)
                     for output, target in zip(outputs, target_vector)]
    
    # 出力ニューロンの重みに関する勾配
    output_grads = [[output_deltas[i] * hidden_output
                     for hidden_output in hidden_outputs + [1]]
                    for i, output_neuron in enumerate(network[-1])]
    
    # 隠れニューロンの活性化出力前の勾配
    hidden_deltas = [hidden_output * (1 - hidden_output) *
                     dot(output_deltas, [n[i] for n in network[-1]])
                     for i, hidden_output in enumerate(hidden_outputs)]
    
    # 隠れニューロンの重みに関する勾配
    hidden_grads = [[hidden_deltas[i] * input for input in input_vector + [1]]
                    for i, hidden_neuron in enumerate(network[0])]
    
    return [hidden_grads, output_grads]

import random
random.seed(0)

# 学習用データ
xs = [[0., 0], [0., 1], [1., 0], [1., 1]]
ys = [[0.], [1.], [1.], [0.]]

# ランダムな重みで初期化する
network = [ #隠れ層：2入力->2出力
           [[random.random() for _ in range(2 + 1)],  # 1番目の隠れ層のニューロン
            [random.random() for _ in range(2 + 1)]], # 2番目の隠れ層のニューロン
           # 出力層：2入力->1出力
           [[random.random() for _ in range(2 + 1)]]  # 1番目の出力ニューロン
           ]

from scratch.gradient_descent import gradient_step
import tqdm

learning_rate = 1.0

for epoch in tqdm.trange(20000, desc="neural net for xor"):
    for x, y in zip(xs, ys):
        gradients = sqerror_gradients(network, x, y)
        
        # 各層のニューロンそれぞれにgradient_stepを適用する
        network = [[gradient_step(neuron, grad, -learning_rate)
                    for neuron, grad in zip(layer, layer_grad)]
                   for layer, layer_grad in zip(network, gradients)]

# 学習済みXORゲートを確認する
assert feed_forward(network, [0, 0])[-1][0] < 0.01
assert feed_forward(network, [1, 0])[-1][0] > 0.99
assert feed_forward(network, [0, 1])[-1][0] > 0.99
assert feed_forward(network, [1, 1])[-1][0] < 0.01

def fizz_buzz_encode(x: int) -> Vector:
    if x % 15 == 0:
        return [0, 0, 0, 1]
    elif x % 5 == 0:
        return [0, 0, 1, 0]
    elif x % 3 == 0:
        return [0, 1, 0, 0]
    else:
        return [1, 0, 0, 0]
    
assert fizz_buzz_encode(2)  == [1, 0, 0, 0]
assert fizz_buzz_encode(6)  == [0, 1, 0, 0]
assert fizz_buzz_encode(10) == [0, 0, 1, 0]
assert fizz_buzz_encode(30) == [0, 0, 0, 1]

def binary_encode(x: int) -> Vector:
    binary: list[float] = []
    
    for i in range(10):
        binary.append(x % 2)
        x = x // 2
    return binary

#                             1  2  4  8 16 32 64 128 256 512
assert binary_encode(0)   == [0, 0, 0, 0, 0, 0, 0, 0,  0,  0]
assert binary_encode(1)   == [1, 0, 0, 0, 0, 0, 0, 0,  0,  0]
assert binary_encode(10)  == [0, 1, 0, 1, 0, 0, 0, 0,  0,  0]
assert binary_encode(101) == [1, 0, 1, 0, 0, 1, 1, 0,  0,  0]
assert binary_encode(999) == [1, 1, 1, 0, 0, 1, 1, 1,  1,  1]

xs = [binary_encode(n) for n in range(101, 1024)]
ys = [fizz_buzz_encode(n) for n in range(101, 1024)]

NUM_HIDDEN = 25

network = [
    # 隠れ層：10個の入力 -> NUM_HIDDEN個の出力
    [[random.random() for _ in range(10 + 1)] for _ in range(NUM_HIDDEN)],
    
    # 出力層：NUM_HIDDEN個の入力 -> 4個の出力
    [[random.random() for _ in range(NUM_HIDDEN + 1)] for _ in range(4)]
]

from scratch.linear_algebra import squared_distance

learning_rate = 1.0

with tqdm.trange(500) as t:
    for epoch in t:
        epoch_loss = 0.0
        for x, y in zip(xs, ys):
            predicted = feed_forward(network, x)[-1]
            epoch_loss += squared_distance(predicted, y)
            gradients = sqerror_gradients(network, x, y)
            
            # 各層の各ニューロンにgradient_stopを適用する
            network = [[gradient_step(neuron, grad, -learning_rate)
                        for neuron, grad in zip(layer, layer_grad)]
                       for layer, layer_grad in zip(network, gradients)]
        
        t.set_description(f"fizz buzz (loss: {epoch_loss:.2f})")
        
def argmax(xs: list) -> int:
    """最大値のインデックスを返す"""
    return max(range(len(xs)), key=lambda i: xs[i])

assert argmax([0, -1]) == 0             # items[0]が最大
assert argmax([-1, 0]) == 1             # items[1]が最大
assert argmax([-1, 10, 5, 20, -3]) == 3 # items[3]が最大


num_correct = 0

for n in range(1, 101):
    x = binary_encode(n)
    predicted = argmax(feed_forward(network, x)[-1])
    actual = argmax(fizz_buzz_encode(n))
    labels = [str(n), "fizz", "buzz", "fizzbuzz"]
    print(n, labels[predicted], labels[actual])
    
    if predicted == actual:
        num_correct += 1
print(num_correct, "/", 100)