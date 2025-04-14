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
    print(outputs)
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