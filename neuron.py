"""
Neural network code.
"""

import math 
import random

from engine import NNData


class Neuron:
  def __init__(self, n_in):
    self.bias = NNData(0.0)
    self.n_in = n_in
    self.weights = [NNData(random.uniform(-1, 1)) for _ in range(n_in)]

  def __call__(self, x):
    wx_sum = self.bias
    for input, weight in zip(x, self.weights):
      wx_sum += weight * input 

    out = NNData.tanh(wx_sum)

    return out

"""
def main():
  x = Neuron(3)
  out = x([2.0, 3.0, -1.0])

  out.backward()

  for weight in x.weights:
    print("WEIGHT")
    print(weight.grad)

main()
"""
