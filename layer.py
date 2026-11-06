"""
Class for a neural network layer. 
"""

from neuron import Neuron

class Layer:
  def __init__(self, n_in, n_out):
    self.n_in = n_in
    self.n_out = n_out 
    self.neurons = [Neuron(n_in) for _ in range(n_out)]

  def __call__(self, x):
    outputs = []
    for neuron in self.neurons:
      outputs.append(neuron(x))

    return outputs
    
  