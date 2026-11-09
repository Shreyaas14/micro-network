"""
A class to store the values which pass through a neural network.
"""

import math

class NNData:
  # constructor
  def __init__(self, data, _children = (), _op = ''):
    self.data = data # storing the data
    self.grad = 0.0 # the gradient value
    self._backward = lambda: None # the function to calculate the gradient
    self._children = _children
    self._op = _op

  def __repr__(self):
    return f"{self.data}"

  def __add__(self, other):
    other = other if isinstance(other, NNData) else NNData(other)

    out = NNData(self.data + other.data, (self, other), '+')

    def backward():
      self.grad += out.grad 
      other.grad += out.grad

    out._backward = backward 

    return out 

  def __mul__(self, other):
    other = other if isinstance(other, NNData) else NNData(other)

    out = NNData(self.data * other.data, (self, other), '*')

    def backward():
      self.grad += out.grad * other.data
      other.grad += out.grad * self.data

    out._backward = backward

    return out 

  def __radd__(self, other):
    return self.__add__(other)
  
  def __rmul__(self, other):
    return self.__mul__(other)

  def backward(self):
    self.grad = 1.0 # gradient with respect to itself is 1
    nodes = []
    visited = set()

    self.top_sort(self, nodes, visited)

    nodes.reverse()

    for node in nodes:
      node._backward()

  def top_sort(self, root, nodes, visited):
    if root in visited:
      return 
    else:
      visited.add(root)
      for child in root._children:
        self.top_sort(child, nodes, visited) 
      nodes.append(root)

  def tanh(self):
    data = math.tanh(self.data)

    out = NNData(data, (self,), 'tanh')

    def backward():
      self.grad += (1 - out.data**2) * out.grad

    out._backward = backward

    return out 


"""
def main():
  a = NNData(2)
  b = NNData(-3)
  c = NNData(10)
  f = NNData(-2)

  e = a * b 
  d = e + c

  L = d * f 

  L.backward()
  print("### test autograd 1")
  print(a.grad, b.grad)

  x = NNData(0.5)
  y = x.tanh()
  y.backward()
  print("### test tanh")
  print(x.grad)



main()
"""
