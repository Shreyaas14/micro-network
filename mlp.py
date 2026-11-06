"""
Multi-layer perceptron.
"""

from layer import Layer 

class MLP:
  def __init__(self, n_in, n_out):
    self.n_in = n_in 
    self.n_out = n_out

    def build_layers():
      l = []
      sizes = [n_in] + n_out

      for i in range(len(sizes) - 1):
        j = i + 1
        curr_layer = Layer(sizes[i], sizes[j])
        l.append(curr_layer)

      return l

    self.layers = build_layers()

  def __call__(self, x):
    for layer in self.layers:
      x = layer(x)

    return x

def main():
  mlp = MLP(3, [4, 4, 1])
  x = mlp([2.0, 3.0, -1.0])
  print(x)

main()


  

  
    