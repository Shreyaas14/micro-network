"""
Multi-layer perceptron.
"""

## Imports
from layer import Layer 
from engine import NNData
import matplotlib.pyplot as plt

## Consts
LEARNING_RATE = 0.05
TURNS = 100

## Multilayer Perceptron class. 
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

## Helper methods for training the MLP.
def least_squared_error(expected, actual):
  if len(expected) != len(actual):
    raise IndexError("Expected and Actual are not the same length.")
  score = NNData(0.0)

  for expected, actual in zip(expected, actual):
    diff = actual + (-1 * expected)
    score += diff * diff

  return score

## Instantiating the actual network.
mlp = MLP(3, [4, 4, 1])
x = [[2.0, 3.0, -1.0], [5.0, 2.5, 7.1], [6.7, 6.9, 4.2]]
targets = [0.67, 0.69, 0.42]

## The training loop
def train(mlp, x, targets):
  out = []
  weights = []

  for a in x:
    b = mlp(a)
    out.extend(b)

  for layer in mlp.layers:
    for neuron in layer.neurons:
      weights.extend(neuron.weights)
      weights.append(neuron.bias)

  score = least_squared_error(targets, out)
  score.backward()

  for weight in weights:
    weight.data -= LEARNING_RATE * weight.grad
    weight.grad = 0.0

  return score.data, [prediction.data for prediction in out]


def main():
  losses = []
  recent_predictions = []
  for turn in range(TURNS):
    loss, predictions = train(mlp, x, targets)
    losses.append(loss)
    if turn >= TURNS - 5:
      recent_predictions.append((turn + 1, predictions))

  print("Targets: [" + ", ".join(f"{target:.6f}" for target in targets) + "]")
  print("Iteration  Predictions")
  for turn, predictions in recent_predictions:
    values = ", ".join(f"{prediction:.6f}" for prediction in predictions)
    print(f"{turn:>9}  [{values}]")

  plt.plot(range(1, len(losses) + 1), losses, marker="o")
  plt.xlabel("Training iteration")
  plt.ylabel("Loss (sum of squared errors)")
  plt.title("Training loss")
  plt.xticks(range(1, len(losses) + 1))
  plt.grid(True, alpha=0.3)
  plt.tight_layout()
  plt.show()

if __name__ == "__main__":
  main()
