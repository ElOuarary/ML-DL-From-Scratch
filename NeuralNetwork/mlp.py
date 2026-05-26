import numpy as np

class Neuron:
    def __init__(self, nin):
        self.nin = nin
        self.grad = 0
        self._b = np.random.uniform(-1, 1)
        self._w = np.random.uniform(-1, 1, nin)
        
    def __call__(self, x):
        print(self._w)
        if len(x) != len(self._w):
            raise 
        return self._w.T @ x + self._b
    
class Layer:
    pass
    
class MLP:
    pass


if __name__ == "__main__":
    n = Neuron(5)
    x = np.random.random(5)
    print(n(x))