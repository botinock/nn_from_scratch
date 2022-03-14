import numpy as np

from .core import Activation

import warnings
warnings.filterwarnings('ignore')

class ReLU(Activation):
    """Rectifier linear unit."""
    def forward(self, x: np.ndarray) -> np.ndarray:
        super().forward(x)
        return np.maximum(x, 0)

    def backward(self, y: np.ndarray) -> np.ndarray:
        super().backward(y)
        return y*np.where(self.x > 0, 1, 0)

class Sigmoid(Activation):
    """Sigmoid"""
    def __init__(self, eps: float = 1e-6):
        self.eps = eps

    def forward(self, x: np.ndarray) -> np.ndarray:
        super().forward(x)
        return self.sigmoid(x)

    def backward(self, y: np.ndarray) -> np.ndarray:
        super().backward(y)
        return y*self.sigmoid(self.x)*(1 - self.sigmoid(self.x))

    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))

class Tanh(Activation):
    """Tangent hyperbolic."""
    def forward(self, x: np.ndarray) -> np.ndarray:
        super().forward(x)
        return np.tanh(x)

    def backward(self, y: np.ndarray) -> np.ndarray:
        super().backward(y)
        return y*(1 - np.tanh(self.x)**2)
