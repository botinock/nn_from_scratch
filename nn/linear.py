from typing import Callable

import numpy as np

from .core import Module
from .weights import xavier_normal

class Linear(Module):
    """Linear projection"""
    def __init__(
        self, 
        in_size: int,
        out_size: int,
        bias: bool = True,
        bias_init: float = 1,
        weights_init: Callable = xavier_normal
    ):
        super().__init__(in_size, out_size)
        self.bias = bias
        self.W: np.ndarray = weights_init(in_size, out_size).T.astype(np.float32)
        self.b: np.ndarray = np.full(out_size, bias_init, dtype=np.float32) if bias else 0
        self.x = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        super().forward(x)
        assert x.shape[1] == self.in_size
        return self.W @ x.T + self.b

    def backward(self, y: np.ndarray) -> np.ndarray:
        super().backward(y)
        assert y.shape[0] == self.out_size
        self.g = {'W': y @ self.x}
        if self.bias:
            self.g['b'] = y.sum()
        return self.W.T @ y

    def get_weights(self) -> dict:
        w = {'W': self.W}
        if self.bias:
            w['b'] = self.b
        return w

    def get_grads(self) -> dict:
        return self.g

    def set_weights(self, w: dict):
        self.W = w['W']
        if self.bias:
            self.b = w['b']