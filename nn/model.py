from typing import Sequence

from .core import Module, Optimizer

class Sequential:
    def __init__(
        self,
        layers: Sequence[Module],
        opt: Optimizer
    ):
        self.layers = layers
        self.opt = opt

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def backward(self, y):
        for layer in self.layers[::-1]:
            y = layer.backward(y)
        grads = {}
        weights = {}
        for i, layer in enumerate(self.layers):
            weights[f"layer_{i}"], grads[f"layer_{i}"] = layer.get_weights_and_grads()
        weights = self.opt.update(weights, grads)
        for i, layer in enumerate(self.layers):
            layer.set_weights(weights[f"layer_{i}"])