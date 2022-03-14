from abc import ABC, abstractmethod
from copy import deepcopy

import numpy as np

class Module(ABC):
    def __init__(self, in_size, out_size):
        """
        Create weights for this layer if needed.
        """
        self.in_size = in_size
        self.out_size = out_size
    
    def __call__(self, *args):
        return self.forward(*args)

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward call.
        (N, M) -> (N, O) where 
        N is batch size;
        M is input size;
        O is output size.
        """
        assert len(x.shape) == 2
        self.x = x

    @abstractmethod
    def backward(self, y: np.ndarray) -> np.ndarray:
        """
        Backward call.
        (N, O) -> (N, M) 
        O is gradients from next layer;
        M is gradients from this layer;
        """
        pass
        # assert len(y.shape) <= 1

    @abstractmethod
    def get_weights(self) -> dict:
        """
        Should return learnable weights.
        """
        pass

    @abstractmethod
    def get_grads(self) -> dict:
        """
        Should return grads for learnable weights.
        """
        pass

    def get_weights_and_grads(self) -> tuple[dict]:
        return self.get_weights(), self.get_grads()

    @abstractmethod
    def set_weights(self, w_dict: dict):
        """
        Set weights to this layer.
        """
        pass

class Activation(Module, ABC):
    """
    Class for activation function.
    """
    def __init__(self):
        pass

    def get_weights(self) -> dict:
        return {}

    def get_grads(self) -> dict:
        return {}

    def set_weights(self, w_dict: dict) -> None:
        pass

class Loss(Module, ABC):
    """
    Class for loss function.
    """
    def __init__(self):
        pass

    def get_weights(self) -> dict:
        return {}

    def get_grads(self) -> dict:
        return {}

    def set_weights(self, w: dict):
        pass

class Optimizer(ABC):

    @abstractmethod
    def calculate_updates(self, g: dict):
        """
        Calculate weight updates from grads.
        """
        pass

    def update(self, weights: dict, g: dict):
        """
        Update model weights.
        """
        w = deepcopy(weights)
        dw = self.calculate_updates(g)
        return self._update(w, dw)
        

    def _update(self, w: dict, dw: dict):
        keys = w.keys()
        for key in keys:
            if isinstance(w[key], dict):
                w[key] = self._update(w[key], dw[key])
            else:
                w[key] += dw[key]
        return w