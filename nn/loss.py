import numpy as np

from .core import Loss

class BCE(Loss):
    def __init__(self, eps: float = 1e-6):
        self.eps = eps

    def forward(self, y: np.ndarray, y_target: np.ndarray) -> np.ndarray:
        y_t = y_target.copy()
        if len(y_target.shape) < len(y.shape):
            y_t = np.reshape(y_target, (-1, 1))
        loss = np.mean(-y_t * np.log(y + self.eps) - (1 - y_t) * np.log(1 - y + self.eps))
        # self.dy = np.mean((y - y_target) / (y * (1 - y) + self.eps))
        self.dy = (-y_t/(y + self.eps) + (1 - y_t)/(1 - y + self.eps))/len(y)
        return loss

    def backward(self) -> np.ndarray:
        return self.dy

class L1(Loss):
    def forward(self, y: np.ndarray, y_target: np.ndarray) -> np.ndarray:
        loss = np.mean(np.abs(y - y_target))
        self.dy = np.sign(y - y_target)/len(y)
        return loss

    def backward(self) -> np.ndarray:
        return self.dy

class L2(Loss):
    def forward(self, y: np.ndarray, y_target: np.ndarray) -> np.ndarray:
        loss = np.mean((y - y_target)**2 / 2)
        self.dy = (y - y_target)/len(y)
        return loss

    def backward(self) -> np.ndarray:
        return self.dy