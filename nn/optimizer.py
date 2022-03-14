import numpy as np
from copy import deepcopy

from .core import Optimizer

class SGD(Optimizer):
    def __init__(
        self,
        lr: float = 1e-3
    ):
        self.lr = lr

    def calculate_updates(self, g: dict) -> dict:
        dw = deepcopy(g)
        dw = self._calculate_updates(g, dw)
        return dw

    def _calculate_updates(self, g: dict, dw: dict) -> dict:
        keys = g.keys()
        for key in keys:
            if isinstance(g[key], dict):
                dw[key] = self._calculate_updates(g[key], dw[key])
            else:
                dw[key] = -self.lr*g[key]
        return dw


class Nesterov(Optimizer):
    def __init__(
        self,
        lr: float = 1e-3,
        mu: float = 0.9,
        tau: float = 0
    ):
        self.lr = lr
        self.mu = mu
        self.tau = tau
        self.v = None

    def calculate_updates(self, g: dict) -> dict:
        dw = deepcopy(g)
        if self.v is None:
            self.v = deepcopy(g)
            self.v = init_dict(self.v, g)
            self.prev_g = deepcopy(g)
        
        self._calculate_updates(self, g, dw, self.prev_g, self.v)
        self.prev_g = deepcopy(g)
        return dw

    def _calculate_updates(self, g: dict, dw: dict, prev_g: dict, v: dict) -> dict:
        keys = g.keys()
        for key in keys:
            if isinstance(g[key], dict):
                self._calculate_updates(g[key], dw[key], prev_g[key], v[key])
            else:
                v[key] = self.mu * v[key] + (1 - self.tau) * g[key]
                dw[key] = -self.lr*(prev_g[key] + self.mu*v[key])


class Adam(Optimizer):
    def __init__(
        self,
        lr: float = 1e-3,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        eps: float = 1e-6
    ):
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps
        self.t = 0
        self.s = None
        self.r = None

    def calculate_updates(self, g: dict) -> dict:
        self.t += 1
        dw = deepcopy(g)
        if self.s is None:
            self.s = deepcopy(g)
            self.s = init_dict(self.s, g)
            self.r = deepcopy(g)
            self.r = init_dict(self.r, g)
            self.prev_g = deepcopy(g)
        
        self._calculate_updates(self, g, dw, self.s, self.r)
        return dw

    def _calculate_updates(self, g: dict, dw: dict, s: dict, r: dict) -> dict:
        keys = g.keys()
        for key in keys:
            if isinstance(g[key], dict):
                self._calculate_updates(g[key], dw[key], s[key], r[key])
            else:
                s[key] = self.beta_1*s[key] + (1 - self.beta_1) * g[key]
                r[key] = self.beta_2*r[key] + (1 - self.beta_2) * g[key] * g[key]
                dw[key] = -self.lr*((s[key]/(1 - self.beta_1**self.t)) / (np.sqrt((r[key]/(1 - self.beta_2**self.t))) + self.eps))


def init_dict(d: dict, g: dict) -> dict:
    keys = g.keys()
    for key in keys:
        if isinstance(g[key], dict):
            init_dict(d[key], g[key])
        else:
            d[key] = 0
    return d