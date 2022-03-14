import numpy as np

def xavier_normal(in_size, out_size):
    std = np.sqrt(2 / (in_size + out_size))
    return np.random.normal(scale=std, size=(in_size, out_size))

def xavier_uniform(in_size, out_size):
    limit = np.sqrt(6 / (in_size + out_size))
    return np.random.uniform(low=-limit, high=limit, size=(in_size, out_size))

def he_normal(in_size, out_size):
    std = np.sqrt(2 / in_size)
    return np.random.normal(scale=std, size=(in_size, out_size))

def he_uniform(in_size, out_size):
    limit = np.sqrt(6 / in_size)
    return np.random.uniform(low=-limit, high=limit, size=(in_size, out_size))