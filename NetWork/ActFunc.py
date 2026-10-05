import numpy as np

def sigmoid(array:np.array):
    return 1 / (1 + np.exp(-array))

def ReLU(array:np.array):
    return np.maximum(0, array)

def tanh(array:np.array):
    return np.tanh(array)