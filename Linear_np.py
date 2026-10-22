import numpy as np

from Module_np import Module
from Tensor_np import Tensor_np


class Linear(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weights = Tensor_np(np.ones((in_features, out_features)) * 0.01, requires_grad=True)
        self.bias = Tensor_np(np.zeros((out_features,)), requires_grad=True)

    def __call__(self, x):
        return x @ self.weights + self.bias


__all__ = ["Linear"]
