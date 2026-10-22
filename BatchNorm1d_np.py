import numpy as np

from Tensor_np import Tensor_np


class BatchNorm1d_np:
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        self.gamma = Tensor_np(np.ones(num_features), requires_grad=True)
        self.beta = Tensor_np(np.zeros(num_features), requires_grad=True)
        self.eps = eps
        self.momentum = momentum
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
        self.training = True
        self.gamma.grad = np.zeros_like(self.gamma.data)
        self.beta.grad = np.zeros_like(self.beta.data)

    def __call__(self, *args, **kwds):
        x = args[0]
        mu = np.mean(x.data, axis=0)
        var = np.var(x.data, axis=0)
        x_hat = (x.data - mu) / np.sqrt(var + self.eps)
        out = self.gamma.data * x_hat + self.beta.data

        out_data = Tensor_np(out, prev=(x, self.gamma, self.beta), requires_grad=(self.gamma.requires_grad or x.requires_grad))

        def _backward():
            dout = out_data.grad
            gamma_g = np.sum((dout * x_hat), axis=0)
            beta_g = np.sum((dout), axis=0)
            if self.gamma.grad is None:
                self.gamma.grad = np.zeros_like(self.gamma.data)
            self.gamma.grad += gamma_g
            if self.beta.grad is None:
                self.beta.grad = np.zeros_like(self.beta.data)
            self.beta.grad += beta_g
            dx_hat = dout * self.gamma.data

            N = x.data.shape[0]
            std = np.sqrt(var + self.eps)
            dx = (1 / (N * std)) * (N * dx_hat - np.sum((dx_hat), axis=0) - x_hat * np.sum((dx_hat * x_hat), axis=0))
            if x.grad is None:
                x.grad = np.zeros_like(x.data)
            x.grad += dx

        out_data._backward = _backward
        return out_data


__all__ = ["BatchNorm1d_np"]
