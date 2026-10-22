import numpy as np


class Tensor_np:
    def __init__(self, data, op='', prev=None, requires_grad=False):
        self.data = np.asarray(data, dtype=np.float32)
        self.requires_grad = requires_grad
        self._prev = prev
        if self._prev is None:
            self._prev = set()
        self._op = op
        self._backward = lambda: None
        self.grad = None

    def backward(self):
        visited = set()
        topo = []

        def build(v):
            if v in visited:
                return
            visited.add(v)
            for p in v._prev:
                build(p)
            topo.append(v)

        build(self)
        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            node._backward()

    @staticmethod
    def unbroadcast(grad, target_shape):
        if grad.shape == target_shape:
            return grad

        while grad.ndim > len(target_shape):
            grad = grad.sum(axis=0)

        for i in range(len(target_shape)):
            if target_shape[i] == 1 and grad.shape[i] > 1:
                grad = grad.sum(axis=i, keepdims=True)

        return grad.reshape(target_shape)

    def __add__(self, other):
        out = Tensor_np(self.data + other.data, prev=(self, other), op='+', requires_grad=False)

        def _backward():
            self_g = out.grad
            self_g = Tensor_np.unbroadcast(self_g, self.data.shape)
            if self.grad is None:
                self.grad = np.zeros_like(self.data)
            self.grad += self_g

            other_g = out.grad
            other_g = Tensor_np.unbroadcast(other_g, other.data.shape)
            if other.grad is None:
                other.grad = np.zeros_like(other.data)
            other.grad += other_g

        out.requires_grad = self.requires_grad or other.requires_grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        out = Tensor_np(self.data * other.data, prev=(self, other), op='*', requires_grad=False)

        def _backward():
            self_g = out.grad
            self_g = self_g * other.data
            self_g = Tensor_np.unbroadcast(self_g, self.data.shape)
            if self.grad is None:
                self.grad = np.zeros_like(self.data)
            self.grad += self_g

            other_g = out.grad
            other_g = other_g * self.data
            other_g = Tensor_np.unbroadcast(other_g, other.data.shape)
            if other.grad is None:
                other.grad = np.zeros_like(other.data)
            other.grad += other_g

        out.requires_grad = self.requires_grad or other.requires_grad
        out._backward = _backward
        return out

    def __matmul__(self, other):
        out = Tensor_np(self.data @ other.data, prev=(self, other), op='@', requires_grad=False)

        def _backward():
            self_g = out.grad
            self_g = self_g @ other.data.T
            self_g = Tensor_np.unbroadcast(self_g, self.data.shape)
            if self.grad is None:
                self.grad = np.zeros_like(self.data)
            self.grad += self_g

            other_g = out.grad
            other_g = self.data.T @ other_g
            other_g = Tensor_np.unbroadcast(other_g, other.data.shape)
            if other.grad is None:
                other.grad = np.zeros_like(other.data)
            other.grad += other_g

        out.requires_grad = self.requires_grad or other.requires_grad
        out._backward = _backward
        return out

    def sum(self):
        out = Tensor_np(self.data.sum(), prev=(self,), op='sum', requires_grad=False)

        def _backward():
            self_g = out.grad
            self_g = np.ones_like(self.data) * self_g
            if self.grad is None:
                self.grad = np.zeros_like(self.data)
            self.grad += self_g

        out.requires_grad = self.requires_grad
        out._backward = _backward
        return out

    def relu(self):
        data = self.data.copy()
        data[data < 0] = 0
        out = Tensor_np(data, prev=(self,), op='relu', requires_grad=False)

        def _backward():
            self_g = out.grad
            self_g = self_g * (self.data > 0)
            if self.grad is None:
                self.grad = np.zeros_like(self.data)
            self.grad += self_g

        out.requires_grad = self.requires_grad
        out._backward = _backward
        return out

    def bcewithlogits(self, target):
        logits = self.data
        target_data = target.data
        bce = (np.maximum(logits, 0) - logits * target_data + np.log(1 + np.exp(-np.abs(logits)))).mean()

        out = Tensor_np(bce, prev=(self, target), op='bcewithlogits', requires_grad=True)

        def _backward():
            self_g = out.grad
            self_g = (self_g) * (1 / (1 + np.exp(-logits)) - target_data) / logits.size
            if self.grad is None:
                self.grad = np.zeros_like(self.data)
            self.grad += self_g

        out.requires_grad = self.requires_grad or target.requires_grad
        out._backward = _backward
        return out


__all__ = ["Tensor_np"]
