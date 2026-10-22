class SGD:
    def __init__(self, params, lr):
        self.params = params
        self.lr = lr

    def step(self):
        for param in self.params:
            if param.grad is not None:
                param.data = param.data - (self.lr * param.grad)

    def zero_grad(self):
        for param in self.params:
            param.grad = None


__all__ = ["SGD"]
