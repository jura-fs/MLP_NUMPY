from Tensor_np import Tensor_np


class Module:
    def __init__(self):
        pass

    def parameters(self):
        params = []
        for value in self.__dict__.values():
            if isinstance(value, Tensor_np) and value.requires_grad:
                params.append(value)
            if isinstance(value, Module):
                params.extend(value.parameters())
        return params


__all__ = ["Module"]
