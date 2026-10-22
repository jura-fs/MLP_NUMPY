from BatchNorm1d_np import BatchNorm1d_np
from Linear_np import Linear
from Module_np import Module


class MLP(Module):
    def __init__(self, in_dim, hidden_dim, out_dim=1):
        super().__init__()
        self.fc1 = Linear(in_dim, hidden_dim)
        self.bn = BatchNorm1d_np(hidden_dim)
        self.fc2 = Linear(hidden_dim, out_dim)

    def __call__(self, x):
        x = self.fc1(x)
        x = self.bn(x)
        x = x.relu()
        x = self.fc2(x)
        return x


__all__ = ["MLP"]
