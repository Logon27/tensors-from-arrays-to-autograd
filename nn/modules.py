from __future__ import annotations

from nn.tensor import Tensor
from nn.functional import matmul
from nn.array_backend import xp


class Module:
    """Base class for all neural network modules."""
    def forward(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def parameters(self) -> list[Tensor]:
        return []


class Linear(Module):
    def __init__(self, in_features: int, out_features: int):
        # Initialize weights to small random values
        limit  = xp.sqrt(1 / in_features)
        w_data = xp.random.uniform(
            low   = -limit,
            high  = limit,
            size  = (in_features, out_features),
        ).astype(xp.float32)
        # Initialize biases to zeros
        b_data = xp.zeros(out_features, dtype=xp.float32)

        self.weight = Tensor(w_data, requires_grad=True)
        self.bias   = Tensor(b_data, requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        # matmul(x, weight) has output shape (batch, out_features), while bias is (out_features,).
        # Adding them broadcasts the bias across every sample in the batch. Hence why we needed to implement _reduce_grad() for backpropagation.
        return matmul(x, self.weight) + self.bias

    def parameters(self) -> list[Tensor]:
        return [self.weight, self.bias]


class SGD:
    """Stochastic Gradient Descent optimizer."""
    def __init__(self, parameters: list[Tensor], learning_rate: float = 0.01):
        self.parameters    = parameters
        self.learning_rate = learning_rate

    def step(self):
        for p in self.parameters:
            if p.grad is not None:
                p.data -= self.learning_rate * p.grad

    def zero_grad(self):
        for p in self.parameters:
            p.grad = None

