from __future__ import annotations
from nn.tensor import Tensor
from nn.operations import MatMul, ReLU, Softmax, CrossEntropyLoss


# Functional API wrappers

def matmul(x: Tensor, w: Tensor) -> Tensor:
    """Matrix multiplication."""
    return Tensor._apply_op(MatMul, x, w)

def relu(x: Tensor) -> Tensor:
    """Apply ReLU activation."""
    return Tensor._apply_op(ReLU, x)

def softmax(x: Tensor) -> Tensor:
    """Apply softmax to logits."""
    return Tensor._apply_op(Softmax, x)

def cross_entropy_loss(probs: Tensor, targets: Tensor) -> Tensor:
    """Cross-entropy loss from probabilities."""
    return Tensor._apply_op(CrossEntropyLoss, probs, targets)

