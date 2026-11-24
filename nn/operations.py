from __future__ import annotations

from typing import Any, TYPE_CHECKING
from nn.array_backend import xp

# Prevents circular import while still allowing typechecking.
if TYPE_CHECKING:
    from nn.tensor import Tensor

# -------------------------------------------------------------------------
# Computation Graph Nodes (Functions)
# -------------------------------------------------------------------------

class Operation:
    """Base class for all computation graph operations."""
    def __init__(self, *tensors: Tensor):
            self.parents = tensors
            self.saved_tensors: list[Any] = []

    def forward(self, *args: xp.ndarray, **kwargs: Any) -> xp.ndarray:
        raise NotImplementedError

    def backward(self, grad_output: xp.ndarray) -> xp.ndarray:
        raise NotImplementedError

    def save_for_backward(self, *tensors: Any) -> None:
        """Saves arbitrary data (usually input arrays) needed for the backward pass."""
        self.saved_tensors.extend(tensors)
    
    @property
    def name(self) -> str:
        """Returns the name of the operation class. Purely used for visualization purposes."""
        return self.__class__.__name__


class Add(Operation):
    def forward(self, x: xp.ndarray, y: xp.ndarray) -> xp.ndarray:
        return x + y

    def backward(self, grad_output: xp.ndarray) -> tuple[xp.ndarray, xp.ndarray]:
        # Forward: f(x, y) = x + y
        # Partial derivatives: ∂f/∂x = 1, ∂f/∂y = 1
        # Structural alignment: match variable positions from forward pass (x + y)
        grad_input_x = grad_output * 1  # Derivative w.r.t x (first term)
        grad_input_y = 1 * grad_output  # Derivative w.r.t y (second term)
        return grad_input_x, grad_input_y

class Sub(Operation):
    def forward(self, x: xp.ndarray, y: xp.ndarray) -> xp.ndarray:
        return x - y

    def backward(self, grad_output: xp.ndarray) -> tuple[xp.ndarray, xp.ndarray]:
        # Forward: f(x, y) = x - y
        # Partial derivatives: ∂f/∂x = 1, ∂f/∂y = -1
        # Structural alignment: match variable positions from forward pass (x - y)
        grad_input_x = grad_output * 1    # Derivative w.r.t x (first term)
        grad_input_y = -1 * grad_output   # Derivative w.r.t y (second term)
        return grad_input_x, grad_input_y

class Mul(Operation):
    def forward(self, x: xp.ndarray, y: xp.ndarray) -> xp.ndarray:
        self.save_for_backward(x, y)
        return x * y

    def backward(self, grad_output: xp.ndarray) -> tuple[xp.ndarray, xp.ndarray]:
        x, y = self.saved_tensors
        # Forward: f(x, y) = x * y
        # Partial derivatives: ∂f/∂x = y, ∂f/∂y = x
        # Structural alignment: match variable positions from forward pass (x * y)
        grad_input_x = grad_output * y  # Derivative w.r.t x (first term)
        grad_input_y = x * grad_output  # Derivative w.r.t y (second term)
        return grad_input_x, grad_input_y

class Div(Operation):
    def forward(self, x: xp.ndarray, y: xp.ndarray) -> xp.ndarray:
        self.save_for_backward(x, y)
        return x / y

    def backward(self, grad_output: xp.ndarray) -> tuple[xp.ndarray, xp.ndarray]:
        x, y = self.saved_tensors
        # Forward: f(x, y) = x / y
        # Partial derivatives: ∂f/∂x = 1 / y, ∂f/∂y = -x / (y^2)
        # Structural alignment: match variable positions from forward pass (x / y)
        grad_input_x = grad_output / y                 # Derivative w.r.t x (first term)
        grad_input_y = (-x / (y ** 2)) * grad_output   # Derivative w.r.t y (second term)
        return grad_input_x, grad_input_y

# Only used for demonstration purposes in the demo.py file.
class Sum(Operation):
    def forward(self, x: xp.ndarray) -> xp.ndarray:
        self.save_for_backward(x)
        return xp.asarray(xp.sum(x), dtype=x.dtype)

    def backward(self, grad_output: xp.ndarray) -> xp.ndarray:
        (x,) = self.saved_tensors
        return xp.ones_like(x) * grad_output

# MatMul is not commutative, so we need to handle the backward pass carefully.
class MatMul(Operation):
    def forward(self, x: xp.ndarray, w: xp.ndarray) -> xp.ndarray:
        self.save_for_backward(x, w)
        # Explicitly calling dot product instead of using the @ symbol
        return x.dot(w)

    def backward(self, grad_output: xp.ndarray) -> tuple[xp.ndarray, xp.ndarray]:
        x, w = self.saved_tensors
        # dL/dx = grad_output dot w.T
        # dL/dw = x.T dot grad_output
        grad_input_x = grad_output.dot(w.T)
        grad_input_w = x.T.dot(grad_output)
        return grad_input_x, grad_input_w

class ReLU(Operation):
    def forward(self, x: xp.ndarray) -> xp.ndarray:
        self.save_for_backward(x)
        return xp.maximum(0, x)

    def backward(self, grad_output: xp.ndarray) -> xp.ndarray:
        x, = self.saved_tensors
        # The gradient is 1 where x > 0, and 0 otherwise.
        grad_input_x = grad_output * (x > 0)
        return grad_input_x

# Softmax is typically used in conjunction with CrossEntropyLoss, but they are implemented separately.
# Softmax is also batch aware, by operating along axis 1 (the feature dimension).
class Softmax(Operation):
    """
    Softmax operation for converting logits to probabilities.
    Uses numerically stable implementation (subtracts max for numerical stability).
    """

    def forward(self, x: xp.ndarray) -> xp.ndarray:
        # Subtract max for numerical stability (prevents overflow)
        shifted_x = x - xp.max(x, axis=1, keepdims=True)
        exps      = xp.exp(shifted_x)
        probs     = exps / xp.sum(exps, axis=1, keepdims=True)

        # Save probabilities for backward pass
        self.save_for_backward(probs)

        return probs

    def backward(self, grad_output: xp.ndarray) -> xp.ndarray:
        probs, = self.saved_tensors

        # Softmax Jacobian: diag(probs) - probs @ probs.T
        # Simplified for batch processing
        grad_input_x = probs * grad_output
        grad_input_x -= probs * xp.sum(
            grad_output * probs, axis=1, keepdims=True
        )

        return grad_input_x

# Cross entropy loss is batch aware and reduces the batch dimension to a single scalar.
# Which is why we average the loss over the batch dimension.
class CrossEntropyLoss(Operation):
    """
    Cross-Entropy loss that expects probabilities (not logits).
    Targets should be one-hot encoded.

    This is separate from Softmax for clarity.
    """

    def forward(self, probs: xp.ndarray, targets: xp.ndarray) -> xp.ndarray:
        batch_size = probs.shape[0]

        # Cross Entropy: -sum(targets * log(probs)) / batch_size
        loss = -xp.sum(targets * xp.log(probs + 1e-7)) / batch_size

        self.save_for_backward(probs, targets)

        return xp.array(loss, dtype=probs.dtype)

    def backward(self, grad_output: xp.ndarray) -> tuple[xp.ndarray, None]:
        probs, targets = self.saved_tensors
        batch_size     = probs.shape[0]

        # Gradient of cross entropy w.r.t. probabilities
        grad_input_probs   = -(targets / (probs + 1e-7)) / batch_size * grad_output
        grad_input_targets = None  # Targets don't require gradients

        return grad_input_probs, grad_input_targets

