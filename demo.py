"""Walkthrough of Tensor creation, operations, computation graphs, and gradients.

Covers:
  1. Creating tensors from constants, Python lists, and NumPy arrays.
  2. Adding two scalar tensors.
  3. Composing operations and visualizing the resulting computation graph.
  4. Backpropagation when each input is used once.
  5. Gradient accumulation when an input is used more than once.
  6. Backpropagation through broadcasting, including gradient reduction.

Run with:
uv run demo.py
"""

import numpy as np
from nn import Tensor


def banner(title: str) -> None:
    print("\n\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ---------------------------------------------------------------------------
# 1. Creating tensors
# ---------------------------------------------------------------------------
banner("1. Creating tensors")

# From a Python constant (scalar)
a = Tensor(2.0)

# From a Python list
b = Tensor([1.0, 2.0, 3.0])

# From an existing NumPy array
array = np.array([[1.0, 2.0], [3.0, 4.0]])
c = Tensor(array)

print("a (from constant)   :", a, ", shape =", a.shape)
print("b (from list)       :", b, ", shape =", b.shape)
print("c (from numpy array):\n", c, "\nshape =", c.shape)


# ---------------------------------------------------------------------------
# 2. Simple operation
# ---------------------------------------------------------------------------
banner("2. Simple add")

x = Tensor(3.0)
y = Tensor(4.0)

add = x + y

print("x + y =", add)


# ---------------------------------------------------------------------------
# 3. Composing operations without reusing variables
# ---------------------------------------------------------------------------
banner("3. Composing operations")

a = Tensor(2.0, requires_grad=True)
b = Tensor(3.0, requires_grad=True)
c = Tensor(4.0, requires_grad=True)
d = Tensor(5.0, requires_grad=True)

# Take the equation e = (a + b) * (c - d)
e = (a + b) * (c - d)
# Note that each operator creates a new tensor.
# This means that two intermediate tensors are created behind the scenes for (a + b) and (c - d) before multiplying them together.
# In this case we have...
# 4 input tensors: a, b, c, and d
# 2 intermediate tensors: (a + b) and (c - d)
# 1 output tensor: e

print("\nTextual graph:")
e.print_graph()

# Much easier to visualize this with the Mermaid graph.
print("\nMermaid graph for `e`:")
e.mermaid_graph()

# Important Note: Only tensors that have requires_grad=True will be tracked in the computation graph.
# The computation graph is a gradient tracking graph, so we can prune any nodes that don't require gradients.
# For example try changing BOTH `c` and `d` to have requires_grad=False and see how the graph changes.

# ---------------------------------------------------------------------------
# 4. Gradients with backward() — no shared variables
# ---------------------------------------------------------------------------
banner("4. Gradients with backward() (no shared variables)")

a = Tensor(2.0, requires_grad=True)
b = Tensor(3.0, requires_grad=True)
c = Tensor(4.0, requires_grad=True)

z = a * b + c

print("z = a * b + c =", z)

# `z` inherits requires_grad=True from its inputs:
# If any input tensor for the operation has requires_grad=True, then the output tensor will also have requires_grad=True.
# This is what allows backward() to build the gradient path back to the input tensors.
print("z.requires_grad =", z.requires_grad)

print("\nMermaid graph with operations as explicit nodes:")
z.mermaid_graph(ops_as_nodes=True)

# Backward computes gradients through the graph using the chain rule.
z.backward()
# After its execution, the gradients are stored in the `.grad` attribute of each tensor that has requires_grad=True.

# z = a * b + c
# Each variable is used exactly ONCE, so each gradient flows along a single path.
# dz/da = b      = 3
# dz/db = a      = 2
# dz/dc = 1      = 1
print("\nAfter backward():")
print("z.grad =", z.grad)
print("a.grad =", a.grad)
print("b.grad =", b.grad)
print("c.grad =", c.grad)


# ===========================================================================
# ADVANCED — the two edge cases of backpropagation.
# Sections 1-4 cover the core idea. These two are optional if you're
# just getting started.
# ===========================================================================


# ---------------------------------------------------------------------------
# 5. Gradients with backward() — a shared variable
# ---------------------------------------------------------------------------
banner("5. Gradients with backward() (shared variable)")

a = Tensor(2.0, requires_grad=True)
b = Tensor(3.0, requires_grad=True)

# `a` is used TWICE, so its gradient is accumulated from two paths.
z = a * (a * b)

print("z = a * (a * b) =", z)
print("z.operation =", z.operation.name)

print("\nMermaid graph with operations as explicit nodes:")
z.mermaid_graph(ops_as_nodes=True)

# Backward computes gradients through the graph using the chain rule.
z.backward()

# z = a * (a * b)
# we can simplify this to the equivalent formula...
# z = a^2 * b,
# dz/da = 2 * a * b = 12    a appears TWICE -> its gradient is accumulated from two paths
# dz/db = a^2       = 4     b appears once  -> a single path (no accumulation)
print("\nAfter backward():")
print("z.grad =", z.grad)
print("a.grad =", a.grad)
print("b.grad =", b.grad)


# ---------------------------------------------------------------------------
# 6. Gradients with backward() — broadcasting
# ---------------------------------------------------------------------------
banner("6. Gradients with backward() (broadcasting)")

a = Tensor(2.0, requires_grad=True)
b = Tensor([1.0, 2.0, 3.0], requires_grad=True)

# The scalar `a` is broadcast to each element of `b`.
y = a * b
print("y = a * b =", y)
print("y.shape =", y.shape)

# backward() needs a scalar output, so sum the broadcasted result.
z = y.sum()
z.mermaid_graph(ops_as_nodes=True)
z.backward()

# For z = sum(a * b), Mul.backward() produces a vector gradient for `a`:
#   dz/da before reduction = b = [1, 2, 3]
# _reduce_grad() sums it back to `a`'s scalar shape:
#   dz/da = sum(b) = 6
# The gradient for b has the same shape as b:
#   dz/db = [a, a, a] = [2, 2, 2]
print("\nAfter backward():")
print("z =", z)
print("z.grad =", z.grad)
print("y.grad =", y.grad)
print("a.grad =", a.grad)
print("b.grad =", b.grad)

