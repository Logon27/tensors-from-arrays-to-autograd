from __future__ import annotations

from typing import Any, Sequence
from nn.array_backend import xp
from nn.operations import Operation, Add, Sub, Mul, Div, Sum
import numpy as _np

# Used purely for generating the Mermaid graph visualization of the computation graph.
import json
import zlib
import base64


# -------------------------------------------------------------------------
# Tensor implementation
# -------------------------------------------------------------------------

class Tensor:
    """
    A multi-dimensional array containing elements of a single data type.
    Tracks operations to facilitate automatic differentiation (autograd).
    """

    def __init__(
        self,
        data         : Any,
        dtype        : str | xp.dtype = "float32",
        operation    : Operation | None = None,
        requires_grad: bool = False,
    ):
        # Ensure the data is a proper array of the specified dtype, using the backend (NumPy or CuPy).
        self.data = self._ensure_array(data, dtype)

        # The operation that resulted in the creation of this tensor (None if leaf/buffer)
        self.operation = operation

        # The gradient is initialized to None and populated during backward(), resulting in an ndarray of the same shape as self.data.
        self.grad: xp.ndarray | None = None
        self.requires_grad = requires_grad

    def _ensure_array(self, data: Any, dtype: str | xp.dtype) -> xp.ndarray:
        """Ensures the incoming data is cast to the correct array type and dtype, ensuring consistency across the tensor operations."""
        target_dtype = xp.dtype(dtype)

        if isinstance(data, xp.ndarray):
            array = data
        else:
            try:
                array = xp.asarray(data)
            except Exception as exc:
                raise TypeError(f"Unable to convert data to array: {exc}") from exc

        if array.dtype != target_dtype:
            array = array.astype(target_dtype, copy=False)
        
        return array

    # -------------------------------------------------------------------------
    # Factory Methods
    # -------------------------------------------------------------------------
    @classmethod
    def _apply_op(cls, op_class: type[Operation], *tensors: Tensor, **kwargs: Any) -> Tensor:
        """Runs the forward pass, wires up the graph, and builds the resulting Tensor."""
        # 1. Instantiate the operation class with the input tensors
        operation = op_class(*tensors)
        
        # 2. Extract raw ndarrays from the tensors
        inputs = [t.data for t in tensors]
        
        # 3. Execute the forward pass using ndarrays (not tensors)
        out_data = operation.forward(*inputs, **kwargs)
        
        # 4. Determine if gradient tracking is needed
        requires_grad = any(t.requires_grad for t in tensors)
        
        # 5. Create and return the resulting Tensor
        return cls(
            data          = out_data,
            dtype         = out_data.dtype,
            operation     = operation if requires_grad else None, # If requires_grad is False, we do not need to track the operation for backpropagation.
            requires_grad = requires_grad
        )

    # -------------------------------------------------------------------------
    # Math Operations (Forward Pass)
    # -------------------------------------------------------------------------

    # Converts scalars or arrays to Tensors for graph tracking. This is used in operator overloads like __add__ and __mul__ to ensure that the other operand is a Tensor.
    def _ensure_tensor(self, value: Any) -> Tensor:
        """Promotes scalars or arrays to Tensors for graph tracking."""
        if isinstance(value, Tensor):
            return value
        
        array = xp.asarray(value, dtype=self.dtype)
        return Tensor(
            data          = array,
            dtype         = array.dtype,
            requires_grad = False,
        )

    def __add__(self, other: Any) -> Tensor:
        return self._apply_op(Add, self, self._ensure_tensor(other))

    # Addition is commutative, so we can reuse __add__ for the right-hand side.
    def __radd__(self, other: Any) -> Tensor:
        return self.__add__(other)

    def __sub__(self, other: Any) -> Tensor:
        return self._apply_op(Sub, self, self._ensure_tensor(other))

    def __rsub__(self, other: Any) -> Tensor:
        return self._apply_op(Sub, self._ensure_tensor(other), self)

    def __mul__(self, other: Any) -> Tensor:
        return self._apply_op(Mul, self, self._ensure_tensor(other))

    # Multiplication is commutative, so we can reuse __mul__ for the right-hand side.
    def __rmul__(self, other: Any) -> Tensor:
        return self.__mul__(other)

    def __truediv__(self, other: Any) -> Tensor:
        return self._apply_op(Div, self, self._ensure_tensor(other))

    def __rtruediv__(self, other: Any) -> Tensor:
        return self._apply_op(Div, self._ensure_tensor(other), self)

    def sum(self) -> Tensor:
        """Return a scalar tensor containing the sum of all elements."""
        return self._apply_op(Sum, self)


    # -------------------------------------------------------------------------
    # Automatic Differentiation (Backward Pass)
    # -------------------------------------------------------------------------

    def _topological_sort(self) -> list[Tensor]:
        """
        Return tensors in computation order, with inputs coming before outputs.

        We use depth-first search to traverse the computation graph.
        `backward()` reverses the order to propagate gradients backward through the graph.
        """
        visited: set[int]     = set()
        order  : list[Tensor] = []

        def visit(node: Tensor) -> None:
            if id(node) in visited:
                return
            visited.add(id(node))
            if node.operation is not None:
                for parent in node.operation.parents:
                    visit(parent)
            order.append(node)

        visit(self)
        return order

    def backward(self) -> None:
        """
        Computes the gradient of current tensor w.r.t. graph leaves.
        """
        if not self.requires_grad:
            raise RuntimeError("Cannot call backward on a tensor that does not require grad.")

        if self.data.size != 1:
            raise RuntimeError("backward() can only be called on a scalar tensor.")

        # The gradient of the loss is seeded as the scalar value 1 because of the identity rule in differential calculus:
        # The derivative of any variable with respect to itself is always 1.
        self.grad = xp.ones_like(self.data, dtype=self.dtype)

        # Traverse the topological sort in reverse order to propagate gradients backward through the computation graph.
        # Each node is just a tensor in the graph.
        for node in reversed(self._topological_sort()):
            # Leaf tensors have no operation, so there are no parents to propagate gradients to.
            if node.operation is None:
                continue

            # grad_output / node.grad is the "upstream gradient" — the gradient of the loss w.r.t. this node's output.
            grad_output = node.grad

            # The operation's backward() computes the input gradients ("downstream gradients")
            # by multiplying the "local gradients" with the output gradient ("upstream gradient").
            # Returns one grad_input per parent tensor (grad_input_x, grad_input_y, ...).
            grad_inputs = node.operation.backward(grad_output)

            # Safety check: ensure grad_inputs is a tuple for the subsequent zip function. (e.g. An operation like ReLU has only 1 input)
            if not isinstance(grad_inputs, tuple):
                grad_inputs = (grad_inputs,)

            # Propagate the operation's input gradients to the parent tensors, accumulating them if necessary.
            for parent, grad_input in zip(node.operation.parents, grad_inputs):
                # grad_input can be None: operations return None for inputs that don't need gradients (e.g. CrossEntropyLoss targets).
                if parent.requires_grad and grad_input is not None:
                    reduced_grad = Tensor._reduce_grad(grad_input, parent.shape)
                    # Accumulate the gradient into the parent tensor's ".grad" attribute.
                    parent._accumulate_grad(reduced_grad)

    def _accumulate_grad(self, grad: xp.ndarray) -> None:
        """Accumulate gradients when multiple downstream paths flow into the same tensor."""
        if self.grad is None:
            self.grad = grad
        else:
            self.grad = self.grad + grad

    @staticmethod
    def _reduce_grad(grad: xp.ndarray, shape: Sequence[int]) -> xp.ndarray:
        """
        Reduces a broadcasted gradient back to the original tensor shape.
        """
        target_shape = tuple(shape)

        # If the gradient already matches the target shape, return it as is.
        if grad.shape == target_shape:
            return grad

        # CASE 1 — rank grew: the gradient has MORE dimensions than the parent tensor.
        # Broadcasting prepends leading axes that did not exist in the parent.
        # e.g. Add(x, y) with x=(3,) and y=(2, 3): x is expanded to shape (2, 3) and then
        # added to y, producing an output of shape (2, 3), so dx also comes back as (2, 3).
        # Summing axis 0 repeatedly removes these extra leading axes until the rank matches
        # len(target_shape). This also applies when the parent is a scalar () added to an
        # array, where dx arrives with that array's rank and is reduced back to ().
        while grad.ndim > len(target_shape):
            grad = grad.sum(axis=0)

        # CASE 2 — rank unchanged, but a size-1 axis was inflated to a larger size.
        # e.g. Add(x, y) with x=(3, 1) and y=(3, 5): x is treated as shape (3, 5) in the
        # forward pass, producing an output of shape (3, 5), so dx also comes back as (3, 5).
        # Ranks already match, so we sum along the axis that was 1 (here axis 1) with
        # keepdims=True to shrink 5 -> 1 without changing the number of dimensions.
        for axis, target_dim in enumerate(target_shape):
            if target_dim == 1 and grad.shape[axis] != 1:
                grad = grad.sum(axis=axis, keepdims=True)

        return grad.reshape(target_shape)


    # -------------------------------------------------------------------------
    # Utility Properties and Methods
    # -------------------------------------------------------------------------

    # Example:
    # Tensor.zeros((2, 2), requires_grad=True)
    @classmethod
    def zeros(
        cls,
        shape: Sequence[int],
        dtype: str | xp.dtype = "float32",
        requires_grad: bool = False,
    ) -> Tensor:
        data = xp.zeros(shape, dtype=dtype)
        return cls(data=data, dtype=dtype, requires_grad=requires_grad)

    @property
    def dtype(self) -> xp.dtype:
        return self.data.dtype
    
    @property
    def shape(self) -> tuple[int, ...]:
        """Returns the shape of the tensor as a tuple of integers."""
        return self.data.shape

    # When printing the tensor as a string, just display the underlying ndarray data.
    def __str__(self) -> str:
        return str(self.data)

    def numpy(self) -> _np.ndarray:
        """Moves data to CPU and returns as a NumPy array."""
        # If the backend is CuPy, we use asnumpy() to move the data to CPU. If it's already a NumPy array, we simply return it.
        if hasattr(xp, "asnumpy"):
            return xp.asnumpy(self.data)
        return xp.asarray(self.data)


    # -------------------------------------------------------------------------
    # Visualization and Debugging
    # -------------------------------------------------------------------------

    # Example: print(repr(Tensor(xp.array([[1, 2], [3, 4]]), requires_grad=True)))
    def __repr__(self) -> str:
        # 1. Format data string (indent if it contains newlines)
        raw_data = str(self.data)
        data_str = f"\n    {raw_data.replace('\n', '\n    ')}" if "\n" in raw_data else f"{raw_data}"

        # 2. Format grad string (handle None, and indent if it contains newlines)
        if self.grad is None:
            grad_str = "None"
        else:
            raw_grad = str(self.grad)
            grad_str = f"\n    {raw_grad.replace('\n', '\n    ')}" if "\n" in raw_grad else f"{raw_grad}"

        # 3. Build a cleaner block template
        return (
            f"Tensor(\n"
            f"    shape = {self.shape}, dtype = {self.dtype}, requires_grad = {self.requires_grad},\n"
            f"    data = {data_str},\n"
            f"    grad = {grad_str}\n"
            f")"
        )

    @property
    def parents(self) -> tuple[Tensor, ...]:
        """Returns the direct parent tensors in the computation graph."""
        return self.operation.parents if self.operation is not None else ()

    def print_graph(self, indent: int = 0) -> None:
        """Prints the computation graph tree starting from this tensor."""
        prefix = "  " * indent
        op_name = self.operation.name if self.operation is not None else "Leaf Tensor"
        print(f"{prefix}{op_name}: shape={self.shape}, requires_grad={self.requires_grad}")
        
        for parent in self.parents:
            parent.print_graph(indent + 1)

    def mermaid_graph(self, ops_as_nodes: bool = True, include_grad: bool = True) -> str:
        """Returns a Mermaid Live Editor URL for the autograd tree.

        :param include_grad: Include requires_grad info in tensor labels.
        :param ops_as_nodes: If True, renders operations as explicit intermediate nodes
                            between tensor inputs and outputs.
        """
        seen: set[int] = set()
        seen_ops: set[int] = set()
        lines = ["graph TD"]
        edges: list[tuple[str, str]] = []

        def visit(node: Tensor) -> str:
            node_id = f"n{id(node)}"

            if id(node) in seen:
                return node_id

            seen.add(id(node))

            shape_text = "(" + ", ".join(str(dim) for dim in node.shape) + ")" if node.shape else "()"
            grad_text = f"<br/>requires_grad={node.requires_grad}" if include_grad else ""

            if ops_as_nodes:
                # Tensors are labeled simply as Tensors or Buffers
                lines.append(f'    {node_id}["Tensor<br/>shape={shape_text}{grad_text}"]')

                if node.operation is not None:
                    op_id = f"op{id(node.operation)}"
                    
                    # Ensure operation node is defined only once
                    if id(node.operation) not in seen_ops:
                        seen_ops.add(id(node.operation))
                        lines.append(f'    {op_id}(["Op: {node.operation.name}"])')

                    # Wire: Operation -> Output Tensor
                    edges.append((op_id, node_id))

                    # Wire: Parent Tensors -> Operation
                    for parent in node.operation.parents:
                        parent_id = visit(parent)
                        edges.append((parent_id, op_id))
            else:
                # Default behavior: fusion of tensor and operation metadata into one node
                label = node.operation.name if node.operation is not None else "Tensor"
                lines.append(f'    {node_id}["{label}<br/>shape={shape_text}{grad_text}"]')

                if node.operation is not None:
                    for parent in node.operation.parents:
                        parent_id = visit(parent)
                        edges.append((parent_id, node_id))

            return node_id

        visit(self)

        for src, dst in edges:
            lines.append(f"    {src} --> {dst}")

        code = "\n".join(lines)

        # Mermaid Live Editor state
        state = {
            "code": code,
            "mermaid": json.dumps({"theme": "default"}),
        }

        # JSON -> zlib/DEFLATE -> URL-safe base64
        data = json.dumps(state, separators=(",", ":")).encode("utf-8")
        compressed = zlib.compress(data, level=9)
        encoded = base64.urlsafe_b64encode(compressed).decode("ascii")
        
        url = f"https://mermaid.live/view#pako:{encoded}"
        print(f"Mermaid Web Visualization URL: \n{url}")
        return url

