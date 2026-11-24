"""
This is a code equivalent of the backpropagation example in the first part of the youtube video.

Run with:
uv run video-part1.py
"""

from nn import Tensor

a = Tensor(4, requires_grad=True)
b = Tensor(1, requires_grad=True)
c = Tensor(5, requires_grad=True)
d = Tensor(-2, requires_grad=True)

L = ((a - b) + c) * d

print(f"Value of L: {L} \n")

L.mermaid_graph()

