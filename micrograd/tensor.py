from typing import Union, Tuple
import math


class Value:
    """
    This class emulates a Tensor (from pytorch, for example) and computes
    its gradient.

    Args:
        data: int or float as a single node to be computed its gradient.
    """
    def __init__(self, data: Union[int, float], _children: Tuple = ()):
        self.data = data
        self.grad = 0.0

        # Internal attributes used for building autograd graph.
        self._previous = set(_children)
        self._backward = lambda: None

    def __repr__(self) -> str:
        return f'Value(data={self.data}, grad={self.grad})'

    def __add__(self, other):
        output = Value(
            data=self.data + other.data,
            _children=(self, other)
        )

        # Backward for add operation. Just remembering an add operation is just
        # a distributor of gradient.
        def _backward():
            self.grad += output.grad
            other.grad += output.grad

        output._backward = _backward

        return output

    def __mul__(self, other):
        output = Value(
            data=self.data * other.data,
            _children=(self, other)
        )

        # Backward for multiplication operation. Just remembering a mul
        # operation is composed by a multiplication between the data and the
        # gradient.
        def _backward():
            self.grad += other.data * output.grad
            other.grad += self.data * output.grad

        output._backward = _backward

        return output

    def tanh(self):
        x = self.data
        tan = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        output = Value(
            data=tan,
            _children=(self, )
        )

        # Backward for hyperbolic tangent operation.
        def _backward():
            self.grad += (1 - tan ** 2) * output.grad

        output._backward = _backward

        return output
