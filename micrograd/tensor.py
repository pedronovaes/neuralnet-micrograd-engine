from typing import Union, Tuple


class Value:
    """
    This class emulates a Tensor (from pytorch, for example) and computes
    its gradient.
    """
    def __init__(self, data: Union[int, float], _children: Tuple = ()):
        self.data = data
        self.grad = 0.0

        # Internal attributes used for building autograd graph.
        self._previous = set(_children)
        self._backward = lambda: None

    def __repr__(self) -> str:
        return f'Value(data={self.data})'
