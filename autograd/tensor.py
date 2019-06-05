import numpy as np
from typing import Union, Sequence, List, NamedTuple, Callable, Optional
from . import functional as F

Arrayable = Union[Sequence[float], float, np.ndarray]


class Dependency(NamedTuple):
    """
    """

    tensor: "Tensor"
    grad_fn: Callable[[np.ndarray], np.ndarray]


def ensure_array(arrayable: Arrayable) -> np.ndarray:
    if isinstance(arrayable, np.ndarray):
        return arrayable.astype("float32")
    else:
        return np.array(arrayable, dtype="float32")


class Tensor:
    """
    A tensor object is a multidimensional array.
    It keeps a memory of the tensors used to construct it in order to backpropagate
    the gradient.

    :example: 
    w = Tensor(u) + Tensor(v)
    dw/du = np.ones_like(u)
    dw/dv = np.ones_like(v)
    """

    def __init__(
        self,
        data: Arrayable,
        requires_grad: bool = False,
        depends_on: List[Dependency] = None,
    ) -> None:

        self.data = ensure_array(data)
        self.requires_grad = requires_grad
        self.depends_on = depends_on or []
        self.grad:Optional[np.ndarray] = None

        if self.requires_grad:
            self.zero_grad()

    
    def __add__(self, other:"Tensor") -> "Tensor":
        return F.add(self, other)

    def __radd__(self, other:"Tensor") -> "Tensor":
        return F.add(self, other)

    def __neg__(self) -> "Tensor":
        return F.opposite(self)

    def __sub__(self, other:"Tensor") -> "Tensor":
        return self + -other

    def __mul__(self, other:"Tensor") -> "Tensor":
        return F.multiply(self, other)

    def __matmul__(self, other:"Tensor") -> "Tensor":
        return F.dot(self, other)

    def __truediv__(self, other:"Tensor") -> "Tensor":
        return F.divide(self, other)

    def __inv__(self) -> "Tensor":
        return F.invert(self)

    @property
    def shape(self) -> tuple:
        return self.data.shape

    def sum(self) -> "Tensor":
        return F.sum(self)

    def numpy(self) -> np.ndarray:
        return self.data

    def list(self) -> list:
        return self.numpy().tolist()

    def backward(self, grad: Arrayable=None) -> np.ndarray:
        assert self.requires_grad, "called backward on a non requires_grad tensor"
        if grad is None:
            if self.shape == ():
                grad = np.array(1.)
            else: 
                raise RuntimeError("Must have a backward grad")

        else:
            grad = ensure_array(grad)

        self.grad = self.grad +  grad
        
        for dependency in self.depends_on:
            backward_grad = dependency.grad_fn(self.grad)
            dependency.tensor.backward(backward_grad)

    def zero_grad(self):
        self.grad = np.zeros_like(self.data)


    def sqrt(self):
        return F.sqrt(self)
