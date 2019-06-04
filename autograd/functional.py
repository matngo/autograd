import numpy as np
from . import tensor as T


def add(t1: "Tensor", t2: "Tensor"):
    """
    Add two tensors

    w = u + v

    if z = f(w)
    dz/dw = grad
    dz/du = dz/dw * dw/du = grad * np.ones_like(u)
    dz/dv = dz/dw * dw/dv = grad * np.ones_like(w)
    
    """
    data = t1.data + t2.data
    requires_grad = t1.requires_grad or t2.requires_grad
    depends_on: list = []
    if t1.requires_grad:
        pass

    if t2.requires_grad:
        pass

    return T.Tensor(data, requires_grad, depends_on)


def sum(t: "Tensor") -> "Tensor":
    """
    Sums up a tensor

    w = sum(u)

    if z = f(w)
    dz/dw = grad -> shape(0)
    dz/du = grad * np.ones_like(u)
    """
    data = t.data.sum()
    requires_grad = t.requires_grad
    depends_on = []
    if requires_grad:

        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad * np.ones_like(t)

        depends_on.append(T.Dependency(t, grad_fn))

    return T.Tensor(data, requires_grad, depends_on)


def opposite(t: "Tensor") -> "Tensor":
    v = T.Tensor(-t.data, -t.requires_grad, t)
    if t.grad is not None:
        v.grad = -t.grad
    return v
