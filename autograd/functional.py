import numpy as np
from . import tensor as T
from functools import partial


def add(u: "Tensor", v: "Tensor"):
    """
    Add two tensors

    w = u + v

    if z = f(w)
    dz/dw = grad
    dz/du = dz/dw * dw/du = grad * np.ones_like(u)
    dz/dv = dz/dw * dw/dv = grad * np.ones_like(w)
    
    """

    def grad_fn(tensor: "Tensor", grad: np.ndarray) -> np.ndarray:
        added_dims = grad.ndim - tensor.numpy().ndim
        for _ in range(added_dims):
            grad = grad.sum(axis=0)

        for axis, size in enumerate(tensor.shape):
            if size == 1:
                grad = grad.sum(axis=axis, keepdims=True)

        return grad * np.ones_like(tensor)

    data = u.data + v.data
    requires_grad = u.requires_grad or v.requires_grad
    depends_on: list = [
        T.Dependency(t, partial(grad_fn, t)) for t in [u, v] if t.requires_grad
    ]

    return T.Tensor(data, requires_grad, depends_on)


def sum(u: "Tensor") -> "Tensor":
    """
    Sums up a tensor

    w = sum(u)

    if z = f(w)
    dz/dw = grad -> shape(0)
    dz/du = grad * np.ones_like(u)
    """
    data = u.data.sum()
    requires_grad = u.requires_grad
    depends_on = []
    if requires_grad:

        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad * np.ones_like(u)

        depends_on.append(T.Dependency(u, grad_fn))

    return T.Tensor(data, requires_grad, depends_on)


def opposite(u: "Tensor") -> "Tensor":
    """
    Returns the opposite of the tensor

    v = -u
    if z = f(v)
    dz/dv = grad -> shape(0)
    dz/du = grad * -1 * np.ones_like(u)
    """
    depends_on = []

    if u.requires_grad:

        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return -grad

        depends_on.append(T.Dependency(u, grad_fn))

    return T.Tensor(-u.data, u.requires_grad, depends_on)


def multiply(u: "Tensor", v: "Tensor") -> "Tensor":
    """
    returns u * v
    w = u * v
    if z = f(w)
    dz/dw = grad
    dz/du = dz/dw * dw/du = grad * v
    dz/dv = dz/dw * dw/dv = grad * u
    """
    def grad_fn(this: "Tensor", other: "Tensor", grad: np.ndarray) -> np.ndarray:
        added_dims = grad.ndim - this.numpy().ndim
        grad = other.data * grad

        for _ in range(added_dims):
            grad = grad.sum(axis=0)

        for axis, size in enumerate(other.shape):
            if size == 1:
                grad = grad.sum(axis=axis, keepdims=True)

        return grad

    data = u.data * v.data
    requires_grad = u.requires_grad or v.requires_grad
    depends_on = [
        T.Dependency(x, partial(grad_fn, x, y))
        for x, y in zip([v, u], [u, v])
        if x.requires_grad
    ]

    return T.Tensor(data, requires_grad, depends_on)

def inverse(u: "Tensor") -> "Tensor":
    """
    returns 1/u
    w = 1/u
    if z = f(w)
    dz/dw = grad
    dz/du = dz/dw * dw/du = grad * (1/ (2 * sqrt (u)))
    """
    data = 1./ u.data
    requires_grad = u.requires_grad

    def grad_fn(grad: np.ndarray) -> np.ndarray:
        return - grad / (u.data ** 2)

    depends_on = [T.Dependency(u, grad_fn)]

    return T.Tensor(data, requires_grad, depends_on)

def divide(u: "Tensor", v: "Tensor") -> "Tensor":
    return u * inverse(v)


def dot(u: "Tensor", v: "Tensor") -> "Tensor":
    data = u.data @ v.data
    requires_grad = u.requires_grad or v.requires_grad
    depends_on = []
    if u.requires_grad:
        def grad_fn_1(grad: np.ndarray) -> np.ndarray:
            return grad @ v.data.T

        depends_on.append(T.Dependency(u, grad_fn_1))


    if v.requires_grad:
        def grad_fn_2(grad: np.ndarray) -> np.ndarray:
            return u.data.T @ grad

        depends_on.append(T.Dependency(v, grad_fn_2))

    return T.Tensor(data, requires_grad, depends_on)

def sqrt(u: "Tensor") -> "Tensor":
    data = np.sqrt(u.data)
    requires_grad = u.requires_grad
    depends_on = []
    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad / 2 * np.sqrt(u.data)

        depends_on.append(T.Dependency(u, grad_fn))

    return T.Tensor(data, requires_grad, depends_on)
