from autograd.nn import Linear, Identity, Module
from autograd.tensor import Tensor
import numpy as np

def test_identity():
    u = Tensor([2,3], requires_grad=True)
    identity = Identity(2)
    v = identity.forward(u)
    
    # test forward
    np.testing.assert_equal(v.data, u.data)
    
    v.backward(1)
    # test backward
    np.testing.assert_equal(u.grad, [1, 1])
    for _, p in identity.parameters:
        assert p.grad is not None

def test_module():
    u = Tensor([2,3], requires_grad=True)
    mod = Module([
        Identity(2),
        Identity(2),
    ])

    v = mod.forward(u)

    # test forward
    np.testing.assert_equal(v.data, u.data)

    l = v.sum()
    l.backward()

    # test backward
    np.testing.assert_equal(u.grad, [1, 1])
    
    for _, p in mod.parameters:
        assert p.grad is not None
