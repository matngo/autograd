from autograd.tensor import Tensor
import numpy as np

def test_init_tensor():
   t = Tensor([1,2,3]) 
   assert t.shape == (3,)

def test_sum_tensor():
    u = Tensor([1,2,3], requires_grad=True)
    s = u.sum()
    s.backward(2)
    np.testing.assert_equal(6, s.numpy())
    assert len(s.depends_on) == 1
    assert s.requires_grad
    np.testing.assert_equal(2, s.grad)
    np.testing.assert_equal([2, 2, 2], u.grad)


def test_neg_tensor():
    u = Tensor([1, 2, 3], requires_grad=True)
    v = -u
    np.testing.assert_equal([-1, -2, -3], v.numpy())

