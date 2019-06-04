from autograd.tensor import Tensor
import autograd.functional as F
import numpy as np


def test_init_tensor():
    t = Tensor([1, 2, 3])
    assert t.shape == (3,)


def test_sum_tensor():
    u = Tensor([1, 2, 3], requires_grad=True)
    s = u.sum()
    s.backward(2)
    np.testing.assert_equal(6, s.numpy())
    assert len(s.depends_on) == 1
    assert s.requires_grad
    np.testing.assert_equal(s.grad, 2)
    np.testing.assert_equal(u.grad, [2, 2, 2])


def test_add_tensor_forward():
    u = Tensor([1, 2, 3])
    v = Tensor([1, 2, 3])
    w = u + v

    np.testing.assert_equal([2, 4, 6], w.numpy())


def test_add_tensor_backward():
    u = Tensor([1, 2, 3], requires_grad=True)
    v = Tensor([1, 2, 3], requires_grad=True)
    w = u + v

    w.backward([1, 1, 1])
    np.testing.assert_equal(w.grad, [1, 1, 1])
    np.testing.assert_equal(u.grad, [1, 1, 1])
    np.testing.assert_equal(v.grad, [1, 1, 1])


def test_add_tensor_backward_broadcast():
    u = Tensor([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ], requires_grad=True)
    v = Tensor([1, 2, 3], requires_grad=True)

    w = u + v
    backward_grad = [ 
        [1, 2, 3],    
        [1, 2, 3],    
        [1, 2, 3]
    ]

    w.backward(backward_grad)
    
    np.testing.assert_equal(u.grad, backward_grad)
    np.testing.assert_equal(v.grad, [3, 6, 9])

def test_add_tensor_backward_broadcast_kept_dims():
    u = Tensor([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ], requires_grad=True)
    v = Tensor([[1, 2, 3]], requires_grad=True)

    w = u + v
    backward_grad = [ 
        [1, 2, 3],    
        [1, 2, 3],    
        [1, 2, 3]
    ]

    w.backward(backward_grad)
    
    np.testing.assert_equal(u.grad, backward_grad)
    np.testing.assert_equal(v.grad, [[3, 6, 9]])


def test_neg_tensor():
    u = Tensor([1, 2, 3], requires_grad=True)
    v = -u
    v.backward([1, 1, 1])
    np.testing.assert_equal(v.numpy(), [-1, -2, -3])
    np.testing.assert_equal(v.grad, [1, 1, 1])
    np.testing.assert_equal(u.grad, [-1, -1, -1])


def test_sub_tensors():
    u = Tensor([1, 2, 3], requires_grad=True)
    v = Tensor([1, 2, 3], requires_grad=True)
    w = u - v
    w.backward([1, 1, 1])
    np.testing.assert_equal(w.numpy(), [0, 0, 0])
    np.testing.assert_equal(v.grad, [-1, -1, -1])
    np.testing.assert_equal(u.grad, [1, 1, 1])


def test_mul_tensors():
    u = Tensor([1, 2, 3], requires_grad=True)
    v = Tensor([4, 5, 6], requires_grad=True)
    w = u * v
    w.backward([5, 5, 5])
    np.testing.assert_equal(w.data, [4, 10, 18])
    np.testing.assert_equal(w.grad, [5, 5, 5])
    np.testing.assert_equal(u.grad, [20, 25, 30])
    np.testing.assert_equal(v.grad, [5, 10, 15])


def test_mul_tensors_broadcast():
    u = Tensor([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ], requires_grad=True)

    v = Tensor([4, 5, 6], requires_grad=True)
    w = u * v
    expected_w = [
        [4, 10, 18],
        [16, 25, 36],
        [28, 40, 54]
    ]

    expected_grad_u = [
        [8, 10, 12],
        [8, 10, 12],
        [8, 10, 12]
    ]

    expected_grad_v = [24, 30, 36]

    w.backward(np.ones_like(u) * 2)
    np.testing.assert_equal(w.numpy(), expected_w)
    np.testing.assert_equal(u.grad, expected_grad_u)
    np.testing.assert_equal(v.grad, expected_grad_v)


def test_inverse():
    u = Tensor([1, 4, 9], requires_grad=True)
    v = F.inverse(u)
    v.backward([1, 1, 1])
    np.testing.assert_almost_equal(v.data, [1, 1/4, 1/9])
    np.testing.assert_almost_equal(u.grad, [1/2, 1/4, 1/6])


def test_divide():
    u = Tensor([1, 2, 3])
    v = u / u
    np.testing.assert_equal([1, 1, 1], v.data)


def test_divide_backward():
    u = Tensor([4, 6, 12], requires_grad=True)
    v = Tensor([1, 4, 9], requires_grad=True)
    w = u/v
    w.backward([1, 1, 1])
    np.testing.assert_almost_equal(u.grad, 1/v.data)
    np.testing.assert_almost_equal(v.grad, [2, 1.5, 2.])
