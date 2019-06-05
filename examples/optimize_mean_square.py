from autograd.tensor import Tensor
import numpy as np


NUM_ITER = 10000
LEARNING_RATE = 0.3
x = Tensor(np.random.randn(200), requires_grad=True)
y = Tensor(np.random.randn(200))
N = Tensor(x.shape[0])


for i in range(NUM_ITER):
    x.zero_grad()
    w = x - y
    z = (w * w).sqrt().sum() / N
    z.backward()
    x -= LEARNING_RATE * x.grad
    print(i, z.data)
