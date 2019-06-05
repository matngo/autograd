from .tensor import Tensor
import numpy as np
from typing import Dict, Sequence, Union


def init_parameter(*shape):
    return Tensor(np.random.randn(*shape), requires_grad=True)


NetUnit = Union["Layer", "Module"]


class Layer:
    def __init__(self):
        self._parameters: Dict[str, Tensor] = {}

    def forward(self, input: Tensor) -> Tensor:
        raise NotImplementedError

    @property
    def parameters(self):
        for name, parameter in self._parameters.items():
            yield name, parameter

    def __call__(self, inputs):
        return self.forward(inputs)


class Module(Layer):
    def __init__(self, layers):
        self.layers = layers

    def forward(self, input: Tensor) -> Tensor:
        intermediate_results = [input]
        for layer in self.layers:
            intermediate_results.append(layer(intermediate_results[-1]))

        return intermediate_results[-1]

    @property
    def parameters(self):
        for layer in self.layers:
            yield from layer.parameters


class Linear(Layer):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self._parameters["weights"] = init_parameter(in_features, out_features)
        self._parameters["bias"] = init_parameter(in_features)

    def forward(self, input) -> Tensor:
        return input @ self._parameters["weights"] + self._parameters["bias"]


class Identity(Layer):
    def __init__(self, in_features):
        super().__init__()
        self._parameters["weights"] = Tensor(
            np.identity(in_features), requires_grad=True
        )

    def forward(self, input) -> Tensor:
        return input @ self._parameters["weights"]
