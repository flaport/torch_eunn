#   Copyright 2018 Floris Laporte
#   MIT License

#   Permission is hereby granted, free of charge, to any person obtaining a copy of this
#   software and associated documentation files (the "Software"), to deal in the Software
#   without restriction, including without limitation the rights to use, copy, modify,
#   merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
#   permit persons to whom the Software is furnished to do so, subject to the following
#   conditions:

#   The above copyright notice and this permission notice shall be included in all copies
#   or substantial portions of the Software.

#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
#   INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
#   PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
#   HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
#   CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR
#   THE USE OR OTHER DEALINGS IN THE SOFTWARE.

""" An Efficient Unitary Neural Network implementation for PyTorch

based on https://arxiv.org/abs/1612.05231

"""


## Properties

name = "torch_eunn"
__author__ = "Floris Laporte"
__version__ = "0.2.0"


## Imports

import torch
from math import pi


## Useful Functions


def cm(x, y):
    """ Complex multiplication between two torch tensors

        Args:
            x (torch.tensor): A torch float tensor with the real and imaginary part
                stored in the last dimension of the tensor; i.e. x.shape = [a, ...,b, 2]
            y (torch.tensor): A torch float tensor with the real and imaginary part
                stored in the last dimension of the tensor; i.e. y.shape = [a, ...,b, 2]
    """
    result = torch.stack(
        [
            x[..., 0] * y[..., 0] - x[..., 1] * y[..., 1],
            x[..., 0] * y[..., 1] + x[..., 1] * y[..., 0],
        ],
        -1,
    )
    return result


## Modular ReLU


class ModReLU(torch.nn.Module):
    """ A modular ReLU activation function for complex-valued tensors """

    def __init__(self, size):
        super(ModReLU, self).__init__()
        self.bias = torch.nn.Parameter(torch.rand(1, size))
        self.relu = torch.nn.ReLU()

    def forward(self, x, eps=1e-5):
        """ ModReLU forward

        Args:
            x (torch.tensor): A torch float tensor with the real and imaginary part
                stored in the last dimension of the tensor; i.e. x.shape = [a, ...,b, 2]
        Kwargs:
            eps (float): A small number added to the norm of the complex tensor for
                numerical stability.
        """
        x_re, x_im = x[..., 0], x[..., 1]
        norm = torch.sqrt(x_re ** 2 + x_im ** 2) + 1e-5
        phase_re, phase_im = x_re / norm, x_im / norm
        activated_norm = self.relu(norm + self.bias)
        modrelu = torch.stack(
            [activated_norm * phase_re, activated_norm * phase_im], -1
        )
        return modrelu


## Feed-forward Layer


class EUNN(torch.nn.Module):
    """ Efficient Unitary Neural Network Layer

    This layer works similarly as a torch.nn.Linear layer. The difference in this case
    is however that the action of this layer can be represented by a unitary matrix.

    This EUNN is loosely based on the tunable version of the EUNN proposed in
    https://arxiv.org/abs/1612.05231. However, the last diagonal matrix of phases
    was removed in favor of periodic boundary conditions. This makes the algorithm
    considerably faster and more stable.

    """

    def __init__(self, hidden_size, capacity=None):
        """ EUNN __init__

        Args:
            hidden_size (int): the size of the unitary matrix this cell represents.
            capacity (int): 0 < capacity <= hidden_size. This number represents the
                number of layers containing unitary rotations. The higher the capacity,
                the more of the unitary matrix space can be filled. This obviously
                introduces a speed penalty. In recurrent neural networks, a small
                capacity is usually preferred.
        """
        # validate parameters
        if hidden_size % 2 != 0:
            raise ValueError("EUNN hidden_size should be even")
        if capacity is None:
            capacity = hidden_size
        elif capacity % 2 != 0:
            raise ValueError("EUNN capacity should be even")

        self.hidden_size = int(round(hidden_size))
        self.capacity = int(round(capacity))

        # initialize
        super(EUNN, self).__init__()

        # phi and theta for the even layers (count starts at 0)
        self.phi0 = torch.nn.Parameter(
            2 * pi * torch.randn(self.hidden_size // 2, self.capacity // 2)
        )
        self.theta0 = torch.nn.Parameter(
            2 * pi * torch.randn(self.hidden_size // 2, self.capacity // 2)
        )

        # phi and theta for the odd layers (count starts at 0)
        self.phi1 = torch.nn.Parameter(
            2 * pi * torch.randn(self.hidden_size // 2, self.capacity // 2)
        )
        self.theta1 = torch.nn.Parameter(
            2 * pi * torch.randn(self.hidden_size // 2, self.capacity // 2)
        )

    def forward(self, x):
        """ forward pass through the layer

        Args:
            x (torch.tensor): Tensor with shape (batch_size, hidden_size, 2=(real|imag))
        """

        # get and validate shape of input tensor:
        bs, hidden_size, ri = x.shape
        if hidden_size != self.hidden_size:
            raise ValueError(
                "Input tensor for EUNN Layer has size %i, "
                "but the EUNN Layer expects a size of %i"
                % (hidden_size, self.hidden_size)
            )
        elif ri != 2:
            raise ValueError(
                "Input tensor for EUNN Layer should be complex, "
                "with the complex components stored in the last dimension (x.shape[2]==2)"
            )

        # calculate the sin and cos of rotaion angles
        cos_phi0 = torch.cos(self.phi0)
        sin_phi0 = torch.sin(self.phi0)
        cos_theta0 = torch.cos(self.theta0)
        sin_theta0 = torch.sin(self.theta0)
        cos_phi1 = torch.cos(self.phi1)
        sin_phi1 = torch.sin(self.phi1)
        cos_theta1 = torch.cos(self.theta1)
        sin_theta1 = torch.sin(self.theta1)

        # calculate the rotation vectors
        # shape = (capacity//2, 1, hidden_size, 2=(real|imag))
        zeros = torch.zeros_like(cos_theta0)
        diag0 = (
            torch.stack(
                [
                    torch.stack([cos_phi0 * cos_theta0, cos_theta0], 1).view(
                        -1, self.capacity // 2
                    ),
                    torch.stack([sin_phi0 * cos_theta0, zeros], 1).view(
                        -1, self.capacity // 2
                    ),
                ],
                -1,
            )
            .unsqueeze(0)
            .permute(2, 0, 1, 3)
        )
        offdiag0 = (
            torch.stack(
                [
                    torch.stack([-cos_phi0 * sin_theta0, sin_theta0], 1).view(
                        -1, self.capacity // 2
                    ),
                    torch.stack([-sin_phi0 * sin_theta0, zeros], 1).view(
                        -1, self.capacity // 2
                    ),
                ],
                -1,
            )
            .unsqueeze(0)
            .permute(2, 0, 1, 3)
        )

        diag1 = (
            torch.stack(
                [
                    torch.stack([cos_phi1 * cos_theta1, cos_theta1], 1).view(
                        -1, self.capacity // 2
                    ),
                    torch.stack([sin_phi1 * cos_theta1, zeros], 1).view(
                        -1, self.capacity // 2
                    ),
                ],
                -1,
            )
            .unsqueeze(0)
            .permute(2, 0, 1, 3)
        )
        offdiag1 = (
            torch.stack(
                [
                    torch.stack([-cos_phi1 * sin_theta1, sin_theta1], 1).view(
                        -1, self.capacity // 2
                    ),
                    torch.stack([-sin_phi1 * sin_theta1, zeros], 1).view(
                        -1, self.capacity // 2
                    ),
                ],
                -1,
            )
            .unsqueeze(0)
            .permute(2, 0, 1, 3)
        )

        # loop over the capacity
        for d0, d1, o0, o1 in zip(diag0, diag1, offdiag0, offdiag1):
            # first layer
            x_perm = torch.stack([x[:, 1::2], x[:, ::2]], 2).view(
                bs, self.hidden_size, 2
            )
            x = cm(x, d0) + cm(x_perm, o0)

            # periodic boundary conditions
            x = torch.cat([x[:, 1:], x[:, :1]], 1)

            # second layer
            x_perm = torch.stack([x[:, 1::2], x[:, ::2]], 2).view(
                bs, self.hidden_size, 2
            )
            x = cm(x, d1) + cm(x_perm, o1)

            # periodic boundary conditions
            x = torch.cat([x[:, -1:], x[:, :-1]], 1)

        return x


## Recurrent Unit


class EURNN(torch.nn.Module):
    """ Pytorch EURNN Recurrent unit

    This recurrent cell works similarly as a torch.nn.RNN layer. The difference in this
    case is however that the action of the internal weight matrix can be represented by
    a unitary matrix.

    """

    def __init__(
        self, input_size, hidden_size, capacity=2, output_type="real", batch_first=False
    ):
        """ EURNN __init__

        Args:
            input_size (int): the size of the input vector
            hidden_size (int): the size of the internal unitary matrix.
            capacity (int): 0 < capacity <= hidden_size. This number represents the
                number of layers containing unitary rotations. The higher the capacity,
                the more of the unitary matrix space can be filled. This obviously
                introduces a speed penalty. In recurrent neural networks, a small
                capacity is usually preferred.
            output_type (str): how to return the output. Allowed values are
                'complex', real', 'imag', 'abs'.
            batch_first (bool): if the first dimension of the input vector is the
                batch or the sequence.
        """
        super(EURNN, self).__init__()
        self.hidden_size = int(hidden_size)
        self.batch_first = int(batch_first)
        self.output_function = {
            "real": lambda x: x[..., 0],
            "imag": lambda x: x[..., 1],
            "complex": lambda x: x,
            "abs": lambda x: torch.sqrt(torch.sum(x ** 2, -1)),
        }[output_type]
        self.output_type = output_type

        self.input_layer = torch.nn.Linear(input_size, hidden_size, bias=True)
        self.hidden_layer = EUNN(hidden_size, capacity=capacity)
        self.modrelu = ModReLU(hidden_size)

    def forward(self, input, state=None):
        """ forward pass through the RNN

        Args:
            input (torch.tensor): Tensor with shape
                (sequence_length, batch_size, feature_size, 2) if batch_first==False
                (batch_size, sequence_length, feature_size, 2) if batch_first==True

        """

        # apply the input layer to the input up front
        input_shape = input.shape
        input = self.input_layer(input.view(-1, input_shape[-1])).view(
            *(input_shape[:-1] + (-1,))
        )

        # make input complex by adding an all-zero dimension
        input = torch.stack([input, torch.zeros_like(input)], -1)

        # unstack the input
        if self.batch_first:
            input = torch.unbind(input, 1)
        else:
            input = torch.unbind(input, 0)

        # if no internal state is given, create one
        if state is None:
            with torch.no_grad():
                state = torch.zeros_like(input[0])

        # recurrent loop
        output = []
        for inp in input:
            state = self.modrelu(inp + self.hidden_layer(state))
            output.append(self.output_function(state))  # take real part

        # stack output
        if self.batch_first:
            output = torch.stack(output, 1)
        else:
            output = torch.stack(output, 0)

        # return output and internal state
        return output, state
