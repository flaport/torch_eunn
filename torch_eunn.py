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

''' An Efficient Unitary Neural Network implementation for PyTorch

based on https://arxiv.org/abs/1612.05231

'''

name = 'torch_eunn'
__author__ = 'Floris Laporte'
__version__ = '0.1.3'


#############
## Imports ##
#############

import torch
from math import pi


######################
## Useful Functions ##
######################

def cm(x, y):
    ''' Complex multiplication between two torch tensors

        Args:
            x (torch.tensor): A torch float tensor with the real and imaginary part
                stored in the last dimension of the tensor; i.e. x.shape = [a, ...,b, 2]
            y (torch.tensor): A torch float tensor with the real and imaginary part
                stored in the last dimension of the tensor; i.e. y.shape = [a, ...,b, 2]
    '''
    result = torch.stack([
        x[...,0]*y[...,0] - x[...,1]*y[...,1],
        x[...,0]*y[...,1] + x[...,1]*y[...,0]
    ], -1)
    return result


##################
## Modular ReLU ##
##################

class ModReLU(torch.nn.Module):
    ''' A modular ReLU activation function for complex-valued tensors '''
    def __init__(self):
        super(ModReLU, self).__init__()
        self.relu = torch.nn.ReLU()

    def forward(self, x, eps=1e-5):
        ''' ModReLU forward

        Args:
            x (torch.tensor): A torch float tensor with the real and imaginary part
                stored in the last dimension of the tensor; i.e. x.shape = [a, ...,b, 2]
        Kwargs:
            eps (float): A small number added to the norm of the complex tensor for
                numerical stability.
        '''
        x_re, x_im = x[...,0], x[...,1]
        norm = torch.sqrt(x_re**2 + x_im**2) + 1e-5
        phase_re, phase_im = x_re/norm, x_im/norm
        activated_norm = self.relu(norm)
        modrelu = torch.stack([activated_norm*phase_re, activated_norm*phase_im], -1)
        return modrelu


###################
## Forward Layer ##
###################

class EUNNLayer(torch.nn.Module):
    ''' Efficient Unitary Neural Network Layer

    This layer works similarly as a torch.nn.Linear layer. The difference in this case
    is however that the action of this layer can be represented by a unitary matrix.

    This EUNNLayer was based on the tunable version of the EUNN proposed in
    https://arxiv.org/abs/1612.05231.
    '''
    def __init__(self, hidden_size, capacity=None):
        ''' EUNNLayer __init__

        Args:
            hidden_size (int): the size of the unitary matrix this cell represents.
            capacity (int): 0 < capacity <= hidden_size. This number represents the
                number of layers containing unitary rotations. The higher the capacity,
                the more of the unitary matrix space can be filled. This obviously
                introduces a speed penalty. In recurrent neural networks, a small
                capacity is usually preferred.
        '''

        super(EUNNLayer, self).__init__()

        # handle inputs
        self.hidden_size = int(hidden_size)
        self.capacity = int(capacity) if capacity else self.hidden_size
        self.even_hidden_size = self.hidden_size//2
        self.odd_hidden_size = (self.hidden_size-1)//2
        self.even_capacity = (self.capacity+1)//2
        self.odd_capacity = self.capacity//2

        # Create parameters
        self.omega = torch.nn.Parameter(torch.rand(self.hidden_size)*2*pi-pi)
        self.even_theta = torch.nn.Parameter(torch.rand(self.even_capacity, self.even_hidden_size)*2*pi-pi)
        self.odd_theta = torch.nn.Parameter(torch.rand(self.odd_capacity, self.odd_hidden_size)*2*pi-pi)
        self.even_phi = torch.nn.Parameter(torch.rand(self.even_capacity, self.even_hidden_size)*2*pi-pi)
        self.odd_phi = torch.nn.Parameter(torch.rand(self.odd_capacity, self.odd_hidden_size)*2*pi-pi)

        # Permutation indices for off diagonal rotation multiplications
        if self.hidden_size%2==0:
            self._permutation_indices = {
                0 : [int(i+0.5) for i in torch.arange(self.even_hidden_size*2).view(-1,2)[:,[1,0]].view(-1).numpy()],
                1 : [0] + [int(i+0.5) for i in torch.arange(1, self.odd_hidden_size*2+1).view(-1,2)[:,[1,0]].view(-1).numpy()] + [self.odd_hidden_size*2+1]
            }
        else:
            self._permutation_indices = {
                0 : [int(i+0.5) for i in torch.arange(self.even_hidden_size*2).view(-1,2)[:,[1,0]].view(-1).numpy()] + [self.even_hidden_size*2],
                1 : [0] + [int(i+0.5) for i in torch.arange(1, self.odd_hidden_size*2+1).view(-1,2)[:,[1,0]].view(-1).numpy()]
            }

    def _v_diag(self, i, cos_theta, sin_phi, cos_phi):
        ''' vector with diagonal elements of rotation '''
        even = i%2 == 0
        zeros = torch.zeros_like(cos_theta)
        zero = zeros[:1]
        one = torch.ones_like(zero)

        v_diag_re = torch.stack([(cos_phi*cos_theta), cos_theta], 1).view(-1)
        v_diag_im = torch.stack([(sin_phi*cos_theta), zeros], 1).view(-1)

        if not even:
            v_diag_re = torch.cat([one, v_diag_re])
            v_diag_im = torch.cat([zero, v_diag_im])
            if self.hidden_size%2 == 0:
                v_diag_re = torch.cat([v_diag_re, one])
                v_diag_im = torch.cat([v_diag_im, zero])
        elif self.hidden_size%2:
            v_diag_re = torch.cat([v_diag_re, one])
            v_diag_im = torch.cat([v_diag_im, zero])

        return torch.stack([v_diag_re, v_diag_im], -1)

    def _v_off_diag(self, i, sin_theta, sin_phi, cos_phi):
        ''' vector with off-diagonal elements of rotation '''
        even = i%2 == 0
        zeros = torch.zeros_like(sin_theta)
        zero = zeros[:1]

        v_off_diag_re = torch.stack([(-cos_phi*sin_theta), sin_theta], 1).view(-1)
        v_off_diag_im = torch.stack([(-sin_phi*sin_theta), zeros], 1).view(-1)

        if not even:
            v_off_diag_re = torch.cat([zero, v_off_diag_re])
            v_off_diag_im = torch.cat([zero, v_off_diag_im])
            if self.hidden_size%2 == 0:
                v_off_diag_re = torch.cat([v_off_diag_re, zero])
                v_off_diag_im = torch.cat([v_off_diag_im, zero])
        elif self.hidden_size%2:
            v_off_diag_re = torch.cat([v_off_diag_re, zero])
            v_off_diag_im = torch.cat([v_off_diag_im, zero])

        return torch.stack([v_off_diag_re, v_off_diag_im], -1)

    def _permute(self, i, x):
        ''' permute vector before off-diagonal multiplication '''
        idxs = self._permutation_indices[i%2]
        return x[:,idxs,:]

    def _get_params(self):
        ''' derived parameters '''
        params = {
            'cos_even_theta' : torch.cos(self.even_theta),
            'sin_even_theta' : torch.sin(self.even_theta),
            'cos_odd_theta' : torch.cos(self.odd_theta),
            'sin_odd_theta' : torch.sin(self.odd_theta),
            'cos_even_phi' : torch.cos(self.even_phi),
            'sin_even_phi' : torch.sin(self.even_phi),
            'cos_odd_phi' : torch.cos(self.odd_phi),
            'sin_odd_phi' : torch.sin(self.odd_phi),
        }
        return params

    def _rotation_vectors(self, i, params):
        ''' choose the parameters to use '''
        if i%2:
            cos_theta, sin_theta, cos_phi, sin_phi = (params['cos_odd_theta'][i//2], # cos_theta
                                                      params['sin_odd_theta'][i//2], # sin_theta
                                                      params['cos_odd_phi'][i//2], # cos_phi
                                                      params['sin_odd_phi'][i//2]) # sin_phi
        else:
            cos_theta, sin_theta, cos_phi, sin_phi = (params['cos_even_theta'][i//2], # cos_theta
                                                      params['sin_even_theta'][i//2], # sin_theta
                                                      params['cos_even_phi'][i//2], # cos_phi
                                                      params['sin_even_phi'][i//2]) # sin_phi

        # get diagonal rotation vector
        v_diag = self._v_diag(i, cos_theta, cos_phi, sin_phi)

        # get off-diagonal rotation vector
        v_off_diag = self._v_off_diag(i, sin_theta, cos_phi, sin_phi)

        return v_diag, v_off_diag

    def forward(self, x):
        ''' forward pass through the layer

        Args:
            x (torch.tensor): Tensor with shape (batch_size, feature_size, 2=(real|imag))
        '''

        # calculate derived parameters once:
        params = self._get_params()

        # Loop over the capacity of the matrix
        for i in range(self.capacity):

            # get rotation vectors
            v_diag, v_off_diag = self._rotation_vectors(i, params)

            # perform diagonal part of rotation
            x_diag = cm(x, v_diag)

            # perform off-diagonal part of rotation
            x_off_diag = self._permute(i, x)
            x_off_diag = cm(x_off_diag, v_off_diag)

            # sum results of diagonal and off-diagonal rotation
            x = x_diag + x_off_diag

        # add a final phase
        x = cm(x, torch.stack([torch.cos(self.omega), torch.sin(self.omega)], -1))

        # return real and imaginary part of the multiplication:
        return x


####################
## Recurrent Unit ##
####################

class EUNN(torch.nn.Module):
    ''' Pytorch EUNN Recurrent unit

    This recurrent cell works similarly as a torch.nn.RNN layer. The difference in this
    case is however that the action of the internal weight matrix can be represented by
    a unitary matrix.

    This EUNN was based on the tunable version of the EUNN proposed in
    https://arxiv.org/abs/1612.05231.
    '''
    def __init__(self, input_size, hidden_size, capacity=2, output_type='real', batch_first=False):
        ''' EUNN __init__

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
        '''
        super(EUNN, self).__init__()
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.output_function = {
            'real':lambda x:x[...,0],
            'imag':lambda x:x[...,1],
            'complex':lambda x:x,
            'abs':lambda x: torch.sqrt(torch.sum(x**2, -1))
        }[output_type]
        self.output_type = output_type

        self.input_layer = torch.nn.Linear(input_size, hidden_size, bias=False)
        self.hidden_layer = EUNNLayer(hidden_size, capacity=capacity)
        self.modrelu = ModReLU()


    def forward(self, input, state=None):
        ''' forward pass through the RNN

        Args:
            input (torch.tensor): Tensor with shape
                (sequence_length, batch_size, feature_size, 2) if batch_first==False
                (batch_size, sequence_length, feature_size, 2) if batch_first==True

        '''

        # apply the input layer to the input up front
        input_shape = input.shape
        input = self.input_layer(input.view(-1, input_shape[-1])).view(*(input_shape[:-1] + (-1,)))

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
            output.append(self.output_function(state)) # take real part

        # stack output
        if self.batch_first:
            output = torch.stack(output, 1)
        else:
            output = torch.stack(output, 0)

        # return output and internal state
        return output, state
