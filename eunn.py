#############
## Imports ##
#############

# torch
import torch
from torch.nn import Parameter, Module

# numpy
import numpy as np
from numpy import pi


######################
## Useful Functions ##
######################

def cm(ar, ai, br, bi):
    ''' Complex multiplication

        Args:
            ar: real part of first complex number
            bi: imag part of first complex number
            br: real part of second complex number
            bi: imag part of second complex number
    '''
    return ar*br - ai*bi, ar*bi + ai*br


#############
## Modules ##
#############

class EUNNCell(Module):
    ''' Efficient Unitary Neural Network Cell

    This cell works similarly as a torch.nn.Linear cell. The difference in this case
    is however that the internal matrix is a Unitary Matrix.

    This EUNNCell was based on the tunable version of the EUNN proposed in
    https://arxiv.org/abs/1612.05231.
    '''
    def __init__(self, hidden_size, capacity=None):
        ''' EUNNCell __init__

        Args:
            hidden_size (int): the size of the unitary matrix this cell represents.
            capacity (int): 0 < capacity <= hidden_size. This number represents the number of
            layers containing unitary rotations. The higher the capacity, the more of the
            unitary matrix space can be filled. This obviously introduces a speed penalty.
            In recurrent neural networks, a small capacity is usually preferred.
        '''

        Module.__init__(self)

        # handle inputs
        self.hidden_size = int(hidden_size)
        self.capacity = int(capacity) if capacity else self.hidden_size
        self.even_hidden_size = self.hidden_size//2
        self.odd_hidden_size = (self.hidden_size-1)//2
        self.even_capacity = (self.capacity+1)//2
        self.odd_capacity = self.capacity//2

        # Create parameters
        self.omega = Parameter(torch.rand(self.hidden_size)*2*pi-pi)
        self.even_theta = Parameter(torch.rand(self.even_capacity, self.even_hidden_size)*2*pi-pi)
        self.odd_theta = Parameter(torch.rand(self.odd_capacity, self.odd_hidden_size)*2*pi-pi)
        self.even_phi = Parameter(torch.rand(self.even_capacity, self.even_hidden_size)*2*pi-pi)
        self.odd_phi = Parameter(torch.rand(self.odd_capacity, self.odd_hidden_size)*2*pi-pi)

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

    def _v1(self, i, cos_theta, sin_phi, cos_phi):
        ''' vector with diagonal elements of rotation '''
        even = i%2 == 0
        zeros = torch.zeros_like(cos_theta)
        zero = zeros[:1]
        one = torch.ones_like(zero)

        v1_re = torch.cat([(cos_phi*cos_theta)[:,None], cos_theta[:,None]], 1).view(-1)
        v1_im = torch.cat([(sin_phi*cos_theta)[:,None], zeros[:,None]], 1).view(-1)

        if not even:
            v1_re = torch.cat([one, v1_re])
            v1_im = torch.cat([zero, v1_im])
            if self.hidden_size%2 == 0:
                v1_re = torch.cat([v1_re, one])
                v1_im = torch.cat([v1_im, zero])
        elif self.hidden_size%2:
            v1_re = torch.cat([v1_re, one])
            v1_im = torch.cat([v1_im, zero])

        return v1_re, v1_im

    def _v2(self, i, sin_theta, sin_phi, cos_phi):
        ''' vector with off-diagonal elements of rotation '''
        even = i%2 == 0
        zeros = torch.zeros_like(sin_theta)
        zero = zeros[:1]

        v2_re = torch.cat([(-cos_phi*sin_theta)[:,None], sin_theta[:,None]], 1).view(-1)
        v2_im = torch.cat([(-sin_phi*sin_theta)[:,None], zeros[:,None]], 1).view(-1)

        if not even:
            v2_re = torch.cat([zero, v2_re])
            v2_im = torch.cat([zero, v2_im])
            if self.hidden_size%2 == 0:
                v2_re = torch.cat([v2_re, zero])
                v2_im = torch.cat([v2_im, zero])
        elif self.hidden_size%2:
            v2_re = torch.cat([v2_re, zero])
            v2_im = torch.cat([v2_im, zero])

        return v2_re, v2_im

    def _permute(self, i, x_re, x_im):
        ''' permute vector before off-diagonal multiplication '''
        idxs = self._permutation_indices[i%2]
        return x_re[:,idxs], x_im[:,idxs]

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

    def _choose_params(self, i, params):
        ''' choose the parameters to use '''
        if i%2:
            return (params['cos_odd_theta'][i//2], # cos_theta
                    params['sin_odd_theta'][i//2], # sin_theta
                    params['cos_odd_phi'][i//2], # cos_phi
                    params['sin_odd_phi'][i//2]) # sin_phi
        else:
            return (params['cos_even_theta'][i//2], # cos_theta
                    params['sin_even_theta'][i//2], # sin_theta
                    params['cos_even_phi'][i//2], # cos_phi
                    params['sin_even_phi'][i//2]) # sin_phi

    def forward(self, x_re, x_im):
        # calculate derived parameters once:
        params = self._get_params()

        # Loop over the capacity of the matrix
        for i in range(self.capacity):

            # choose even or odd parameters:
            cos_theta, sin_theta, cos_phi, sin_phi = self._choose_params(i, params)

            # get diagonal rotation vector
            v1_re, v1_im = self._v1(i, cos_theta, cos_phi, sin_phi)

            # get off-diagonal rotation vector
            v2_re, v2_im = self._v2(i, sin_theta, cos_phi, sin_phi)

            # perform diagonal part of rotation
            x_re1, x_im1 = cm(x_re, x_im, v1_re, v1_im)

            # perform off-diagonal part of rotation
            x_re2, x_im2 = self._permute(i, x_re, x_im)
            x_re2, x_im2 = cm(x_re2, x_im2, v2_re, v2_im)

            # sum results of diagonal and off-diagonal rotation
            x_re = x_re1 + x_re2
            x_im = x_im1 + x_im2

        # add a final phase
        x_re, x_im = cm(x_re, x_im, torch.cos(self.omega), torch.sin(self.omega))

        # return real and imaginary part of the multiplication:
        return x_re, x_im
