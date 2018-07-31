# EUNN

This repository contains a simple pytorch implementation of a Tunable Efficient Unitary
Neural Network Cell. This implementation was based on this paper:
[https://arxiv.org/abs/1612.05231](https://arxiv.org/abs/1612.05231).

## Roadmap
* Tunable Efficient Unitary ***Recurrent*** Neural Network
* A collection of benchmark tests.

## Possible Bugs

Normally, one would assume that the EUNNCell can approach any unitary matrix. However
the implementation presented here fails to completely converge to the desired matrix
(see the universality check). I am not sure if this is due to the optimization algorithm
or due to a bad implementation of the EUNNCell. All clarification welcome.

## Checks

Some simple checks were performed to test the performance and behavior of the EUNNCell.

### Check Speed
We compare the action of this unitary matrix to the action of a normal complex matrix multiplication


```python
M = 500 # size of the unitary matrix

cell = EUNNCell(M)
x_re = Parameter(torch.rand(M,M))
x_im = Parameter(torch.rand(M,M))
y_re = torch.rand(M,M)
y_im = torch.rand(M,M)

def linear(z_re, z_im): # complex version of torch.nn.Linear()
    return z_re*y_re - z_im*y_im, z_re*y_im + z_im*y_re # complex multiplication

def forward_backward(func, x_re, x_im):
    y_re, y_im = func(x_re, x_im)
    loss = ((1-y_re)**2).sum()
    loss.backward()

%time forward_backward(cell, x_re, x_im)
%time forward_backward(linear, x_re, x_im)
```

    CPU times: user 16.9 s, sys: 1.47 s, total: 18.4 s
    Wall time: 4.66 s
    CPU times: user 15.7 ms, sys: 1.2 ms, total: 16.9 ms
    Wall time: 3.82 ms


We see that this implementation is *much* slower than a normal matrix multiplication. The execution time also increases *very fast* for increasing matrix size.

### Check Unitarity
First we check if the action performed by our EUNNCell is unitary


```python
# dimensionality of the cell
M = 50

# create new cell
cell = EUNNCell(M)

# get result of action of cell on identity matrix:
x_re = torch.eye(M)
x_im = torch.zeros_like(x_re)
x_re, x_im = cell(x_re, x_im)
X = x_re.data.numpy() + 1j*x_im.data.numpy()

# check unitarity of result
print(np.abs(X@X.T.conj()))
```

    [[1. 0. 0. ... 0. 0. 0.]
     [0. 1. 0. ... 0. 0. 0.]
     [0. 0. 1. ... 0. 0. 0.]
     ...
     [0. 0. 0. ... 1. 0. 0.]
     [0. 0. 0. ... 0. 1. 0.]
     [0. 0. 0. ... 0. 0. 1.]]


We see that the operation of this EUNNCell is clearly unitary.

### Check Universality
Next we check if a full capacity cell can approximate any unitary matrix


```python
# dimensionality of the cell
M = 10

# create unitary matrix to approximate
U, _, _ = np.linalg.svd(np.random.randn(M,M) + 1j*np.random.randn(M,M)) # unitary matrix U

# create new cell
cell = EUNNCell(M, M)

# train the cell so that the action of the cell on U.T.conj() yields the identity
real_matrix = torch.tensor(np.real(U.T.conj()), dtype=torch.float32)
imag_matrix = torch.tensor(np.imag(U.T.conj()), dtype=torch.float32)
real_target = torch.eye(M)
imag_target = torch.zeros((M,M))
lossfunc = torch.nn.MSELoss()
optimizer = torch.optim.Adam(cell.parameters(), lr=0.0020)
steps = range(5000)
for _ in steps:
    optimizer.zero_grad()
    real_result, imag_result = cell(real_matrix, imag_matrix)
    loss = lossfunc(real_result, real_target) + lossfunc(imag_result, imag_target)
    loss.backward()
    optimizer.step()

result = real_result.detach().numpy() + 1j*imag_result.detach().numpy()
print(abs(result)**2)
```

    [[0.97 0.   0.   0.   0.   0.   0.   0.   0.   0.01]
     [0.   0.99 0.   0.   0.   0.   0.   0.   0.   0.01]
     [0.   0.   0.95 0.   0.01 0.01 0.   0.   0.01 0.02]
     [0.   0.   0.   0.98 0.   0.   0.   0.   0.   0.01]
     [0.   0.   0.01 0.   0.97 0.   0.   0.   0.   0.01]
     [0.   0.   0.01 0.   0.   0.96 0.   0.   0.   0.02]
     [0.   0.   0.   0.   0.   0.   0.98 0.   0.   0.02]
     [0.01 0.   0.   0.   0.   0.   0.   0.96 0.   0.02]
     [0.   0.   0.   0.   0.   0.   0.   0.   0.94 0.05]
     [0.01 0.01 0.02 0.01 0.01 0.03 0.01 0.03 0.04 0.83]]





We can approach the desired matrix closely, but not completely. I am not sure if this is a bug, or if I should optimize with an even smaller learning rate.