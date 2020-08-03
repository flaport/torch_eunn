# torch_eunn

This repository contains a simple PyTorch implementation of a Tunable
Efficient Unitary Neural Network (EUNN) Cell.

The implementation is loosely based on the tunable EUNN presented in
this paper:
[https://arxiv.org/abs/1612.05231](https://arxiv.org/abs/1612.05231).

## Installation

```
    pip install torch_eunn
```

## Usage

```python
    from torch_eunn import EUNN # feed forward layer
    from torch_eunn import EURNN # recurrent unit
```

#### Note

The `hidden_size` of the EUNN needs to be **_even_**, as explained in
the section _"Difference with original implementation"_.

## Examples

- 00: [Simple Tests](examples/00_simple_tests.ipynb)
- 01: [Copying Task](examples/01_copying_task.ipynb)
- 02: [MNIST Task](examples/02_mnist.ipynb)

## Requirements

- [PyTorch](http://pytorch.org) >= 0.4.0: `conda install pytorch -c pytorch`

## Difference with original implementation

This implementation of the EUNN has a major difference with the
original implementation proposed in
[https://arxiv.org/abs/1612.05231](https://arxiv.org/abs/1612.05231),
which is outlined below.

In the original implementation, the first output of the top mixing
unit of a capacity-2 sublayer skips the second layer of mixing units
(indicated with dots in the ascii figure below) to connect to the next
capacity-2 sublayer of the EUNN. The reverse happens at the bottom,
where the first layer of the capacity-2 sublayer is skipped. This way,
a `(2*n+1)`-dimensional unitary matrix representation is created, with
`n` the number of mixing units in each capacity-1 sublayer.

```
  __  __......
    \/
  __/\____  __
          \/
  __  ____/\__
    \/
  __/\____  __
          \/
  ......__/\__
```

For each capacity-1 sublayer with `N=2*n+1` inputs (`N` odd), we thus
have `N-1` parameters (each mixing unit has 2 parameters). Thus to
have a unitary matrix representation which spans the full unitary
space, one needs `N` capacity-1 layers **_and_** `N` _extra_ phases
appended to the back of the capacity-`N` sublayer to bring the total
number of parameters in the unitary-matrix representation to `N**2`
(the total number of independent parameters in a unitary matrix).

In the implementation proposed here, the dots in each capacity-2
sublayer are connected onto themselves (periodic boundaries). This has
the implication that for each capacity-1 sublayer with `n` mixing
units, there are `N=2*n` inputs and as many independent parameters.
This means that we just need `N` capacity-1 sublayers and **no**
_extra_ phases to span the full unitary space with `N` parameters.

This, however, has the implication that the `hidden_size = N = 2*n` of
the unitary matrix should always be _even_. Moreover, this periodic
boundary representation removes the physical interpretability, where
the mixing units can for example be represented by Mach Zehnder
intereferometers, as physically, it's practically infeasible to make
connections from the top of the mesh to the bottom of the mesh.
However, from a purely Machine Learning point of view, the
representation used here is _more stable_ and _converges faster_.

## License

© Floris Laporte, MIT license.
