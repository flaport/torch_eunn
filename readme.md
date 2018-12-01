# torch_eunn

This repository contains a simple PyTorch implementation of a Tunable Efficient Unitary
Neural Network (EUNN) Cell. This implementation was based on the tunable EUNN presented in this paper:
[https://arxiv.org/abs/1612.05231](https://arxiv.org/abs/1612.05231).

## Installation

```
    pip install torch_eunn
```

## Usage
```python
    from torch_eunn import EUNNLayer # feed forward layer
    from torch_eunn import EUNN # recurrent unit
```

## Examples

* 00: [Simple Tests](examples/00_simple_tests.ipynb)
* 01: [Copying Task](examples/01_copying_task.ipynb)

## Requirements

* [PyTorch](http://pytorch.org) >= 0.4.0: `conda install pytorch -c pytorch`
