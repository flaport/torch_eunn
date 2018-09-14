# torch_eunn

This repository contains a simple pytorch implementation of a Tunable Efficient Unitary
Neural Network (EUNN) Cell. This implementation was based on the tunable EUNN presented in this paper:
[https://arxiv.org/abs/1612.05231](https://arxiv.org/abs/1612.05231).

## Installation

```
    pip install torch_eunn
```

## Usage
```python
    from torch_eunn import EUNNLayer # feed forward layer
    from torch_eunn import EUNN # Recurrent unit
```

## Requirements

* [PyTorch](http://pytorch.org) >= 0.4.0: `conda install pytorch -c pytorch`
