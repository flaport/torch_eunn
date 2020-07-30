""" tests for torch_eunn """

import torch
import numpy as np
from torch_eunn import EUNN


def test_unitarity():
    num_hidden = 50
    cell = EUNN(num_hidden)
    x = torch.stack(
        [torch.eye(num_hidden, num_hidden), torch.zeros(num_hidden, num_hidden)], -1
    )
    y = cell(x)
    y = y[..., 0].detach().numpy() + 1j * y[..., 1].detach().numpy()
    diag = np.abs(y @ y.T.conj())

    np.testing.assert_array_almost_equal(diag, np.eye(num_hidden))


def test_universality():
    num_hidden = 8
    cell = EUNN(num_hidden, num_hidden)
    random_state = np.random.RandomState(seed=42)
    U, _, _ = np.linalg.svd(
        random_state.randn(num_hidden, num_hidden)
        + 1j * random_state.randn(num_hidden, num_hidden)
    )
    U_torch = torch.stack(
        [
            torch.tensor(np.real(U.T.conj()), dtype=torch.float32),
            torch.tensor(np.imag(U.T.conj()), dtype=torch.float32),
        ],
        -1,
    )
    I_torch = torch.stack(
        [torch.eye(num_hidden), torch.zeros((num_hidden, num_hidden)),], -1
    )
    lossfunc = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(cell.parameters(), lr=0.5)
    range_ = range(400)
    for _ in range_:
        optimizer.zero_grad()
        I_approx = cell(U_torch)
        loss = lossfunc(I_approx, I_torch)
        loss.backward()
        optimizer.step()
    result = (
        np.abs(
            I_approx[..., 0].detach().numpy() + 1j * I_approx[..., 1].detach().numpy()
        )
        ** 2
    )

    np.testing.assert_array_almost_equal(result, np.eye(num_hidden), decimal=2)


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
