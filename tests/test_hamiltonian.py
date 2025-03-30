import pytest
import torch
import numpy as np
from tblite_gpu.hamiltonian import shift_operator

def test_shift_operator():
    # Test with a simple 1D tensor
    input_tensor = torch.tensor([0, 1, 2, 3, 4])
    result = shift_operator(input_tensor)
    expected = torch.tensor([4, 0, 1, 2, 3])
    assert torch.equal(result, expected)

    # You can also test with a numpy array converted to a torch tensor
    input_array = np.array([10, 20, 30, 40])
    input_tensor = torch.tensor(input_array)
    result = shift_operator(input_tensor)
    expected = torch.tensor([40, 10, 20, 30])
    assert torch.equal(result, expected)
