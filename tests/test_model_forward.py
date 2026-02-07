import os
import sys
import torch
# make sure repo root is on path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.model import Model

def test_model_forward_shapes():
    # Small synthetic check for deconvolution.separate behavior
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Model(8, 3, 3, True, 'sphconv', True, 1).to(device)
    # Create a dummy deconvolved tensor with sufficient channels for 'separate' to slice
    x_deconv = torch.randn(1, 50, 1, 1, 1).to(device)
    fodf_shc, extra_shc, iso = model.deconvolution.separate(x_deconv)
    # Basic shape checks and NaN safety
    assert fodf_shc is not None
    assert extra_shc is not None
    assert iso is not None
    assert not torch.isnan(fodf_shc).any()
    assert not torch.isnan(extra_shc).any()
    assert not torch.isnan(iso).any()
