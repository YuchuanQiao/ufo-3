import os
import sys
import pytest
# make sure repo root is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data import create_dataset_train


def test_create_dataset_train_returns_iterable():
    # Use existing small dataset file if available
    train_txt = './data/3025.txt'
    if not os.path.exists(train_txt):
        pytest.skip('Small dataset file ./data/3025.txt not found; skip data loading test')
    cfg = {
        'size_3d_patch': 9,
        'dataset_mode': 'chcps',
        'train_params': {
            'train_txtfile': train_txt,
            'featrues_CHCP': 1,
            'max_order': 8
        }
    }
    train_list = [train_txt]
    # Some environments may not have the optional dataset implementations; skip in that case
    try:
        import importlib
        importlib.import_module('data.chcps_dataset')
    except Exception:
        pytest.skip('Dataset implementation data.chcps_dataset not available; skip data loading test')

    dataset = create_dataset_train(cfg, train_list, 1)
    # Expect an iterable with at least one batch
    iterator = iter(dataset)
    batch = next(iterator, None)
    assert batch is not None
    assert 'fodlr' in batch
    assert 'Y' in batch
    assert 'G' in batch
    assert 'mask' in batch
