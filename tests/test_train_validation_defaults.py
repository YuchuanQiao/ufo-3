import os
import sys
import yaml
import pytest
import torch
# make repo importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import train


def test_train_runs_without_top_level_val_and_patience(monkeypatch, tmp_path):
    # Load base smoke config and remove top-level val_iters/patience
    cfg = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), '..', 'train_smoke.yaml'), 'r'))
    cfg.pop('val_iters', None)
    cfg.pop('patience', None)

    # small experiment artifacts go to tmp
    cfg['save_dir'] = str(tmp_path)
    cfg['train_params']['n_dense_epochs'] = 1
    cfg['train_params']['initial_epoch'] = 0
    # skip validation in this quick unit test (we only check no crash for missing keys)
    cfg['val_iters'] = 0

    # Provide tiny dataset that yields a single safe batch
    class OneBatchDataset:
        def __iter__(self):
            batch = {
                'nside16sh8': torch.tensor([0]),
                'fodlr': torch.zeros(1, 45, 1, 1, 1),
                'mask': torch.ones(1, 1, 1, 1),
                'fodgt': torch.zeros(1, 45, 1, 1, 1),
                'table': torch.zeros(1),
                'Y': torch.zeros(1),
                'G': torch.zeros(1),
            }
            yield batch
        def __len__(self):
            return 1

    monkeypatch.setattr(train, 'create_dataset_train', lambda cfg_, lst, prev_nf: OneBatchDataset())
    monkeypatch.setattr(train, 'create_dataset_val', lambda cfg_, lst, prev_nf: OneBatchDataset())

    # ensure save dir exists (main() will attempt to write files)
    os.makedirs(os.path.join(cfg['save_dir'], cfg['name']), exist_ok=True)

    # monkeypatch swanlab.log to avoid requiring swanlab.init() in tests
    monkeypatch.setattr(train.swanlab, 'log', lambda *a, **k: None)

    # Tiny deterministic model (no NaNs) - return None for deconvolved SHC to skip einsum
    class TinyModel:
        def __init__(self, *a, **k):
            import torch
            self._p = torch.nn.Parameter(torch.zeros(1))
        def to(self, device):
            # move param to device but keep leaf status
            import torch
            self._p = torch.nn.Parameter(self._p.data.to(device))
            return self
        def load_state_dict(self, *a, **k):
            return
        def state_dict(self):
            return {'_p': self._p.detach().cpu()}
        def parameters(self):
            return [self._p]
        def named_parameters(self):
            return [('param_0', self._p)]
        def eval(self):
            return self
        def train(self):
            return self
        def __call__(self, inputs, nside16sh8, table, A):
            # return: reconstructed, fod_pred_shc, extra_trapped_shc
            # create outputs from parameter (avoid reusing input storage which may be modified in-place later)
            outputs = (1.0 + self._p) * torch.ones_like(inputs)
            # We return None for the SHC deconvolved tensors to avoid running einsum in training
            return outputs, None, None

    monkeypatch.setattr(train, 'Model', TinyModel)

    # run - should not raise
    train.main(cfg)
