import os
import sys
import yaml
import pytest
import torch
# make repo importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import train


def test_nan_dump_is_created_when_model_outputs_nan(monkeypatch, tmp_path):
    cfg = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), '..', 'train_smoke.yaml'), 'r'))
    cfg['save_dir'] = str(tmp_path)
    cfg['train_params']['n_dense_epochs'] = 1
    cfg['train_params']['initial_epoch'] = 0
    cfg['dump_on_nan'] = True

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

    # Fake model that returns NaNs in outputs to trigger dump
    class NaNModel:
        def __init__(self, *a, **k):
            import torch
            self._p = torch.nn.Parameter(torch.zeros(1))
        def to(self, device):
            # move param to device in a way that preserves leaf status (wrap data)
            import torch
            self._p = torch.nn.Parameter(self._p.data.to(device))
            return self
        def load_state_dict(self, *a, **k):
            return
        def state_dict(self):
            return {'_p': self._p.detach().cpu()}
        def parameters(self):
            return [self._p]
        def eval(self):
            return self
        def train(self):
            return self
        def __call__(self, inputs, nside16sh8, table, A):
            # produce NaN tensors on same device as inputs
            outputs = torch.full_like(inputs, float('nan'))
            fod_pred = inputs.new_full((inputs.size(0), 45, *inputs.shape[3:]), float('nan'))
            return outputs, fod_pred, None

    monkeypatch.setattr(train, 'Model', NaNModel)

    # run - should not raise and should produce a dump
    train.main(cfg)

    # check for dump directory and at least one file
    save_path = os.path.join(cfg['save_dir'], cfg['name'])
    dump_dir = os.path.join(save_path, 'nan_dumps')
    # small wait if file system is slow
    assert os.path.exists(dump_dir), f"Expected dump dir {dump_dir} to exist"
    files = os.listdir(dump_dir)
    assert len(files) > 0, 'Expected at least one nan dump file'

    monkeypatch.setattr(train, 'Model', NaNModel)

    # run - should not raise and should produce a dump
    train.main(cfg)

    # check for dump directory and at least one file
    save_path = os.path.join(cfg['save_dir'], cfg['name'])
    dump_dir = os.path.join(save_path, 'nan_dumps')
    # small wait if file system is slow
    assert os.path.exists(dump_dir), f"Expected dump dir {dump_dir} to exist"
    files = os.listdir(dump_dir)
    assert len(files) > 0, 'Expected at least one nan dump file'
