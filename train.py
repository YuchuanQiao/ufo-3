import sys
#!/usr/bin/env python
# coding=utf-8
'''
Author: YuchuanQiao@fudan.edu.cn
Date: 2025-07-24 11:20:30
LastEditTime: 2025-08-07 16:47:27
LastEditors: Yuchuan Qiao
Description: 
FilePath: /dwi2FOD/ufo-3/train.py
'''
from tqdm import tqdm
import swanlab
import os
import numpy as np
import time
import yaml
from utils.loss import Loss
from model.model import Model
use_swanlab=True
import random
import torch
torch.autograd.set_detect_anomaly(True)
#from torch.utils.tensorboard import Summary#writer
import pytorch_warmup as warmup
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
sys.path.append('./data')
from data import create_dataset_val,create_dataset_train
from data.chcp_dataset import pre_generator,matrixB
import argparse


def calculate_acc(y_true, y_pred, mask,epsilon=1e-8):
    """
    Computing Angular Correlation Coefficient (ACC).
    
    Params:
    y_true (Tensor): ground truth,shape: (batch, channel, X, Y, Z).
    y_pred (Tensor): predicted value,shape: (batch, channel, X, Y, Z).
    epsilon (float):a small value to avoid division by zero.

    return:
    Tensor: ACC .
    """
    # Compute dot product for each sample
    y_true[y_true < 0] = 0  # Negative values have no physical meaning, set to 0
    y_pred[y_pred < 0] = 0

    y_pred = y_pred.squeeze(1).permute(0, 2, 3, 4, 1)
    y_true = y_true.permute(0, 2, 3, 4, 1)

    dot_product = torch.sum(y_true * y_pred, dim=-1)  # Sum over channel dimension

    # Compute norm for each sample
    norm_true = torch.sqrt(torch.sum(y_true ** 2, dim=-1))
    norm_pred = torch.sqrt(torch.sum(y_pred ** 2, dim=-1))

    # Create a mask to exclude samples with invalid regions
    valid_mask = mask > 0.25

    # Compute ACC (angular correlation coefficient), only for valid samples
    acc = torch.zeros_like(mask)  # shape: [B, D, H, W]
    acc[valid_mask] = dot_product[valid_mask] / (norm_true[valid_mask] * norm_pred[valid_mask] + epsilon)

    # Compute mean ACC within the batch, only averaging over valid samples
    if torch.any(valid_mask):
        acc = torch.mean(acc[valid_mask])
    else:
        acc = 0  # Set ACC to 0 if no valid samples are found

    return acc

def channel_decorrelation_loss(x):
    # x shape: (batch_size, channels, X, Y, Z)
    b, c, _, _, _ = x.shape
    x_flat = x.view(b, c, -1)  # Flatten x to (batch_size, channels, num_voxels)
    
    # Compute the mean of each channel
    x_mean = x_flat.mean(dim=-1, keepdim=True)
    
    # Subtract the mean
    x_centered = x_flat - x_mean
    
    # Compute covariance matrix between channels
    cov_matrix = torch.bmm(x_centered, x_centered.transpose(1, 2)) / x_flat.size(-1)
    
    # Sum of squared off-diagonal elements as decorrelation penalty
    off_diag_cov = cov_matrix - torch.diag_embed(torch.diagonal(cov_matrix, dim1=-2, dim2=-1))
    loss = torch.sum(off_diag_cov ** 2)
    
    return loss

def maskmse(input, target):
    # Compute masked mean squared error
    input = input.squeeze(1)
    target = target.squeeze(1)
    mse = (input - target) ** 2  # Element-wise squared error

    masked_mse = mse[input != 0].mean()  # Mean over non-zero regions only
    return masked_mse

def main(cfg):
    """Train a model
    Args:
        cfg (dict): Configuration dictionary containing all training parameters
    """
    best_val_acc=0
    # Load the shell and the graph samplings
    set_seed(2026)
    feature_in = 1
    
    # Extract variables from cfg
    batch_size = cfg['batch_size']
    lr = float(cfg['lr'])
    n_epoch = cfg['n_epoch']
    filter_start = cfg['filter_start']
    sh_degree = cfg['sh_degree']
    kernel_sizeSph = cfg['kernel_sizeSph']
    kernel_sizeSpa = cfg['kernel_sizeSpa']
    depth = cfg['depth']
    n_side = cfg['n_side']
    normalize = cfg['normalize']
    size_3d_patch = cfg['size_3d_patch']
    graph_sampling = cfg['graph_sampling']
    conv_name = cfg['conv_name']
    isoSpa = not cfg['anisoSpa']
    concatenate = cfg['concatenate']
    loss_fn_intensity = cfg['loss_intensity']
    intensity_weight = float(cfg['intensity_weight'])
    loss_fn_sparsity = cfg['loss_sparsity']
    sigma_sparsity = float(cfg['sigma_sparsity'])
    sparsity_weight = float(cfg['sparsity_weight'])
    loss_fn_non_negativity = cfg['loss_non_negativity']
    nn_fodf_weight = float(cfg['nn_fodf_weight'])
    pve_weight = float(cfg['pve_weight'])
    load_state = cfg['load_state']
    rf_name = cfg['rf_name']
    wm = cfg['wm']
    gm = cfg['gm']
    csf = cfg['csf']
    save_every = cfg['save_every']
    save_path = os.path.join(cfg['save_dir'], cfg['name'])
    middle_voxel = cfg['middle_voxel']
    
    # Create the deconvolution model
    model = Model(filter_start, kernel_sizeSph, kernel_sizeSpa, normalize, conv_name, isoSpa, feature_in)
    if load_state:
        print(load_state)
        model.load_state_dict(torch.load(load_state), strict=False)
    # Load model in GPU
    model = model.to(DEVICE)
    torch.save(model.state_dict(), os.path.join(save_path, 'epoch_0.pth'))

    #Constrain Set
    coord,ConstraintSet,__build_class__,trg= pre_generator('./',cfg['constrain_points'])
    BS=matrixB(coord, cfg['train_params']['max_order'])
    BC=torch.from_numpy(BS[ConstraintSet,:].transpose()).float().to(DEVICE)

    # dataset
    train_list=cfg['train_params']['train_txtfile']
    val_list=cfg['train_params']['val_txtfile']
    if cfg['dataset_mode'] =='hcp':
        train_vol_names = np.loadtxt(cfg['train_params']['train_txtfile'],dtype='str',ndmin=2)
        prev_nf = cfg['train_params']['featrues_HCP']
    if cfg['dataset_mode'] =='chcp':
        prev_nf = cfg['train_params']['featrues_CHCP']
    if cfg['dataset_mode'] =='chcps':
        prev_nf = cfg['train_params']['featrues_CHCP']
        train_list='./data/3025.txt'
        val_list='./data/3025.txt'
    print('dataset_mode:',cfg['dataset_mode'])
    print('prev_nf:',prev_nf)
    print('train_list:',train_list)
    print('val_list:',val_list)
    train_list=np.loadtxt(train_list,dtype='str',ndmin=2)
    val_list=np.loadtxt(val_list,dtype='str',ndmin=2)

    # Loss
    intensity_criterion = Loss(loss_fn_intensity)
    non_negativity_criterion = Loss(loss_fn_non_negativity)
    sparsity_criterion = Loss(loss_fn_sparsity, sigma_sparsity)
  
    # Optimizer/Scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    warmup_period = cfg['optimizer']['warmup_steps']
   
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.975)

    warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period)
    save_loss = {}
    save_loss['train'] = {}
  
    tb_j = 0
    counter = 0
    early_stop = False
    # AMP & cudnn settings from config
    use_amp = cfg.get('use_amp', False)
    if use_amp and not torch.cuda.is_available():
        print("Warning: use_amp=True but CUDA not available. Disabling AMP.")
        use_amp = False
    cudnn_benchmark = cfg.get('cudnn_benchmark', False)
    torch.backends.cudnn.benchmark = cudnn_benchmark
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # Start training timer
    training_start_time = time.time()
    total_epochs = cfg['train_params']['n_dense_epochs'] - cfg['train_params']['initial_epoch']
    print(f"Starting training for {total_epochs} epochs... (use_amp={use_amp}, cudnn_benchmark={cudnn_benchmark})")

    # Training loop
    for epoch in range(cfg['train_params']['initial_epoch'], cfg['train_params']['n_dense_epochs']):
        # Start epoch timer
        epoch_start_time = time.time()
        model.train()
        # Initialize loss to save and plot.
        loss_intensity_ = 0
        loss_sparsity_ = 0
        loss_non_negativity_fodf_ = 0
        loss_pve_fodf_coff_ = 0
        loss_pve_extra_trapped_ = 0
        
        if epoch % 800 == 0:#1
            indices = np.random.choice(train_list.shape[0],cfg['bs'], replace=False)
            train_list_batch=train_list[indices]
            train_dataset = create_dataset_train(cfg,train_list_batch,prev_nf)
            n_batch = len(train_dataset)
            valindices = np.random.choice(val_list.shape[0],2, replace=False)
            val_list_batch=val_list[valindices]
            val_dataset = create_dataset_val(cfg,val_list_batch,prev_nf)
            val_temp_dataloader = iter(val_dataset) 

        epoch_loss = []
        t = tqdm(train_dataset, desc='Epoch %d' % epoch)
        
        for i, data in enumerate(t):
            nside16sh8 = data['nside16sh8']
            nside16sh8 = nside16sh8.to(DEVICE)
            if data['fodlr']==None:
                continue
            start = time.time()

            optimizer.zero_grad()
            to_print = ''
            # Initialize per-batch loss components (safe defaults)
            loss_sparsity = torch.tensor(0.).to(DEVICE)
            loss_non_negativity_fodf = torch.tensor(0.).to(DEVICE)
            loss_pve_fodf_coff = torch.tensor(0.).to(DEVICE)
            loss_pve_extra_trapped = torch.tensor(0.).to(DEVICE)

            # Load the data in the DEVICE
            input = data['fodlr'].to(DEVICE).unsqueeze(1)
            table = data['table'].to(DEVICE)
          
            Y = data['Y'].to(DEVICE)
            G = data['G'].to(DEVICE)
            A=Y*G
            mask = data['mask'].to(DEVICE)
            output = input.squeeze(1)

            # Forward + Loss (supports AMP when enabled)
            if use_amp:
                with torch.cuda.amp.autocast():
                    x_reconstructed, x_deconvolved_fodf_coff_shc, x_deconvolved_extra_trapped_shc = model(input,nside16sh8,table,A)
                    # Check for NaNs in model outputs
                    if torch.isnan(x_reconstructed).any() or (x_deconvolved_fodf_coff_shc is not None and torch.isnan(x_deconvolved_fodf_coff_shc).any()) or (x_deconvolved_extra_trapped_shc is not None and torch.isnan(x_deconvolved_extra_trapped_shc).any()):
                        print('Warning: NaN detected in model outputs; skipping this batch')
                        # optionally dump batch for postmortem debugging
                        if cfg.get('dump_on_nan', False):
                            try:
                                dump_dir = os.path.join(save_path, 'nan_dumps')
                                os.makedirs(dump_dir, exist_ok=True)
                                dump_path = os.path.join(dump_dir, f"epoch{epoch}_iter{i}_model_outputs_nan.pt")
                                torch.save({
                                    'epoch': epoch, 'iter': i, 'reason': 'model_outputs_nan',
                                    'data': {k: v.detach().cpu() for k, v in data.items() if isinstance(v, torch.Tensor)},
                                    'x_reconstructed': x_reconstructed.detach().cpu() if 'x_reconstructed' in locals() else None,
                                    'x_deconvolved_fodf_coff_shc': x_deconvolved_fodf_coff_shc.detach().cpu() if 'x_deconvolved_fodf_coff_shc' in locals() and isinstance(x_deconvolved_fodf_coff_shc, torch.Tensor) else None,
                                    'x_deconvolved_extra_trapped_shc': x_deconvolved_extra_trapped_shc.detach().cpu() if 'x_deconvolved_extra_trapped_shc' in locals() and isinstance(x_deconvolved_extra_trapped_shc, torch.Tensor) else None,
                                }, dump_path)
                                print(f"Saved NaN dump to {dump_path}")
                            except Exception as e:
                                print('Failed to save NaN dump:', e)
                        continue
                    # Loss
                    loss = torch.tensor(0.).to(DEVICE)
                    # Intensity loss
                    loss_intensity = intensity_criterion(x_reconstructed, output, mask[:, None].expand(-1, output.shape[1], -1, -1, -1))
                    loss_intensity_ += loss_intensity.item()
                    loss = loss + intensity_weight * loss_intensity
                    to_print += ', Intensity: {0:.10f}'.format(loss_intensity.item())
                    if not x_deconvolved_fodf_coff_shc is None:
                        try:
                            x_deconvolved_fodf_coff = torch.einsum('agbijk,bl->aglijk', x_deconvolved_fodf_coff_shc, BC)
                            if torch.isnan(x_deconvolved_fodf_coff).any():
                                print('Warning: NaN in x_deconvolved_fodf_coff after einsum (AMP); skipping batch')
                                if cfg.get('dump_on_nan', False):
                                    try:
                                        dump_dir = os.path.join(save_path, 'nan_dumps')
                                        os.makedirs(dump_dir, exist_ok=True)
                                        dump_path = os.path.join(dump_dir, f"epoch{epoch}_iter{i}_einsum_nan_amp.pt")
                                        torch.save({
                                            'epoch': epoch, 'iter': i, 'reason': 'einsum_nan_amp',
                                            'data': {k: v.detach().cpu() for k, v in data.items() if isinstance(v, torch.Tensor)},
                                            'x_deconvolved_fodf_coff_shc': x_deconvolved_fodf_coff_shc.detach().cpu() if 'x_deconvolved_fodf_coff_shc' in locals() and isinstance(x_deconvolved_fodf_coff_shc, torch.Tensor) else None,
                                            'x_deconvolved_fodf_coff': x_deconvolved_fodf_coff.detach().cpu() if 'x_deconvolved_fodf_coff' in locals() and isinstance(x_deconvolved_fodf_coff, torch.Tensor) else None,
                                        }, dump_path)
                                        print(f"Saved NaN dump to {dump_path}")
                                    except Exception as e:
                                        print('Failed to save NaN dump:', e)
                                continue
                        except Exception as e:
                            print('EINSUM ERROR (AMP) computing x_deconvolved_fodf_coff:', e)
                            raise
                        # Sparsity loss
                        loss_sparsity = torch.mean(torch.abs(x_deconvolved_fodf_coff))
                        loss_sparsity_ += loss_sparsity.item()
                        try:
                            term = sparsity_weight * loss_sparsity
                            loss = loss + term
                        except Exception as e:
                            print('DEBUG: sparsity add failed', type(loss), getattr(loss,'shape',None), type(sparsity_weight), type(loss_sparsity), getattr(loss_sparsity,'shape',None))
                            raise
                        to_print += ', fodf_coff Sparsity: {0:.10f}'.format(loss_sparsity.item())
                        # Non negativity loss
                        fodf_neg = torch.min(x_deconvolved_fodf_coff, torch.zeros_like(x_deconvolved_fodf_coff))
                        fodf_neg_zeros = torch.zeros(fodf_neg.shape).to(DEVICE)
                        loss_non_negativity_fodf = non_negativity_criterion(fodf_neg, fodf_neg_zeros, mask[:, None, None].expand(-1, fodf_neg_zeros.shape[1], fodf_neg_zeros.shape[2], -1, -1, -1))
                        loss_non_negativity_fodf_ += loss_non_negativity_fodf.item()
                        loss = loss + nn_fodf_weight * loss_non_negativity_fodf
                        to_print += ', fodf_coff NN: {0:.10f}'.format(loss_non_negativity_fodf.item())
                        # Partial volume regularizer (guarded)
                        masked = x_deconvolved_fodf_coff_shc[:, :, 0][mask[:, None].expand(-1, x_deconvolved_fodf_coff_shc.shape[1], -1, -1, -1)==1]
                        if masked.numel() == 0:
                            loss_pve_fodf_coff = torch.tensor(0.).to(DEVICE)
                        else:
                            loss_pve_fodf_coff = 1/(torch.mean(masked)*np.sqrt(4*np.pi) + 1e-16)
                        loss_pve_fodf_coff_ += loss_pve_fodf_coff.item()
                        loss = loss + pve_weight * loss_pve_fodf_coff
                        to_print += ', fodf_coff regularizer: {0:.10f}'.format(loss_pve_fodf_coff.item())
                    index = 0
                    if gm:
                        masked = x_deconvolved_extra_trapped_shc[:, index, 0][mask==1]
                        if masked.numel() == 0:
                            loss_pve_extra_trapped = torch.tensor(0.).to(DEVICE)
                        else:
                            loss_pve_extra_trapped = 1/(torch.mean(masked)*np.sqrt(4*np.pi) + 1e-16)
                        loss_pve_extra_trapped_ += loss_pve_extra_trapped.item()
                        loss += pve_weight * loss_pve_extra_trapped
                        to_print += ', extra_trapped regularizer GM: {0:.10f}'.format(loss_pve_extra_trapped.item())
                        index += 1
                    if csf:
                        masked = x_deconvolved_extra_trapped_shc[:, index, 0][mask==1]
                        if masked.numel() == 0:
                            loss_pve_extra_trapped = torch.tensor(0.).to(DEVICE)
                        else:
                            loss_pve_extra_trapped = 1/(torch.mean(masked)*np.sqrt(4*np.pi) + 1e-16)
                        loss_pve_extra_trapped_ += loss_pve_extra_trapped.item()
                        loss += pve_weight * loss_pve_extra_trapped
                        to_print += ', extra_trapped regularizer CSF: {0:.10f}'.format(loss_pve_extra_trapped.item())
            else:
                x_reconstructed, x_deconvolved_fodf_coff_shc, x_deconvolved_extra_trapped_shc = model(input,nside16sh8,table,A)
                # Check for NaNs in model outputs
                if torch.isnan(x_reconstructed).any() or (x_deconvolved_fodf_coff_shc is not None and torch.isnan(x_deconvolved_fodf_coff_shc).any()) or (x_deconvolved_extra_trapped_shc is not None and torch.isnan(x_deconvolved_extra_trapped_shc).any()):
                    print('Warning: NaN detected in model outputs; skipping this batch')
                    if cfg.get('dump_on_nan', False):
                        try:
                            dump_dir = os.path.join(save_path, 'nan_dumps')
                            os.makedirs(dump_dir, exist_ok=True)
                            dump_path = os.path.join(dump_dir, f"epoch{epoch}_iter{i}_model_outputs_nan_nonamp.pt")
                            torch.save({
                                'epoch': epoch, 'iter': i, 'reason': 'model_outputs_nan_nonamp',
                                'data': {k: v.detach().cpu() for k, v in data.items() if isinstance(v, torch.Tensor)},
                                'x_reconstructed': x_reconstructed.detach().cpu() if 'x_reconstructed' in locals() else None,
                                'x_deconvolved_fodf_coff_shc': x_deconvolved_fodf_coff_shc.detach().cpu() if 'x_deconvolved_fodf_coff_shc' in locals() and isinstance(x_deconvolved_fodf_coff_shc, torch.Tensor) else None,
                            }, dump_path)
                            print(f"Saved NaN dump to {dump_path}")
                        except Exception as e:
                            print('Failed to save NaN dump:', e)
                    continue
                # Loss
                loss = torch.tensor(0.).to(DEVICE)
                # Intensity loss
                loss_intensity = intensity_criterion(x_reconstructed, output, mask[:, None].expand(-1, output.shape[1], -1, -1, -1))
                loss_intensity_ += loss_intensity.item()
                loss = loss + intensity_weight * loss_intensity
                to_print += ', Intensity: {0:.10f}'.format(loss_intensity.item())
                if not x_deconvolved_fodf_coff_shc is None:
                    try:
                        x_deconvolved_fodf_coff = torch.einsum('agbijk,bl->aglijk', x_deconvolved_fodf_coff_shc, BC)
                        if torch.isnan(x_deconvolved_fodf_coff).any():
                            print('Warning: NaN in x_deconvolved_fodf_coff after einsum (non-AMP); skipping batch')
                            if cfg.get('dump_on_nan', False):
                                try:
                                    dump_dir = os.path.join(save_path, 'nan_dumps')
                                    os.makedirs(dump_dir, exist_ok=True)
                                    dump_path = os.path.join(dump_dir, f"epoch{epoch}_iter{i}_einsum_nan_nonamp.pt")
                                    torch.save({
                                        'epoch': epoch, 'iter': i, 'reason': 'einsum_nan_nonamp',
                                        'data': {k: v.detach().cpu() for k, v in data.items() if isinstance(v, torch.Tensor)},
                                        'x_deconvolved_fodf_coff_shc': x_deconvolved_fodf_coff_shc.detach().cpu() if 'x_deconvolved_fodf_coff_shc' in locals() and isinstance(x_deconvolved_fodf_coff_shc, torch.Tensor) else None,
                                        'x_deconvolved_fodf_coff': x_deconvolved_fodf_coff.detach().cpu() if 'x_deconvolved_fodf_coff' in locals() and isinstance(x_deconvolved_fodf_coff, torch.Tensor) else None,
                                    }, dump_path)
                                    print(f"Saved NaN dump to {dump_path}")
                                except Exception as e:
                                    print('Failed to save NaN dump:', e)
                            continue
                    except Exception as e:
                        print('EINSUM ERROR (non-AMP) computing x_deconvolved_fodf_coff:', e)
                        raise
                    loss_sparsity = torch.mean(torch.abs(x_deconvolved_fodf_coff))
                    loss_sparsity_ += loss_sparsity.item()
                    try:
                        term = sparsity_weight * loss_sparsity
                        loss = loss + term
                    except Exception as e:
                        print('DEBUG: sparsity add failed (non-AMP)', type(loss), getattr(loss,'shape',None), type(sparsity_weight), type(loss_sparsity), getattr(loss_sparsity,'shape',None))
                        raise
                    to_print += ', fodf_coff Sparsity: {0:.10f}'.format(loss_sparsity.item())
                    fodf_neg = torch.min(x_deconvolved_fodf_coff, torch.zeros_like(x_deconvolved_fodf_coff))
                    fodf_neg_zeros = torch.zeros(fodf_neg.shape).to(DEVICE)
                    loss_non_negativity_fodf = non_negativity_criterion(fodf_neg, fodf_neg_zeros, mask[:, None, None].expand(-1, fodf_neg_zeros.shape[1], fodf_neg_zeros.shape[2], -1, -1, -1))
                    loss_non_negativity_fodf_ += loss_non_negativity_fodf.item()
                    loss = loss + nn_fodf_weight * loss_non_negativity_fodf
                    to_print += ', fodf_coff NN: {0:.10f}'.format(loss_non_negativity_fodf.item())
                    masked = x_deconvolved_fodf_coff_shc[:, :, 0][mask[:, None].expand(-1, x_deconvolved_fodf_coff_shc.shape[1], -1, -1, -1)==1]
                    if masked.numel() == 0:
                        loss_pve_fodf_coff = torch.tensor(0.).to(DEVICE)
                    else:
                        loss_pve_fodf_coff = 1/(torch.mean(masked)*np.sqrt(4*np.pi) + 1e-16)
                    loss_pve_fodf_coff_ += loss_pve_fodf_coff.item()
                    loss = loss + pve_weight * loss_pve_fodf_coff
                    to_print += ', fodf_coff regularizer: {0:.10f}'.format(loss_pve_fodf_coff.item())
                index = 0
                if gm:
                    masked = x_deconvolved_extra_trapped_shc[:, index, 0][mask==1]
                    if masked.numel() == 0:
                        loss_pve_extra_trapped = torch.tensor(0.).to(DEVICE)
                    else:
                        loss_pve_extra_trapped = 1/(torch.mean(masked)*np.sqrt(4*np.pi) + 1e-16)
                    loss_pve_extra_trapped_ += loss_pve_extra_trapped.item()
                    loss += pve_weight * loss_pve_extra_trapped
                    to_print += ', extra_trapped regularizer GM: {0:.10f}'.format(loss_pve_extra_trapped.item())
                    index += 1
                if csf:
                    masked = x_deconvolved_extra_trapped_shc[:, index, 0][mask==1]
                    if masked.numel() == 0:
                        loss_pve_extra_trapped = torch.tensor(0.).to(DEVICE)
                    else:
                        loss_pve_extra_trapped = 1/(torch.mean(masked)*np.sqrt(4*np.pi) + 1e-16)
                    loss_pve_extra_trapped_ += loss_pve_extra_trapped.item()
                    loss += pve_weight * loss_pve_extra_trapped
                    to_print += ', extra_trapped regularizer CSF: {0:.10f}'.format(loss_pve_extra_trapped.item())

            #if not x_deconvolved_extra_trapped_shc is None:
            #    ###############################################################################################
            #    # Partial volume regularizer
            #    loss_pve_extra_trapped = 1/torch.mean(x_deconvolved_extra_trapped_shc[:, :, 0][mask[:, None].expand(-1, x_deconvolved_extra_trapped_shc.shape[1], -1, -1, -1)==1])*np.sqrt(4*np.pi)
            #    loss_pve_extra_trapped_ += loss_pve_extra_trapped.item()
            #    loss += pve_weight * loss_pve_extra_trapped
            #    to_print += ', extra_trapped regularizer: {0:.10f}'.format(loss_pve_extra_trapped.item())
            ###############################################################################################
                # Partial volume regularizer
            index = 0
            if gm:
                loss_pve_extra_trapped = 1/(torch.mean(x_deconvolved_extra_trapped_shc[:, index, 0][mask==1])*np.sqrt(4*np.pi) + 1e-16)
                loss_pve_extra_trapped_ += loss_pve_extra_trapped.item()
                loss += pve_weight * loss_pve_extra_trapped
                to_print += ', extra_trapped regularizer GM: {0:.10f}'.format(loss_pve_extra_trapped.item())
                index += 1
            if csf:
                loss_pve_extra_trapped = 1/(torch.mean(x_deconvolved_extra_trapped_shc[:, index, 0][mask==1])*np.sqrt(4*np.pi) + 1e-16)
                loss_pve_extra_trapped_ += loss_pve_extra_trapped.item()
                loss += pve_weight * loss_pve_extra_trapped
                to_print += ', extra_trapped regularizer CSF: {0:.10f}'.format(loss_pve_extra_trapped.item())
            #VAR
            # variance_l1_loss=maskmse(torch.var(output, dim=1, unbiased=True) ,torch.var(x_reconstructed, dim=1, unbiased=True) )
            # loss+=variance_l1_loss
            ###############################################################################################
            # Tensorboard
            tb_j += 1
            # Log all loss components in a single swanlab.log call
            swanlab.log({
                'Batch/train_intensity': loss_intensity.item(),
                'Batch/train_sparsity': loss_sparsity.item(),
                "Batch/lr": optimizer.param_groups[0]['lr'],
                #'Batch/train_var': variance_l1_loss.item(),
                'Batch/train_nn': loss_non_negativity_fodf.item(),
                'Batch/train_pve_fodf_coff': loss_pve_fodf_coff.item(),
                'Batch/train_pve_extra_trapped': loss_pve_extra_trapped.item(),
                'Batch/train_total': loss.item()
            })
            #################################F##############################################################
            # Loss backward (AMP aware) and gradient logging
            if scaler is not None:
                scaler.scale(loss).backward()
                # Unscale before gradient clipping and logging
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1, norm_type=2)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1, norm_type=2)

            for name, parameter in model.named_parameters():
                if parameter.grad is not None:
                    if not torch.isnan(parameter.grad).any():
                        grad_stats = {
                            f"gradients/{name}_mean": parameter.grad.mean().item(),
                            f"gradients/{name}_std": parameter.grad.std().item(),
                            f"gradients/{name}_max": parameter.grad.max().item(),
                            f"gradients/{name}_min": parameter.grad.min().item()
                        }
                        swanlab.log(grad_stats)
                    else:
                        print(name, 'nan')
                        continue
                else:
                    continue

            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            
            with warmup_scheduler.dampening():
                if warmup_scheduler.last_step + 1 >= warmup_period and optimizer.param_groups[0]['lr']>0.00005:
                    lr_scheduler.step()
            ###############################################################################################
            # To print loss
            end = time.time()
            to_print += ', Elapsed time: {0} s'.format(end - start)
            to_print = 'Epoch [{0}/{1}], Iter [{2}/{3}]: Loss: {4:.10f}'.format(epoch + 1, n_epoch,
                                                                                i + 1, n_batch,
                                                                                loss.item()) \
                       + to_print
            print(to_print, end="\r")

            if (i + 1) % 500 == 0:
                torch.save(model.state_dict(), os.path.join(save_path, 'epoch_{0}.pth'.format(epoch + 1)))
            
        ###############################################################################################
        # Save and print mean loss for the epoch
        loss_ = 0
        # Mean results of the last epoch
        save_loss['train'][epoch] = {}

        save_loss['train'][epoch]['loss_intensity'] = loss_intensity_ / n_batch
        save_loss['train'][epoch]['weight_loss_intensity'] = intensity_weight
        loss_ += intensity_weight * loss_intensity_
        to_print += ', Intensity: {0:.10f}'.format(loss_intensity_ / n_batch)

        save_loss['train'][epoch]['loss_sparsity'] = loss_sparsity_ / n_batch
        save_loss['train'][epoch]['weight_loss_sparsity'] = sparsity_weight
        loss_ += sparsity_weight * loss_sparsity_
        to_print += ', Sparsity: {0:.10f}'.format(loss_sparsity_ / n_batch)

        save_loss['train'][epoch]['loss_non_negativity_fodf'] = loss_non_negativity_fodf_ / n_batch
        save_loss['train'][epoch]['weight_loss_non_negativity_fodf'] = nn_fodf_weight
        loss_ += nn_fodf_weight * loss_non_negativity_fodf_
        to_print += ', WM fODF NN: {0:.10f}'.format(loss_non_negativity_fodf_ / n_batch)

        save_loss['train'][epoch]['loss_pve_fodf_coff'] = loss_pve_fodf_coff_ / n_batch
        save_loss['train'][epoch]['weight_loss_pve_fodf_coff'] = pve_weight
        loss_ += pve_weight * loss_pve_fodf_coff_
        to_print += ', fodf_coff regularizer: {0:.10f}'.format(loss_pve_fodf_coff_ / n_batch)

        save_loss['train'][epoch]['loss_pve_extra_trapped'] = loss_pve_extra_trapped_ / n_batch
        save_loss['train'][epoch]['weight_loss_pve_extra_trapped'] = pve_weight
        loss_ += pve_weight * loss_pve_extra_trapped_
        to_print += ', extra_trapped regularizer: {0:.10f}'.format(loss_pve_extra_trapped_ / n_batch)

        save_loss['train'][epoch]['loss'] = loss_ / n_batch
        to_print = 'Epoch [{0}/{1}], Train Loss: {2:.10f}'.format(epoch + 1, n_epoch, loss_ / n_batch) + to_print
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        elapsed_time = time.time() - training_start_time
        remaining_epochs = cfg['train_params']['n_dense_epochs'] - (epoch + 1)
        avg_epoch_time = elapsed_time / (epoch - cfg['train_params']['initial_epoch'] + 1)
        estimated_remaining_time = remaining_epochs * avg_epoch_time
        
        print(to_print)
        print(f'Epoch time: {epoch_time:.2f}s | Total elapsed: {elapsed_time/3600:.2f}h | ETA: {estimated_remaining_time/3600:.2f}h')

        swanlab.log({'epoch': epoch + 1, 'Epoch/learning_rate': lr_scheduler.optimizer.param_groups[0]['lr'],
                    'Epoch/train_total': loss_ / n_batch,
                    'Epoch/train_intensity': loss_intensity_ / n_batch,
                    'Epoch/train_sparsity': loss_sparsity_ / n_batch,
                    'Epoch/train_nn': loss_non_negativity_fodf_ / n_batch,
                    'Epoch/train_pve_fodf_coff': loss_pve_fodf_coff_ / n_batch,
                    'Epoch/train_pve_extra_trapped': loss_pve_extra_trapped_ / n_batch,
                    'Training/epoch_time': epoch_time,
                    'Training/total_elapsed_time': elapsed_time,
                    'Training/avg_epoch_time': avg_epoch_time,
                    'Training/eta_hours': estimated_remaining_time / 3600
            })
        # Persist epoch metrics to CSV for easy aggregation
        metrics_csv = os.path.join(save_path, 'metrics.csv')
        if not os.path.exists(metrics_csv):
            with open(metrics_csv, 'w') as f:
                f.write('epoch,train_total,train_intensity,train_sparsity,train_nn,train_pve_fodf_coff,train_pve_extra_trapped,val_loss,val_acc,epoch_time,avg_epoch_time\n')
        val_loss_val = val_loss.item() if 'val_loss' in locals() else float('nan')
        val_acc_val = val_acc.item() if 'val_acc' in locals() else float('nan')
        with open(metrics_csv, 'a') as f:
            f.write(f"{epoch+1},{loss_/n_batch},{loss_intensity_/n_batch},{loss_sparsity_/n_batch},{loss_non_negativity_fodf_/n_batch},{loss_pve_fodf_coff_/n_batch},{loss_pve_extra_trapped_/n_batch},{val_loss_val},{val_acc_val},{epoch_time},{avg_epoch_time}\n")
        
        # if use_swanlab:
        #     swanlab.log(#to_swanlab)

        ###############################################################################################
        #validation
        if epoch % 1 == 0:
            # determine validation iterations with sensible defaults
            val_iters = cfg.get('val_iters', cfg.get('train_params', {}).get('val_iters', 0))
            patience = cfg.get('patience', cfg.get('train_params', {}).get('patience', 5))

            j_val_loss=torch.tensor(0).float().to(DEVICE)
            j_val_loss_fod=torch.tensor(0).float().to(DEVICE)
            j_val_acc=torch.tensor(0).float().to(DEVICE)
            with torch.no_grad():
                for j in range(val_iters):
                    data_list = next(val_temp_dataloader, 'reset_val_dataloader')
                    if data_list == 'reset_val_dataloader':
                        val_temp_dataloader = iter(val_dataset)
                        data_list = next(val_temp_dataloader)

                    inputs= data_list['fodlr'].float().to(DEVICE).unsqueeze(1)
                    mask= data_list['mask'].float().to(DEVICE)
                    #val_dti= data_list['fodlr6'].float().to(DEVICE)
                    #val_dti =torch.cat([val_dti, inputs], dim=1)
                    fodgt=data_list['fodgt'].float().to(DEVICE)

                    table=data_list['table'].float().to(DEVICE)
                    Y=data_list['Y'].float().to(DEVICE)
                    G=data_list['G'].float().to(DEVICE)
                    nside16sh8 = data_list['nside16sh8'].to(DEVICE)
                    A=Y*G
                    #val_dti=torch.einsum('agb,abijk->agijk',A.transpose(1,2),inputs)#bs,10,X,Y,Z bs,45,10
                    model.eval()
                    outputs,fod_pred,_ = model(inputs,nside16sh8,table,A)
                    #outputs=(outputs-dmean)/dstd

                    j_val_loss+=maskmse(inputs,outputs)
                    j_val_loss_fod+=maskmse(fodgt,fod_pred[:,:45])
                    j_val_acc+=calculate_acc(fodgt,fod_pred[:,:45],mask)

                if val_iters > 0:
                    val_loss=j_val_loss/val_iters
                    val_loss_fod=j_val_loss_fod/val_iters
                    val_acc=j_val_acc/val_iters
                else:
                    val_loss=torch.tensor(float('nan')).to(DEVICE)
                    val_loss_fod=torch.tensor(float('nan')).to(DEVICE)
                    val_acc=torch.tensor(float('nan')).to(DEVICE)

                if val_acc > best_val_acc:
                    counter = 0
                    print(f"ACC has improved from {best_val_acc} to {val_acc}")
                    best_val_acc = val_acc
                    #model_path = os.path.join(save_path, 'model_%d_epoch%04d_step%04d.pt' % (1,epoch+1,i+1))
                    model_path = os.path.join(save_path, 'best.pt')
                    torch.save(model.state_dict(), model_path)
                else:
                    counter += 1
                    if counter >= patience:
                        early_stop = True
                    print(f"Current ACC: {val_acc}, Best ACC: {best_val_acc}, early stopping counter: {counter}")
                if epoch%20==0 and epoch >0:
                    model_path = os.path.join(save_path, 'epoch_{0}.pth'.format(epoch + 1))
                    torch.save(model.state_dict(), model_path)
                    
            print('epoch:',epoch,'val_loss:',val_loss.item(),'val_loss_fod:',val_loss_fod.item(),'val_acc:',val_acc.item())
            swanlab.log({
            "val_loss" : torch.mean(val_loss),
            "val_acc" : torch.mean(val_acc),
            "val_loss_fod" : torch.mean(val_loss_fod),
            "epoch" : epoch+1,
            })   
      
        if len(epoch_loss)==0:
            continue
        if early_stop == True:
            print('Early stopping')
            break

    # Training completed - log final time statistics
    total_training_time = time.time() - training_start_time
    completed_epochs = epoch - cfg['train_params']['initial_epoch'] + 1
    avg_epoch_time = total_training_time / completed_epochs
    
    print(f"\nTraining completed!")
    print(f"Total training time: {total_training_time/3600:.2f} hours ({total_training_time:.0f} seconds)")
    print(f"Completed epochs: {completed_epochs}")
    print(f"Average time per epoch: {avg_epoch_time:.2f} seconds")
    
    # Log final training statistics
    swanlab.log({
        'Training/total_training_time_hours': total_training_time / 3600,
        'Training/total_training_time_seconds': total_training_time,
        'Training/completed_epochs': completed_epochs,
        'Training/final_avg_epoch_time': avg_epoch_time
    })

    # final model save
    final_model_path = os.path.join(save_path, 'final_model_%d_epoch%04d_step%04d.pt' % (1,epoch+1,i+1))
    torch.save(model.state_dict(), final_model_path)
    
    return final_model_path

def set_seed(seed):
    random.seed(seed)  
    np.random.seed(seed)  
    torch.manual_seed(seed)  

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False  

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Train UFO3 model')
    parser.add_argument('--config', type=str, default='./train.yaml', 
                        help='Path to YAML configuration file')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
        run = swanlab.init(
            project="UFO3",
            name = str(cfg['name']),
            # mode="offline",
        )    

    # Create save directory
    save_path = os.path.join(cfg['save_dir'], cfg['name'])
    if not os.path.exists(save_path):
        print('Create new directory: {0}'.format(save_path))
        os.makedirs(save_path)
    print('Save path: {0}'.format(save_path))
    
    # Call main function with cfg
    main(cfg)
