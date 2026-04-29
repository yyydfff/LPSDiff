# -*- coding: utf-8 -*-
# --- LPSDiff.py: latent phase-state diffusion version ---
import argparse
import copy
import logging
import os

import numpy as np
import torch
import torch.nn as nn
import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR

from models.basic_template import TrainTask
from utils.ema import EMA
from utils.loss_function import PerceptualLoss  # kept for external compatibility
from utils.measure import *
from util.util import get_logger, save_images, compute_ssim, compute_rmse, compute_psnr2D
from .LPSDiff_wrapper import Network, WeightNet
from .LPSDiff_modules import LatentPhaseStateDiffusion

try:
    import wandb
except Exception:
    wandb = None


class LPSDiff(TrainTask):


    @staticmethod
    def build_options():
        parser = argparse.ArgumentParser('Private arguments for latent phase-state diffusion')

        parser.add_argument("--in_channels", default=1, type=int)
        parser.add_argument("--out_channels", default=1, type=int)
        parser.add_argument("--init_lr", default=2e-4, type=float)

        parser.add_argument('--update_ema_iter', default=10, type=int)
        parser.add_argument('--start_ema_iter', default=2000, type=int)
        parser.add_argument('--ema_decay', default=0.01, type=float)

        parser.add_argument('--T', default=10, type=int)
        parser.add_argument('--sampling_routine', default='ddim', type=str)
        parser.add_argument('--only_adjust_two_step', action='store_true')
        parser.add_argument('--start_adjust_iter', default=1, type=int)

        parser.add_argument('--log_file', default='latent_phase_state.log', type=str)
        parser.add_argument('--log_mode', default='newrun', choices=['append', 'newrun'])
        parser.add_argument('--epochs', default=100, type=int)
        parser.add_argument('--warmup_epochs', default=0, type=int)
        parser.add_argument('--resume_iter', default=0, type=int)
        parser.add_argument('--wandb', action='store_true')

        parser.add_argument('--dose', default=1.0, type=float)
        parser.add_argument('--context', default=False, action='store_true')
        parser.add_argument('--self_condition_prob', default=0.5, type=float)

        # Latent phase-state mechanism. These are not extra modules; they define
        # the learned state variable and its transition structure.
        parser.add_argument('--use_latent_phase_state', default=True,
                            type=lambda x: str(x).lower() in ['true', '1', 'yes'])
        parser.add_argument('--phase_lambda_identity', default=1.0, type=float)
        parser.add_argument('--phase_lambda_phase', default=1.0, type=float)
        parser.add_argument('--phase_lambda_energy', default=0.5, type=float)
        parser.add_argument('--phase_ref_power', default=1.0, type=float)
        parser.add_argument('--phase_soft_temp', default=0.25, type=float)
        parser.add_argument('--phase_loss_weight', default=0.1, type=float)
        parser.add_argument('--phase_align_weight', default=1.0, type=float)
        parser.add_argument('--phase_smooth_weight', default=0.1, type=float)
        parser.add_argument('--phase_identity_weight', default=0.05, type=float)
        parser.add_argument('--phase_mismatch_sparse_weight', default=1e-3, type=float)
        parser.add_argument('--phase_confidence_weight', default=1e-3, type=float)
        parser.add_argument('--phase_transition_weight', default=0.05, type=float)

        parser.add_argument('--save_phase_diagnostics', action='store_true')
        parser.add_argument('--diag_max_batches', default=4, type=int)

        parser.add_argument('--save_val_pred', action='store_true')
        parser.add_argument('--save_test_pred', action='store_true')
        return parser

    def set_model(self):
        opt = self.opt
        self.ema = EMA(opt.ema_decay)
        self.update_ema_iter = opt.update_ema_iter
        self.start_ema_iter = opt.start_ema_iter
        self.dose = opt.dose
        self.T = opt.T
        self.sampling_routine = opt.sampling_routine
        self.context = opt.context

        effective_in_channels = 3 if opt.context else 1
        denoise_fn = Network(
            in_channels=effective_in_channels,
            out_channels=opt.out_channels,
            context=opt.context,
        )
        model = LatentPhaseStateDiffusion(
            denoise_fn=denoise_fn,
            image_size=512,
            channels=opt.out_channels,
            timesteps=opt.T,
            context=opt.context,
            use_latent_phase_state=opt.use_latent_phase_state,
            self_condition_prob=opt.self_condition_prob,
            phase_lambda_identity=opt.phase_lambda_identity,
            phase_lambda_phase=opt.phase_lambda_phase,
            phase_lambda_energy=opt.phase_lambda_energy,
            phase_ref_power=opt.phase_ref_power,
            phase_soft_temp=opt.phase_soft_temp,
            phase_loss_weight=opt.phase_loss_weight,
            phase_align_weight=opt.phase_align_weight,
            phase_smooth_weight=opt.phase_smooth_weight,
            phase_identity_weight=opt.phase_identity_weight,
            phase_mismatch_sparse_weight=opt.phase_mismatch_sparse_weight,
            phase_confidence_weight=opt.phase_confidence_weight,
            phase_transition_weight=opt.phase_transition_weight,
        )
        ema_model = copy.deepcopy(model)

        model = model.to(self.device)
        ema_model = ema_model.to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), opt.init_lr)

        self.optimizer = optimizer
        self.logger.modules = [model, ema_model, optimizer]
        self.model = model
        self.ema_model = ema_model

        warmup_epochs = max(0, getattr(self.opt, 'warmup_epochs', 0))
        main_epochs = max(1, getattr(self.opt, 'epochs', 1) - warmup_epochs)
        self.lr_scheduler = CosineAnnealingLR(self.optimizer, T_max=main_epochs, eta_min=1e-6)

        self.lossfn = nn.MSELoss()
        self.lossfn_sub1 = nn.MSELoss()
        self.best_val_psnr = float('-inf')
        self.best_psnr = self.best_val_psnr

        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        self.ckpt_dir = project_root
        self.result_dir = os.path.join(project_root, 'test_results')
        os.makedirs(self.result_dir, exist_ok=True)
        self.best_model_path = os.path.join(self.result_dir, 'best.model')

        log_dir = self.result_dir
        os.makedirs(log_dir, exist_ok=True)
        self.shared_log_path = os.path.join(log_dir, getattr(opt, 'log_file', 'latent_phase_state.log'))

        self.run_logger = logging.getLogger('latent_phase_state_logger')
        self.run_logger.setLevel(logging.INFO)
        for h in list(self.run_logger.handlers):
            self.run_logger.removeHandler(h)

        fh = logging.FileHandler(self.shared_log_path, mode='a', encoding='utf-8')
        fmt = logging.Formatter(
            '[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S,%f'
        )
        fh.setFormatter(fmt)
        self.run_logger.addHandler(fh)
        self.run_logger.propagate = False

        if getattr(opt, 'log_mode', 'newrun') == 'newrun':
            self.run_logger.info('===== NEW RUN: latent phase-state diffusion =====')
            self.run_logger.info('Strict split: train uses train_loader; best checkpoint uses val_loader; final test uses test_loader once.')

        self.reset_parameters()

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self, n_iter):
        if n_iter < self.start_ema_iter:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def _prepare_batch(self, low_dose, full_dose):
        low_dose = low_dose.to(self.device)
        full_dose = full_dose.to(self.device)

        if low_dose.dim() == 5 and low_dose.size(1) == 1:
            low_dose = low_dose.squeeze(1)
        if full_dose.dim() == 5 and full_dose.size(1) == 1:
            full_dose = full_dose.squeeze(1)

        if full_dose.dim() == 3:
            full_dose = full_dose.unsqueeze(1)
        elif full_dose.dim() == 4 and full_dose.size(1) != 1:
            full_dose = full_dose[:, :1]

        if self.context:
            if low_dose.dim() == 4 and low_dose.size(1) == 1:
                low_dose = low_dose.repeat(1, 3, 1, 1)
            elif low_dose.dim() == 3:
                low_dose = low_dose.unsqueeze(1).repeat(1, 3, 1, 1)
        else:
            if low_dose.dim() == 3:
                low_dose = low_dose.unsqueeze(1)
            elif low_dose.dim() == 4 and low_dose.size(1) > 1:
                low_dose = low_dose[:, 0:1]
        return low_dose.float(), full_dose.float()

    def train(self, inputs, n_iter):
        opt = self.opt
        self.model.train()
        self.ema_model.train()

        low_dose, full_dose = inputs
        low_dose, full_dose = self._prepare_batch(low_dose, full_dose)
        self.optimizer.zero_grad()

        x_recon, x_mix, x_recon_sub1, x_mix_sub1 = self.model(
            low_dose,
            full_dose,
            n_iter,
            only_adjust_two_step=opt.only_adjust_two_step,
            start_adjust_iter=opt.start_adjust_iter,
        )

        loss_main = self.lossfn(x_recon, full_dose)
        loss_aux = self.lossfn_sub1(x_recon_sub1, full_dose)
        loss = loss_main + 0.5 * loss_aux

        phase_loss_val = 0.0
        if isinstance(self.model.last_phase_loss, torch.Tensor):
            loss = loss + self.model.last_phase_loss
            phase_loss_val = float(self.model.last_phase_loss.detach())

        loss.backward()
        self.optimizer.step()

        lr = self.optimizer.param_groups[0]['lr']
        log_dict = {
            'iter': n_iter,
            'loss': float(loss.detach()),
            'loss_main': float(loss_main.detach()),
            'loss_aux': float(loss_aux.detach()),
            'loss_phase_state': phase_loss_val,
            'lr': float(lr),
        }
        for k, v in getattr(self.model, 'last_phase_stats', {}).items():
            log_dict[f'phase_state/{k}'] = v

        if hasattr(self.model.denoise_fn, 'export_phase_diagnostics'):
            local_diag = self.model.denoise_fn.export_phase_diagnostics()
            for k in ['local_phase_mean', 'velocity_mean', 'update_gate_mean']:
                if k in local_diag:
                    log_dict[f'local_phase/{k}'] = float(local_diag[k].detach())

        if getattr(opt, 'wandb', False) and wandb is not None:
            try:
                if n_iter == getattr(opt, 'resume_iter', 0) + 1:
                    wandb.init(project="latent-phase-state-diffusion")
                wandb.log(log_dict)
            except Exception:
                pass

        if n_iter % self.update_ema_iter == 0:
            self.step_ema(n_iter)

        return log_dict

    @torch.no_grad()
    def _save_diagnostics(self, batch_idx, phase='test'):
        if not getattr(self.opt, 'save_phase_diagnostics', False):
            return
        if batch_idx >= getattr(self.opt, 'diag_max_batches', 4):
            return
        diag = self.ema_model.export_phase_diagnostics()
        if not diag:
            return
        diag_root = os.path.join(self.result_dir, 'phase_diagnostics', phase)
        os.makedirs(diag_root, exist_ok=True)
        arrays = {}
        for k, v in diag.items():
            if isinstance(v, torch.Tensor):
                arrays[k] = v.detach().float().cpu().numpy()
        if arrays:
            np.savez(os.path.join(diag_root, f'diag_batch_{batch_idx:03d}.npz'), **arrays)

    @torch.no_grad()
    def _evaluate_loader(self, loader, n_iter, epoch=None, phase='val', save_best=False, save_pred=False):
        self.ema_model.eval()

        pred_root = os.path.join(self.result_dir, 'pred', phase)
        if save_pred:
            os.makedirs(pred_root, exist_ok=True)

        psnr_list, ssim_list, rmse_list = [], [], []
        epoch_tag = int(epoch) + 1 if epoch is not None else 0
        eval_bar = tqdm.tqdm(
            total=len(loader),
            desc=f'{phase} epoch {epoch_tag}',
            leave=True,
            dynamic_ncols=True,
        )

        for batch_idx, (low_dose, full_dose) in enumerate(loader, start=1):
            low_dose, full_dose = self._prepare_batch(low_dose, full_dose)
            y_pred, _, _ = self.ema_model.sample(
                batch_size=low_dose.size(0),
                img=low_dose,
                t=self.T,
                sampling_routine=self.sampling_routine,
                n_iter=n_iter,
                start_adjust_iter=self.opt.start_adjust_iter,
            )
            y = full_dose

            psnr2d = compute_psnr2D(y_pred, y)
            ssim2d = compute_ssim(y_pred, y)
            rmse = compute_rmse(y_pred, y)

            psnr_list.append(float(psnr2d.detach().cpu()))
            ssim_list.append(float(ssim2d.detach().cpu()))
            rmse_list.append(float(rmse.detach().cpu()))

            if save_pred:
                save_images(y_pred, root=pred_root, phase='', index=batch_idx, normalize=False)

            if phase == 'test':
                self._save_diagnostics(batch_idx - 1, phase=phase)

            eval_bar.update(1)

        eval_bar.close()

        def _mean_std(vals):
            if not vals:
                return 0.0, 0.0
            a = np.asarray(vals, dtype=np.float32)
            return float(a.mean()), float(a.std())

        average_psnr2d, std_psnr2d = _mean_std(psnr_list)
        average_ssim2d, std_ssim2d = _mean_std(ssim_list)
        average_rmse, std_rmse = _mean_std(rmse_list)

        lr = self.optimizer.param_groups[0]['lr']
        if epoch is None:
            prefix = f'[{phase}] '
        else:
            prefix = f'[{phase} epoch {int(epoch) + 1}/{self.opt.epochs}] '

        tag = ''
        if save_best and average_psnr2d > self.best_val_psnr:
            self.best_val_psnr = average_psnr2d
            self.best_psnr = average_psnr2d
            tag = ' (BEST_VAL↑)'
            torch.save(self.ema_model.state_dict(), self.best_model_path)
            self.run_logger.info(f'Saved validation-best model: {self.best_model_path}')

        message = (
            f'{prefix}| lr {lr:.2e} | '
            f'psnr2d: {average_psnr2d:.4f}±{std_psnr2d:.4f}, '
            f'ssim2d: {average_ssim2d:.4f}±{std_ssim2d:.4f}, '
            f'rmse: {average_rmse:.6f}±{std_rmse:.6f}{tag}'
        )
        self.run_logger.info(message)

        tqdm.tqdm.write(
            f'{phase.upper()} | '
            f'PSNR2D: {average_psnr2d:.4f}±{std_psnr2d:.4f}, '
            f'SSIM2D: {average_ssim2d:.4f}±{std_ssim2d:.4f}, '
            f'RMSE: {average_rmse:.6f}±{std_rmse:.6f}{tag}'
        )

        return {
            'psnr2d': average_psnr2d,
            'ssim2d': average_ssim2d,
            'rmse': average_rmse,
            'std_psnr2d': std_psnr2d,
            'std_ssim2d': std_ssim2d,
            'std_rmse': std_rmse,
        }

    def validate(self, n_iter, epoch=None):
        if not hasattr(self, 'val_loader') or self.val_loader is None:
            raise RuntimeError('val_loader is None. Training mode must build validation loader.')

        return self._evaluate_loader(
            loader=self.val_loader,
            n_iter=n_iter,
            epoch=epoch,
            phase='val',
            save_best=True,
            save_pred=getattr(self.opt, 'save_val_pred', False),
        )

    @torch.no_grad()
    def test(self, n_iter, epoch=None):
        if not hasattr(self, 'test_loader') or self.test_loader is None:
            raise RuntimeError(
                'test_loader is None. Final test should be run with test.py or --mode test, '
                'not during training.'
            )

        return self._evaluate_loader(
            loader=self.test_loader,
            n_iter=n_iter,
            epoch=epoch,
            phase='test',
            save_best=False,
            save_pred=getattr(self.opt, 'save_test_pred', False),
        )
