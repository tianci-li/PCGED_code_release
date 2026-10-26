


import os
import sys
import time
import math
import json
from pathlib import Path
from typing import Dict, Optional, Any, Union
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm


try:

    from torch.cuda.amp import GradScaler
    from torch.amp import autocast as new_autocast
    HAS_NEW_AUTOCAST = True
except ImportError:
    try:

        from torch.cuda.amp import GradScaler, autocast as old_autocast
        HAS_NEW_AUTOCAST = False
    except ImportError:
        GradScaler = None
        HAS_NEW_AUTOCAST = False

from .distributed import (
    DistributedManager,
    is_main_process,
    get_rank,
    get_world_size,
)



class AutocastContext:


    def __init__(self, device_type: str = 'cuda', dtype=torch.float16, enabled: bool = True):
        self.enabled = enabled
        self.dtype = dtype
        self.device_type = device_type
        self._autocast = None

    def __enter__(self):
        if not self.enabled:
            return self

        try:

            from torch.amp import autocast
            self._autocast = autocast(device_type=self.device_type, dtype=self.dtype)
        except (ImportError, TypeError):
            try:

                from torch.cuda.amp import autocast
                self._autocast = autocast(enabled=True, dtype=self.dtype)
            except TypeError:

                from torch.cuda.amp import autocast
                self._autocast = autocast(enabled=True)

        if self._autocast is not None:
            self._autocast.__enter__()
        return self

    def __exit__(self, *args):
        if self._autocast is not None:
            self._autocast.__exit__(*args)


@dataclass
class TrainerConfig:


    epochs: int = 100
    batch_size: int = 16
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0


    lr: float = 1e-4
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8


    scheduler_type: str = 'cosine'
    warmup_steps: int = 2000
    warmup_ratio: float = 0.0
    min_lr_ratio: float = 0.1


    gene_loss_weight: float = 1.0
    expr_bin_loss_weight: float = 0.5
    expr_level_loss_weight: float = 0.3
    regression_loss_weight: float = 0.2


    mixed_precision: bool = True
    amp_dtype: str = 'bf16'


    save_dir: str = './checkpoints'
    save_every_n_steps: int = 1000
    save_every_n_epochs: int = 1
    keep_last_n: int = 5
    resume_from: Optional[str] = None


    log_dir: str = './logs'
    log_every_n_steps: int = 100


    val_every_n_steps: int = 500
    val_every_n_epochs: int = 1


class Trainer:


    def __init__(
        self,
        model: nn.Module,
        train_dataloader,
        val_dataloader=None,
        config: Optional[TrainerConfig] = None,
        dist_manager: Optional[DistributedManager] = None,
    ):
        self.config = config or TrainerConfig()
        self.dist_manager = dist_manager


        if dist_manager is not None:
            self.device = dist_manager.device
            self.is_main = dist_manager.is_main_process
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.is_main = True


        self.model = model.to(self.device)
        if dist_manager is not None and dist_manager.is_distributed:
            self.model = dist_manager.wrap_model(self.model)


        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader


        self.optimizer = self._create_optimizer()


        self.total_steps = len(train_dataloader) * self.config.epochs
        self.scheduler = self._create_scheduler()


        self.scaler = None
        self.amp_dtype = torch.float32
        if self.config.mixed_precision:
            self._setup_amp()


        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float('inf')


        if self.is_main:
            Path(self.config.save_dir).mkdir(parents=True, exist_ok=True)
            Path(self.config.log_dir).mkdir(parents=True, exist_ok=True)


        if self.config.resume_from:
            self._load_checkpoint(self.config.resume_from)

    def _create_optimizer(self) -> AdamW:


        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if 'bias' in name or 'norm' in name or 'embedding' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        param_groups = [
            {'params': decay_params, 'weight_decay': self.config.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ]

        return AdamW(
            param_groups,
            lr=self.config.lr,
            betas=self.config.betas,
            eps=self.config.eps,
        )

    def _create_scheduler(self) -> LambdaLR:

        warmup_steps = self.config.warmup_steps
        if self.config.warmup_ratio > 0:
            warmup_steps = int(self.total_steps * self.config.warmup_ratio)

        min_lr_ratio = self.config.min_lr_ratio

        if self.config.scheduler_type == 'cosine':
            def lr_lambda(step):
                if step < warmup_steps:
                    return step / max(1, warmup_steps)
                progress = (step - warmup_steps) / max(1, self.total_steps - warmup_steps)
                return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

        elif self.config.scheduler_type == 'linear':
            def lr_lambda(step):
                if step < warmup_steps:
                    return step / max(1, warmup_steps)
                progress = (step - warmup_steps) / max(1, self.total_steps - warmup_steps)
                return min_lr_ratio + (1 - min_lr_ratio) * (1 - progress)

        else:
            def lr_lambda(step):
                if step < warmup_steps:
                    return step / max(1, warmup_steps)
                return 1.0

        return LambdaLR(self.optimizer, lr_lambda)

    def _setup_amp(self):

        requested_dtype = self.config.amp_dtype


        if requested_dtype == 'bf16':
            if torch.cuda.is_available():

                device = torch.cuda.current_device()
                capability = torch.cuda.get_device_capability(device)
                if capability[0] < 8:
                    print(f"WARN GPU compute capability {capability[0]}.{capability[1]} < 8.0, "
                          f"bf16 not supported. Falling back to fp16.")
                    requested_dtype = 'fp16'

        if requested_dtype == 'bf16':
            self.amp_dtype = torch.bfloat16

            self.scaler = None
        else:
            self.amp_dtype = torch.float16
            if GradScaler is not None:
                self.scaler = GradScaler()
            else:
                print("WARN GradScaler not available, disabling loss scaling")
                self.scaler = None

    def _compute_loss(self, outputs: Dict, batch: Dict) -> Dict[str, torch.Tensor]:

        losses = {}


        gene_logits = outputs['gene_logits']
        expr_bin_logits = outputs['expr_bin_logits']
        expr_level_logits = outputs['expr_level_logits']


        gene_tokens = batch['gene_tokens'][:, 1:].contiguous()
        expr_bin_tokens = batch['expr_bin_tokens'][:, 1:].contiguous()
        expr_level_tokens = batch['expr_level_tokens'][:, 1:].contiguous()


        gene_logits = gene_logits[:, :-1, :].contiguous()
        expr_bin_logits = expr_bin_logits[:, :-1, :].contiguous()
        expr_level_logits = expr_level_logits[:, :-1, :].contiguous()


        if self.global_step < 5:
            for name, logits in [('gene', gene_logits), ('expr_bin', expr_bin_logits), ('expr_level', expr_level_logits)]:
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    print(f"WARN  {name}_logits has NaN/Inf at step {self.global_step}")
                    print(f"   Shape: {logits.shape}")
                    print(f"   NaN count: {torch.isnan(logits).sum().item()}")
                    print(f"   Inf count: {torch.isinf(logits).sum().item()}")


        gene_logits_flat = gene_logits.view(-1, gene_logits.size(-1))
        gene_tokens_flat = gene_tokens.view(-1)

        valid_gene_mask = (gene_tokens_flat != 0) & (gene_tokens_flat < gene_logits.size(-1))
        if valid_gene_mask.sum() > 0:
            gene_loss = F.cross_entropy(
                gene_logits_flat,
                gene_tokens_flat,
                ignore_index=0,
            )
        else:
            gene_loss = torch.tensor(0.0, device=gene_logits.device)
        losses['gene_loss'] = gene_loss * self.config.gene_loss_weight


        expr_bin_logits_flat = expr_bin_logits.view(-1, expr_bin_logits.size(-1))
        expr_bin_tokens_flat = expr_bin_tokens.view(-1)
        valid_bin_mask = (expr_bin_tokens_flat != 0) & (expr_bin_tokens_flat < expr_bin_logits.size(-1))
        if valid_bin_mask.sum() > 0:
            expr_bin_loss = F.cross_entropy(
                expr_bin_logits_flat,
                expr_bin_tokens_flat,
                ignore_index=0,
            )
        else:
            expr_bin_loss = torch.tensor(0.0, device=expr_bin_logits.device)
        losses['expr_bin_loss'] = expr_bin_loss * self.config.expr_bin_loss_weight


        expr_level_logits_flat = expr_level_logits.view(-1, expr_level_logits.size(-1))
        expr_level_tokens_flat = expr_level_tokens.view(-1)
        valid_level_mask = (expr_level_tokens_flat != 0) & (expr_level_tokens_flat < expr_level_logits.size(-1))
        if valid_level_mask.sum() > 0:
            expr_level_loss = F.cross_entropy(
                expr_level_logits_flat,
                expr_level_tokens_flat,
                ignore_index=0,
            )
        else:
            expr_level_loss = torch.tensor(0.0, device=expr_level_logits.device)
        losses['expr_level_loss'] = expr_level_loss * self.config.expr_level_loss_weight


        losses['total_loss'] = sum(losses.values())


        if torch.isnan(losses['total_loss']) or torch.isinf(losses['total_loss']):
            print(f"\nERR NaN/Inf loss detected at step {self.global_step}:")
            print(f"   gene_loss: {losses['gene_loss'].item()}")
            print(f"   expr_bin_loss: {losses['expr_bin_loss'].item()}")
            print(f"   expr_level_loss: {losses['expr_level_loss'].item()}")
            print(f"   gene_tokens range: [{gene_tokens_flat.min().item()}, {gene_tokens_flat.max().item()}]")
            print(f"   expr_bin_tokens range: [{expr_bin_tokens_flat.min().item()}, {expr_bin_tokens_flat.max().item()}]")
            print(f"   expr_level_tokens range: [{expr_level_tokens_flat.min().item()}, {expr_level_tokens_flat.max().item()}]")
            print(f"   gene_logits vocab size: {gene_logits.size(-1)}")
            print(f"   expr_bin_logits vocab size: {expr_bin_logits.size(-1)}")
            print(f"   expr_level_logits vocab size: {expr_level_logits.size(-1)}")

        return losses

    def _validate_batch(self, batch: Dict):


        model = self.model.module if hasattr(self.model, 'module') else self.model
        config = model.config

        max_gene_id = config.n_genes + config.n_special_tokens - 1

        max_expr_bin = config.n_expr_bins - 1
        max_expr_level = config.n_expr_levels - 1


        gene_tokens = batch['gene_tokens']
        gene_max = gene_tokens.max().item()
        gene_min = gene_tokens.min().item()
        if gene_max > max_gene_id or gene_min < 0:
            raise ValueError(
                f"gene_tokens     !      [0, {max_gene_id}], "
                f"     [{gene_min}, {gene_max}]"
            )


        expr_bin_tokens = batch['expr_bin_tokens']
        expr_bin_max = expr_bin_tokens.max().item()
        expr_bin_min = expr_bin_tokens.min().item()
        if expr_bin_max > max_expr_bin or expr_bin_min < 0:
            raise ValueError(
                f"expr_bin_tokens     !      [0, {max_expr_bin}], "
                f"     [{expr_bin_min}, {expr_bin_max}]"
            )


        expr_level_tokens = batch['expr_level_tokens']
        expr_level_max = expr_level_tokens.max().item()
        expr_level_min = expr_level_tokens.min().item()
        if expr_level_max > max_expr_level or expr_level_min < 0:
            raise ValueError(
                f"expr_level_tokens     !      [0, {max_expr_level}], "
                f"     [{expr_level_min}, {expr_level_max}]"
            )

    def _check_for_nan(self, tensor: torch.Tensor, name: str, step: int) -> bool:

        has_nan = torch.isnan(tensor).any().item()
        has_inf = torch.isinf(tensor).any().item()
        if has_nan or has_inf:
            print(f"\nWARN  NaN/Inf detected in '{name}' at step {step}:")
            print(f"   Shape: {tensor.shape}")
            print(f"   Has NaN: {has_nan}, Has Inf: {has_inf}")
            print(f"   Min: {tensor[~torch.isnan(tensor)].min().item() if not torch.isnan(tensor).all() else 'all NaN'}")
            print(f"   Max: {tensor[~torch.isnan(tensor)].max().item() if not torch.isnan(tensor).all() else 'all NaN'}")
            return True
        return False

    def _debug_nan(self, batch: Dict, outputs: Dict, losses: Dict):

        print(f"\n{'='*60}")
        print(f"  NaN Debug Info at step {self.global_step}")
        print(f"{'='*60}")


        print("\n Input tensors:")
        self._check_for_nan(batch['pathway_embedding'], 'pathway_embedding', self.global_step)
        self._check_for_nan(batch['gene_tokens'].float(), 'gene_tokens', self.global_step)
        self._check_for_nan(batch['expr_bin_tokens'].float(), 'expr_bin_tokens', self.global_step)
        self._check_for_nan(batch['expr_level_tokens'].float(), 'expr_level_tokens', self.global_step)
        self._check_for_nan(batch['attention_mask'].float(), 'attention_mask', self.global_step)


        print("\n  Output tensors:")
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                self._check_for_nan(value, f'outputs.{key}', self.global_step)


        print("\n  Losses:")
        for key, value in losses.items():
            if isinstance(value, torch.Tensor):
                val = value.item()
                print(f"   {key}: {val} (NaN: {math.isnan(val)}, Inf: {math.isinf(val)})")


        print("\n Model parameters with NaN/Inf gradients:")
        model = self.model.module if hasattr(self.model, 'module') else self.model
        nan_params = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    nan_params.append(name)
            if torch.isnan(param).any() or torch.isinf(param).any():
                print(f"   WARN  Parameter '{name}' has NaN/Inf values!")

        if nan_params:
            print(f"   Parameters with NaN/Inf gradients: {nan_params[:10]}...")
        else:
            print("   No NaN/Inf gradients detected in parameters")

        print(f"{'='*60}\n")

    def _train_step(self, batch: Dict) -> Dict[str, float]:

        self.model.train()


        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}


        if self.global_step < 10:
            try:
                self._validate_batch(batch)
            except ValueError as e:
                print(f"\nERR Batch validation failed at step {self.global_step}:")
                print(f"   {e}")
                print(f"   gene_tokens shape: {batch['gene_tokens'].shape}")
                print(f"   expr_bin_tokens shape: {batch['expr_bin_tokens'].shape}")
                raise


        with AutocastContext(device_type='cuda', dtype=self.amp_dtype, enabled=self.config.mixed_precision):
            outputs = self.model(
                pathway_embedding=batch['pathway_embedding'],
                gene_tokens=batch['gene_tokens'],
                expr_bin_tokens=batch['expr_bin_tokens'],
                expr_level_tokens=batch['expr_level_tokens'],
                attention_mask=batch['attention_mask'].float(),
                return_loss=False,
            )
            losses = self._compute_loss(outputs, batch)
            loss = losses['total_loss'] / self.config.gradient_accumulation_steps


        if self.global_step < 5 or torch.isnan(loss) or torch.isinf(loss):
            if torch.isnan(loss) or torch.isinf(loss):
                self._debug_nan(batch, outputs, losses)

                print(f"WARN  Skipping batch {self.global_step} due to NaN/Inf loss")
                return {k: float('nan') for k in losses.keys()}


        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        return {k: v.item() for k, v in losses.items()}

    def _optimizer_step(self):

        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()

        self.optimizer.zero_grad()
        self.scheduler.step()

    @torch.no_grad()
    def _validate(self) -> Dict[str, float]:

        if self.val_dataloader is None:
            return {}

        self.model.eval()
        total_losses = {}
        num_batches = 0

        for batch in self.val_dataloader:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            with AutocastContext(device_type='cuda', dtype=self.amp_dtype, enabled=self.config.mixed_precision):
                outputs = self.model(
                    pathway_embedding=batch['pathway_embedding'],
                    gene_tokens=batch['gene_tokens'],
                    expr_bin_tokens=batch['expr_bin_tokens'],
                    expr_level_tokens=batch['expr_level_tokens'],
                    attention_mask=batch['attention_mask'].float(),
                    return_loss=False,
                )
                losses = self._compute_loss(outputs, batch)

            for k, v in losses.items():
                total_losses[k] = total_losses.get(k, 0) + v.item()
            num_batches += 1


        avg_losses = {f'val_{k}': v / num_batches for k, v in total_losses.items()}


        if self.dist_manager is not None and self.dist_manager.is_distributed:
            for k in avg_losses:
                tensor = torch.tensor(avg_losses[k], device=self.device)
                tensor = self.dist_manager.all_reduce_mean(tensor)
                avg_losses[k] = tensor.item()

        return avg_losses

    def _save_checkpoint(self, name: str = 'checkpoint'):

        if not self.is_main:
            return


        model_to_save = self.model
        if hasattr(self.model, 'module'):
            model_to_save = self.model.module

        checkpoint = {
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'current_epoch': self.current_epoch,
            'best_val_loss': self.best_val_loss,
            'config': self.config.__dict__,
        }

        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        path = Path(self.config.save_dir) / f'{name}.pt'
        torch.save(checkpoint, path)
        print(f"  Saved checkpoint to {path}")


        self._cleanup_checkpoints()

    def _load_checkpoint(self, path: str):

        print(f" Loading checkpoint from {path}")

        map_location = f'cuda:{get_rank()}' if torch.cuda.is_available() else 'cpu'
        checkpoint = torch.load(path, map_location=map_location)


        model_to_load = self.model
        if hasattr(self.model, 'module'):
            model_to_load = self.model.module
        model_to_load.load_state_dict(checkpoint['model_state_dict'])


        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])


        self.global_step = checkpoint.get('global_step', 0)
        self.current_epoch = checkpoint.get('current_epoch', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))


        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        print(f"OK Resumed from epoch {self.current_epoch}, step {self.global_step}")

    def _cleanup_checkpoints(self):

        save_dir = Path(self.config.save_dir)
        checkpoints = sorted(save_dir.glob('checkpoint_step_*.pt'),
                           key=lambda x: int(x.stem.split('_')[-1]))

        while len(checkpoints) > self.config.keep_last_n:
            oldest = checkpoints.pop(0)
            oldest.unlink()
            print(f"    Removed old checkpoint: {oldest}")

    def _log(self, metrics: Dict[str, float], prefix: str = ''):

        if not self.is_main:
            return

        log_str = f"Step {self.global_step} | Epoch {self.current_epoch}"
        for k, v in metrics.items():

            if 'lr' in k.lower():
                log_str += f" | {prefix}{k}: {v:.2e}"
            else:
                log_str += f" | {prefix}{k}: {v:.4f}"
        print(log_str)

    def train(self):

        print(f" Starting training for {self.config.epochs} epochs")
        print(f"   Total steps: {self.total_steps}")
        print(f"   Device: {self.device}")

        start_time = time.time()

        for epoch in range(self.current_epoch, self.config.epochs):
            self.current_epoch = epoch


            if hasattr(self.train_dataloader, 'set_epoch'):
                self.train_dataloader.set_epoch(epoch)

            epoch_losses = {}
            num_batches = 0
            epoch_start_time = time.time()
            nan_count = 0

            total_batches = len(self.train_dataloader)


            pbar = tqdm(
                enumerate(self.train_dataloader),
                total=total_batches,
                desc=f"Epoch {epoch}",
                disable=not self.is_main,
                ncols=120,
                dynamic_ncols=True,
            )

            for step, batch in pbar:

                losses = self._train_step(batch)


                has_nan = any(math.isnan(v) if isinstance(v, float) else False for v in losses.values())
                if has_nan:
                    nan_count += 1
                    if nan_count <= 5:
                        pbar.write(f"WARN  NaN detected at step {self.global_step}, batch {step}")
                    continue


                for k, v in losses.items():
                    if not math.isnan(v):
                        epoch_losses[k] = epoch_losses.get(k, 0) + v
                num_batches += 1


                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    self._optimizer_step()
                    self.global_step += 1


                    loss_val = losses.get('total_loss', 0)
                    lr = self.scheduler.get_last_lr()[0]
                    pbar.set_postfix({
                        'step': self.global_step,
                        'loss': f'{loss_val:.4f}',
                        'lr': f'{lr:.2e}',
                    }, refresh=True)


                    if self.global_step % self.config.log_every_n_steps == 0:
                        losses['lr'] = lr
                        pbar.write(f"Step {self.global_step} | loss: {loss_val:.4f} | lr: {lr:.2e}")
                        if nan_count > 0:
                            pbar.write(f"   WARN  NaN count so far: {nan_count}")


                    if self.global_step % self.config.val_every_n_steps == 0:
                        val_metrics = self._validate()
                        if val_metrics:
                            val_loss = val_metrics.get('val_total_loss', 0)
                            pbar.write(f"Step {self.global_step} | val_loss: {val_loss:.4f}")


                            if val_loss < self.best_val_loss:
                                self.best_val_loss = val_loss
                                self._save_checkpoint('best_model')


                    if self.global_step % self.config.save_every_n_steps == 0:
                        self._save_checkpoint(f'checkpoint_step_{self.global_step}')

            pbar.close()


            epoch_time = time.time() - epoch_start_time
            avg_losses = {k: v / max(num_batches, 1) for k, v in epoch_losses.items()}
            if self.is_main:
                print(f"\n{'='*60}")
                print(f"  Epoch {epoch} completed in {epoch_time/60:.1f} min")
                print(f"   Valid batches: {num_batches}/{total_batches}, NaN batches: {nan_count}")
                self._log(avg_losses, prefix='epoch_avg_')
                print(f"{'='*60}")


            if nan_count > total_batches * 0.5:
                print(f"ERR Too many NaN batches ({nan_count}/{total_batches}), stopping training!")
                break


            if (epoch + 1) % self.config.val_every_n_epochs == 0:
                val_metrics = self._validate()
                if val_metrics and self.is_main:
                    self._log(val_metrics)


            if (epoch + 1) % self.config.save_every_n_epochs == 0:
                self._save_checkpoint(f'checkpoint_epoch_{epoch}')


            if self.dist_manager is not None:
                self.dist_manager.barrier()


        total_time = time.time() - start_time
        if self.is_main:
            print(f"\n Training completed in {total_time/3600:.2f} hours")
            self._save_checkpoint('final_model')

        return self.best_val_loss