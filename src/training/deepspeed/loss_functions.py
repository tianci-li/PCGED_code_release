


import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class PCGEDLoss(nn.Module):


    def __init__(
        self,
        gene_loss_weight: float = 1.0,
        expr_bin_loss_weight: float = 0.5,
        expr_level_loss_weight: float = 0.3,
        pad_token_id: int = 0,
        n_genes: int = 60669,
        n_expr_bins: int = 16,
        n_expr_levels: int = 9,
    ):
        super().__init__()
        self.gene_loss_weight = gene_loss_weight
        self.expr_bin_loss_weight = expr_bin_loss_weight
        self.expr_level_loss_weight = expr_level_loss_weight
        self.pad_token_id = pad_token_id


        self.n_genes = n_genes
        self.n_expr_bins = n_expr_bins
        self.n_expr_levels = n_expr_levels

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:


        gene_logits = outputs['gene_logits']
        expr_bin_logits = outputs['expr_bin_logits']
        expr_level_logits = outputs['expr_level_logits']


        gene_tokens = batch['gene_tokens'][:, 1:].contiguous()
        expr_bin_tokens = batch['expr_bin_tokens'][:, 1:].contiguous()
        expr_level_tokens = batch['expr_level_tokens'][:, 1:].contiguous()


        gene_logits = gene_logits[:, :-1, :].contiguous()
        expr_bin_logits = expr_bin_logits[:, :-1, :].contiguous()
        expr_level_logits = expr_level_logits[:, :-1, :].contiguous()


        gene_loss = F.cross_entropy(
            gene_logits.view(-1, gene_logits.size(-1)),
            gene_tokens.view(-1),
            ignore_index=self.pad_token_id,
        )


        expr_bin_loss = F.cross_entropy(
            expr_bin_logits.view(-1, expr_bin_logits.size(-1)),
            expr_bin_tokens.view(-1),
            ignore_index=0,
        )


        expr_level_loss = F.cross_entropy(
            expr_level_logits.view(-1, expr_level_logits.size(-1)),
            expr_level_tokens.view(-1),
            ignore_index=0,
        )


        total_loss = (
            self.gene_loss_weight * gene_loss +
            self.expr_bin_loss_weight * expr_bin_loss +
            self.expr_level_loss_weight * expr_level_loss
        )

        loss_dict = {
            'gene_loss': gene_loss.detach(),
            'expr_bin_loss': expr_bin_loss.detach(),
            'expr_level_loss': expr_level_loss.detach(),
            'total_loss': total_loss.detach(),
        }

        return total_loss, loss_dict


def pipeline_loss_fn(
    outputs: torch.Tensor,
    labels: torch.Tensor,
    n_genes: int = 60669,
    n_expr_bins: int = 16,
    n_expr_levels: int = 9,
    gene_loss_weight: float = 1.0,
    expr_bin_loss_weight: float = 0.5,
    expr_level_loss_weight: float = 0.3,
) -> torch.Tensor:


    L_out = outputs.shape[1]
    L_label = labels.shape[1]



    if L_out != L_label:

        min_len = min(L_out, L_label)
        outputs = outputs[:, :min_len, :]
        labels = labels[:, :min_len, :]



    gene_logits = outputs[..., :n_genes]
    expr_bin_logits = outputs[..., n_genes:n_genes + n_expr_bins]
    expr_level_logits = outputs[..., n_genes + n_expr_bins:]



    gene_targets = labels[..., 0].long()
    expr_bin_targets = labels[..., 1].long()
    expr_level_targets = labels[..., 2].long()






    gene_logits_shifted = gene_logits[:, :-1, :]
    expr_bin_logits_shifted = expr_bin_logits[:, :-1, :]
    expr_level_logits_shifted = expr_level_logits[:, :-1, :]

    gene_targets_shifted = gene_targets[:, 1:]
    expr_bin_targets_shifted = expr_bin_targets[:, 1:]
    expr_level_targets_shifted = expr_level_targets[:, 1:]







    gene_loss = F.cross_entropy(
        gene_logits_shifted.reshape(-1, n_genes),
        gene_targets_shifted.reshape(-1),
        ignore_index=0,
    )

    expr_bin_loss = F.cross_entropy(
        expr_bin_logits_shifted.reshape(-1, n_expr_bins),
        expr_bin_targets_shifted.reshape(-1),
        ignore_index=0,
    )

    expr_level_loss = F.cross_entropy(
        expr_level_logits_shifted.reshape(-1, n_expr_levels),
        expr_level_targets_shifted.reshape(-1),
        ignore_index=0,
    )


    total_loss = (
        gene_loss_weight * gene_loss +
        expr_bin_loss_weight * expr_bin_loss +
        expr_level_loss_weight * expr_level_loss
    )


    DEBUG_GRADIENT = os.environ.get('DEBUG_GRADIENT', '0') == '1'
    DEBUG_LOSS = os.environ.get('DEBUG_LOSS', '0') == '1'

    if DEBUG_LOSS or DEBUG_GRADIENT:
        rank = int(os.environ.get('RANK', 0))
        if rank == 0:
            print("[pipeline_loss_fn] Loss breakdown:")
            print(f"  gene_loss: {gene_loss.item():.4f}")
            print(f"  expr_bin_loss: {expr_bin_loss.item():.4f}")
            print(f"  expr_level_loss: {expr_level_loss.item():.4f}")
            print(f"  total_loss: {total_loss.item():.4f}")

    if DEBUG_GRADIENT:
        rank = int(os.environ.get('RANK', 0))
        print("[Rank {}] pipeline_loss_fn DEBUG:".format(rank))
        print("  outputs.requires_grad: {}".format(outputs.requires_grad))
        print("  gene_loss.requires_grad: {}".format(gene_loss.requires_grad))
        print("  total_loss.requires_grad: {}".format(total_loss.requires_grad))
        print("  total_loss.grad_fn: {}".format(total_loss.grad_fn))
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print("  WARN  total_loss is NaN/Inf!")






    has_nan = (
        torch.isnan(gene_loss) or torch.isnan(expr_bin_loss) or torch.isnan(expr_level_loss) or
        torch.isinf(gene_loss) or torch.isinf(expr_bin_loss) or torch.isinf(expr_level_loss)
    )

    if has_nan:

        if not hasattr(pipeline_loss_fn, '_nan_count'):
            pipeline_loss_fn._nan_count = 0
            pipeline_loss_fn._total_count = 0
        pipeline_loss_fn._nan_count += 1
        pipeline_loss_fn._total_count += 1

        if pipeline_loss_fn._nan_count % 100 == 1:
            rank = int(os.environ.get('RANK', 0))
            if rank == 0:
                nan_ratio = pipeline_loss_fn._nan_count / pipeline_loss_fn._total_count * 100
                print(f"WARN  NaN detected in loss! Count: {pipeline_loss_fn._nan_count}/"
                      f"{pipeline_loss_fn._total_count} ({nan_ratio:.1f}%)")
                print(f"    gene_loss: {gene_loss.item() if not torch.isnan(gene_loss) else 'NaN'}")
                print(f"    expr_bin_loss: {expr_bin_loss.item() if not torch.isnan(expr_bin_loss) else 'NaN'}")
                print(f"    expr_level_loss: {expr_level_loss.item() if not torch.isnan(expr_level_loss) else 'NaN'}")



        total_loss = total_loss.clamp(min=-1e4, max=1e4)
    else:

        if not hasattr(pipeline_loss_fn, '_total_count'):
            pipeline_loss_fn._total_count = 0
        pipeline_loss_fn._total_count += 1


        total_loss = total_loss.clamp(min=-1e4, max=1e4)


    if DEBUG_GRADIENT:
        print("  After processing:")
        print("    total_loss.requires_grad: {}".format(total_loss.requires_grad))
        print("    total_loss.grad_fn: {}".format(total_loss.grad_fn))

    return total_loss


class PipelineLossWrapper:


    def __init__(
        self,
        n_genes: int = 60669,
        n_expr_bins: int = 16,
        n_expr_levels: int = 9,
        gene_loss_weight: float = 1.0,
        expr_bin_loss_weight: float = 0.5,
        expr_level_loss_weight: float = 0.3,
    ):
        self.n_genes = n_genes
        self.n_expr_bins = n_expr_bins
        self.n_expr_levels = n_expr_levels
        self.gene_loss_weight = gene_loss_weight
        self.expr_bin_loss_weight = expr_bin_loss_weight
        self.expr_level_loss_weight = expr_level_loss_weight

    def __call__(self, outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return pipeline_loss_fn(
            outputs=outputs,
            labels=labels,
            n_genes=self.n_genes,
            n_expr_bins=self.n_expr_bins,
            n_expr_levels=self.n_expr_levels,
            gene_loss_weight=self.gene_loss_weight,
            expr_bin_loss_weight=self.expr_bin_loss_weight,
            expr_level_loss_weight=self.expr_level_loss_weight,
        )