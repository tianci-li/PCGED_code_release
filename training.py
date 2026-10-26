


import os
import sys
import logging
import signal




def detect_accelerator():

    import torch


    if torch.cuda.is_available():
        try:
            device_name = torch.cuda.get_device_name(0).lower()

            if 'amd' in device_name or 'radeon' in device_name or 'mi' in device_name:
                return 'rocm'

            if 'dcu' in device_name or 'hygon' in device_name:
                return 'dcu'

            return 'cuda'
        except:
            return 'cuda'


    if os.environ.get('ROCM_HOME') or os.environ.get('HIP_VISIBLE_DEVICES'):
        return 'rocm'

    return 'cpu'

ACCELERATOR_TYPE = detect_accelerator()


if (ACCELERATOR_TYPE in ('rocm', 'dcu')):

    os.environ['DISABLE_FLASH_ATTN'] = '1'

    os.environ.setdefault('PYTORCH_HIP_ALLOC_CONF', 'expandable_segments:True')
else:

    os.environ.setdefault('DISABLE_FLASH_ATTN', '0')

    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')


os.environ['DEEPSPEED_LOG_LEVEL'] = 'ERROR'

import argparse
import gc
import json
import random
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


if (ACCELERATOR_TYPE in ('rocm', 'dcu')):

    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
else:

    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)

import yaml

try:
    import deepspeed
    from deepspeed import DeepSpeedEngine
    from deepspeed.utils import logger as ds_logger

    ds_logger.setLevel(logging.ERROR)

    logging.getLogger('deepspeed.runtime.fp16.unfused_optimizer').setLevel(logging.ERROR)
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False
    print("WARN  DeepSpeed not installed. Please install with: pip install deepspeed")


project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.model.model_pathway_aux import PCGED, PCGEDConfig


try:
    from src.data.dataset import SparseCellDataset
    from src.data.dataloader import create_dataloaders
    DATASET_AVAILABLE = True
except ImportError as e:
    DATASET_AVAILABLE = False
    SparseCellDataset = None
    create_dataloaders = None
    print(f"WARN  Dataset modules not available: {e}")


try:
    from src.training.deepspeed import (
        PCGEDPipelineModel,
        create_pipeline_model,
        pack_pipeline_input,
        PipelineLossWrapper,
    )

    from src.training.deepspeed.pipeline_model_v2 import (
        create_pipeline_model_v2,
        pack_pipeline_input as pack_pipeline_input_v2,
    )
    PIPELINE_AVAILABLE = True
except ImportError as e:
    PIPELINE_AVAILABLE = False
    print(f"WARN  Pipeline modules not available: {e}")






@dataclass
class TrainingArgs:


    config: str = "configs/train_config.yaml"
    deepspeed_config: str = "configs/ds_config_zero2.json"


    pipeline_parallel_size: int = 1
    fixed_seq_length: int = -1


    max_steps: int = -1
    max_epochs: int = 5


    output_dir: str = "./outputs"
    logging_dir: str = ""
    save_steps: int = 1000
    save_total_limit: int = 5
    resume_from_checkpoint: Optional[str] = None
    resume_mode: str = "full"


    logging_steps: int = 100
    eval_steps: int = 500


    test_mode: bool = False
    seed: int = 42
    local_rank: int = -1






def set_seed(seed: int):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_yaml_config(path: str) -> dict:

    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_json_config(path: str) -> dict:

    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def is_main_process() -> bool:

    return int(os.environ.get('RANK', 0)) == 0


def get_rank() -> int:

    return int(os.environ.get('RANK', 0))


def get_world_size() -> int:

    return int(os.environ.get('WORLD_SIZE', 1))


def print_rank_0(message: str):

    if is_main_process():
        print(message)






def create_model_from_config(config: dict, n_pathways: int = None) -> PCGED:

    model_config = config['model']


    actual_n_pathways = n_pathways or model_config.get('n_pathways', 2000)


    pathway_masking_block_size_range = (
        model_config.get('pathway_masking_block_size_min', 0.1),
        model_config.get('pathway_masking_block_size_max', 0.4),
    )


    pathway_masking_total_steps = model_config.get('pathway_masking_total_steps', 500000)

    pcged_config = PCGEDConfig(
        n_genes=model_config.get('n_genes', 60664),
        n_expr_bins=model_config.get('n_expr_bins', 16),
        n_expr_levels=model_config.get('n_expr_levels', 9),
        n_pathways=actual_n_pathways,
        d_model=model_config.get('d_model', 768),
        n_layers=model_config.get('n_layers', 12),
        n_heads=model_config.get('n_heads', 12),
        d_ff=model_config.get('d_ff', 3072),
        dropout=model_config.get('dropout', 0.1),
        max_seq_len=model_config.get('max_seq_len', 8192),
        pathway_encoder_type=model_config.get('pathway_encoder_type', 'discrete'),
        n_pathway_tokens=model_config.get('n_pathway_tokens', 8),
        n_global_pathway_tokens=model_config.get('n_global_pathway_tokens', 2),
        pathway_hidden_dim=model_config.get('pathway_hidden_dim', 2048),
        pathway_fusion_layers=model_config.get('pathway_fusion_layers', 2),
        use_flash_attention=model_config.get('use_flash_attention', True),
        use_gradient_checkpointing=model_config.get('use_gradient_checkpointing', True),
        tie_word_embeddings=model_config.get('tie_word_embeddings', False),

        pathway_masking_enabled=model_config.get('pathway_masking_enabled', True),
        pathway_curriculum_phase0_end=model_config.get('pathway_curriculum_phase0_end', 0.10),
        pathway_curriculum_phase1_end=model_config.get('pathway_curriculum_phase1_end', 0.20),
        pathway_curriculum_phase2_end=model_config.get('pathway_curriculum_phase2_end', 0.60),
        pathway_curriculum_k_phase1=model_config.get('pathway_curriculum_k_phase1', 5),
        pathway_curriculum_k_max_ratio=model_config.get('pathway_curriculum_k_max_ratio', 0.55),
        pathway_curriculum_phase3_dropout=model_config.get('pathway_curriculum_phase3_dropout', 0.1),
        pathway_masking_total_steps=pathway_masking_total_steps,
        pathway_interpolation_enabled=model_config.get('pathway_interpolation_enabled', False),
        pathway_interpolation_mode=model_config.get('pathway_interpolation_mode', 'off'),
        pathway_interpolation_strategy=model_config.get('pathway_interpolation_strategy', 'linear'),
        pathway_interpolation_mix_ratio=model_config.get('pathway_interpolation_mix_ratio', 0.5),
        pathway_aux_head_enabled=model_config.get('pathway_aux_head_enabled', False),
    )

    return PCGED(pcged_config)






def create_datasets(config: dict):

    if not DATASET_AVAILABLE:
        raise ImportError("Dataset modules not available. Please check src/data/dataset.py")

    data_config = config['data']
    model_config = config.get('model', {})


    dataset = SparseCellDataset(
        index_folder=data_config['index_folder'],
        csv_filepath=data_config['csv_filepath'],
        global_index=data_config['global_index'],
        cell_description=data_config['cell_description'],
        gene_location_list_path=data_config['gene_location_list_path'],
        max_len=data_config.get('max_len', 8192),
        anonymous=data_config.get('anonymous', True),
        pathway_db_path=data_config.get('pathway_db_path', './pathway_data/pathway_database_ensembl.pkl'),
        pathway_method=data_config.get('pathway_method', 'combined'),
        min_pathway_genes=data_config.get('min_pathway_genes', 5),
        max_pathway_genes=data_config.get('max_pathway_genes', 500),
        n_expr_bins=data_config.get('n_expr_bins', 20),
        use_global_binning=data_config.get('use_global_binning', True),
        return_pathway_scores=(
            model_config.get('pathway_interpolation_enabled', False)
            or model_config.get('pathway_aux_head_enabled', False)
        ),
    )


    n_pathways = dataset.n_pathways


    total_len = len(dataset)
    train_len = int(total_len * 0.8)
    val_len = int(total_len * 0.1)
    test_len = total_len - train_len - val_len

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_len, val_len, test_len],
        generator=torch.Generator().manual_seed(config.get('seed', 42))
    )

    print_rank_0(f"  Dataset info:")
    print_rank_0(f"   Total samples: {total_len:,}")
    print_rank_0(f"   Train samples: {train_len:,} (80%)")
    print_rank_0(f"   Val samples:   {val_len:,} (10%)")
    print_rank_0(f"   Test samples:  {test_len:,} (10%)")
    print_rank_0(f"   N pathways: {n_pathways}")

    return train_dataset, val_dataset, test_dataset, n_pathways






def chunked_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_chunks: int = 8,
    ignore_index: int = -100,
) -> torch.Tensor:


    if logits.size(-1) <= 1024:
        return F.cross_entropy(
            logits.float().reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=ignore_index,
        )


    logits_chunks = logits.chunk(num_chunks, dim=1)
    labels_chunks = labels.chunk(num_chunks, dim=1)

    total_loss = torch.tensor(0.0, device=logits.device, dtype=torch.float32)
    total_valid_tokens = 0

    for logits_chunk, labels_chunk in zip(logits_chunks, labels_chunks):


        logits_chunk_fp32 = logits_chunk.float()


        valid_mask = labels_chunk != ignore_index
        n_valid = valid_mask.sum().item()

        if n_valid == 0:
            continue


        chunk_loss = F.cross_entropy(
            logits_chunk_fp32.reshape(-1, logits_chunk_fp32.size(-1)),
            labels_chunk.reshape(-1),
            ignore_index=ignore_index,
            reduction='sum',
        )

        total_loss = total_loss + chunk_loss
        total_valid_tokens += n_valid


    if total_valid_tokens > 0:
        return total_loss / total_valid_tokens
    else:
        return total_loss


class PCGEDLoss(nn.Module):


    def __init__(
        self,
        gene_loss_weight: float = 1.0,
        expr_bin_loss_weight: float = 0.5,
        expr_level_loss_weight: float = 0.3,
        pathway_loss_weight: float = 0.0,
        pathway_loss_type: str = 'smooth_l1',
        pathway_loss_use_sigmoid: bool = True,
        pathway_num_bins: int = 5,
        eos_loss_weight: float = 0.0,
        eos_pos_weight: float = 16.0,
        eos_token_id: int = 2,
        pad_token_id: int = 0,
        num_output_chunks: int = 8,
    ):
        super().__init__()
        self.gene_loss_weight = gene_loss_weight
        self.expr_bin_loss_weight = expr_bin_loss_weight
        self.expr_level_loss_weight = expr_level_loss_weight
        self.pathway_loss_weight = pathway_loss_weight
        self.pathway_loss_type = pathway_loss_type.lower()
        self.pathway_loss_use_sigmoid = pathway_loss_use_sigmoid
        self.pathway_target_max = max(int(pathway_num_bins) - 1, 1)
        self.eos_loss_weight = eos_loss_weight
        self.eos_pos_weight = float(eos_pos_weight)
        self.eos_token_id = int(eos_token_id)
        self.pad_token_id = pad_token_id
        self.num_output_chunks = num_output_chunks
        self._logged_first_step = False

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        gene_logits = outputs['gene_logits']
        expr_bin_logits = outputs['expr_bin_logits']
        expr_level_logits = outputs['expr_level_logits']


        gene_targets = batch['gene_tokens'][:, 1:].contiguous()
        expr_bin_targets = batch['expr_bin_tokens'][:, 1:].contiguous()
        expr_level_targets = batch['expr_level_tokens'][:, 1:].contiguous()


        gene_logits = gene_logits[:, :-1, :].contiguous()
        expr_bin_logits = expr_bin_logits[:, :-1, :].contiguous()
        expr_level_logits = expr_level_logits[:, :-1, :].contiguous()


        if not self._logged_first_step and is_main_process():
            self._logged_first_step = True
            print(f"  [PCGEDLoss] gene_logits dtype: {gene_logits.dtype}, "
                  f"shape: {gene_logits.shape}, "
                  f"num_chunks: {self.num_output_chunks}")
            print(f"   -> gene_loss         CE (chunked fp32 softmax)")
            print(f"   -> expr_bin/level_loss      fp32 CE")
            if self.pathway_loss_weight > 0:
                print(
                    f"   -> pathway_loss   : weight={self.pathway_loss_weight}, "
                    f"type={self.pathway_loss_type}, sigmoid={self.pathway_loss_use_sigmoid}"
                )
            if self.eos_loss_weight > 0:
                print(
                    f"   -> eos_loss   : weight={self.eos_loss_weight}, "
                    f"pos_weight={self.eos_pos_weight}, eos_token_id={self.eos_token_id}"
                )





        gene_loss = chunked_cross_entropy(
            gene_logits,
            gene_targets,
            num_chunks=self.num_output_chunks,
            ignore_index=self.pad_token_id,
        )





        expr_bin_loss = F.cross_entropy(
            expr_bin_logits.float().reshape(-1, expr_bin_logits.size(-1)),
            expr_bin_targets.reshape(-1),
            ignore_index=0,
        )

        expr_level_loss = F.cross_entropy(
            expr_level_logits.float().reshape(-1, expr_level_logits.size(-1)),
            expr_level_targets.reshape(-1),
            ignore_index=0,
        )

        pathway_loss = None
        if self.pathway_loss_weight > 0:
            pathway_aux_pred = outputs.get('pathway_aux_pred')
            pathway_targets = batch.get('pathway_scores')


            if pathway_targets is None:
                pathway_bins = batch.get('pathway_embedding')
                if pathway_bins is not None:
                    pathway_targets = pathway_bins.float() / float(self.pathway_target_max)

            if pathway_aux_pred is not None and pathway_targets is not None:
                pred = pathway_aux_pred.float()
                target = pathway_targets.float().clamp(0.0, 1.0)

                if pred.shape != target.shape:
                    n_common = min(pred.size(-1), target.size(-1))
                    pred = pred[:, :n_common]
                    target = target[:, :n_common]

                if self.pathway_loss_use_sigmoid:
                    pred = torch.sigmoid(pred)

                if self.pathway_loss_type == 'mse':
                    pathway_loss = F.mse_loss(pred, target)
                else:
                    pathway_loss = F.smooth_l1_loss(pred, target)

        eos_loss = None
        if self.eos_loss_weight > 0:
            valid_mask = gene_targets.ne(self.pad_token_id)
            if valid_mask.any():
                eos_logits = gene_logits.float()[..., self.eos_token_id]
                eos_targets = gene_targets.eq(self.eos_token_id).float()
                pos_weight = torch.tensor(
                    self.eos_pos_weight,
                    device=eos_logits.device,
                    dtype=eos_logits.dtype,
                )
                eos_loss = F.binary_cross_entropy_with_logits(
                    eos_logits[valid_mask],
                    eos_targets[valid_mask],
                    pos_weight=pos_weight,
                )


        total_loss = (
            self.gene_loss_weight * gene_loss +
            self.expr_bin_loss_weight * expr_bin_loss +
            self.expr_level_loss_weight * expr_level_loss
        )

        if pathway_loss is not None:
            total_loss = total_loss + self.pathway_loss_weight * pathway_loss

        if eos_loss is not None:
            total_loss = total_loss + self.eos_loss_weight * eos_loss

        loss_dict = {
            'gene_loss': gene_loss.detach(),
            'expr_bin_loss': expr_bin_loss.detach(),
            'expr_level_loss': expr_level_loss.detach(),
            'total_loss': total_loss.detach(),
        }

        if pathway_loss is not None:
            loss_dict['pathway_loss'] = pathway_loss.detach()
            loss_dict['pathway_loss_weighted'] = (
                self.pathway_loss_weight * pathway_loss
            ).detach()

        if eos_loss is not None:
            loss_dict['eos_loss'] = eos_loss.detach()
            loss_dict['eos_loss_weighted'] = (
                self.eos_loss_weight * eos_loss
            ).detach()

        return total_loss, loss_dict






class DeepSpeedTrainer:


    def __init__(
        self,
        model: nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: Optional[torch.utils.data.DataLoader],
        test_dataloader: Optional[torch.utils.data.DataLoader],
        args: TrainingArgs,
        ds_config: dict,
        loss_fn: nn.Module,
    ):
        self.args = args
        self.ds_config = ds_config
        self.loss_fn = loss_fn
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader


        self.gradient_accumulation_steps = ds_config.get("gradient_accumulation_steps", 1)




        if ACCELERATOR_TYPE in ('rocm', 'dcu'):

            print_rank_0(f"   WARN ROCm/DCU detected - using torch.optim.AdamW (FusedAdam not supported)")
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=ds_config.get("optimizer", {}).get("params", {}).get("lr", 1e-4),
                betas=tuple(ds_config.get("optimizer", {}).get("params", {}).get("betas", [0.9, 0.95])),
                eps=ds_config.get("optimizer", {}).get("params", {}).get("eps", 1e-8),
                weight_decay=ds_config.get("optimizer", {}).get("params", {}).get("weight_decay", 0.1),
            )

            ds_config_copy = ds_config.copy()
            if "optimizer" in ds_config_copy:
                del ds_config_copy["optimizer"]

            ds_config_copy["zero_allow_untested_optimizer"] = True

            self.model_engine, self.optimizer, _, self.lr_scheduler = deepspeed.initialize(
                model=model,
                optimizer=optimizer,
                model_parameters=model.parameters(),
                config=ds_config_copy,
            )
        else:

            print_rank_0(f"   OK NVIDIA detected - using DeepSpeed FusedAdam (CUDA fused kernel)")
            ds_config_copy = ds_config.copy()

            ds_config_copy.pop("zero_allow_untested_optimizer", None)






            if int(os.environ.get('DS_OVERLAP_COMM', '1')) == 0:
                if 'zero_optimization' in ds_config_copy:
                    ds_config_copy['zero_optimization']['overlap_comm'] = False
                print_rank_0(f"   WARN overlap_comm DISABLED via DS_OVERLAP_COMM=0 (A/B test)")

            self.model_engine, self.optimizer, _, self.lr_scheduler = deepspeed.initialize(
                model=model,
                model_parameters=model.parameters(),
                config=ds_config_copy,
            )


        inner_opt = getattr(self.optimizer, 'optimizer', self.optimizer)
        inner_opt_name = type(inner_opt).__name__
        print_rank_0(f"    Actual optimizer: {type(self.optimizer).__name__} (inner: {inner_opt_name})")






        self.global_step = 0
        self._optimizer_step = 0
        self._masking_step_offset = 0
        self._masking_reset_on_resume = os.environ.get('MASKING_RESET_ON_RESUME', '0') == '1'
        self.current_epoch = 0
        self._epoch_step = 0
        self.best_val_loss = float('inf')


        self._stop_requested = False


        self._grad_norm_sum = 0.0
        self._grad_norm_count = 0
        self._grad_norm_max = 0.0


        self.logging_dir = Path(args.logging_dir) if args.logging_dir else (Path(args.output_dir) / 'logs')
        self.metrics_jsonl_path: Optional[Path] = None
        if is_main_process():
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)
            self.logging_dir.mkdir(parents=True, exist_ok=True)
            self.metrics_jsonl_path = self.logging_dir / 'metrics.jsonl'
            print_rank_0(f"     metrics JSONL: {self.metrics_jsonl_path}")


        print_rank_0(f"   gradient_accumulation_steps: {self.gradient_accumulation_steps}")
        print_rank_0(f"   -> logging/eval/save/masking    optimizer step    ")


        scheduler_params = self.ds_config.get('scheduler', {}).get('params', {})
        ds_total_steps = scheduler_params.get('total_num_steps')
        if isinstance(ds_total_steps, (int, float)) and ds_total_steps > 0:
            if self.args.max_steps > 0:
                expected_opt_steps = math.ceil(
                    self.args.max_steps / max(self.gradient_accumulation_steps, 1)
                )
            else:
                expected_opt_steps = math.ceil(
                    (len(self.train_dataloader) * self.args.max_epochs) /
                    max(self.gradient_accumulation_steps, 1)
                )
            ratio = float(ds_total_steps) / max(expected_opt_steps, 1)
            print_rank_0(
                f"   scheduler.total_num_steps={int(ds_total_steps)}, "
                f"expected_opt_steps~{expected_opt_steps} (ratio={ratio:.4f})"
            )
            if ratio < 0.8:
                print_rank_0(
                    "   WARN scheduler total_num_steps   ,LR         warmup_min_lr,"
                    "     '      /     '."
                )
                print_rank_0(
                    f"         total_num_steps      {expected_opt_steps}(         )"
                )


        if args.resume_from_checkpoint:
            self._load_checkpoint(args.resume_from_checkpoint, resume_mode=args.resume_mode)
            if self._masking_reset_on_resume and self._optimizer_step > 0 and self._masking_step_offset == 0:

                self._masking_step_offset = self._optimizer_step
                print_rank_0(
                    f"  Curriculum step reset enabled: masking_step = opt_step - {self._masking_step_offset}"
                )
            elif (not self._masking_reset_on_resume) and self._masking_step_offset > 0:


                old_off = self._masking_step_offset
                self._masking_step_offset = 0
                print_rank_0(
                    f"  MASKING_RESET_ON_RESUME=0: cleared inherited offset {old_off} -> 0 "
                    f"(curriculum now follows absolute opt_step={self._optimizer_step})"
                )

    def _get_effective_masking_step(self) -> int:

        if self._masking_step_offset > 0:
            return max(self._optimizer_step - self._masking_step_offset, 0)
        return self._optimizer_step

    def _should_reset_data_cursor_on_resume(self, checkpoint_path: Path, resume_mode: str) -> bool:

        if resume_mode != 'full':
            return False
        if os.environ.get('DATA_RESET_ON_RESUME', '0') == '1':
            return True
        if checkpoint_path.name == 'best_model' and os.environ.get('DATA_RESET_ON_BEST_RESUME', '1') == '1':
            return True
        return False

    def _reset_data_cursor_after_resume(self, reason: str):

        self.current_epoch = 0
        self._epoch_step = 0
        if hasattr(self.train_dataloader, 'reset_resume'):
            self.train_dataloader.reset_resume()
        print_rank_0(
            f"     Data cursor reset ({reason}): epoch=0, epoch_step=0 (global/optimizer steps preserved)"
        )

    @property
    def device(self) -> torch.device:
        return self.model_engine.device

    def _profile_train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:

        def _get_nccl_seq_num() -> int:
            if not torch.distributed.is_initialized():
                return 0
            try:
                pg = torch.distributed.group.WORLD
                if hasattr(pg, '_get_sequence_number_for_group'):
                    return int(pg._get_sequence_number_for_group())
            except Exception:
                pass
            return 0

        timings = {}
        torch.cuda.synchronize()
        t_total = time.time()


        self.model_engine.train()


        t0 = time.time()
        batch = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        if self.model_engine.fp16_enabled():
            target_dtype = torch.float16
        elif self.model_engine.bfloat16_enabled():
            target_dtype = torch.bfloat16
        else:
            target_dtype = torch.float32
        for k, v in batch.items():
            if isinstance(v, torch.Tensor) and v.dtype in (torch.float32, torch.float64):
                batch[k] = v.to(target_dtype, non_blocking=True)


        token_count = 0
        max_seq_tokens = 0
        mean_seq_tokens = 0.0
        attn_mask = batch.get('attention_mask')
        if isinstance(attn_mask, torch.Tensor):
            try:
                if attn_mask.dtype != torch.bool:
                    attn_mask = attn_mask != 0
                per_seq = attn_mask.sum(dim=1)
                if per_seq.numel() > 0:
                    token_count = int(per_seq.sum().item())
                    max_seq_tokens = int(per_seq.max().item())
                    mean_seq_tokens = float(per_seq.float().mean().item())
            except Exception:
                pass
        timings['tokens'] = float(token_count)
        timings['max_seq_tokens'] = float(max_seq_tokens)
        timings['mean_seq_tokens'] = float(mean_seq_tokens)

        torch.cuda.synchronize()
        timings['data_transfer'] = (time.time() - t0) * 1000


        t0 = time.time()
        if hasattr(self.model_engine.module, 'set_masking_step'):
            self.model_engine.module.set_masking_step(self._get_effective_masking_step())
        timings['masking_update'] = (time.time() - t0) * 1000


        t0 = time.time()
        outputs = self.model_engine(
            pathway_embedding=batch['pathway_embedding'],
            pathway_scores=batch.get('pathway_scores', None),
            gene_tokens=batch['gene_tokens'],
            expr_bin_tokens=batch['expr_bin_tokens'],
            expr_level_tokens=batch['expr_level_tokens'],
            attention_mask=batch['attention_mask'],
            return_loss=False,
        )
        torch.cuda.synchronize()
        timings['forward'] = (time.time() - t0) * 1000


        t0 = time.time()
        loss, loss_dict = self.loss_fn(outputs, batch)
        torch.cuda.synchronize()
        timings['loss_compute'] = (time.time() - t0) * 1000

        if torch.isnan(loss) or torch.isinf(loss):
            return {k: float('nan') for k in loss_dict.keys()}




        _pre_bwd_latency = 0.0
        if torch.distributed.is_initialized():
            _lat_t = torch.ones(1, device=self.device)
            torch.cuda.synchronize()
            _lat_t0 = time.time()
            for _ in range(10):
                torch.distributed.all_reduce(_lat_t)
            torch.cuda.synchronize()
            _pre_bwd_latency = (time.time() - _lat_t0) * 1000
            del _lat_t

        _seq_before_bwd = _get_nccl_seq_num()
        t0 = time.time()
        self.model_engine.backward(loss)
        t_bwd_launch = time.time()
        torch.cuda.synchronize()
        t_bwd_sync = time.time()
        _seq_after_bwd = _get_nccl_seq_num()
        timings['backward'] = (t_bwd_sync - t0) * 1000
        timings['backward_launch'] = (t_bwd_launch - t0) * 1000
        timings['backward_wait'] = (t_bwd_sync - t_bwd_launch) * 1000


        t0 = time.time()
        prev_ds_global_steps = self.model_engine.global_steps
        self.model_engine.step()
        t_opt_launch = time.time()
        torch.cuda.synchronize()
        t_opt_sync = time.time()
        _seq_after_step = _get_nccl_seq_num()
        timings['optimizer_step'] = (t_opt_sync - t0) * 1000
        timings['opt_launch'] = (t_opt_launch - t0) * 1000
        timings['opt_wait'] = (t_opt_sync - t_opt_launch) * 1000
        timings['nccl_ops_backward'] = float(max(_seq_after_bwd - _seq_before_bwd, 0))
        timings['nccl_ops_step'] = float(max(_seq_after_step - _seq_after_bwd, 0))
        timings['nccl_ops_total'] = float(max(_seq_after_step - _seq_before_bwd, 0))

        if self.model_engine.global_steps > prev_ds_global_steps:
            self._optimizer_step += 1


        tok_k = max(float(token_count) / 1000.0, 1e-6)
        timings['forward_per_1k_tok'] = timings['forward'] / tok_k
        timings['backward_per_1k_tok'] = timings['backward'] / tok_k
        timings['optimizer_step_per_1k_tok'] = timings['optimizer_step'] / tok_k


        mem_alloc_gb = torch.cuda.memory_allocated() / 1024**3
        mem_reserved_gb = torch.cuda.memory_reserved() / 1024**3



        cuda_mem_stats = {}
        try:
            stats = torch.cuda.memory_stats()
            cuda_mem_stats['num_segments'] = stats.get('segment.all.current', 0)
            cuda_mem_stats['num_blocks'] = stats.get('allocation.all.current', 0)
            cuda_mem_stats['inactive_splits'] = stats.get('inactive_split.all.current', 0)
            cuda_mem_stats['alloc_retries'] = stats.get('num_alloc_retries', 0)
        except Exception:
            pass





        ds_internal_state = {}
        try:
            opt = self.model_engine.optimizer

            for attr_name in dir(opt):
                if attr_name.startswith('__'):
                    continue
                try:
                    val = getattr(opt, attr_name)
                    if callable(val):
                        continue
                    if isinstance(val, torch.Tensor):
                        continue
                    if isinstance(val, list) and len(val) > 0:
                        ds_internal_state[f'opt.{attr_name}'] = len(val)
                    elif isinstance(val, dict) and len(val) > 0:
                        ds_internal_state[f'opt.{attr_name}'] = len(val)
                    elif isinstance(val, (set, tuple)) and len(val) > 0:
                        ds_internal_state[f'opt.{attr_name}'] = len(val)
                    elif hasattr(val, '__len__') and not isinstance(val, (str, bytes)):
                        n = len(val)
                        if isinstance(n, int) and n > 0:
                            ds_internal_state[f'opt.{attr_name}'] = n
                except Exception:
                    continue


            ipg_buckets = getattr(opt, 'ipg_buckets', None)
            if isinstance(ipg_buckets, dict):
                for bkey, bucket in ipg_buckets.items():
                    prefix = f'opt.ipg_buckets[{bkey}]'
                    for fld in ['index', 'elements', 'grads_in_partition']:
                        try:
                            fv = getattr(bucket, fld)
                            if isinstance(fv, (int, float)):
                                ds_internal_state[f'{prefix}.{fld}'] = float(fv)
                        except Exception:
                            continue
                    for fld in ['buffer', 'params', 'grads']:
                        try:
                            fv = getattr(bucket, fld)
                            if hasattr(fv, '__len__'):
                                ds_internal_state[f'{prefix}.{fld}_len'] = len(fv)
                        except Exception:
                            continue


            for attr_name in ['tput_timer', 'timers', 'skipped_steps',
                              'compression_scheduler', 'all_reduce_list']:
                val = getattr(self.model_engine, attr_name, None)
                if val is not None:
                    if isinstance(val, (list, dict)):
                        ds_internal_state[f'engine.{attr_name}'] = len(val)
                    elif hasattr(val, 'elapsed') and isinstance(val.elapsed, list):
                        ds_internal_state[f'engine.{attr_name}.elapsed'] = len(val.elapsed)
                    elif hasattr(val, 'step_elapsed'):
                        ds_internal_state[f'engine.{attr_name}.step_elapsed'] = float(val.step_elapsed)

            try:
                ds_internal_state['engine.grad_acc_boundary'] = int(self.model_engine.is_gradient_accumulation_boundary())
            except Exception:
                pass
            micro_steps = getattr(self.model_engine, 'micro_steps', None)
            if isinstance(micro_steps, int):
                ds_internal_state['engine.micro_steps'] = micro_steps


            if torch.distributed.is_initialized():
                try:
                    from torch.distributed.distributed_c10d import _world
                    if hasattr(_world, 'pg_map'):
                        ds_internal_state['dist.pg_count'] = len(_world.pg_map)
                    elif hasattr(_world, 'pg_names'):
                        ds_internal_state['dist.pg_count'] = len(_world.pg_names)
                except Exception:
                    pass

                try:
                    ds_internal_state['nccl_seq_num'] = _seq_after_step if _seq_after_step > 0 else _get_nccl_seq_num()
                except Exception:
                    pass
        except Exception:
            pass


        nccl_times = {}
        if torch.distributed.is_initialized():

            small_tensor = torch.ones(1, device=self.device)
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(10):
                torch.distributed.all_reduce(small_tensor)
            torch.cuda.synchronize()
            nccl_times['allreduce_1elem_x10'] = (time.time() - t0) * 1000


            medium_tensor = torch.ones(8 * 1024 * 1024, device=self.device)
            torch.cuda.synchronize()
            t0 = time.time()
            torch.distributed.all_reduce(medium_tensor)
            torch.cuda.synchronize()
            nccl_times['allreduce_32MB'] = (time.time() - t0) * 1000


            world_size = torch.distributed.get_world_size()
            chunk = torch.ones(8 * 1024 * 1024 // world_size, device=self.device)
            gather_list = [torch.empty_like(chunk) for _ in range(world_size)]
            torch.cuda.synchronize()
            t0 = time.time()
            torch.distributed.all_gather(gather_list, chunk)
            torch.cuda.synchronize()
            nccl_times['allgather_32MB'] = (time.time() - t0) * 1000

            del small_tensor, medium_tensor, chunk, gather_list


        rss_gb = 0.0
        try:
            with open('/proc/self/status') as f:
                for line in f:
                    if line.startswith('VmRSS:'):
                        rss_gb = int(line.split()[1]) / 1024 / 1024
                        break
        except Exception:
            pass

        timings['total'] = (time.time() - t_total) * 1000


        if is_main_process():
            lines = []
            lines.append(f"{'='*70}")
            lines.append(f"  PROFILE step {self.global_step} (opt_step={self._optimizer_step})")
            lines.append(f"{'='*70}")

            if (hasattr(self.model_engine.module, '_get_current_n_visible') and
                    getattr(self.model_engine.module.config, 'pathway_masking_enabled', False)):
                _nv = self.model_engine.module._get_current_n_visible()
                _phase = self.model_engine.module._get_current_phase()
                _N = self.model_engine.module.config.n_pathways
                _total = self.model_engine.module.config.pathway_masking_total_steps
                lines.append(f"    curriculum: Phase {_phase}, n_visible={_nv}/{_N} ({_nv/_N*100:.1f}%), total_steps={_total}")
            if hasattr(self, '_profile_data_load_ms'):
                lines.append(f"  data_load:      {self._profile_data_load_ms:>8.1f} ms")
            lines.append(f"  batch tokens:   {int(timings['tokens']):>8d}  (max_seq={int(timings['max_seq_tokens'])}, mean_seq={timings['mean_seq_tokens']:.1f})")
            lines.append(f"  data_transfer:  {timings['data_transfer']:>8.1f} ms")
            lines.append(f"  masking_update: {timings['masking_update']:>8.1f} ms")
            lines.append(f"  forward:        {timings['forward']:>8.1f} ms")
            lines.append(f"  loss_compute:   {timings['loss_compute']:>8.1f} ms")
            lines.append(f"  backward:       {timings['backward']:>8.1f} ms  (launch={timings['backward_launch']:.1f} + sync={timings['backward_wait']:.1f})")
            lines.append(f"  optimizer_step: {timings['optimizer_step']:>8.1f} ms  (launch={timings['opt_launch']:.1f} + sync={timings['opt_wait']:.1f})")
            lines.append(f"  backward/1k tok:{timings['backward_per_1k_tok']:>8.1f} ms")
            lines.append(f"  opt_step/1k tok:{timings['optimizer_step_per_1k_tok']:>8.1f} ms")
            lines.append(f"  NCCL ops (step): backward={int(timings['nccl_ops_backward'])}, step={int(timings['nccl_ops_step'])}, total={int(timings['nccl_ops_total'])}")
            lines.append(f"                                   ")
            lines.append(f"  total (synced): {timings['total']:>8.1f} ms")
            if _pre_bwd_latency > 0:
                lines.append(f"     NCCL latency check   ")
                lines.append(f"  pre-backward  1elemx10: {_pre_bwd_latency:>7.1f} ms")
                lines.append(f"  post-step     1elemx10: (see below)")
            lines.append(f"  GPU mem alloc:  {mem_alloc_gb:.2f} GB / reserved: {mem_reserved_gb:.2f} GB")

            try:
                _streams_idle = torch.cuda.current_stream().query()
                lines.append(f"  CUDA streams idle: {_streams_idle}")
            except Exception:
                pass
            lines.append(f"  Process RSS:    {rss_gb:.2f} GB")
            if cuda_mem_stats:
                lines.append(f"     CUDA allocator state   ")
                lines.append(f"  segments: {cuda_mem_stats['num_segments']:>6d}  |  blocks: {cuda_mem_stats['num_blocks']:>6d}  "
                             f"|  inactive_splits: {cuda_mem_stats['inactive_splits']:>6d}  |  alloc_retries: {cuda_mem_stats['alloc_retries']:>6d}")
            if ds_internal_state:
                lines.append(f"     DeepSpeed internal state audit   ")
                total_items = sum(
                    v for k, v in ds_internal_state.items()
                    if isinstance(v, (int, float)) and not str(k).startswith('nccl_seq_')
                )
                lines.append(f"  total tracked items(excl seq counters): {total_items:.0f}")
                for k, v in sorted(ds_internal_state.items(),
                                   key=lambda x: -x[1] if isinstance(x[1], (int, float)) else 0):
                    if isinstance(v, int):
                        lines.append(f"    {k}: {v}")
                    elif isinstance(v, float):
                        lines.append(f"    {k}: {v:.1f}")
            if nccl_times:
                lines.append(f"     NCCL communication benchmark   ")
                lines.append(f"  allreduce 1elemx10: {nccl_times['allreduce_1elem_x10']:>7.1f} ms  (latency)")
                lines.append(f"  allreduce 32MB:     {nccl_times['allreduce_32MB']:>7.1f} ms  (bandwidth)")
                lines.append(f"  allgather 32MB:     {nccl_times['allgather_32MB']:>7.1f} ms  (opt_step sim)")

            _ds_total = sum(
                v for k, v in ds_internal_state.items()
                if isinstance(v, (int, float)) and not str(k).startswith('nccl_seq_')
            )
            if hasattr(self, '_profile_history'):
                self._profile_history.append({
                    'step': self.global_step,
                    **timings,
                    **{f'nccl_{k}': v for k, v in nccl_times.items()},
                    'rss_gb': rss_gb,
                    'mem_alloc_gb': mem_alloc_gb,
                    'mem_reserved_gb': mem_reserved_gb,
                    **{f'cuda_{k}': v for k, v in cuda_mem_stats.items()},
                    'ds_internal_total': _ds_total,
                    'nccl_seq_num': ds_internal_state.get('nccl_seq_num', 0),
                })
                if len(self._profile_history) >= 2:
                    first = self._profile_history[0]
                    prev = self._profile_history[-2]
                    lines.append(f"     Trend vs first profile (step {first['step']})   ")
                    for key in ['data_transfer', 'forward', 'backward', 'backward_launch', 'backward_wait',
                                 'optimizer_step', 'opt_launch', 'opt_wait',
                                 'backward_per_1k_tok', 'optimizer_step_per_1k_tok',
                                 'nccl_ops_backward', 'nccl_ops_step', 'total']:
                        if key in first and key in timings:
                            delta_first = timings[key] - first[key]
                            delta_prev = timings[key] - prev.get(key, timings[key])
                            lines.append(f"    {key:18s}: {delta_first:>+7.1f} ms vs first, {delta_prev:>+7.1f} ms vs prev")

                    for nk in ['allreduce_32MB', 'allgather_32MB']:
                        full_key = f'nccl_{nk}'
                        if full_key in first and full_key in self._profile_history[-1]:
                            d1 = nccl_times[nk] - first[full_key]
                            d2 = nccl_times[nk] - prev.get(full_key, nccl_times[nk])
                            lines.append(f"    nccl_{nk:14s}: {d1:>+7.1f} ms vs first, {d2:>+7.1f} ms vs prev")
                    if 'tokens' in first and 'tokens' in timings:
                        lines.append(
                            f"    tokens           : {timings['tokens'] - first['tokens']:>+7.0f} vs first, "
                            f"{timings['tokens'] - prev.get('tokens', timings['tokens']):>+7.0f} vs prev"
                        )
                    lines.append(f"    RSS:               {rss_gb - first['rss_gb']:>+7.2f} GB vs first")
                    lines.append(f"    reserved_gb:       {mem_reserved_gb - first.get('mem_reserved_gb', mem_reserved_gb):>+7.2f} GB vs first")

                    _seq_now = ds_internal_state.get('nccl_seq_num', 0)
                    _seq_first = first.get('nccl_seq_num', 0)
                    _seq_prev = prev.get('nccl_seq_num', 0)
                    if _seq_now > 0:
                        lines.append(f"    nccl_seq_num      : {_seq_now:>7d} total, {_seq_now - _seq_first:>+7d} vs first, {_seq_now - _seq_prev:>+7d} vs prev")

                    for ck in ['num_segments', 'num_blocks', 'inactive_splits', 'alloc_retries']:
                        full_key = f'cuda_{ck}'
                        if full_key in first and full_key in self._profile_history[-1]:
                            d1 = self._profile_history[-1][full_key] - first[full_key]
                            d2 = self._profile_history[-1][full_key] - prev.get(full_key, 0)
                            lines.append(f"    {ck:18s}: {d1:>+7d} vs first, {d2:>+7d} vs prev")

                    if 'ds_internal_total' in first:
                        d1 = self._profile_history[-1].get('ds_internal_total', 0) - first.get('ds_internal_total', 0)
                        d2 = self._profile_history[-1].get('ds_internal_total', 0) - prev.get('ds_internal_total', 0)
                        lines.append(f"    ds_internal_total : {d1:>+7.0f} vs first, {d2:>+7.0f} vs prev")
            else:
                self._profile_history = [{
                    'step': self.global_step,
                    **timings,
                    **{f'nccl_{k}': v for k, v in nccl_times.items()},
                    'rss_gb': rss_gb,
                    'mem_alloc_gb': mem_alloc_gb,
                    'mem_reserved_gb': mem_reserved_gb,
                    **{f'cuda_{k}': v for k, v in cuda_mem_stats.items()},
                    'ds_internal_total': _ds_total,
                    'nccl_seq_num': ds_internal_state.get('nccl_seq_num', 0),
                }]
            lines.append(f"{'='*70}")

            report = '\n'.join(lines)
            print(f"\n{report}\n")


            profile_path = Path(self.args.output_dir) / 'profile_log.txt'
            with open(profile_path, 'a') as pf:
                from datetime import datetime
                pf.write(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]\n")
                pf.write(report + '\n')

        return {k: v.item() for k, v in loss_dict.items()}

    def _train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:

        self.model_engine.train()


        batch = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}


        if self.model_engine.fp16_enabled():
            target_dtype = torch.float16
        elif self.model_engine.bfloat16_enabled():
            target_dtype = torch.bfloat16
        else:
            target_dtype = torch.float32


        for k, v in batch.items():
            if isinstance(v, torch.Tensor) and v.dtype in (torch.float32, torch.float64):
                batch[k] = v.to(target_dtype, non_blocking=True)


        if self.global_step == 0 and is_main_process():
            print("=" * 60)
            print("  [v2.4.0 DTYPE DIAGNOSTIC] First step dtype check:")
            print(f"   DeepSpeed fp16: {self.model_engine.fp16_enabled()}")
            print(f"   DeepSpeed bf16: {self.model_engine.bfloat16_enabled()}")
            print(f"   Input target_dtype: {target_dtype}")
            first_param = next(self.model_engine.parameters())
            print(f"   Model weight dtype: {first_param.dtype}")
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    print(f"   batch['{k}']: {v.dtype} {v.shape}")
            print(f"   WARN Gene loss: chunked fp32 CE (8 chunks)")
            print(f"   WARN Expr bin/level loss:    fp32 CE")
            print(f"   gradient_accumulation_steps: {self.gradient_accumulation_steps}")
            print(f"   -> masking step: {self._get_effective_masking_step()} (opt_step={self._optimizer_step}, offset={self._masking_step_offset})")
            print("=" * 60)


        if hasattr(self.model_engine.module, 'set_masking_step'):
            self.model_engine.module.set_masking_step(self._get_effective_masking_step())


        outputs = self.model_engine(
            pathway_embedding=batch['pathway_embedding'],
            pathway_scores=batch.get('pathway_scores', None),
            gene_tokens=batch['gene_tokens'],
            expr_bin_tokens=batch['expr_bin_tokens'],
            expr_level_tokens=batch['expr_level_tokens'],
            attention_mask=batch['attention_mask'],
            return_loss=False,
        )


        loss, loss_dict = self.loss_fn(outputs, batch)


        if torch.isnan(loss) or torch.isinf(loss):
            print_rank_0(f"WARN  NaN/Inf loss at step {self.global_step} (opt_step={self._optimizer_step}), skipping...")
            return {k: float('nan') for k in loss_dict.keys()}


        self.model_engine.backward(loss)


        prev_ds_global_steps = self.model_engine.global_steps


        self.model_engine.step()




        if self.global_step > 0 and (self.global_step + 1) % self.args.logging_steps == 0:
            try:
                grad_norm = self.model_engine.get_global_grad_norm()
                if grad_norm is None or not isinstance(grad_norm, (int, float)):
                    grad_norm = float(grad_norm) if grad_norm is not None else 0.0
                self._grad_norm_sum += grad_norm
                self._grad_norm_count += 1
                self._grad_norm_max = max(self._grad_norm_max, grad_norm)
            except Exception:
                grad_norm = -1.0



        if self.model_engine.global_steps > prev_ds_global_steps:
            self._optimizer_step += 1

        return {k: v.item() for k, v in loss_dict.items()}

    @torch.no_grad()
    def _validate(self, max_batches: int = 25) -> Dict[str, float]:

        if self.val_dataloader is None:
            return {}

        self.model_engine.eval()

        if hasattr(self.model_engine.module, 'set_masking_step'):
            self.model_engine.module.set_masking_step(self._get_effective_masking_step())
        total_losses = {}
        num_batches = 0


        if self.model_engine.fp16_enabled():
            target_dtype = torch.float16
        elif self.model_engine.bfloat16_enabled():
            target_dtype = torch.bfloat16
        else:
            target_dtype = torch.float32


        if self.args.test_mode:
            max_batches = 10

        for batch in self.val_dataloader:
            batch = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}


            for k, v in batch.items():
                if isinstance(v, torch.Tensor) and v.dtype in (torch.float32, torch.float64):
                    batch[k] = batch[k].to(target_dtype, non_blocking=True)

            outputs = self.model_engine(
                pathway_embedding=batch['pathway_embedding'],
                pathway_scores=batch.get('pathway_scores', None),
                gene_tokens=batch['gene_tokens'],
                expr_bin_tokens=batch['expr_bin_tokens'],
                expr_level_tokens=batch['expr_level_tokens'],
                attention_mask=batch['attention_mask'],
                return_loss=False,
            )

            _, loss_dict = self.loss_fn(outputs, batch)

            for k, v in loss_dict.items():
                total_losses[k] = total_losses.get(k, 0) + v.item()
            num_batches += 1


            if num_batches >= max_batches:
                break



        self.model_engine.train()




        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        if num_batches == 0:
            return {}
        avg_losses = {f'val_{k}': v / num_batches for k, v in total_losses.items()}
        return avg_losses

    @torch.no_grad()
    def _test(self) -> Dict[str, float]:

        if self.test_dataloader is None:
            return {}

        self.model_engine.eval()

        if hasattr(self.model_engine.module, 'set_masking_step'):
            self.model_engine.module.set_masking_step(self._get_effective_masking_step())
        total_losses = {}
        num_batches = 0


        if self.model_engine.fp16_enabled():
            target_dtype = torch.float16
        elif self.model_engine.bfloat16_enabled():
            target_dtype = torch.bfloat16
        else:
            target_dtype = torch.float32

        for batch in self.test_dataloader:
            batch = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}


            for k, v in batch.items():
                if isinstance(v, torch.Tensor) and v.dtype in (torch.float32, torch.float64):
                    batch[k] = batch[k].to(target_dtype, non_blocking=True)

            outputs = self.model_engine(
                pathway_embedding=batch['pathway_embedding'],
                pathway_scores=batch.get('pathway_scores', None),
                gene_tokens=batch['gene_tokens'],
                expr_bin_tokens=batch['expr_bin_tokens'],
                expr_level_tokens=batch['expr_level_tokens'],
                attention_mask=batch['attention_mask'],
                return_loss=False,
            )

            _, loss_dict = self.loss_fn(outputs, batch)

            for k, v in loss_dict.items():
                total_losses[k] = total_losses.get(k, 0) + v.item()
            num_batches += 1


        self.model_engine.train()

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        if num_batches == 0:
            return {}
        avg_losses = {f'test_{k}': v / num_batches for k, v in total_losses.items()}
        return avg_losses

    def _save_checkpoint(self, tag: str = None):

        tag = tag or f"step_{self.global_step}"
        output_root = Path(self.args.output_dir)
        save_dir = output_root / tag


        self.model_engine.save_checkpoint(
            save_dir=str(output_root),
            tag=tag,
            client_state={
                'global_step': self.global_step,
                'optimizer_step': self._optimizer_step,
                'masking_step_offset': self._masking_step_offset,
                'current_epoch': self.current_epoch,
                'epoch_step': self._epoch_step,
                'best_val_loss': self.best_val_loss,
            }
        )


        if is_main_process():
            universal_path = save_dir / "universal_model.pt"
            try:
                state_dict = self.model_engine.module.state_dict()
                torch.save({
                    'model_state_dict': state_dict,
                    'global_step': self.global_step,
                    'optimizer_step': self._optimizer_step,
                    'masking_step_offset': self._masking_step_offset,
                    'current_epoch': self.current_epoch,
                    'epoch_step': self._epoch_step,
                    'best_val_loss': self.best_val_loss,
                }, str(universal_path))
                print_rank_0(f"  Saved universal checkpoint to {universal_path}")
            except Exception as e:
                print_rank_0(f"WARN  Failed to save universal checkpoint: {e}")

        print_rank_0(f"  Saved checkpoint to {save_dir} (global_step={self.global_step}, opt_step={self._optimizer_step})")


        self._cleanup_checkpoints()

    def _load_checkpoint(self, path: str, resume_mode: str = 'full'):

        resume_mode = (resume_mode or 'full').lower()
        if resume_mode not in ('full', 'weights_only'):
            raise ValueError(f"Unknown resume_mode: {resume_mode}")

        print_rank_0(f" Loading checkpoint from {path}")
        print_rank_0(f"     Resume mode: {resume_mode}")

        checkpoint_path = Path(path)



        if checkpoint_path.is_file():
            print_rank_0(f"   INFO  resume path is a file, using parent directory: {checkpoint_path.parent}")
            checkpoint_path = checkpoint_path.parent

        should_reset_data_cursor = self._should_reset_data_cursor_on_resume(checkpoint_path, resume_mode)

        path = str(checkpoint_path)
        universal_path = checkpoint_path / "universal_model.pt"


        if resume_mode == 'weights_only':
            if not universal_path.exists():
                raise FileNotFoundError(
                    f"weights_only mode requires {universal_path}, but file is missing."
                )

            checkpoint = torch.load(str(universal_path), map_location='cpu')
            state_dict = checkpoint['model_state_dict']


            keys_to_remove = [k for k in state_dict if 'rope.' in k and '_cached' in k]
            for k in keys_to_remove:
                del state_dict[k]

            self.model_engine.module.load_state_dict(state_dict, strict=False)


            self.global_step = 0
            self._optimizer_step = 0
            self._masking_step_offset = 0
            self.current_epoch = 0
            self._epoch_step = 0
            self.best_val_loss = float('inf')

            if hasattr(self.train_dataloader, 'reset_resume'):
                self.train_dataloader.reset_resume()

            print_rank_0("OK Loaded model weights only from universal checkpoint")
            print_rank_0("     Counters reset: epoch=0, global_step=0, opt_step=0, epoch_step=0")
            print_rank_0("   INFO  Optimizer/LR state is intentionally NOT restored")
            return


        if is_main_process():
            print(f"     Checkpoint directory contents:")
            if checkpoint_path.exists():
                for item in sorted(checkpoint_path.iterdir()):
                    if item.is_dir():
                        sub_items = list(item.iterdir())
                        print(f"      {item.name}/ ({len(sub_items)} files)")
                    else:
                        size_mb = item.stat().st_size / (1024 * 1024)
                        print(f"      {item.name} ({size_mb:.1f} MB)")
            else:
                print(f"      ERR Directory does not exist!")


        native_attempts = []
        _seen_native_attempts = set()

        def _add_native_attempt(load_dir: Path, tag: Optional[str], desc: str):
            key = (str(load_dir), tag)
            if key in _seen_native_attempts:
                return
            _seen_native_attempts.add(key)
            native_attempts.append((load_dir, tag, desc))


        _add_native_attempt(checkpoint_path, None, 'direct-latest')


        step_tag = checkpoint_path.name
        if step_tag:
            _add_native_attempt(checkpoint_path.parent, step_tag, 'parent+tag')

            _add_native_attempt(checkpoint_path, step_tag, 'dir+tag')

        native_errors = []
        for load_dir, tag, desc in native_attempts:
            try:
                print_rank_0(
                    f"     Native load attempt [{desc}]: load_dir={load_dir}, tag={tag}"
                )
                load_path, client_state = self.model_engine.load_checkpoint(
                    load_dir=str(load_dir),
                    tag=tag,
                )



                if not load_path:
                    raise RuntimeError("load_checkpoint returned empty load_path (checkpoint not actually restored)")

                if client_state:
                    self.global_step = client_state.get('global_step', 0)
                    self._optimizer_step = client_state.get('optimizer_step', 0)
                    self._masking_step_offset = int(client_state.get('masking_step_offset', 0) or 0)
                    self.current_epoch = client_state.get('current_epoch', 0)
                    self._epoch_step = client_state.get('epoch_step', 0)
                    self.best_val_loss = client_state.get('best_val_loss', float('inf'))

                    if self._optimizer_step == 0 and self.global_step > 0:
                        self._optimizer_step = self.global_step // self.gradient_accumulation_steps
                        print_rank_0(f"   WARN  Old checkpoint without optimizer_step, estimated: {self._optimizer_step}")
                else:


                    loaded_dir = Path(load_path)
                    meta_path = loaded_dir / "universal_model.pt"
                    restored_meta = False
                    if meta_path.exists():
                        try:
                            meta = torch.load(str(meta_path), map_location='cpu')
                            self.global_step = int(meta.get('global_step', 0))
                            self._optimizer_step = int(meta.get('optimizer_step', 0))
                            self._masking_step_offset = int(meta.get('masking_step_offset', 0) or 0)
                            self.current_epoch = int(meta.get('current_epoch', 0))
                            self._epoch_step = int(meta.get('epoch_step', 0))
                            self.best_val_loss = float(meta.get('best_val_loss', float('inf')))
                            if self._optimizer_step == 0 and self.global_step > 0:
                                self._optimizer_step = self.global_step // self.gradient_accumulation_steps
                            restored_meta = True
                            print_rank_0(
                                f"   INFO  Native client_state empty; restored counters from {meta_path.name}"
                            )
                        except Exception as meta_e:
                            print_rank_0(f"   WARN  Failed to restore counters from {meta_path}: {meta_e}")


                    if not restored_meta:
                        loaded_name = loaded_dir.name
                        if loaded_name.startswith('step_'):
                            try:
                                est_step = int(''.join(ch for ch in loaded_name if ch.isdigit()))
                                if est_step > 0:
                                    self.global_step = est_step
                                    if self._optimizer_step == 0:
                                        self._optimizer_step = est_step // self.gradient_accumulation_steps
                                    print_rank_0(
                                        f"   INFO  Native client_state empty; estimated step from tag: {est_step}"
                                    )
                            except Exception:
                                pass


                try:
                    current_lr = self.model_engine.get_lr()
                    print_rank_0(f"     LR after native load: {current_lr}")
                except Exception:
                    print_rank_0(f"     LR after native load: (unable to read)")


                if is_main_process():
                    sample_params = {}
                    for name, param in list(self.model_engine.module.named_parameters())[:3]:
                        sample_params[name] = (param.data.float().mean().item(), param.data.float().std().item())
                    print(f"     Weight sanity check (first 3 params):")
                    for name, (mean, std) in sample_params.items():
                        print(f"      {name}: mean={mean:.6f}, std={std:.6f}")

                print_rank_0(
                    f"OK DeepSpeed NATIVE load succeeded via [{desc}] (load_path={load_path}): epoch {self.current_epoch}, "
                    f"global_step {self.global_step}, opt_step {self._optimizer_step}, mask_offset {self._masking_step_offset}, "
                    f"epoch_step {self._epoch_step}"
                )
                if should_reset_data_cursor:
                    self._reset_data_cursor_after_resume(reason=checkpoint_path.name)
                return

            except Exception as e:
                native_errors.append((desc, str(load_dir), tag, str(e)))

        if native_errors:
            first_desc, first_dir, first_tag, first_err = native_errors[0]
            print_rank_0(
                f"WARN  DeepSpeed native load failed: [{first_desc}] load_dir={first_dir}, "
                f"tag={first_tag}, error={first_err}"
            )
            if len(native_errors) > 1:
                print_rank_0(f"   INFO  Other native attempts also failed: {len(native_errors) - 1}")
        print_rank_0(f"   Trying universal checkpoint...")


        if universal_path.exists():
            checkpoint = torch.load(str(universal_path), map_location='cpu')
            state_dict = checkpoint['model_state_dict']


            model_keys = set(self.model_engine.module.state_dict().keys())
            ckpt_keys = set(state_dict.keys())


            keys_to_remove = [k for k in state_dict if 'rope.' in k and '_cached' in k]
            for k in keys_to_remove:
                del state_dict[k]
                ckpt_keys.discard(k)
                print_rank_0(f"    Skipped non-weight buffer: {k}")


            matched_keys = model_keys & ckpt_keys
            missing_in_ckpt = model_keys - ckpt_keys
            unexpected_keys = ckpt_keys - model_keys
            print_rank_0(f"     Key matching stats:")
            print_rank_0(f"      Model keys:    {len(model_keys)}")
            print_rank_0(f"      Ckpt keys:     {len(ckpt_keys)}")
            print_rank_0(f"      OK Matched:     {len(matched_keys)}")
            print_rank_0(f"      ERR Missing (model has, ckpt lacks): {len(missing_in_ckpt)}")
            print_rank_0(f"        Unexpected (ckpt has, model lacks): {len(unexpected_keys)}")

            if missing_in_ckpt:
                for k in sorted(missing_in_ckpt)[:20]:
                    print_rank_0(f"         MISSING: {k}")
                if len(missing_in_ckpt) > 20:
                    print_rank_0(f"         ... and {len(missing_in_ckpt) - 20} more")

            if unexpected_keys:
                for k in sorted(unexpected_keys)[:20]:
                    print_rank_0(f"         UNEXPECTED: {k}")
                if len(unexpected_keys) > 20:
                    print_rank_0(f"         ... and {len(unexpected_keys) - 20} more")


            if len(missing_in_ckpt) > 10:
                print_rank_0(f"     WARNING: {len(missing_in_ckpt)} model keys not found in checkpoint!")
                print_rank_0(f"     These layers will keep RANDOM initialization -> loss spike!")

            self.model_engine.module.load_state_dict(state_dict, strict=False)


            if is_main_process():
                sample_params = {}
                for name, param in list(self.model_engine.module.named_parameters())[:3]:
                    sample_params[name] = (param.data.float().mean().item(), param.data.float().std().item())
                print(f"     Weight sanity check (first 3 params):")
                for name, (mean, std) in sample_params.items():
                    print(f"      {name}: mean={mean:.6f}, std={std:.6f}")

            self.global_step = checkpoint.get('global_step', 0)
            self._optimizer_step = checkpoint.get('optimizer_step', 0)
            self._masking_step_offset = int(checkpoint.get('masking_step_offset', 0) or 0)
            self.current_epoch = checkpoint.get('current_epoch', 0)
            self._epoch_step = checkpoint.get('epoch_step', 0)
            self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))

            if self._optimizer_step == 0 and self.global_step > 0:
                self._optimizer_step = self.global_step // self.gradient_accumulation_steps

            if should_reset_data_cursor:
                self._reset_data_cursor_after_resume(reason=checkpoint_path.name)

            print_rank_0(f"OK Resumed from universal checkpoint: epoch {self.current_epoch}, "
                         f"global_step {self.global_step}, opt_step {self._optimizer_step}, mask_offset {self._masking_step_offset}")
            print_rank_0(f"   WARN  Optimizer/LR state reset (cross-scale resume)")
        else:
            raise FileNotFoundError(
                f"Cannot load checkpoint from {path}. "
                f"Neither DeepSpeed checkpoint nor universal_model.pt found."
            )

    def _cleanup_checkpoints(self):

        if not is_main_process():
            return


        if self.args.save_total_limit <= 0:
            return

        output_dir = Path(self.args.output_dir)
        if not output_dir.exists():
            return




        checkpoint_dirs = []
        for d in output_dir.iterdir():
            if not d.name.startswith('step_'):
                continue
            if not d.is_dir() or d.is_symlink():
                continue
            checkpoint_dirs.append(d)

        checkpoints = sorted(checkpoint_dirs, key=lambda x: x.stat().st_mtime)

        while len(checkpoints) > self.args.save_total_limit:
            oldest = checkpoints.pop(0)
            import shutil
            try:
                shutil.rmtree(oldest)
                print_rank_0(f"    Removed old checkpoint: {oldest}")
            except Exception as e:
                print_rank_0(f"WARN  Failed to remove old checkpoint {oldest}: {e}")

    def _soft_reset(self):

        import ctypes

        t0 = time.time()


        for _ in range(3):
            gc.collect()





        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        try:
            torch._C._cuda_clearCublasWorkspaces()
        except AttributeError:
            pass



        try:
            libc = ctypes.CDLL('libc.so.6')
            libc.malloc_trim(0)
        except (OSError, AttributeError):
            pass


        import re
        import linecache
        import sys
        re.purge()
        linecache.clearcache()
        if hasattr(sys, '_clear_type_cache'):
            sys._clear_type_cache()




        _ds_reset_info = {}
        try:

            if hasattr(self.model_engine, 'timers') and hasattr(self.model_engine.timers, 'timers'):
                for tname, timer in self.model_engine.timers.timers.items():
                    for attr in ('elapsed', 'elapsed_records'):
                        lst = getattr(timer, attr, None)
                        if isinstance(lst, list) and len(lst) > 0:
                            _ds_reset_info[f'timer.{tname}.{attr}'] = len(lst)
                            lst.clear()

            opt = self.model_engine.optimizer
            for attr_name in dir(opt):
                if attr_name.startswith('__'):
                    continue
                try:
                    val = getattr(opt, attr_name)
                    if isinstance(val, list) and len(val) > 100:
                        _ds_reset_info[f'opt.{attr_name}'] = len(val)
                except Exception:
                    continue
        except Exception:
            pass


        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        elapsed_ms = (time.time() - t0) * 1000
        _reset_detail = ', '.join(f'{k}={v}' for k, v in _ds_reset_info.items()) if _ds_reset_info else 'none'
        print_rank_0(f"  Soft reset completed in {elapsed_ms:.0f}ms "
                     f"(gcx3 + CUDA sync + cuBLAS clear + malloc_trim + "
                     f"python caches | DS state: {_reset_detail})")

    def _log_metrics(self, metrics: Dict[str, float], prefix: str = ""):

        if not is_main_process():
            return


        scalar_metrics: Dict[str, float] = {}
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                if v.numel() != 1:
                    continue
                scalar_metrics[k] = float(v.detach().item())
            elif isinstance(v, (np.floating, np.integer)):
                scalar_metrics[k] = float(v)
            elif isinstance(v, (float, int)):
                scalar_metrics[k] = float(v)

        log_str = f"Step {self.global_step}"
        for k, v in scalar_metrics.items():
            if 'lr' in k.lower():
                log_str += f" | {prefix}{k}: {v:.2e}"
            else:
                log_str += f" | {prefix}{k}: {v:.4f}"
        print(log_str)

        if self.metrics_jsonl_path is not None:
            stage = prefix[:-1] if prefix.endswith('_') else (prefix if prefix else 'metrics')
            record = {
                'timestamp_utc': datetime.now(timezone.utc).isoformat(),
                'global_step': int(self.global_step),
                'optimizer_step': int(self._optimizer_step),
                'stage': stage,
            }
            for k, v in scalar_metrics.items():
                out_key = f"{prefix}{k}" if prefix else k
                record[out_key] = v
            try:
                with open(self.metrics_jsonl_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
            except Exception as e:
                print_rank_0(f"WARN Failed to write metrics JSONL: {e}")

    def train(self):

        print_rank_0(f"\n{'='*60}")
        print_rank_0(f" Starting DeepSpeed training")
        print_rank_0(f"   Epochs: {self.args.max_epochs}")
        print_rank_0(f"   Max steps: {self.args.max_steps}")
        print_rank_0(f"   Device: {self.device}")
        print_rank_0(f"   World size: {get_world_size()}")
        print_rank_0(f"{'='*60}\n")


        if self.global_step > 0 and self.val_dataloader is not None:
            print_rank_0(f"  [Resume Sanity Check] Validating loaded model (expecting loss ~ pre-save level)...")
            try:
                sanity_metrics = self._validate(max_batches=20)
                sanity_loss = sanity_metrics.get('val_total_loss', float('nan'))
                print_rank_0(f"   Sanity check loss: {sanity_loss:.4f}")
                if sanity_loss > 8.0:
                    print_rank_0(f"     CRITICAL: Loss {sanity_loss:.4f} >> expected ~4.5!")
                    print_rank_0(f"     Model weights likely NOT loaded correctly!")
                    print_rank_0(f"     Check logs above for key mismatch or native load failure.")
                elif sanity_loss > 6.0:
                    print_rank_0(f"   WARN  WARNING: Loss {sanity_loss:.4f} higher than expected, partial weight load?")
                else:
                    print_rank_0(f"   OK Loss looks reasonable, checkpoint loaded correctly.")
            except Exception as e:
                print_rank_0(f"   WARN  Sanity check failed: {e}")

        start_time = time.time()
        total_steps = self.args.max_steps if self.args.max_steps > 0 else (
            len(self.train_dataloader) * self.args.max_epochs
        )


        if (self.args.test_mode):
            total_steps = min(100, total_steps)
            print_rank_0(f" Test mode: running {total_steps} steps only")

        from tqdm import tqdm





        _WORKER_RESTART_INTERVAL = int(os.environ.get(
            'WORKER_RESTART_INTERVAL', str(self.args.eval_steps)))




        _GC_INTERVAL = int(os.environ.get('GC_INTERVAL', '0'))


        _SOFT_RESET_AFTER_VAL = int(os.environ.get('SOFT_RESET_AFTER_VAL', '1'))

        for epoch in range(self.current_epoch, self.args.max_epochs):
            self.current_epoch = epoch
            epoch_losses = {}
            num_batches = 0
            nan_count = 0


            if hasattr(self.train_dataloader, 'set_epoch'):
                self.train_dataloader.set_epoch(epoch)
            elif hasattr(self.train_dataloader, 'sampler') and hasattr(self.train_dataloader.sampler, 'set_epoch'):
                self.train_dataloader.sampler.set_epoch(epoch)


            if epoch == self.current_epoch and self._epoch_step > 0:
                _initial_skip = self._epoch_step
            else:
                _initial_skip = 0

            _epoch_finished = False
            while not _epoch_finished:
                skip_batches = self._epoch_step if self._epoch_step > 0 else _initial_skip


                if skip_batches > 0 and hasattr(self.train_dataloader, 'set_resume_step'):
                    self.train_dataloader.set_resume_step(skip_batches)
                    if skip_batches == _initial_skip:
                        print_rank_0(f"  Resuming epoch {epoch}: skipping first {skip_batches:,} batches "
                                     f"(sampler-level, zero data loading overhead)")


                remaining_len = len(self.train_dataloader)
                full_len = remaining_len + skip_batches

                pbar = tqdm(
                    enumerate(self.train_dataloader),
                    total=full_len,
                    desc=f"Epoch {epoch}",
                    disable=not is_main_process(),
                    ncols=120,
                    initial=skip_batches,
                )

                _need_worker_restart = False
                _iter_t0 = time.time()
                for step, batch in pbar:
                    _data_load_ms = (time.time() - _iter_t0) * 1000


                    self._epoch_step = step + skip_batches


                    if self.args.max_steps > 0 and self.global_step >= self.args.max_steps:
                        break


                    if self._stop_requested:
                        print_rank_0(f"\n    Stop signal received at step {self.global_step} (opt_step={self._optimizer_step})")
                        print_rank_0(f"   Saving checkpoint before exit...")
                        self._save_checkpoint(f'interrupted_step_{self.global_step}')
                        print_rank_0(f"   OK Checkpoint saved. Exiting gracefully.")
                        return self.best_val_loss


                    _is_profile_step = (self.args.eval_steps > 10 and
                                        self.global_step > 0 and
                                        (self.global_step + 10) % self.args.eval_steps == 0)
                    if _is_profile_step:
                        self._profile_data_load_ms = _data_load_ms
                        losses = self._profile_train_step(batch)
                    else:
                        losses = self._train_step(batch)


                    has_nan = any(math.isnan(v) if isinstance(v, float) else False for v in losses.values())
                    if has_nan:
                        nan_count += 1
                        continue


                    for k, v in losses.items():
                        if not math.isnan(v):
                            epoch_losses[k] = epoch_losses.get(k, 0) + v
                    num_batches += 1
                    self.global_step += 1


                    loss_val = losses.get('total_loss', 0)

                    lr = 0.0
                    try:
                        lr = self.model_engine.get_lr()[0]
                    except (AttributeError, IndexError, TypeError):

                        try:
                            lr = self.lr_scheduler.get_last_lr()[0]
                        except Exception:
                            lr = 0.0
                    pbar.set_postfix({
                        'step': self.global_step,
                        'opt_step': self._optimizer_step,
                        'loss': f'{loss_val:.4f}',
                        'lr': f'{lr:.2e}',
                    })


                    if self.global_step % self.args.logging_steps == 0:
                        losses['lr'] = lr
                        losses['optimizer_step'] = float(self._optimizer_step)
                        self._log_metrics(losses, prefix='train_')

                        if (is_main_process() and
                            hasattr(self.model_engine.module, '_get_current_n_visible') and
                            getattr(self.model_engine.module.config, 'pathway_masking_enabled', False)):
                            _nv = self.model_engine.module._get_current_n_visible()
                            _phase = self.model_engine.module._get_current_phase()
                            _N = self.model_engine.module.config.n_pathways
                            print(f"     curriculum: Phase {_phase}, n_visible={_nv}/{_N} ({_nv/_N*100:.1f}%)")

                        if is_main_process() and self._grad_norm_count > 0:
                            avg_grad_norm = self._grad_norm_sum / self._grad_norm_count
                            print(f"     grad_norm: avg={avg_grad_norm:.4f}, max={self._grad_norm_max:.4f}, "
                                  f"clip_threshold=1.0, {'WARN CLIPPING LIKELY' if self._grad_norm_max > 1.0 else 'OK OK'}")

                            self._grad_norm_sum = 0.0
                            self._grad_norm_count = 0
                            self._grad_norm_max = 0.0


                    if self.global_step % self.args.eval_steps == 0:
                        val_metrics = self._validate()
                        if val_metrics:
                            self._log_metrics(val_metrics)





                        local_val_loss = val_metrics.get('val_total_loss', float('inf')) if val_metrics else float('inf')
                        if torch.distributed.is_initialized():
                            vl_tensor = torch.tensor([local_val_loss], device=self.device, dtype=torch.float64)
                            torch.distributed.all_reduce(vl_tensor)
                            val_loss = vl_tensor.item() / torch.distributed.get_world_size()
                        else:
                            val_loss = local_val_loss

                        if val_loss < self.best_val_loss:
                            self.best_val_loss = val_loss
                            self._save_checkpoint('best_model')


                        if _SOFT_RESET_AFTER_VAL:
                            self._soft_reset()


                    if self.global_step % self.args.save_steps == 0:
                        self._save_checkpoint()


                    elif (_GC_INTERVAL > 0 and
                          self.global_step > 0 and
                          self.global_step % _GC_INTERVAL == 0):
                        gc.collect()


                    if (_WORKER_RESTART_INTERVAL > 0 and
                            self.global_step > 0 and
                            self.global_step % _WORKER_RESTART_INTERVAL == 0):
                        self._epoch_step += 1
                        _need_worker_restart = True
                        break

                    _iter_t0 = time.time()

                pbar.close()

                if _need_worker_restart:
                    print_rank_0(f"  Worker restart at step {self.global_step}: "
                                 f"recycling DataLoader workers to free host memory...")
                    if hasattr(self.train_dataloader, 'restart_workers'):
                        self.train_dataloader.restart_workers()
                    gc.collect()

                    continue


                _epoch_finished = True


            self._epoch_step = 0
            if hasattr(self.train_dataloader, 'reset_resume'):
                self.train_dataloader.reset_resume()


            if num_batches > 0:
                avg_losses = {k: v / num_batches for k, v in epoch_losses.items()}
                print_rank_0(f"\n  Epoch {epoch} completed")
                print_rank_0(f"   Valid batches: {num_batches}, NaN batches: {nan_count}")
                self._log_metrics(avg_losses, prefix='epoch_avg_')


            if self.args.max_steps > 0 and self.global_step >= self.args.max_steps:
                break


        total_time = time.time() - start_time
        print_rank_0(f"\n Training completed in {total_time/3600:.2f} hours")
        self._save_checkpoint('final_model')

        return self.best_val_loss






def parse_args() -> TrainingArgs:

    parser = argparse.ArgumentParser(description='Train PCGED with DeepSpeed')


    parser.add_argument('--config', type=str, default='configs/train_config.yaml',
                        help='Path to training config file')



    parser.add_argument('--pipeline_parallel_size', type=int, default=1,
                        help='Pipeline parallel size (1 = no pipeline)')
    parser.add_argument('--fixed_seq_length', type=int, default=-1,
                        help='Fixed sequence length for pipeline parallel (-1 = use config)')


    parser.add_argument('--max_steps', type=int, default=-1,
                        help='Max training steps (-1 = use epochs)')
    parser.add_argument('--max_epochs', type=int, default=5,
                        help='Max training epochs')


    parser.add_argument('--output_dir', type=str, default='./outputs/deepspeed',
                        help='Output directory')
    parser.add_argument('--logging_dir', type=str, default='',
                        help='Logging directory (default: output_dir/logs)')
    parser.add_argument('--save_steps', type=int, default=1000,
                        help='Save checkpoint every N steps')
    parser.add_argument('--save_total_limit', type=int, default=5,
                        help='Max checkpoints to keep')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--resume_mode', type=str, default=os.environ.get('RESUME_MODE', 'full'), choices=['full', 'weights_only'],
                        help='Restore mode: full (resume states) or weights_only (load model weights and reset counters)')


    parser.add_argument('--logging_steps', type=int, default=100,
                        help='Log every N steps')
    parser.add_argument('--eval_steps', type=int, default=500,
                        help='Evaluate every N steps')


    parser.add_argument('--test_mode', action='store_true',
                        help='Test mode (run only 100 steps)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')


    parser.add_argument('--local_rank', type=int, default=-1,
                        help='Local rank (set by DeepSpeed)')


    parser = deepspeed.add_config_arguments(parser) if DEEPSPEED_AVAILABLE else parser

    args = parser.parse_args()


    ds_config = getattr(args, 'deepspeed_config', None) or getattr(args, 'deepspeed', None)
    if ds_config is None:
        ds_config = 'configs/ds_config_zero2.json'

    return TrainingArgs(
        config=args.config,
        deepspeed_config=ds_config,
        pipeline_parallel_size=args.pipeline_parallel_size,
        fixed_seq_length=args.fixed_seq_length,
        max_steps=args.max_steps,
        max_epochs=args.max_epochs,
        output_dir=args.output_dir,
        logging_dir=args.logging_dir,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        resume_from_checkpoint=args.resume_from_checkpoint,
        resume_mode=getattr(args, 'resume_mode', os.environ.get('RESUME_MODE', 'full')),
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        test_mode=args.test_mode,
        seed=args.seed,
        local_rank=args.local_rank,
    )


def main():

    if not DEEPSPEED_AVAILABLE:
        print("ERR DeepSpeed is required. Install with: pip install deepspeed")
        sys.exit(1)


    args = parse_args()


    set_seed(args.seed)


    config = load_yaml_config(args.config)
    ds_config = load_json_config(args.deepspeed_config)

    config['seed'] = args.seed


    device_type_str = {
        'cuda': '  NVIDIA GPU (CUDA)',
        'rocm': '  AMD GPU (ROCm)',
        'dcu': '  DCU (HIP)',
        'cpu': '  CPU',
    }.get(ACCELERATOR_TYPE, f'Unknown ({ACCELERATOR_TYPE})')

    print_rank_0("=" * 80)
    print_rank_0(" PCGED DeepSpeed Training")
    print_rank_0("=" * 80)
    print_rank_0(f"    Accelerator: {device_type_str}")
    if torch.cuda.is_available():
        print_rank_0(f"    Device: {torch.cuda.get_device_name(0)}")
    print_rank_0(f"  Config: {args.config}")
    print_rank_0(f"  DeepSpeed config: {args.deepspeed_config}")
    print_rank_0(f"  Output dir: {args.output_dir}")
    if args.logging_dir:
        print_rank_0(f"  Logging dir: {args.logging_dir}")
    print_rank_0(f"  Resume mode: {args.resume_mode}")
    print_rank_0(f"  Masking reset on resume: {os.environ.get('MASKING_RESET_ON_RESUME', '0')}")
    print_rank_0(f"  Seed: {args.seed}")
    print_rank_0(f"    Local rank: {args.local_rank}")
    print_rank_0(f"  World size: {get_world_size()}")
    if args.pipeline_parallel_size > 1:
        print_rank_0(f"  Pipeline parallel size: {args.pipeline_parallel_size}")
        if args.fixed_seq_length > 0:
            print_rank_0(f"  Fixed sequence length: {args.fixed_seq_length}")
    print_rank_0("=" * 80)


    print_rank_0("\n  Creating datasets...")
    train_dataset, val_dataset, test_dataset, n_pathways = create_datasets(config)


    if config['model'].get('n_pathways', 2000) != n_pathways:
        print_rank_0(f"WARN  Updating n_pathways from {config['model'].get('n_pathways', 2000)} to {n_pathways}")
        config['model']['n_pathways'] = n_pathways


    data_config = config.get('data', {})
    batch_size = data_config.get('batch_size', 16)


    if 'train_micro_batch_size_per_gpu' in ds_config:
        batch_size = ds_config['train_micro_batch_size_per_gpu']


    fixed_seq_length = None
    if args.pipeline_parallel_size > 1:

        if args.fixed_seq_length > 0:
            fixed_seq_length = args.fixed_seq_length
        else:
            fixed_seq_length = data_config.get('max_len', 2048)
        print_rank_0(f"   Pipeline mode: fixed sequence length = {fixed_seq_length}")

    dataloaders = create_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=batch_size,
        num_workers=data_config.get('num_workers', 4),
        distributed=get_world_size() > 1,
        world_size=get_world_size(),
        rank=get_rank(),
        fixed_seq_length=fixed_seq_length,
    )

    train_dataloader = dataloaders['train']
    val_dataloader = dataloaders.get('val')
    test_dataloader = dataloaders.get('test')

    print_rank_0(f"   Train batches: {len(train_dataloader):,}")
    if val_dataloader:
        print_rank_0(f"   Val batches: {len(val_dataloader):,}")
    if test_dataloader:
        print_rank_0(f"   Test batches: {len(test_dataloader):,}")


    print_rank_0("\n  Creating model...")


    from src.model.model_pathway_aux import __MODEL_VERSION__
    print_rank_0(f"       Model code version: {__MODEL_VERSION__}")

    model = create_model_from_config(config, n_pathways)

    n_params = sum(p.numel() for p in model.parameters())
    print_rank_0(f"   Total parameters: {n_params:,}")


    loss_config = config.get('training', {}).get('loss_weights', {})
    loss_fn = PCGEDLoss(
        gene_loss_weight=loss_config.get('gene_loss', loss_config.get('gene', 1.0)),
        expr_bin_loss_weight=loss_config.get('expr_bin_loss', loss_config.get('expr_bin', 0.5)),
        expr_level_loss_weight=loss_config.get('expr_level_loss', loss_config.get('expr_level', 0.3)),
        pathway_loss_weight=loss_config.get('pathway_loss', loss_config.get('pathway', 0.0)),
        pathway_loss_type=loss_config.get('pathway_loss_type', 'smooth_l1'),
        pathway_loss_use_sigmoid=loss_config.get('pathway_loss_use_sigmoid', True),
        pathway_num_bins=config.get('data', {}).get('pathway_num_bins', 5),
        eos_loss_weight=loss_config.get('eos_loss', 0.0),
        eos_pos_weight=loss_config.get('eos_pos_weight', 16.0),
        eos_token_id=config.get('model', {}).get('eos_token_id', 2),
    )


    print_rank_0("\n Creating DeepSpeed trainer...")
    trainer = DeepSpeedTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        args=args,
        ds_config=ds_config,
        loss_fn=loss_fn,
    )


    print_rank_0("\n" + "=" * 80)
    print_rank_0(" Starting training...")
    print_rank_0("=" * 80 + "\n")


    def _sigterm_handler(signum, frame):
        print_rank_0(f"\n  Received SIGTERM (signal {signum}), requesting graceful shutdown...")
        trainer._stop_requested = True

    signal.signal(signal.SIGTERM, _sigterm_handler)
    signal.signal(signal.SIGUSR1, _sigterm_handler)

    try:
        best_val_loss = trainer.train()
        print_rank_0(f"\nOK Training completed! Best val loss: {best_val_loss:.4f}")


        if test_dataloader is not None:
            print_rank_0("\n" + "=" * 60)
            print_rank_0(" Running final test evaluation...")
            print_rank_0("=" * 60)
            test_metrics = trainer._test()
            if test_metrics:
                trainer._log_metrics(test_metrics)
                print_rank_0(f"OK Test evaluation completed!")
    except KeyboardInterrupt:
        print_rank_0("\nWARN Training interrupted by user")
        trainer._save_checkpoint('interrupted')
    except Exception as e:
        print_rank_0(f"\nERR Training failed with error: {e}")
        raise


if __name__ == '__main__':
    main()


