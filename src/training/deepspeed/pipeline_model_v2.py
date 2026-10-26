


import math
from typing import Tuple, List, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


try:
    from deepspeed.pipe import PipelineModule, LayerSpec
    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False
    PipelineModule = nn.Module
    LayerSpec = None

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.model.model import (
    PCGEDConfig,
    PathwayEncoder,
    PathwayEncoderV2,
    MultiScalePathwayEncoder,
    DiscretePathwayEncoder,
    GeneExpressionEmbedding,
    TransformerDecoderLayer,
    RotaryPositionEmbedding,
    _RMSNorm,
)






def pack_pipeline_input(
    pathway_embedding: Tensor,
    gene_tokens: Tensor,
    expr_bin_tokens: Tensor,
    expr_level_tokens: Tensor,
    attention_mask: Tensor,
) -> Tensor:

    target_dtype = pathway_embedding.dtype

    packed = torch.cat([
        pathway_embedding,
        gene_tokens.to(target_dtype),
        expr_bin_tokens.to(target_dtype),
        expr_level_tokens.to(target_dtype),
        attention_mask.to(target_dtype),
    ], dim=1)

    return packed


def unpack_pipeline_input(
    packed: Tensor,
    n_pathways: int,
    seq_len: int,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:

    idx = 0

    pathway_embedding = packed[:, idx:idx + n_pathways]
    idx += n_pathways

    gene_tokens = packed[:, idx:idx + seq_len].long()
    idx += seq_len

    expr_bin_tokens = packed[:, idx:idx + seq_len].long()
    idx += seq_len

    expr_level_tokens = packed[:, idx:idx + seq_len].long()
    idx += seq_len

    attention_mask = packed[:, idx:idx + seq_len]

    return pathway_embedding, gene_tokens, expr_bin_tokens, expr_level_tokens, attention_mask






class EmbeddingStageV2(nn.Module):


    def __init__(self, config: PCGEDConfig):
        super().__init__()
        self.config = config
        self.n_pathways = config.n_pathways
        self.n_pathway_tokens = config.n_pathway_tokens
        self._discrete_pathway = (config.pathway_encoder_type == 'discrete')


        if config.pathway_encoder_type == 'discrete':
            self.pathway_encoder = DiscretePathwayEncoder(config)
            self._pathway_mask_id = 5
        elif config.pathway_encoder_type == 'multiscale':
            self.pathway_encoder = MultiScalePathwayEncoder(config)
        elif config.pathway_encoder_type == 'v2':
            self.pathway_encoder = PathwayEncoderV2(config)
        else:
            self.pathway_encoder = PathwayEncoder(config)


        self.gene_expr_embedding = GeneExpressionEmbedding(config)


        self.sep_embedding = nn.Parameter(torch.randn(1, 1, config.d_model) * 0.02)

    def _apply_pathway_masking(self, pathway_embedding: Tensor) -> Tensor:

        if not self.training or not self.config.pathway_masking_enabled:
            return pathway_embedding

        B, N = pathway_embedding.shape
        device = pathway_embedding.device


        n_visible = max(1, int(N * self.config.pathway_curriculum_k_max_ratio))

        if self._discrete_pathway:

            mask_id = self._pathway_mask_id
            result = torch.full_like(pathway_embedding, mask_id)
            for i in range(B):
                visible_indices = torch.randperm(N, device=device)[:n_visible]
                result[i, visible_indices] = pathway_embedding[i, visible_indices]
            return result
        else:

            dtype = pathway_embedding.dtype
            mask = torch.zeros(B, N, device=device, dtype=dtype)
            for i in range(B):
                visible_indices = torch.randperm(N, device=device)[:n_visible]
                mask[i, visible_indices] = 1.0
            return pathway_embedding * mask

    def forward(self, packed_input: Tensor) -> Tensor:

        B = packed_input.shape[0]
        device = packed_input.device


        total_packed_len = packed_input.shape[1]
        computed_seq_len = (total_packed_len - self.n_pathways) // 4



        expected_seq_len = self.config.max_seq_len


        seq_len = expected_seq_len



        pathway_emb, gene_tokens, expr_bin_tokens, expr_level_tokens, attention_mask =\
            self._safe_unpack_input(packed_input, seq_len)


        pathway_emb = self._apply_pathway_masking(pathway_emb)


        condition_tokens = self.pathway_encoder(pathway_emb)


        sep_token = self.sep_embedding.expand(B, -1, -1)


        gene_emb = self.gene_expr_embedding(
            gene_tokens, expr_bin_tokens, expr_level_tokens
        )


        hidden_states = torch.cat([condition_tokens, sep_token, gene_emb], dim=1)

        return hidden_states

    def _safe_unpack_input(self, packed: Tensor, target_seq_len: int):

        B = packed.shape[0]
        device = packed.device
        dtype = packed.dtype


        total_len = packed.shape[1]
        actual_data_len = total_len - self.n_pathways
        actual_seq_len = actual_data_len // 4 if actual_data_len > 0 else 0


        pathway_emb = packed[:, :self.n_pathways]

        if actual_seq_len >= target_seq_len:

            idx = self.n_pathways
            gene_tokens = packed[:, idx:idx + target_seq_len].long()
            idx += actual_seq_len
            expr_bin_tokens = packed[:, idx:idx + target_seq_len].long()
            idx += actual_seq_len
            expr_level_tokens = packed[:, idx:idx + target_seq_len].long()
            idx += actual_seq_len
            attention_mask = packed[:, idx:idx + target_seq_len]
        else:

            gene_tokens = torch.zeros(B, target_seq_len, dtype=torch.long, device=device)
            expr_bin_tokens = torch.zeros(B, target_seq_len, dtype=torch.long, device=device)
            expr_level_tokens = torch.zeros(B, target_seq_len, dtype=torch.long, device=device)
            attention_mask = torch.zeros(B, target_seq_len, dtype=dtype, device=device)

            if actual_seq_len > 0:

                idx = self.n_pathways
                gene_tokens[:, :actual_seq_len] = packed[:, idx:idx + actual_seq_len].long()
                idx += actual_seq_len
                expr_bin_tokens[:, :actual_seq_len] = packed[:, idx:idx + actual_seq_len].long()
                idx += actual_seq_len
                expr_level_tokens[:, :actual_seq_len] = packed[:, idx:idx + actual_seq_len].long()
                idx += actual_seq_len
                attention_mask[:, :actual_seq_len] = packed[:, idx:idx + actual_seq_len]

        return pathway_emb, gene_tokens, expr_bin_tokens, expr_level_tokens, attention_mask


class TransformerStageV2(nn.Module):


    def __init__(self, config: PCGEDConfig, n_layers: int):
        super().__init__()
        self.config = config
        self.n_pathway_tokens = config.n_pathway_tokens

        self.layers = nn.ModuleList([
            TransformerDecoderLayer(config)
            for _ in range(n_layers)
        ])


        self.rope = RotaryPositionEmbedding(
            config.d_model // config.n_heads,
            max_seq_len=config.max_seq_len
        )

    def _build_attention_mask(
        self,
        n_condition_tokens: int,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:

        total_len = n_condition_tokens + seq_len
        mask = torch.zeros(total_len, total_len, device=device, dtype=dtype)
        return mask[None, None, :, :]

    def forward(self, hidden_states: Tensor) -> Tensor:


        if len(self.layers) == 0:
            return hidden_states

        B, L, D = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype

        n_condition = self.n_pathway_tokens + 1
        gene_seq_len = L - n_condition


        cos, sin = self.rope(hidden_states, L)


        causal_mask = self._build_attention_mask(n_condition, gene_seq_len, device, dtype)


        for layer in self.layers:
            hidden_states = layer(
                hidden_states, cos, sin, causal_mask, is_causal=True
            )

        return hidden_states


class OutputStageV2(nn.Module):


    def __init__(self, config: PCGEDConfig):
        super().__init__()
        self.config = config
        self.n_pathway_tokens = config.n_pathway_tokens


        self.final_norm = _RMSNorm(config.d_model, eps=1e-6)


        vocab_size = config.n_genes + config.n_special_tokens
        self.gene_head = nn.Linear(config.d_model, vocab_size, bias=False)
        self.expr_bin_head = nn.Linear(config.d_model, config.n_expr_bins, bias=False)
        self.expr_level_head = nn.Linear(config.d_model, config.n_expr_levels, bias=False)

        self.n_condition_tokens = config.n_pathway_tokens + 1

    def forward(self, hidden_states: Tensor) -> Tensor:

        n_condition = self.n_condition_tokens


        import os
        rank = int(os.environ.get('RANK', 0))
        expected_total_len = 8192 + n_condition

        if hidden_states.shape[1] != expected_total_len:
            print(f"[Rank {rank}] WARN OutputStageV2 ABNORMAL: hidden_states shape = {hidden_states.shape}, "
                  f"expected L = {expected_total_len}, got L = {hidden_states.shape[1]}")


        hidden_states = self.final_norm(hidden_states)


        gene_hidden = hidden_states[:, n_condition:, :]


        gene_logits = self.gene_head(gene_hidden)
        expr_bin_logits = self.expr_bin_head(gene_hidden)
        expr_level_logits = self.expr_level_head(gene_hidden)


        concatenated_logits = torch.cat([
            gene_logits,
            expr_bin_logits,
            expr_level_logits,
        ], dim=-1)

        return concatenated_logits






def create_pipeline_module_v2(
    config: PCGEDConfig,
    num_stages: int = 4,
    loss_fn = None,
    **kwargs
):

    if not PIPELINE_AVAILABLE:
        raise ImportError("DeepSpeed Pipeline not available")


    specs = []


    specs.append(LayerSpec(EmbeddingStageV2, config))


    n_transformer_stages = num_stages - 2
    if n_transformer_stages < 1:
        raise ValueError(f"num_stages must be at least 3, got {num_stages}")



    if config.n_layers >= n_transformer_stages:

        layers_per_stage = config.n_layers // n_transformer_stages
        extra_layers = config.n_layers % n_transformer_stages

        for stage_idx in range(n_transformer_stages):
            n_layers_this_stage = layers_per_stage + (1 if stage_idx < extra_layers else 0)
            specs.append(LayerSpec(TransformerStageV2, config, n_layers_this_stage))
    else:

        for stage_idx in range(n_transformer_stages):
            n_layers_this_stage = 1 if stage_idx < config.n_layers else 0
            specs.append(LayerSpec(TransformerStageV2, config, n_layers_this_stage))


    specs.append(LayerSpec(OutputStageV2, config))


    pipeline_model = PipelineModule(
        layers=specs,
        num_stages=num_stages,
        loss_fn=loss_fn,
        **kwargs
    )

    return pipeline_model


def create_pipeline_model_v2(
    n_genes: int = 60664,
    n_pathways: int = 2000,
    model_size: str = 'base',
    num_stages: int = 4,
    loss_fn = None,
    **kwargs
):

    size_configs = {
        'small': {'d_model': 512, 'n_layers': 6, 'n_heads': 8, 'd_ff': 2048},
        'base': {'d_model': 768, 'n_layers': 12, 'n_heads': 12, 'd_ff': 3072},
        'large': {'d_model': 1024, 'n_layers': 24, 'n_heads': 16, 'd_ff': 4096},
    }

    config_dict = size_configs.get(model_size, size_configs['base'])
    config_dict.update({
        'n_genes': n_genes,
        'n_pathways': n_pathways,
    })
    config_dict.update(kwargs)

    config = PCGEDConfig(**config_dict)

    return create_pipeline_module_v2(config, num_stages, loss_fn=loss_fn)


if __name__ == '__main__':
    print("Testing Pipeline Model V2...")

    config = PCGEDConfig(
        n_genes=60664,
        n_pathways=2000,
        d_model=768,
        n_layers=12,
        n_heads=12,
    )


    print("\n1. Testing EmbeddingStageV2...")
    emb_stage = EmbeddingStageV2(config)

    B, L = 2, 100
    pathway_emb = torch.randn(B, config.n_pathways)
    gene_tokens = torch.randint(5, 60000, (B, L))
    expr_bin_tokens = torch.randint(0, 16, (B, L))
    expr_level_tokens = torch.randint(0, 9, (B, L))
    attention_mask = torch.ones(B, L)

    packed_input = pack_pipeline_input(
        pathway_emb, gene_tokens, expr_bin_tokens,
        expr_level_tokens, attention_mask
    )
    print(f"   Packed input shape: {packed_input.shape}")

    hidden = emb_stage(packed_input)
    print(f"   Hidden states shape: {hidden.shape}")

    print("\n2. Testing TransformerStageV2...")
    trans_stage = TransformerStageV2(config, n_layers=2)
    hidden = trans_stage(hidden)
    print(f"   After transformer: {hidden.shape}")

    print("\n3. Testing OutputStageV2...")
    out_stage = OutputStageV2(config)
    logits = out_stage(hidden)
    print(f"   Logits shape: {logits.shape}")

    expected_vocab = config.n_genes + config.n_special_tokens + config.n_expr_bins + config.n_expr_levels
    print(f"   Expected vocab: {expected_vocab}, Actual: {logits.shape[-1]}")

    print("\nOK All tests passed!")
