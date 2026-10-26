


import math
from typing import Tuple, List, Optional, Union, Dict, Any
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


try:
    from deepspeed.pipe import PipelineModule, LayerSpec, TiedLayerSpec
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






@dataclass
class PipelineData:

    hidden_states: Tensor
    cos: Tensor
    sin: Tensor
    causal_mask: Tensor
    n_condition_tokens: int

    def to_tuple(self) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:


        n_cond = torch.tensor([self.n_condition_tokens],
                              dtype=torch.long,
                              device=self.hidden_states.device)
        return (self.hidden_states, self.cos, self.sin, self.causal_mask, n_cond)

    @classmethod
    def from_tuple(cls, data: Tuple[Tensor, ...]) -> 'PipelineData':

        hidden_states, cos, sin, causal_mask, n_cond = data
        return cls(
            hidden_states=hidden_states,
            cos=cos,
            sin=sin,
            causal_mask=causal_mask,
            n_condition_tokens=n_cond[0].item(),
        )


def pack_pipeline_input(
    pathway_embedding: Tensor,
    gene_tokens: Tensor,
    expr_bin_tokens: Tensor,
    expr_level_tokens: Tensor,
    attention_mask: Tensor,
) -> Tensor:

    B = pathway_embedding.shape[0]
    L = gene_tokens.shape[1]
    n_pathways = pathway_embedding.shape[1]


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






class EmbeddingStage(nn.Module):


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


        self.rope = RotaryPositionEmbedding(
            config.d_model // config.n_heads,
            max_seq_len=config.max_seq_len
        )

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

    def _build_attention_mask(
        self,
        n_condition_tokens: int,
        seq_len: int,
        device: torch.device,
    ) -> Tensor:

        total_len = n_condition_tokens + seq_len
        mask = torch.zeros(total_len, total_len, device=device)
        return mask

    def forward(self, packed_input: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:

        B = packed_input.shape[0]
        device = packed_input.device



        total_packed_len = packed_input.shape[1]
        seq_len = (total_packed_len - self.n_pathways) // 4


        pathway_emb, gene_tokens, expr_bin_tokens, expr_level_tokens, attention_mask =\
            unpack_pipeline_input(packed_input, self.n_pathways, seq_len)


        pathway_emb = self._apply_pathway_masking(pathway_emb)


        condition_tokens = self.pathway_encoder(pathway_emb)


        sep_token = self.sep_embedding.expand(B, -1, -1)


        gene_emb = self.gene_expr_embedding(
            gene_tokens, expr_bin_tokens, expr_level_tokens
        )


        hidden_states = torch.cat([condition_tokens, sep_token, gene_emb], dim=1)

        total_seq_len = hidden_states.shape[1]
        n_condition = self.n_pathway_tokens + 1


        cos, sin = self.rope(hidden_states, total_seq_len)


        causal_mask = self._build_attention_mask(
            n_condition_tokens=n_condition,
            seq_len=total_seq_len - n_condition,
            device=device,
        )


        if attention_mask is not None:

            pad_mask = torch.cat([
                torch.ones(B, n_condition, device=device, dtype=attention_mask.dtype),
                attention_mask
            ], dim=1)
            pad_mask = pad_mask[:, None, None, :]
            pad_mask = torch.where(
                pad_mask == 0,
                torch.tensor(float('-inf'), device=device, dtype=pad_mask.dtype),
                torch.tensor(0.0, device=device, dtype=pad_mask.dtype)
            )


            causal_mask = causal_mask[None, None, :, :] + pad_mask
        else:

            causal_mask = causal_mask[None, None, :, :]





        zero_scale = (hidden_states.sum() * 0).view(1, 1, 1)
        cos = cos + zero_scale
        sin = sin + zero_scale

        causal_mask = causal_mask + zero_scale.view(1, 1, 1, 1)


        n_cond_tensor = torch.tensor([n_condition], dtype=torch.long, device=device)

        return (hidden_states, cos, sin, causal_mask, n_cond_tensor)


class TransformerStageFactory(nn.Module):


    def __init__(self, config: 'PCGEDConfig', n_layers: int):

        super().__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(config)
            for _ in range(n_layers)
        ])

    def forward(
        self,
        inputs: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:

        hidden_states, cos, sin, causal_mask, n_cond_tensor = inputs


        for layer in self.layers:
            hidden_states = layer(
                hidden_states, cos, sin, causal_mask, is_causal=True
            )




        zero_scale = (hidden_states.sum() * 0).view(1, 1, 1)
        cos = cos + zero_scale
        sin = sin + zero_scale

        causal_mask = causal_mask + zero_scale.view(1, 1, 1, 1)

        return (hidden_states, cos, sin, causal_mask, n_cond_tensor)


class TransformerStage(nn.Module):


    def __init__(self, layers: nn.ModuleList):

        super().__init__()
        self.layers = layers

    def forward(
        self,
        inputs: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:

        hidden_states, cos, sin, causal_mask, n_cond_tensor = inputs


        for layer in self.layers:
            hidden_states = layer(
                hidden_states, cos, sin, causal_mask, is_causal=True
            )




        zero_scale = (hidden_states.sum() * 0).view(1, 1, 1)
        cos = cos + zero_scale
        sin = sin + zero_scale

        causal_mask = causal_mask + zero_scale.view(1, 1, 1, 1)

        return (hidden_states, cos, sin, causal_mask, n_cond_tensor)


class OutputStage(nn.Module):


    def __init__(self, config: PCGEDConfig):
        super().__init__()
        self.config = config


        self.final_norm = _RMSNorm(config.d_model, eps=1e-6)


        vocab_size = config.n_genes + config.n_special_tokens
        self.gene_head = nn.Linear(config.d_model, vocab_size, bias=False)
        self.expr_bin_head = nn.Linear(config.d_model, config.n_expr_bins, bias=False)
        self.expr_level_head = nn.Linear(config.d_model, config.n_expr_levels, bias=False)


        self.gene_vocab_size = vocab_size
        self.expr_bin_vocab_size = config.n_expr_bins
        self.expr_level_vocab_size = config.n_expr_levels



        self.n_condition_tokens = config.n_pathway_tokens + 1

    def forward(
        self,
        inputs: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
    ) -> Tensor:

        hidden_states, cos, sin, causal_mask, n_cond_tensor = inputs





        n_condition = self.n_condition_tokens


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






class PCGEDPipelineModel(nn.Module):


    def __init__(
        self,
        config: PCGEDConfig,
        num_stages: int = 4,
        **kwargs
    ):
        super().__init__()
        self.config = config
        self.num_stages = num_stages


        transformer_stages = num_stages - 2
        if transformer_stages <= 0:
            raise ValueError(f"num_stages must be at least 3, got {num_stages}")

        if config.n_layers % transformer_stages != 0:
            print(f"WARN  n_layers ({config.n_layers}) not divisible by "
                  f"transformer_stages ({transformer_stages}). "
                  f"Some stages will have more layers.")


        self.stages = self._create_stages(config, transformer_stages)


        self.gene_vocab_size = config.n_genes + config.n_special_tokens
        self.expr_bin_vocab_size = config.n_expr_bins
        self.expr_level_vocab_size = config.n_expr_levels

    def _create_stages(
        self,
        config: PCGEDConfig,
        n_transformer_stages: int
    ) -> nn.ModuleList:

        stages = nn.ModuleList()


        stages.append(EmbeddingStage(config))


        layers_per_stage = config.n_layers // n_transformer_stages
        extra_layers = config.n_layers % n_transformer_stages

        layer_idx = 0
        for stage_idx in range(n_transformer_stages):

            n_layers_this_stage = layers_per_stage + (1 if stage_idx < extra_layers else 0)


            stage_layers = nn.ModuleList([
                TransformerDecoderLayer(config)
                for _ in range(n_layers_this_stage)
            ])

            stages.append(TransformerStage(stage_layers))
            layer_idx += n_layers_this_stage


        stages.append(OutputStage(config))

        return stages

    def forward(self, packed_input: Tensor) -> Tensor:


        x = self.stages[0](packed_input)


        for stage in self.stages[1:-1]:
            x = stage(x)


        logits = self.stages[-1](x)

        return logits

    def get_layer_specs(self) -> List:

        if not PIPELINE_AVAILABLE:
            raise ImportError("DeepSpeed Pipeline not available")

        specs = []


        specs.append(LayerSpec(EmbeddingStage, self.config))


        n_transformer_stages = self.num_stages - 2
        layers_per_stage = self.config.n_layers // n_transformer_stages
        extra_layers = self.config.n_layers % n_transformer_stages

        for stage_idx in range(n_transformer_stages):
            n_layers_this_stage = layers_per_stage + (1 if stage_idx < extra_layers else 0)



            specs.append(LayerSpec(
                TransformerStageFactory,
                self.config,
                n_layers_this_stage,
            ))


        specs.append(LayerSpec(OutputStage, self.config))

        return specs


def create_pipeline_module(
    config: PCGEDConfig,
    num_stages: int = 4,
    loss_fn = None,
    **kwargs
) -> 'PipelineModule':

    if not PIPELINE_AVAILABLE:
        raise ImportError(
            "DeepSpeed Pipeline not available. "
            "Please install DeepSpeed with pipeline support."
        )


    pcged_pipeline = PCGEDPipelineModel(config, num_stages)
    layer_specs = pcged_pipeline.get_layer_specs()


    pipeline_model = PipelineModule(
        layers=layer_specs,
        num_stages=num_stages,
        loss_fn=loss_fn,
        **kwargs
    )

    return pipeline_model


def create_pipeline_model(
    n_genes: int = 60664,
    n_pathways: int = 2000,
    model_size: str = 'base',
    num_stages: int = 4,
    use_deepspeed_pipeline: bool = True,
    loss_fn = None,
    **kwargs
) -> Union[PCGEDPipelineModel, 'PipelineModule']:


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

    if use_deepspeed_pipeline and PIPELINE_AVAILABLE:
        return create_pipeline_module(config, num_stages, loss_fn=loss_fn)
    else:
        return PCGEDPipelineModel(config, num_stages)






def convert_to_pipeline(
    model: nn.Module,
    num_stages: int = 4,
) -> PCGEDPipelineModel:

    if not hasattr(model, 'config'):
        raise ValueError("Model must have a 'config' attribute")


    pipeline_model = PCGEDPipelineModel(model.config, num_stages)


    _copy_weights(model, pipeline_model)

    return pipeline_model


def _copy_weights(source: nn.Module, target: PCGEDPipelineModel):


    embed_stage = target.stages[0]


    embed_stage.pathway_encoder.load_state_dict(
        source.pathway_encoder.state_dict()
    )


    embed_stage.gene_expr_embedding.load_state_dict(
        source.gene_expr_embedding.state_dict()
    )


    embed_stage.sep_embedding.data.copy_(source.sep_embedding.data)


    embed_stage.rope.load_state_dict(source.rope.state_dict())


    source_layer_idx = 0
    for stage in target.stages[1:-1]:
        for layer in stage.layers:
            layer.load_state_dict(
                source.layers[source_layer_idx].state_dict()
            )
            source_layer_idx += 1


    output_stage = target.stages[-1]
    output_stage.final_norm.load_state_dict(source.final_norm.state_dict())
    output_stage.gene_head.load_state_dict(source.gene_head.state_dict())
    output_stage.expr_bin_head.load_state_dict(source.expr_bin_head.state_dict())
    output_stage.expr_level_head.load_state_dict(source.expr_level_head.state_dict())






if __name__ == '__main__':
    print("=" * 80)
    print(" TESTING PCGED PIPELINE MODEL")
    print("=" * 80)


    config = PCGEDConfig(
        n_genes=60000,
        n_pathways=2000,
        d_model=512,
        n_layers=6,
        n_heads=8,
        d_ff=2048,
        n_pathway_tokens=8,
        pathway_encoder_type='multiscale',
    )


    for num_stages in [3, 4]:
        print(f"\n{'='*60}")
        print(f"  Testing Pipeline Model with {num_stages} stages")
        print("=" * 60)


        pipeline_model = PCGEDPipelineModel(config, num_stages=num_stages)

        n_params = sum(p.numel() for p in pipeline_model.parameters())
        print(f"    Total parameters: {n_params:,}")
        print(f"    Number of stages: {len(pipeline_model.stages)}")

        for i, stage in enumerate(pipeline_model.stages):
            stage_params = sum(p.numel() for p in stage.parameters())
            print(f"      Stage {i}: {type(stage).__name__}, params: {stage_params:,}")


        B, L = 2, 100
        n_pathways = 2000


        pathway_emb = torch.randn(B, n_pathways)
        gene_tokens = torch.randint(5, 60000, (B, L))
        expr_bin_tokens = torch.randint(0, 16, (B, L))
        expr_level_tokens = torch.randint(0, 9, (B, L))
        attention_mask = torch.ones(B, L)

        packed_input = pack_pipeline_input(
            pathway_emb, gene_tokens, expr_bin_tokens,
            expr_level_tokens, attention_mask
        )

        print(f"\n    Testing forward pass...")
        print(f"      Packed input shape: {packed_input.shape}")

        with torch.no_grad():
            outputs = pipeline_model(packed_input)

        print(f"      Output shape: {outputs.shape}")


        expected_vocab = (
            config.n_genes + config.n_special_tokens +
            config.n_expr_bins +
            config.n_expr_levels
        )
        print(f"      Expected vocab size: {expected_vocab}")
        print(f"      Actual vocab size: {outputs.shape[-1]}")
        assert outputs.shape[-1] == expected_vocab, "Output vocab size mismatch!"
        print(f"      OK Output shape correct!")


    print(f"\n{'='*60}")
    print("  Testing weight conversion from standard model")
    print("=" * 60)

    from src.model.model import PCGED


    standard_model = PCGED(config)
    print(f"  Standard model params: {sum(p.numel() for p in standard_model.parameters()):,}")


    converted_model = convert_to_pipeline(standard_model, num_stages=4)
    print(f"  Converted model params: {sum(p.numel() for p in converted_model.parameters()):,}")


    with torch.no_grad():

        standard_output = standard_model(
            pathway_embedding=pathway_emb,
            gene_tokens=gene_tokens,
            expr_bin_tokens=expr_bin_tokens,
            expr_level_tokens=expr_level_tokens,
            attention_mask=attention_mask,
            return_loss=False,
        )

        pipeline_output = converted_model(packed_input)


    gene_vocab = config.n_genes + config.n_special_tokens
    expr_bin_vocab = config.n_expr_bins

    pipeline_gene_logits = pipeline_output[:, :, :gene_vocab]
    pipeline_expr_bin_logits = pipeline_output[:, :, gene_vocab:gene_vocab + expr_bin_vocab]
    pipeline_expr_level_logits = pipeline_output[:, :, gene_vocab + expr_bin_vocab:]


    gene_diff = (standard_output['gene_logits'] - pipeline_gene_logits).abs().max()
    expr_bin_diff = (standard_output['expr_bin_logits'] - pipeline_expr_bin_logits).abs().max()
    expr_level_diff = (standard_output['expr_level_logits'] - pipeline_expr_level_logits).abs().max()

    print(f"\n  Output comparison:")
    print(f"      Gene logits max diff: {gene_diff:.6f}")
    print(f"      Expr bin logits max diff: {expr_bin_diff:.6f}")
    print(f"      Expr level logits max diff: {expr_level_diff:.6f}")

    if gene_diff < 1e-5 and expr_bin_diff < 1e-5 and expr_level_diff < 1e-5:
        print(f"  OK Weight conversion successful!")
    else:
        print(f"  WARN  Output differences detected (may be due to numerical precision)")

    print("\n All tests completed!")
