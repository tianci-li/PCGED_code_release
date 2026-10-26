






__MODEL_VERSION__ = "v3.0.0-discrete-pathway-curriculum-20260308"

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint_utils
from torch import Tensor
from typing import Optional, Dict, Tuple, List
import math
from dataclasses import dataclass
from src.pathway.pathway_perturb_interpolator import PathwayPerturbInterpolator






try:


    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
except AttributeError:

    pass


try:
    from einops import rearrange, repeat
except ImportError:
    rearrange = None
    repeat = None


try:
    from flash_attn import flash_attn_func
    from flash_attn.bert_padding import pad_input, unpad_input
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    flash_attn_func = None



class RMSNorm(nn.Module):


    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:

        target_dtype = self.weight.dtype
        if x.dtype != target_dtype:
            x = x.to(target_dtype)

        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)

        return x / rms * self.weight



if hasattr(nn, 'RMSNorm'):


    _RMSNorm = nn.RMSNorm
else:

    _RMSNorm = RMSNorm


@dataclass
class PCGEDConfig:


    n_genes: int = 60664
    n_expr_bins: int = 16
    n_expr_levels: int = 9
    n_pathways: int = 2000


    pad_token_id: int = 0
    sep_token_id: int = 1
    eos_token_id: int = 2
    mask_token_id: int = 3
    drug_token_id: int = 4
    n_special_tokens: int = 5


    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    d_ff: int = 3072
    dropout: float = 0.1
    max_seq_len: int = 8192


    n_pathway_tokens: int = 8
    pathway_hidden_dim: int = 2048
    pathway_encoder_type: str = 'discrete'
    n_global_pathway_tokens: int = 2
    pathway_fusion_layers: int = 2


    use_flash_attention: bool = True
    use_gradient_checkpointing: bool = False
    tie_word_embeddings: bool = False


    drug_embedding_dim: int = 256
    n_drug_tokens: int = 2


    pathway_masking_enabled: bool = True












    pathway_curriculum_phase0_end: float = 0.10
    pathway_curriculum_phase1_end: float = 0.20
    pathway_curriculum_phase2_end: float = 0.60



    pathway_curriculum_k_phase1: int = 5


    pathway_curriculum_k_max_ratio: float = 0.55


    pathway_curriculum_phase3_dropout: float = 0.1


    pathway_masking_total_steps: int = 500000


    pathway_interpolation_enabled: bool = False
    pathway_interpolation_mode: str = 'off'
    pathway_interpolation_strategy: str = 'linear'
    pathway_interpolation_mix_ratio: float = 0.5

class RotaryPositionEmbedding(nn.Module):


    def __init__(self, dim: int, max_seq_len: int = 8192, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base


        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)


        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)


        self.register_buffer('cos_cached', emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer('sin_cached', emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x: Tensor, seq_len: int) -> Tuple[Tensor, Tensor]:
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
        return (
            self.cos_cached[:, :, :seq_len, :],
            self.sin_cached[:, :, :seq_len, :]
        )


def rotate_half(x: Tensor) -> Tensor:

    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> Tuple[Tensor, Tensor]:

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class SwiGLU(nn.Module):


    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:

        target_dtype = self.w1.weight.dtype
        if x.dtype != target_dtype:
            x = x.to(target_dtype)
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class MultiHeadAttention(nn.Module):


    def __init__(self, config: PCGEDConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.head_dim = config.d_model // config.n_heads
        self.use_flash = config.use_flash_attention and FLASH_ATTN_AVAILABLE

        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        self.dropout = nn.Dropout(config.dropout)
        self.dropout_p = config.dropout
        self.scale = self.head_dim ** -0.5


        if not hasattr(MultiHeadAttention, '_flash_status_printed'):
            MultiHeadAttention._flash_status_printed = True
            if self.use_flash:
                print(f"OK Using Flash Attention 2 (flash_attn library)")
            else:
                print(f"WARN Flash Attention not available, using standard attention (O(L ) memory)")


    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        attention_mask: Optional[Tensor] = None,
        is_causal: bool = True,
        past_kv: Optional[Tuple[Tensor, Tensor]] = None,
        use_cache: bool = False,
    ):
        B, L, _ = x.shape


        target_dtype = self.q_proj.weight.dtype
        if x.dtype != target_dtype:
            x = x.to(target_dtype)
        if cos.dtype != target_dtype:
            cos = cos.to(target_dtype)
        if sin.dtype != target_dtype:
            sin = sin.to(target_dtype)

        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)


        q, k = apply_rotary_pos_emb(q, k, cos, sin)


        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)

        present_kv = (k, v) if use_cache else None


        if self.use_flash and past_kv is None:

            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)


            if q.dtype not in [torch.float16, torch.bfloat16]:
                flash_dtype = target_dtype if target_dtype in [torch.float16, torch.bfloat16] else torch.bfloat16
                q = q.to(flash_dtype)
                k = k.to(flash_dtype)
                v = v.to(flash_dtype)

            attn_output = flash_attn_func(
                q, k, v,
                dropout_p=self.dropout_p if self.training else 0.0,
                causal=is_causal,
                softmax_scale=self.scale,
            )

        else:

            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale


            if is_causal and past_kv is None:
                L_kv = k.shape[2]
                causal_mask = torch.triu(
                    torch.ones(L, L_kv, dtype=torch.bool, device=x.device), diagonal=1
                )
                attn_weights.masked_fill_(causal_mask, float('-inf'))

            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask

            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
            if attn_weights.dtype != v.dtype:
                attn_weights = attn_weights.to(v.dtype)
            attn_weights = self.dropout(attn_weights)
            attn_output = torch.matmul(attn_weights, v)
            attn_output = attn_output.transpose(1, 2)

        attn_output = attn_output.contiguous().view(B, L, self.d_model)
        output = self.o_proj(attn_output)

        if use_cache:
            return output, present_kv
        return output

class TransformerDecoderLayer(nn.Module):


    def __init__(self, config: PCGEDConfig):
        super().__init__()
        self.attn_norm = _RMSNorm(config.d_model, eps=1e-6)
        self.attn = MultiHeadAttention(config)
        self.ff_norm = _RMSNorm(config.d_model, eps=1e-6)
        self.ff = SwiGLU(config.d_model, config.d_ff or config.d_model * 4, config.dropout)


    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        attention_mask: Optional[Tensor] = None,
        is_causal: bool = True,
        past_kv: Optional[Tuple[Tensor, Tensor]] = None,
        use_cache: bool = False,
    ):
        if use_cache:
            attn_out, present_kv = self.attn(
                self.attn_norm(x), cos, sin, attention_mask, is_causal, past_kv, True)
            x = x + attn_out
            x = x + self.ff(self.ff_norm(x))
            return x, present_kv
        else:
            x = x + self.attn(self.attn_norm(x), cos, sin, attention_mask, is_causal)
            x = x + self.ff(self.ff_norm(x))
            return x

class PathwayEncoder(nn.Module):


    def __init__(self, config: PCGEDConfig):
        super().__init__()
        self.n_tokens = config.n_pathway_tokens
        self.d_model = config.d_model


        self.encoder = nn.Sequential(
            nn.Linear(config.n_pathways, config.pathway_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.pathway_hidden_dim),
            nn.Dropout(config.dropout),
            nn.Linear(config.pathway_hidden_dim, config.n_pathway_tokens * config.d_model),
        )


        self.type_embedding = nn.Parameter(torch.randn(1, config.n_pathway_tokens, config.d_model) * 0.02)

    def forward(self, pathway_embedding: Tensor) -> Tensor:

        B = pathway_embedding.shape[0]
        tokens = self.encoder(pathway_embedding)
        tokens = tokens.view(B, self.n_tokens, self.d_model)
        tokens = tokens + self.type_embedding
        return tokens


class DrugEncoder(nn.Module):


    def __init__(self, config: PCGEDConfig):
        super().__init__()
        self.config = config
        self.n_tokens = config.n_drug_tokens
        self.encoder = nn.Sequential(
            nn.Linear(config.drug_embedding_dim, config.d_model),
            nn.GELU(),
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.n_drug_tokens * config.d_model),
        )
        self.type_embedding = nn.Parameter(torch.randn(1, config.n_drug_tokens, config.d_model) * 0.02)

    def forward(self, drug_embedding: Tensor) -> Tensor:
        B = drug_embedding.shape[0]
        tokens = self.encoder(drug_embedding)
        tokens = tokens.view(B, self.n_tokens, self.config.d_model)
        return tokens + self.type_embedding


class PathwayEncoderV2(nn.Module):


    def __init__(self, config: PCGEDConfig):
        super().__init__()
        self.n_tokens = config.n_pathway_tokens
        self.d_model = config.d_model
        self.n_pathways = config.n_pathways


        self.pathways_per_token = math.ceil(config.n_pathways / config.n_pathway_tokens)


        self.token_projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.pathways_per_token, config.d_model),
                nn.GELU(),
                nn.LayerNorm(config.d_model),
                nn.Linear(config.d_model, config.d_model),
            )
            for _ in range(config.n_pathway_tokens)
        ])


        cross_num_heads = 4
        while config.d_model % cross_num_heads != 0 and cross_num_heads > 1:
            cross_num_heads -= 1


        self.cross_attention = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=cross_num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(config.d_model)


        self.type_embedding = nn.Parameter(
            torch.randn(1, config.n_pathway_tokens, config.d_model) * 0.02
        )

    def forward(self, pathway_embedding: Tensor) -> Tensor:

        B = pathway_embedding.shape[0]
        device = pathway_embedding.device



        padded_size = self.n_tokens * self.pathways_per_token
        if self.n_pathways < padded_size:

            padding = torch.zeros(B, padded_size - self.n_pathways, device=device, dtype=pathway_embedding.dtype)
            pathway_embedding = torch.cat([pathway_embedding, padding], dim=1)


        chunks = pathway_embedding.view(B, self.n_tokens, self.pathways_per_token)


        tokens = []
        for i, projector in enumerate(self.token_projectors):
            token = projector(chunks[:, i, :])
            tokens.append(token)

        tokens = torch.stack(tokens, dim=1)


        tokens_attn, _ = self.cross_attention(tokens, tokens, tokens)
        tokens = self.cross_norm(tokens + tokens_attn)


        tokens = tokens + self.type_embedding

        return tokens


class MultiScalePathwayEncoder(nn.Module):


    def __init__(self, config: PCGEDConfig):
        super().__init__()
        self.n_tokens = config.n_pathway_tokens
        self.d_model = config.d_model
        self.n_pathways = config.n_pathways
        self.n_global_tokens = config.n_global_pathway_tokens




        self.global_encoder = nn.Sequential(
            nn.Linear(config.n_pathways, config.pathway_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.pathway_hidden_dim),
            nn.Dropout(config.dropout),
            nn.Linear(config.pathway_hidden_dim, self.n_global_tokens * config.d_model),
        )



        self.n_local_tokens = config.n_pathway_tokens - self.n_global_tokens
        self.pathways_per_local = math.ceil(config.n_pathways / self.n_local_tokens)


        self.local_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.pathways_per_local, config.d_model),
                nn.GELU(),
                nn.LayerNorm(config.d_model),
                nn.Dropout(config.dropout),
                nn.Linear(config.d_model, config.d_model),
            )
            for _ in range(self.n_local_tokens)
        ])




        nhead = config.n_heads // 3 if config.n_heads >= 6 else 2

        while config.d_model % nhead != 0 and nhead > 1:
            nhead -= 1
        if nhead < 1:
            nhead = 1

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=nhead,
            dim_feedforward=config.d_model * 2,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.fusion = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.pathway_fusion_layers,
        )



        self.global_type_embedding = nn.Parameter(
            torch.randn(1, self.n_global_tokens, config.d_model) * 0.02
        )
        self.local_type_embedding = nn.Parameter(
            torch.randn(1, self.n_local_tokens, config.d_model) * 0.02
        )


        self.position_embedding = nn.Parameter(
            torch.randn(1, config.n_pathway_tokens, config.d_model) * 0.02
        )


        self.output_norm = nn.LayerNorm(config.d_model)


        self._init_weights()

    def _init_weights(self):

        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                in_features = module.weight.shape[1]
                out_features = module.weight.shape[0]
                if in_features > 500:

                    nn.init.xavier_uniform_(module.weight, gain=0.1)
                else:
                    nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, pathway_embedding: Tensor) -> Tensor:

        B = pathway_embedding.shape[0]
        device = pathway_embedding.device



        actual_dim = pathway_embedding.shape[1] if pathway_embedding.dim() == 2 else 0
        if actual_dim != self.n_pathways:

            fixed = torch.zeros(B, self.n_pathways, device=device, dtype=pathway_embedding.dtype)
            if actual_dim > 0 and pathway_embedding.dim() == 2:
                copy_dim = min(actual_dim, self.n_pathways)
                fixed[:, :copy_dim] = pathway_embedding[:, :copy_dim]
            pathway_embedding = fixed


        if torch.isnan(pathway_embedding).any() or torch.isinf(pathway_embedding).any():
            pathway_embedding = torch.nan_to_num(pathway_embedding, nan=0.0, posinf=1.0, neginf=0.0)


        global_features = self.global_encoder(pathway_embedding)

        global_tokens = global_features.view(B, self.n_global_tokens, self.d_model)
        global_tokens = global_tokens + self.global_type_embedding



        padded_size = self.n_local_tokens * self.pathways_per_local
        if self.n_pathways < padded_size:

            padding = torch.zeros(B, padded_size - self.n_pathways, device=device, dtype=pathway_embedding.dtype)
            padded_pathway = torch.cat([pathway_embedding, padding], dim=1)
        else:
            padded_pathway = pathway_embedding[:, :padded_size]


        chunks = padded_pathway.view(B, self.n_local_tokens, self.pathways_per_local)


        local_tokens = []
        for i, encoder in enumerate(self.local_encoders):
            local_token = encoder(chunks[:, i, :])
            local_tokens.append(local_token)

        local_tokens = torch.stack(local_tokens, dim=1)
        local_tokens = local_tokens + self.local_type_embedding


        if torch.isnan(local_tokens).any() or torch.isinf(local_tokens).any():
            print(f"WARN  MultiScalePathwayEncoder: local_tokens has NaN/Inf!")



        all_tokens = torch.cat([global_tokens, local_tokens], dim=1)


        all_tokens = all_tokens + self.position_embedding



        all_tokens = self.fusion(all_tokens)


        if torch.isnan(all_tokens).any() or torch.isinf(all_tokens).any():
            print(f"WARN  MultiScalePathwayEncoder: all_tokens has NaN/Inf after fusion!")


        all_tokens = self.output_norm(all_tokens)

        return all_tokens


class DiscretePathwayEncoder(nn.Module):


    def __init__(self, config: PCGEDConfig):
        super().__init__()
        self.n_pathways = config.n_pathways
        self.d_model = config.d_model
        self.n_tokens = config.n_pathway_tokens
        self.n_global_tokens = config.n_global_pathway_tokens
        self.n_local_tokens = config.n_pathway_tokens - config.n_global_pathway_tokens



        self.num_bins = 5
        self.mask_token_id = self.num_bins
        self.embed_dim = min(64, config.d_model // 4)

        self.pathway_embedding = nn.Embedding(
            num_embeddings=self.num_bins + 1,
            embedding_dim=self.embed_dim,
        )
        nn.init.normal_(self.pathway_embedding.weight, std=0.02)


        interpolation_mode = config.pathway_interpolation_mode
        if not config.pathway_interpolation_enabled:
            interpolation_mode = 'off'
        self.perturb_interpolator = PathwayPerturbInterpolator(
            num_bins=self.num_bins,
            mask_token_id=self.mask_token_id,
            mode=interpolation_mode,
            strategy=config.pathway_interpolation_strategy,
            mix_ratio=config.pathway_interpolation_mix_ratio,
        )



        global_input_dim = config.n_pathways * self.embed_dim
        self.global_encoder = nn.Sequential(
            nn.Linear(global_input_dim, config.pathway_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.pathway_hidden_dim),
            nn.Dropout(config.dropout),
            nn.Linear(config.pathway_hidden_dim, self.n_global_tokens * config.d_model),
        )


        self.pathways_per_local = math.ceil(config.n_pathways / self.n_local_tokens)
        local_input_dim = self.pathways_per_local * self.embed_dim

        self.local_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(local_input_dim, config.d_model),
                nn.GELU(),
                nn.LayerNorm(config.d_model),
                nn.Dropout(config.dropout),
                nn.Linear(config.d_model, config.d_model),
            )
            for _ in range(self.n_local_tokens)
        ])


        nhead = config.n_heads // 3 if config.n_heads >= 6 else 2
        while config.d_model % nhead != 0 and nhead > 1:
            nhead -= 1
        if nhead < 1:
            nhead = 1

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=nhead,
            dim_feedforward=config.d_model * 2,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.fusion = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.pathway_fusion_layers,
        )


        self.global_type_embedding = nn.Parameter(
            torch.randn(1, self.n_global_tokens, config.d_model) * 0.02
        )
        self.local_type_embedding = nn.Parameter(
            torch.randn(1, self.n_local_tokens, config.d_model) * 0.02
        )
        self.position_embedding = nn.Parameter(
            torch.randn(1, config.n_pathway_tokens, config.d_model) * 0.02
        )


        self.output_norm = nn.LayerNorm(config.d_model)


        self._init_weights()

    def _init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                in_features = module.weight.shape[1]
                if in_features > 500:
                    nn.init.xavier_uniform_(module.weight, gain=0.1)
                else:
                    nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        pathway_bins: Optional[Tensor] = None,
        pathway_scores: Optional[Tensor] = None,
        pathway_visibility_mask: Optional[Tensor] = None,
    ) -> Tensor:

        if pathway_bins is None and pathway_scores is None:
            raise ValueError("DiscretePathwayEncoder requires pathway_bins or pathway_scores.")

        base_input = pathway_bins if pathway_bins is not None else pathway_scores
        B = base_input.shape[0]
        device = base_input.device


        actual_dim = base_input.shape[1] if base_input.dim() == 2 else 0
        if actual_dim != self.n_pathways:
            if pathway_bins is not None:
                fixed_bins = torch.zeros(B, self.n_pathways, device=device, dtype=torch.long)
            else:
                fixed_bins = None
            if pathway_scores is not None:
                fixed_scores = torch.zeros(B, self.n_pathways, device=device, dtype=pathway_scores.dtype)
            else:
                fixed_scores = None

            if actual_dim > 0 and base_input.dim() == 2:
                copy_dim = min(actual_dim, self.n_pathways)
                if fixed_bins is not None and pathway_bins is not None:
                    fixed_bins[:, :copy_dim] = pathway_bins[:, :copy_dim]
                if fixed_scores is not None and pathway_scores is not None:
                    fixed_scores[:, :copy_dim] = pathway_scores[:, :copy_dim]
            pathway_bins = fixed_bins
            pathway_scores = fixed_scores


        pw_emb = self.perturb_interpolator(
            embedding_table=self.pathway_embedding.weight,
            pathway_bins=pathway_bins,
            pathway_scores=pathway_scores,
            pathway_visibility_mask=pathway_visibility_mask,
        )
        target_dtype = self.pathway_embedding.weight.dtype
        if pw_emb.dtype != target_dtype:
            pw_emb = pw_emb.to(dtype=target_dtype)



        global_input = pw_emb.reshape(B, -1)
        global_features = self.global_encoder(global_input)
        global_tokens = global_features.view(B, self.n_global_tokens, self.d_model)
        global_tokens = global_tokens + self.global_type_embedding



        padded_size = self.n_local_tokens * self.pathways_per_local
        if self.n_pathways < padded_size:

            pad_ids = torch.full(
                (B, padded_size - self.n_pathways), self.mask_token_id,
                device=device, dtype=torch.long
            )
            pad_emb = self.pathway_embedding(pad_ids)
            pw_emb_padded = torch.cat([pw_emb, pad_emb], dim=1)
        else:
            pw_emb_padded = pw_emb[:, :padded_size, :]


        chunks = pw_emb_padded.view(B, self.n_local_tokens, self.pathways_per_local, self.embed_dim)

        chunks_flat = chunks.reshape(B, self.n_local_tokens, -1)

        local_tokens = []
        for i, encoder in enumerate(self.local_encoders):
            local_token = encoder(chunks_flat[:, i, :])
            local_tokens.append(local_token)
        local_tokens = torch.stack(local_tokens, dim=1)
        local_tokens = local_tokens + self.local_type_embedding


        all_tokens = torch.cat([global_tokens, local_tokens], dim=1)
        all_tokens = all_tokens + self.position_embedding
        all_tokens = self.fusion(all_tokens)
        all_tokens = self.output_norm(all_tokens)

        return all_tokens


class GeneExpressionEmbedding(nn.Module):


    def __init__(self, config: PCGEDConfig):
        super().__init__()
        vocab_size = config.n_genes + config.n_special_tokens

        self.gene_embedding = nn.Embedding(vocab_size, config.d_model, padding_idx=config.pad_token_id)

        self.expr_bin_embedding = nn.Embedding(config.n_expr_bins, config.d_model, padding_idx=0)
        self.expr_level_embedding = nn.Embedding(config.n_expr_levels, config.d_model, padding_idx=0)

        self.dropout = nn.Dropout(config.dropout)
        self.norm = _RMSNorm(config.d_model, eps=1e-6)


        nn.init.normal_(self.gene_embedding.weight, std=0.02)
        nn.init.normal_(self.expr_bin_embedding.weight, std=0.02)
        nn.init.normal_(self.expr_level_embedding.weight, std=0.02)

    def forward(
        self,
        gene_tokens: Tensor,
        expr_bin_tokens: Optional[Tensor] = None,
        expr_level_tokens: Optional[Tensor] = None,
    ) -> Tensor:

        emb = self.gene_embedding(gene_tokens)

        if expr_bin_tokens is not None:
            emb = emb + self.expr_bin_embedding(expr_bin_tokens)

        if expr_level_tokens is not None:
            emb = emb + self.expr_level_embedding(expr_level_tokens)

        return self.dropout(self.norm(emb))


class PCGED(nn.Module):


    def __init__(self, config: PCGEDConfig):
        super().__init__()
        self.config = config


        self._current_masking_step: int = 0



        if config.pathway_encoder_type == 'discrete':
            self.pathway_encoder = DiscretePathwayEncoder(config)
            self._discrete_pathway = True
        elif config.pathway_encoder_type == 'multiscale':
            self.pathway_encoder = MultiScalePathwayEncoder(config)
            self._discrete_pathway = False
        elif config.pathway_encoder_type == 'v2':
            self.pathway_encoder = PathwayEncoderV2(config)
            self._discrete_pathway = False
        else:
            self.pathway_encoder = PathwayEncoder(config)
            self._discrete_pathway = False


        self._pathway_mask_id = 5


        self.gene_expr_embedding = GeneExpressionEmbedding(config)


        self.sep_embedding = nn.Parameter(torch.randn(1, 1, config.d_model) * 0.02)


        self.rope = RotaryPositionEmbedding(
            config.d_model // config.n_heads,
            max_seq_len=config.max_seq_len
        )


        self.layers = nn.ModuleList([
            TransformerDecoderLayer(config) for _ in range(config.n_layers)
        ])


        self.final_norm = _RMSNorm(config.d_model, eps=1e-6)


        vocab_size = config.n_genes + config.n_special_tokens
        self.gene_head = nn.Linear(config.d_model, vocab_size, bias=False)
        self.expr_bin_head = nn.Linear(config.d_model, config.n_expr_bins, bias=False)
        self.expr_level_head = nn.Linear(config.d_model, config.n_expr_levels, bias=False)


        self.expr_regression_head = nn.Linear(config.d_model, 1, bias=True)


        if config.tie_word_embeddings:
            self.gene_head.weight = self.gene_expr_embedding.gene_embedding.weight


        self.gradient_checkpointing = config.use_gradient_checkpointing


        self._init_weights()

    def _init_weights(self):

        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):

                in_features = module.weight.shape[1]
                if in_features > 1000:

                    nn.init.xavier_uniform_(module.weight)
                else:

                    nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
            elif isinstance(module, (nn.LayerNorm, _RMSNorm)):
                if hasattr(module, 'weight') and module.weight is not None:
                    nn.init.ones_(module.weight)
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.zeros_(module.bias)

    def load_state_dict(self, state_dict, strict=True, **kwargs):

        keys_to_remove = [k for k in state_dict if 'rope.' in k and '_cached' in k]
        if keys_to_remove:
            from copy import copy
            state_dict = copy(state_dict)
            for k in keys_to_remove:
                del state_dict[k]
        return super().load_state_dict(state_dict, strict=strict, **kwargs)

    def set_masking_step(self, current_step: int, total_steps: Optional[int] = None):

        self._current_masking_step = current_step
        if total_steps is not None:
            self.config.pathway_masking_total_steps = total_steps

    def _get_current_n_visible(self) -> int:

        cfg = self.config
        total = max(cfg.pathway_masking_total_steps, 1)
        ratio = min(self._current_masking_step / total, 1.0)
        N = cfg.n_pathways

        k = cfg.pathway_curriculum_k_phase1
        K_max = max(1, int(N * cfg.pathway_curriculum_k_max_ratio))

        if ratio < cfg.pathway_curriculum_phase0_end:

            return 0

        elif ratio < cfg.pathway_curriculum_phase1_end:

            return k

        elif ratio < cfg.pathway_curriculum_phase2_end:

            t = (ratio - cfg.pathway_curriculum_phase1_end) / (
                cfg.pathway_curriculum_phase2_end - cfg.pathway_curriculum_phase1_end
            )
            n_visible = int(k + t * (K_max - k))
            return min(n_visible, K_max)

        else:

            return K_max

    def _get_current_phase(self) -> int:

        cfg = self.config
        total = max(cfg.pathway_masking_total_steps, 1)
        ratio = min(self._current_masking_step / total, 1.0)

        if ratio < cfg.pathway_curriculum_phase0_end:
            return 0
        elif ratio < cfg.pathway_curriculum_phase1_end:
            return 1
        elif ratio < cfg.pathway_curriculum_phase2_end:
            return 2
        else:
            return 3

    def _build_pathway_visibility_mask(self, pathway_tensor: Tensor) -> Optional[Tensor]:

        if not self.config.pathway_masking_enabled:
            return None

        B, N = pathway_tensor.shape
        device = pathway_tensor.device
        n_visible = self._get_current_n_visible()

        if n_visible <= 0:
            return torch.zeros(B, N, device=device, dtype=torch.bool)
        if n_visible >= N:
            return torch.ones(B, N, device=device, dtype=torch.bool)

        visible_mask = torch.zeros(B, N, device=device, dtype=torch.bool)
        if self.training:
            for i in range(B):
                visible_indices = torch.randperm(N, device=device)[:n_visible]
                visible_mask[i, visible_indices] = True
        else:
            visible_mask[:, :n_visible] = True

        return visible_mask

    def _apply_pathway_masking(self, pathway_embedding: Tensor) -> Tensor:

        if not self.config.pathway_masking_enabled:
            return pathway_embedding

        B, N = pathway_embedding.shape
        device = pathway_embedding.device
        n_visible = self._get_current_n_visible()
        visible_mask = self._build_pathway_visibility_mask(pathway_embedding)

        if self._discrete_pathway:

            mask_id = self._pathway_mask_id

            if n_visible <= 0:

                return torch.full_like(pathway_embedding, mask_id)

            if n_visible >= N:

                return pathway_embedding

            result = torch.full_like(pathway_embedding, mask_id)
            if visible_mask is not None:
                result[visible_mask] = pathway_embedding[visible_mask]
            return result

        else:

            dtype = pathway_embedding.dtype

            if n_visible <= 0:
                return torch.zeros_like(pathway_embedding)
            if n_visible >= N:
                return pathway_embedding

            if visible_mask is None:
                mask = torch.ones(B, N, device=device, dtype=dtype)
            else:
                mask = visible_mask.to(dtype=dtype)

            return pathway_embedding * mask

    def _build_attention_mask(
        self,
        n_condition_tokens: int,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype = None,
    ) -> Tensor:

        if dtype is None:
            dtype = torch.float32

        total_len = n_condition_tokens + seq_len


        mask = torch.zeros(total_len, total_len, device=device, dtype=dtype)

        return mask

    def forward(
        self,
        pathway_embedding: Tensor,
        pathway_scores: Optional[Tensor] = None,
        gene_tokens: Optional[Tensor] = None,
        expr_bin_tokens: Optional[Tensor] = None,
        expr_level_tokens: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Dict[str, Tensor]] = None,
        return_loss: bool = True,
    ) -> Dict[str, Tensor]:

        B = pathway_embedding.shape[0]
        device = pathway_embedding.device


        pathway_visibility_mask = None
        use_continuous_pathway = (
            self._discrete_pathway
            and pathway_scores is not None
            and self.config.pathway_interpolation_enabled
        )

        if use_continuous_pathway:
            pathway_visibility_mask = self._build_pathway_visibility_mask(pathway_scores)
        else:
            pathway_embedding = self._apply_pathway_masking(pathway_embedding)


        if self._discrete_pathway:
            condition_tokens = self.pathway_encoder(
                pathway_bins=pathway_embedding,
                pathway_scores=pathway_scores if use_continuous_pathway else None,
                pathway_visibility_mask=pathway_visibility_mask,
            )
        else:
            condition_tokens = self.pathway_encoder(pathway_embedding)


        sep_token = self.sep_embedding.expand(B, -1, -1)


        if gene_tokens is not None:
            gene_emb = self.gene_expr_embedding(gene_tokens, expr_bin_tokens, expr_level_tokens)

            hidden_states = torch.cat([condition_tokens, sep_token, gene_emb], dim=1)
        else:

            hidden_states = torch.cat([condition_tokens, sep_token], dim=1)

        seq_len = hidden_states.shape[1]
        n_condition = self.config.n_pathway_tokens + 1


        cos, sin = self.rope(hidden_states, seq_len)











        _use_flash = (len(self.layers) > 0
                      and getattr(self.layers[0].attn, 'use_flash', False))

        if _use_flash:



            causal_mask = None
        else:


            if attention_mask is not None:
                pad_mask = torch.cat([
                    torch.ones(B, n_condition, device=device, dtype=attention_mask.dtype),
                    attention_mask
                ], dim=1)

                pad_mask = pad_mask[:, None, None, :].to(dtype=hidden_states.dtype)
                causal_mask = torch.where(
                    pad_mask == 0,
                    torch.tensor(float('-inf'), device=device, dtype=hidden_states.dtype),
                    torch.tensor(0.0, device=device, dtype=hidden_states.dtype)
                )
            else:
                causal_mask = None







        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden_states = checkpoint_utils.checkpoint(
                    layer, hidden_states, cos, sin, causal_mask, True,
                    use_reentrant=False
                )
            else:
                hidden_states = layer(hidden_states, cos, sin, causal_mask, is_causal=True)

        hidden_states = self.final_norm(hidden_states)


        gene_hidden = hidden_states[:, n_condition:, :]


        gene_logits = self.gene_head(gene_hidden)
        expr_bin_logits = self.expr_bin_head(gene_hidden)
        expr_level_logits = self.expr_level_head(gene_hidden)
        expr_regression = self.expr_regression_head(gene_hidden).squeeze(-1)

        outputs = {
            'gene_logits': gene_logits,
            'expr_bin_logits': expr_bin_logits,
            'expr_level_logits': expr_level_logits,
            'expr_regression': expr_regression,
            'hidden_states': hidden_states,
        }


        if return_loss and labels is not None:
            outputs['loss'] = self._compute_loss(
                gene_logits, expr_bin_logits, expr_level_logits, expr_regression,
                labels, attention_mask
            )

        return outputs

    def _compute_loss(
        self,
        gene_logits: Tensor,
        expr_bin_logits: Tensor,
        expr_level_logits: Tensor,
        expr_regression: Tensor,
        labels: Dict[str, Tensor],
        attention_mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:


        gene_labels = labels.get('gene_labels')
        expr_bin_labels = labels.get('expr_bin_labels')
        expr_level_labels = labels.get('expr_level_labels')
        expr_value_labels = labels.get('expr_value_labels')

        losses = {}


        if gene_labels is not None:
            gene_loss = F.cross_entropy(
                gene_logits.view(-1, gene_logits.size(-1)),
                gene_labels.view(-1),
                ignore_index=self.config.pad_token_id,
            )
            losses['gene_loss'] = gene_loss


        if expr_bin_labels is not None:
            expr_bin_loss = F.cross_entropy(
                expr_bin_logits.view(-1, expr_bin_logits.size(-1)),
                expr_bin_labels.view(-1),
                ignore_index=0,
            )
            losses['expr_bin_loss'] = expr_bin_loss


        if expr_level_labels is not None:
            expr_level_loss = F.cross_entropy(
                expr_level_logits.view(-1, expr_level_logits.size(-1)),
                expr_level_labels.view(-1),
                ignore_index=0,
            )
            losses['expr_level_loss'] = expr_level_loss


        if expr_value_labels is not None and attention_mask is not None:
            mask = attention_mask.bool()
            regression_loss = F.smooth_l1_loss(
                expr_regression[mask],
                expr_value_labels[mask],
            )
            losses['regression_loss'] = regression_loss


        total_loss = sum(losses.values())
        losses['total_loss'] = total_loss

        return losses


    @torch.no_grad()
    def generate(
        self,
        pathway_embedding: Tensor,
        pathway_scores: Optional[Tensor] = None,
        max_length: int = 1000,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True,
        eos_token_id: Optional[int] = None,
        use_kv_cache: bool = True,
    ) -> Dict[str, Tensor]:

        B = pathway_embedding.shape[0]
        device = pathway_embedding.device
        eos_token_id = eos_token_id or self.config.eos_token_id


        generated_genes = torch.full((B, 1), self.config.sep_token_id, dtype=torch.long, device=device)
        generated_expr_bins = torch.zeros((B, 1), dtype=torch.long, device=device)
        generated_expr_levels = torch.zeros((B, 1), dtype=torch.long, device=device)


        use_continuous_pathway = (
            self._discrete_pathway
            and pathway_scores is not None
            and self.config.pathway_interpolation_enabled
        )
        if self._discrete_pathway:
            pathway_visibility_mask = self._build_pathway_visibility_mask(
                pathway_scores if use_continuous_pathway else pathway_embedding
            )
            condition_tokens = self.pathway_encoder(
                pathway_bins=pathway_embedding,
                pathway_scores=pathway_scores if use_continuous_pathway else None,
                pathway_visibility_mask=pathway_visibility_mask,
            )
        else:
            condition_tokens = self.pathway_encoder(pathway_embedding)
        sep_token = self.sep_embedding.expand(B, -1, -1)
        n_condition = self.config.n_pathway_tokens + 1

        finished = torch.zeros(B, dtype=torch.bool, device=device)

        if not use_kv_cache:

            for step in range(max_length):
                gene_emb = self.gene_expr_embedding(
                    generated_genes[:, 1:] if step > 0 else generated_genes[:, :0],
                    generated_expr_bins[:, 1:] if step > 0 else None,
                    generated_expr_levels[:, 1:] if step > 0 else None,
                )
                if gene_emb.shape[1] > 0:
                    hidden_states = torch.cat([condition_tokens, sep_token, gene_emb], dim=1)
                else:
                    hidden_states = torch.cat([condition_tokens, sep_token], dim=1)

                seq_len = hidden_states.shape[1]
                cos, sin = self.rope(hidden_states, seq_len)
                _use_flash = (len(self.layers) > 0
                              and getattr(self.layers[0].attn, 'use_flash', False))
                causal_mask = None if _use_flash else self._build_attention_mask(
                    n_condition, seq_len - n_condition, device, dtype=hidden_states.dtype)

                for layer in self.layers:
                    hidden_states = layer(hidden_states, cos, sin, causal_mask, is_causal=True)

                hidden_states = self.final_norm(hidden_states)
                last_hidden = hidden_states[:, -1:, :]
                gene_logits = self.gene_head(last_hidden)[:, 0, :]
                expr_bin_logits = self.expr_bin_head(last_hidden)[:, 0, :]
                expr_level_logits = self.expr_level_head(last_hidden)[:, 0, :]

                next_gene = self._sample_token(gene_logits, temperature, top_k, top_p, do_sample)
                next_expr_bin = self._sample_token(expr_bin_logits, temperature, top_k, top_p, do_sample)
                next_expr_level = self._sample_token(expr_level_logits, temperature, top_k, top_p, do_sample)

                finished = finished | (next_gene == eos_token_id)
                if finished.all():
                    break

                generated_genes = torch.cat([generated_genes, next_gene.unsqueeze(1)], dim=1)
                generated_expr_bins = torch.cat([generated_expr_bins, next_expr_bin.unsqueeze(1)], dim=1)
                generated_expr_levels = torch.cat([generated_expr_levels, next_expr_level.unsqueeze(1)], dim=1)
        else:


            max_total_len = n_condition + max_length
            cos_full, sin_full = self.rope(condition_tokens, max_total_len)

            past_kvs = [None] * len(self.layers)

            for step in range(max_length):
                if step == 0:

                    hidden_states = torch.cat([condition_tokens, sep_token], dim=1)
                    cos_step = cos_full[:, :, :n_condition, :]
                    sin_step = sin_full[:, :, :n_condition, :]
                else:

                    hidden_states = self.gene_expr_embedding(
                        prev_gene.unsqueeze(1),
                        prev_bin.unsqueeze(1),
                        prev_level.unsqueeze(1),
                    )
                    pos = n_condition + step - 1
                    cos_step = cos_full[:, :, pos:pos+1, :]
                    sin_step = sin_full[:, :, pos:pos+1, :]

                new_past_kvs = []
                for i, layer in enumerate(self.layers):
                    hidden_states, kv = layer(
                        hidden_states, cos_step, sin_step,
                        attention_mask=None, is_causal=True,
                        past_kv=past_kvs[i], use_cache=True,
                    )
                    new_past_kvs.append(kv)
                past_kvs = new_past_kvs

                hidden_states = self.final_norm(hidden_states)
                last_hidden = hidden_states[:, -1:, :]

                gene_logits = self.gene_head(last_hidden)[:, 0, :]
                expr_bin_logits = self.expr_bin_head(last_hidden)[:, 0, :]
                expr_level_logits = self.expr_level_head(last_hidden)[:, 0, :]

                next_gene = self._sample_token(gene_logits, temperature, top_k, top_p, do_sample)
                next_expr_bin = self._sample_token(expr_bin_logits, temperature, top_k, top_p, do_sample)
                next_expr_level = self._sample_token(expr_level_logits, temperature, top_k, top_p, do_sample)

                finished = finished | (next_gene == eos_token_id)
                if finished.all():
                    break

                generated_genes = torch.cat([generated_genes, next_gene.unsqueeze(1)], dim=1)
                generated_expr_bins = torch.cat([generated_expr_bins, next_expr_bin.unsqueeze(1)], dim=1)
                generated_expr_levels = torch.cat([generated_expr_levels, next_expr_level.unsqueeze(1)], dim=1)

                prev_gene = next_gene
                prev_bin = next_expr_bin
                prev_level = next_expr_level

        return {
            'gene_ids': generated_genes[:, 1:],
            'expr_bins': generated_expr_bins[:, 1:],
            'expr_levels': generated_expr_levels[:, 1:],
            'sequence_lengths': (generated_genes[:, 1:] != eos_token_id).sum(dim=1),
        }

    def _sample_token(
        self,
        logits: Tensor,
        temperature: float,
        top_k: int,
        top_p: float,
        do_sample: bool,
    ) -> Tensor:

        if not do_sample:
            return logits.argmax(dim=-1)


        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
        logits = logits / temperature
        vocab_size = logits.size(-1)


        if top_k > 0:
            effective_k = min(top_k, vocab_size)
            top_k_values, _ = torch.topk(logits, effective_k, dim=-1)
            threshold = top_k_values[..., -1:]
            logits = torch.where(logits < threshold, torch.full_like(logits, -1e4), logits)


        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
            indices_to_remove.scatter_(-1, sorted_indices, sorted_indices_to_remove)
            logits = torch.where(indices_to_remove, torch.full_like(logits, -1e4), logits)


        probs = F.softmax(logits, dim=-1)


        probs = torch.clamp(probs, min=1e-8, max=1.0)
        probs = probs / probs.sum(dim=-1, keepdim=True)

        return torch.multinomial(probs, num_samples=1).squeeze(-1)


def create_model(
    n_genes: int = 60000,
    n_pathways: int = 2000,
    model_size: str = 'base',
    **kwargs,
) -> PCGED:

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
    return PCGED(config)


if __name__ == '__main__':

    print("=" * 80)
    print(" TESTING PCGED MODEL")
    print("=" * 80)


    for encoder_type in ['original', 'v2', 'multiscale']:
        print(f"\n{'='*60}")
        print(f"  Testing with PathwayEncoder: {encoder_type}")
        print('='*60)


        model = create_model(
            n_genes=60000,
            n_pathways=2000,
            model_size='small',
            pathway_encoder_type=encoder_type,
            n_pathway_tokens=8,
        )

        n_params = sum(p.numel() for p in model.parameters())
        n_encoder_params = sum(p.numel() for p in model.pathway_encoder.parameters())
        print(f"    Total parameters: {n_params:,}")
        print(f"    Pathway encoder parameters: {n_encoder_params:,}")


        B, L = 4, 100
        pathway_emb = torch.randn(B, 2000)
        gene_tokens = torch.randint(5, 60000, (B, L))
        expr_bin_tokens = torch.randint(0, 11, (B, L))
        expr_level_tokens = torch.randint(0, 4, (B, L))
        attention_mask = torch.ones(B, L)

        print("    Testing forward pass...")
        outputs = model(
            pathway_embedding=pathway_emb,
            gene_tokens=gene_tokens,
            expr_bin_tokens=expr_bin_tokens,
            expr_level_tokens=expr_level_tokens,
            attention_mask=attention_mask,
        )

        print(f"    Gene logits shape: {outputs['gene_logits'].shape}")
        print(f"    Expr bin logits shape: {outputs['expr_bin_logits'].shape}")
        print(f"    Expr level logits shape: {outputs['expr_level_logits'].shape}")


    print("\n" + "=" * 80)
    print("  DETAILED TEST: MultiScalePathwayEncoder")
    print("=" * 80)

    config = PCGEDConfig(
        n_pathways=2000,
        d_model=768,
        n_pathway_tokens=8,
        n_global_pathway_tokens=2,
        pathway_hidden_dim=2048,
        pathway_fusion_layers=2,
        dropout=0.1,
        n_heads=12,
    )

    encoder = MultiScalePathwayEncoder(config)
    print(f"\n  Encoder parameters: {sum(p.numel() for p in encoder.parameters()):,}")
    print(f"  - n_global_tokens: {encoder.n_global_tokens}")
    print(f"  - n_local_tokens: {encoder.n_local_tokens}")
    print(f"  - pathways_per_local: {encoder.pathways_per_local}")


    pathway_emb = torch.randn(2, 2000)
    tokens = encoder(pathway_emb)
    print(f"\n  Forward pass:")
    print(f"  Input shape: {pathway_emb.shape}")
    print(f"  Output shape: {tokens.shape}")


    input_dim = 2000
    output_dim = config.n_pathway_tokens * config.d_model
    print(f"\n  Information capacity:")
    print(f"  Input dimension: {input_dim}")
    print(f"  Output dimension: {output_dim}")
    print(f"  Expansion ratio: {output_dim / input_dim:.2f}x")
    print(f"  OK No information bottleneck!" if output_dim >= input_dim else "  WARN Potential bottleneck")


    print("\n  Testing generation with MultiScale encoder...")
    model = create_model(
        n_genes=60000,
        n_pathways=2000,
        model_size='small',
        pathway_encoder_type='multiscale',
        n_pathway_tokens=8,
    )

    generated = model.generate(
        pathway_embedding=pathway_emb[:1],
        max_length=50,
        temperature=0.8,
        top_k=50,
    )

    print(f"  Generated genes shape: {generated['gene_ids'].shape}")
    print(f"  Generated expr bins shape: {generated['expr_bins'].shape}")
    print(f"  Sequence lengths: {generated['sequence_lengths']}")

    print("\n All tests completed!")






