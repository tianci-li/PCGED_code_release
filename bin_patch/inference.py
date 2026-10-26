



import sys
import json
import yaml
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional




PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))





def _install_tokenizer() -> None:
    import importlib
    from bin_patch.tokenizer import GeneTokenizers as FixedGeneTokenizers
    orig_mod = importlib.import_module("src.data.tokenizer")
    orig_mod.GeneTokenizers = FixedGeneTokenizers
    print("[bin_patch] Installed GeneTokenizers")


_install_tokenizer()


from src.model.model_pathway_aux import PCGED, PCGEDConfig


_DEFAULT_CONFIG_PATH = str(
    PROJECT_ROOT / "configs" / "train.yaml"
)


class PCGEDInference:


    def __init__(
        self,
        checkpoint_path: str,
        config_path: str = _DEFAULT_CONFIG_PATH,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        multi_gpu: bool = False,
    ):
        self.device = torch.device(device)
        self.dtype = dtype

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        self.raw_config = config

        gene_list_path = config["data"]["gene_location_list_path"]
        with open(gene_list_path, "r") as f:
            self.gene_list = json.load(f)

        print(f" Loading checkpoint from {checkpoint_path} ...")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)


        keys_to_remove = [k for k in state_dict if "rope." in k and "_cached" in k]
        for k in keys_to_remove:
            del state_dict[k]


        model_config = dict(config["model"])
        d_model = model_config.get("d_model", 2048)
        embed_dim = min(64, d_model // 4)
        global_enc_key = "pathway_encoder.global_encoder.0.weight"
        if global_enc_key in state_dict:
            global_input_dim = state_dict[global_enc_key].shape[1]
            actual_n_pathways = global_input_dim // embed_dim
            cfg_np = model_config.get("n_pathways", 2000)
            if actual_n_pathways != cfg_np:
                print(f"   WARN  n_pathways: config={cfg_np}, ckpt={actual_n_pathways} ->   ckpt")
                model_config["n_pathways"] = actual_n_pathways


        expr_bin_emb_key = "gene_expr_embedding.expr_bin_embedding.weight"
        if expr_bin_emb_key in state_dict:
            actual_bins = state_dict[expr_bin_emb_key].shape[0]
            cfg_bins = model_config.get("n_expr_bins", 26)
            if actual_bins != cfg_bins:
                print(f"   WARN  n_expr_bins: config={cfg_bins}, ckpt={actual_bins} ->   ckpt")
                model_config["n_expr_bins"] = actual_bins

        self.pcged_config = self._build_pcged_config(model_config)
        self.model = PCGED(self.pcged_config)
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"   WARN  missing keys (     ): {len(missing)}  ,   : {missing[:3]}")
        if unexpected:
            print(f"   WARN  unexpected keys (   ): {len(unexpected)}  ,   : {unexpected[:3]}")

        self.model.to(device=self.device, dtype=self.dtype)
        self.model.eval()


        total_steps = self.pcged_config.pathway_masking_total_steps
        self.model.set_masking_step(total_steps, total_steps)

        step_info = ckpt.get("global_step", "unknown")
        opt_step = ckpt.get("optimizer_step", "unknown")
        masking_offset = ckpt.get("masking_step_offset", 0)
        best_val = ckpt.get("best_val_loss", "unknown")
        print(
            f"OK Model loaded (global_step={step_info}, opt_step={opt_step}, "
            f"masking_offset={masking_offset}, best_val_loss={best_val})"
        )
        print(
            f"   n_pathways={self.pcged_config.n_pathways}, "
            f"n_expr_bins={self.pcged_config.n_expr_bins}, "
            f"aux_head={self.pcged_config.pathway_aux_head_enabled}, "
            f"device={self.device}, dtype={self.dtype}"
        )
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"   Parameters: {n_params / 1e9:.2f}B")

        self.N_SPECIAL = self.pcged_config.n_special_tokens
        self.N_GENES = len(self.gene_list)

        n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self.multi_gpu = multi_gpu and n_gpus > 1
        if self.multi_gpu:
            self.gpu_ids = list(range(n_gpus))
            print(f"   Multi-GPU enabled: {n_gpus} GPUs {self.gpu_ids}")
        else:
            self.gpu_ids = [0]

    def _build_pcged_config(self, mc: dict) -> PCGEDConfig:

        return PCGEDConfig(
            n_genes=mc.get("n_genes", 60664),
            n_expr_bins=mc.get("n_expr_bins", 26),
            n_expr_levels=mc.get("n_expr_levels", 9),
            n_pathways=mc.get("n_pathways", 2000),
            d_model=mc.get("d_model", 2048),
            n_layers=mc.get("n_layers", 24),
            n_heads=mc.get("n_heads", 16),
            d_ff=mc.get("d_ff", 5504),
            dropout=0.0,
            max_seq_len=mc.get("max_seq_len", 8192),
            pathway_encoder_type=mc.get("pathway_encoder_type", "discrete"),
            n_pathway_tokens=mc.get("n_pathway_tokens", 8),
            n_global_pathway_tokens=mc.get("n_global_pathway_tokens", 2),
            pathway_hidden_dim=mc.get("pathway_hidden_dim", 2048),
            pathway_fusion_layers=mc.get("pathway_fusion_layers", 2),
            use_flash_attention=mc.get("use_flash_attention", True),
            use_gradient_checkpointing=False,
            tie_word_embeddings=mc.get("tie_word_embeddings", False),
            pathway_masking_enabled=mc.get("pathway_masking_enabled", True),
            pathway_masking_total_steps=mc.get("pathway_masking_total_steps", 15000),
            pathway_curriculum_phase0_end=mc.get("pathway_curriculum_phase0_end", 0.333),
            pathway_curriculum_phase1_end=mc.get("pathway_curriculum_phase1_end", 0.500),
            pathway_curriculum_phase2_end=mc.get("pathway_curriculum_phase2_end", 0.667),
            pathway_curriculum_k_phase1=mc.get("pathway_curriculum_k_phase1", 5),
            pathway_curriculum_k_max_ratio=mc.get("pathway_curriculum_k_max_ratio", 1.0),
            pathway_curriculum_phase3_dropout=mc.get("pathway_curriculum_phase3_dropout", 0.05),
            pathway_interpolation_enabled=mc.get("pathway_interpolation_enabled", False),
            pathway_interpolation_mode=mc.get("pathway_interpolation_mode", "off"),
            pathway_interpolation_strategy=mc.get("pathway_interpolation_strategy", "linear"),
            pathway_interpolation_mix_ratio=mc.get("pathway_interpolation_mix_ratio", 0.5),
            pathway_aux_head_enabled=mc.get("pathway_aux_head_enabled", True),
        )





    @torch.no_grad()
    def generate(
        self,
        pathway_bins: torch.Tensor,
        pathway_scores: Optional[torch.Tensor] = None,
        max_length: int = 8192,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True,
        batch_size: int = 16,
    ) -> List[Dict]:
        return self._generate_on_device(
            self.model, pathway_bins, pathway_scores, self.device,
            max_length, temperature, top_k, top_p, do_sample, batch_size,
        )

    @torch.no_grad()
    def _generate_on_device(
        self,
        model: nn.Module,
        pathway_bins: torch.Tensor,
        pathway_scores: Optional[torch.Tensor],
        device: torch.device,
        max_length: int,
        temperature: float,
        top_k: int,
        top_p: float,
        do_sample: bool,
        batch_size: int,
    ) -> List[Dict]:
        N = pathway_bins.shape[0]
        all_results: List[Dict] = []

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch_pw = pathway_bins[start:end].to(device=device, dtype=torch.long)
            batch_scores = None
            if pathway_scores is not None:
                batch_scores = pathway_scores[start:end].to(device=device, dtype=torch.float32)

            outputs = model.generate(
                pathway_embedding=batch_pw,
                pathway_scores=batch_scores,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample,
            )

            gene_ids = outputs["gene_ids"].cpu()
            expr_bins = outputs["expr_bins"].cpu()
            expr_levels = outputs["expr_levels"].cpu()

            eos_id = self.pcged_config.eos_token_id

            for i in range(gene_ids.shape[0]):
                eos_mask = (gene_ids[i] == eos_id)
                if eos_mask.any():
                    seq_len = eos_mask.nonzero(as_tuple=True)[0][0].item()
                else:
                    seq_len = gene_ids.shape[1]
                result = self._decode_single(
                    gene_ids[i, :seq_len],
                    expr_bins[i, :seq_len],
                    expr_levels[i, :seq_len],
                )
                all_results.append(result)

            done = len(all_results)
            if done % 50 < batch_size or done == N:
                print(f"   [cuda:{device.index or 0}] Generated {done}/{N} cells ...")

        return all_results

    def _decode_single(
        self,
        gene_token_ids: torch.Tensor,
        expr_bin_token_ids: torch.Tensor,
        expr_level_token_ids: torch.Tensor,
    ) -> Dict:
        gene_indices: List[int] = []
        expr_bins: List[int] = []
        expr_levels: List[int] = []

        for j in range(len(gene_token_ids)):
            g_tok = gene_token_ids[j].item()
            b_tok = expr_bin_token_ids[j].item()
            l_tok = expr_level_token_ids[j].item()

            gene_idx = g_tok - self.N_SPECIAL
            if gene_idx < 0 or gene_idx >= self.N_GENES:
                continue
            bin_val = max(b_tok - self.N_SPECIAL, 0)
            level_val = max(l_tok - self.N_SPECIAL, 0)

            gene_indices.append(gene_idx)
            expr_bins.append(bin_val)
            expr_levels.append(level_val)

        expression_vector = np.zeros(self.N_GENES, dtype=np.float32)
        for idx, bval in zip(gene_indices, expr_bins):
            expression_vector[idx] = max(expression_vector[idx], float(bval))

        gene_names = [
            self.gene_list[idx] if idx < len(self.gene_list) else f"UNKNOWN_{idx}"
            for idx in gene_indices
        ]
        return {
            "gene_indices": gene_indices,
            "gene_names": gene_names,
            "expr_bins": expr_bins,
            "expr_levels": expr_levels,
            "n_genes": len(gene_indices),
            "expression_vector": expression_vector,
        }

    @torch.no_grad()
    def extract_embeddings(
        self,
        samples: List[Dict],
        batch_size: int = 8,
    ) -> np.ndarray:
        all_embs = []
        n_cond = self.pcged_config.n_pathway_tokens + 1

        for start in range(0, len(samples), batch_size):
            end = min(start + batch_size, len(samples))
            batch = samples[start:end]

            max_sl = 0
            for s in batch:
                sl = s["seq_length"]
                if isinstance(sl, torch.Tensor):
                    sl = sl.item()
                max_sl = max(max_sl, int(sl))
            trunc_len = min(max_sl + 2, self.pcged_config.max_seq_len)

            pw = torch.stack([s["pathway_embedding"] for s in batch]).to(
                device=self.device, dtype=torch.long)
            gt = torch.stack([s["gene_tokens"][:trunc_len] for s in batch]).to(
                device=self.device, dtype=torch.long)
            ebt = torch.stack([s["expr_bin_tokens"][:trunc_len] for s in batch]).to(
                device=self.device, dtype=torch.long)
            elt = torch.stack([s["expr_level_tokens"][:trunc_len] for s in batch]).to(
                device=self.device, dtype=torch.long)
            amask = torch.stack([s["attention_mask"][:trunc_len] for s in batch]).to(
                device=self.device)

            outputs = self.model(
                pathway_embedding=pw,
                gene_tokens=gt,
                expr_bin_tokens=ebt,
                expr_level_tokens=elt,
                attention_mask=amask,
                return_loss=False,
            )

            hidden = outputs["hidden_states"]
            gene_hidden = hidden[:, n_cond:, :]
            amask_f = amask.float().unsqueeze(-1)
            pooled = (gene_hidden.float() * amask_f).sum(dim=1)
            lengths = amask_f.sum(dim=1).clamp(min=1)
            pooled = pooled / lengths



            all_embs.append(np.array(pooled.detach().cpu().tolist(), dtype=np.float32))

            if end % 100 < batch_size or end == len(samples):
                print(f"   Extracted embeddings: {end}/{len(samples)}")

        return np.vstack(all_embs)

    def _build_generated_samples(
        self,
        gen_results: List[Dict],
        pathway_bins: torch.Tensor,
    ) -> List[Dict]:
        max_len = self.pcged_config.max_seq_len
        N_SP = self.N_SPECIAL
        start_id = self.pcged_config.sep_token_id
        end_id = self.pcged_config.eos_token_id

        samples = []
        for i, result in enumerate(gen_results):
            gi = result["gene_indices"]
            eb = result["expr_bins"]
            el = result["expr_levels"]
            n_genes = len(gi)

            gene_tok = torch.zeros(max_len, dtype=torch.long)
            ebin_tok = torch.zeros(max_len, dtype=torch.long)
            elvl_tok = torch.zeros(max_len, dtype=torch.long)
            amask = torch.zeros(max_len, dtype=torch.long)

            gene_tok[0] = start_id
            ebin_tok[0] = start_id
            elvl_tok[0] = start_id
            amask[0] = 1

            actual = min(n_genes, max_len - 2)
            for j in range(actual):
                p = j + 1
                gene_tok[p] = gi[j] + N_SP
                ebin_tok[p] = eb[j] + N_SP
                elvl_tok[p] = el[j] + N_SP
                amask[p] = 1

            sep_pos = actual + 1
            if sep_pos < max_len:
                gene_tok[sep_pos] = end_id
                ebin_tok[sep_pos] = end_id
                elvl_tok[sep_pos] = end_id
                amask[sep_pos] = 1

            pw = pathway_bins[i]
            if not isinstance(pw, torch.Tensor):
                pw = torch.tensor(pw, dtype=torch.long)

            samples.append({
                "pathway_embedding": pw,
                "gene_tokens": gene_tok,
                "expr_bin_tokens": ebin_tok,
                "expr_level_tokens": elvl_tok,
                "attention_mask": amask,
                "seq_length": actual,
            })

        return samples

    @staticmethod
    def real_sample_to_expression_vector(
        sample: Dict,
        n_genes: int = 60664,
        n_special: int = 5,
    ) -> np.ndarray:
        gene_tokens = sample["gene_tokens"]
        expr_bin_tokens = sample["expr_bin_tokens"]
        seq_len = sample["seq_length"]

        if isinstance(gene_tokens, torch.Tensor):
            gene_tokens = gene_tokens.tolist()
        if isinstance(expr_bin_tokens, torch.Tensor):
            expr_bin_tokens = expr_bin_tokens.tolist()

        vec = np.zeros(n_genes, dtype=np.float32)
        for pos in range(1, min(seq_len + 1, len(gene_tokens))):
            g_tok = gene_tokens[pos]
            b_tok = expr_bin_tokens[pos]
            gene_idx = g_tok - n_special
            bin_val = max(b_tok - n_special, 0)
            if 0 <= gene_idx < n_genes:
                vec[gene_idx] = max(vec[gene_idx], float(bin_val))
        return vec
