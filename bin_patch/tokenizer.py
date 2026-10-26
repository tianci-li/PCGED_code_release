

from __future__ import annotations

import json
from typing import Dict, List, Optional

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing





SPECIAL_TOKENS: List[str] = ["[PAD]", "[CLS]", "[SEP]", "[UNK]", "[MASK]"]


EXPR_BIN_MAX = 20
EXPR_BIN_VOCAB: Dict[str, int] = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
for _i in range(EXPR_BIN_MAX + 1):
    EXPR_BIN_VOCAB[f"BIN_{_i}"] = len(EXPR_BIN_VOCAB)



LEGACY_EXPR_BIN_VOCAB: Dict[str, int] = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
for _i in range(11):
    LEGACY_EXPR_BIN_VOCAB[f"BIN_{_i}"] = len(LEGACY_EXPR_BIN_VOCAB)


LEGACY_EXPR_BIN_VOCAB_MAP: Dict[int, int] = {
    legacy_id: EXPR_BIN_VOCAB[token]
    for token, legacy_id in LEGACY_EXPR_BIN_VOCAB.items()
}


EXPR_LEVEL_VOCAB: Dict[str, int] = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
for _i in range(4):
    EXPR_LEVEL_VOCAB[f"LEVEL_{_i}"] = len(EXPR_LEVEL_VOCAB)


class GeneTokenizers:


    def __init__(
        self,
        gene_location_list_path: str,
        max_vocab_size: int = 50000,
    ):
        with open(gene_location_list_path, "r") as f:
            self.gene_location_list = json.load(f)

        self.gene_tokenizer = self._build_gene_tokenizer()
        self.expr_bin_tokenizer = self._build_expr_bin_tokenizer()
        self.expr_level_tokenizer = self._build_expr_level_tokenizer()




    def _build_gene_tokenizer(self) -> Tokenizer:
        gene_vocab: Dict[str, int] = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
        for i, location in enumerate(self.gene_location_list):
            gene_vocab[f"GENE_{i}_{location}"] = len(gene_vocab)

        tokenizer = Tokenizer(WordLevel(gene_vocab, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer.post_processor = TemplateProcessing(
            single="[CLS] $A [SEP]",
            pair="[CLS] $A [SEP] $B:1 [SEP]:1",
            special_tokens=[
                ("[CLS]", gene_vocab["[CLS]"]),
                ("[SEP]", gene_vocab["[SEP]"]),
            ],
        )
        return tokenizer

    def _build_expr_bin_tokenizer(self) -> Tokenizer:

        tokenizer = Tokenizer(WordLevel(dict(EXPR_BIN_VOCAB), unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer.post_processor = TemplateProcessing(
            single="[CLS] $A [SEP]",
            special_tokens=[
                ("[CLS]", EXPR_BIN_VOCAB["[CLS]"]),
                ("[SEP]", EXPR_BIN_VOCAB["[SEP]"]),
            ],
        )
        return tokenizer

    def _build_expr_level_tokenizer(self) -> Tokenizer:
        tokenizer = Tokenizer(WordLevel(dict(EXPR_LEVEL_VOCAB), unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer.post_processor = TemplateProcessing(
            single="[CLS] $A [SEP]",
            special_tokens=[
                ("[CLS]", EXPR_LEVEL_VOCAB["[CLS]"]),
                ("[SEP]", EXPR_LEVEL_VOCAB["[SEP]"]),
            ],
        )
        return tokenizer




    def get_gene_locations(self, gene_ids: List[int]) -> List[str]:
        out: List[str] = []
        for gene_id in gene_ids:
            if gene_id < len(self.gene_location_list):
                out.append(self.gene_location_list[gene_id])
            else:
                out.append("unknown_location")
        return out

    def _pad_to(self, ids: List[int], pad_id: int, max_length: Optional[int]) -> List[int]:
        if max_length is None:
            return ids
        if len(ids) > max_length:
            return ids[:max_length]
        return ids + [pad_id] * (max_length - len(ids))

    def encode_genes(self, gene_ids: List[int], max_length: Optional[int] = None) -> torch.Tensor:
        gene_locations = self.get_gene_locations(gene_ids)
        gene_tokens = [
            f"GENE_{gene_ids[i]}_{loc}" for i, loc in enumerate(gene_locations)
        ]
        encoding = self.gene_tokenizer.encode(" ".join(gene_tokens))
        tokens = self._pad_to(
            encoding.ids,
            self.gene_tokenizer.token_to_id("[PAD]"),
            max_length,
        )
        return torch.tensor(tokens, dtype=torch.long)

    def encode_expr_bins(self, expr_bins: List[int], max_length: Optional[int] = None) -> torch.Tensor:
        bin_tokens = [f"BIN_{int(v)}" for v in expr_bins]
        encoding = self.expr_bin_tokenizer.encode(" ".join(bin_tokens))
        tokens = self._pad_to(
            encoding.ids,
            self.expr_bin_tokenizer.token_to_id("[PAD]"),
            max_length,
        )
        return torch.tensor(tokens, dtype=torch.long)

    def encode_expr_levels(self, expr_levels: List[int], max_length: Optional[int] = None) -> torch.Tensor:
        level_tokens = [f"LEVEL_{int(v)}" for v in expr_levels]
        encoding = self.expr_level_tokenizer.encode(" ".join(level_tokens))
        tokens = self._pad_to(
            encoding.ids,
            self.expr_level_tokenizer.token_to_id("[PAD]"),
            max_length,
        )
        return torch.tensor(tokens, dtype=torch.long)

    def decode_genes(self, tokens: torch.Tensor) -> List[str]:
        token_ids = tokens.tolist() if isinstance(tokens, torch.Tensor) else tokens
        return self.gene_tokenizer.decode(token_ids).split()

    def decode_expr_bins(self, tokens: torch.Tensor) -> List[str]:
        token_ids = tokens.tolist() if isinstance(tokens, torch.Tensor) else tokens
        return self.expr_bin_tokenizer.decode(token_ids).split()

    def decode_expr_levels(self, tokens: torch.Tensor) -> List[str]:
        token_ids = tokens.tolist() if isinstance(tokens, torch.Tensor) else tokens
        return self.expr_level_tokenizer.decode(token_ids).split()




    def get_vocab_sizes(self) -> Dict[str, int]:
        return {
            "gene_vocab_size": self.gene_tokenizer.get_vocab_size(),
            "expr_bin_vocab_size": self.expr_bin_tokenizer.get_vocab_size(),
            "expr_level_vocab_size": self.expr_level_tokenizer.get_vocab_size(),
        }

    def save_tokenizers(self, save_dir: str) -> None:
        import os
        os.makedirs(save_dir, exist_ok=True)
        self.gene_tokenizer.save(os.path.join(save_dir, "gene_tokenizer.json"))
        self.expr_bin_tokenizer.save(os.path.join(save_dir, "expr_bin_tokenizer.json"))
        self.expr_level_tokenizer.save(os.path.join(save_dir, "expr_level_tokenizer.json"))

    @classmethod
    def load_tokenizers(cls, save_dir: str, gene_location_list_path: str) -> "GeneTokenizers":
        import os
        instance = cls.__new__(cls)
        with open(gene_location_list_path, "r") as f:
            instance.gene_location_list = json.load(f)
        instance.gene_tokenizer = Tokenizer.from_file(os.path.join(save_dir, "gene_tokenizer.json"))
        instance.expr_bin_tokenizer = Tokenizer.from_file(os.path.join(save_dir, "expr_bin_tokenizer.json"))
        instance.expr_level_tokenizer = Tokenizer.from_file(os.path.join(save_dir, "expr_level_tokenizer.json"))
        return instance
