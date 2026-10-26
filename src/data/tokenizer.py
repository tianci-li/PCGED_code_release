import json
import torch
from typing import List, Dict, Union, Optional
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

class GeneTokenizers:


    def __init__(self, gene_location_list_path: str, max_vocab_size: int = 50000):


        with open(gene_location_list_path, 'r') as f:
            self.gene_location_list = json.load(f)


        self.gene_tokenizer = self._build_gene_tokenizer()
        self.expr_bin_tokenizer = self._build_expr_bin_tokenizer()
        self.expr_level_tokenizer = self._build_expr_level_tokenizer()

    def _build_gene_tokenizer(self) -> Tokenizer:


        gene_vocab = {}


        special_tokens = ["[PAD]", "[CLS]", "[SEP]", "[UNK]", "[MASK]"]
        for i, token in enumerate(special_tokens):
            gene_vocab[token] = i


        for i, location in enumerate(self.gene_location_list):
            gene_key = f"GENE_{i}_{location}"
            gene_vocab[gene_key] = len(gene_vocab)


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


        bin_vocab = {}


        special_tokens = ["[PAD]", "[CLS]", "[SEP]", "[UNK]", "[MASK]"]
        for i, token in enumerate(special_tokens):
            bin_vocab[token] = i


        for i in range(11):
            bin_vocab[f"BIN_{i}"] = len(bin_vocab)

        tokenizer = Tokenizer(WordLevel(bin_vocab, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()

        tokenizer.post_processor = TemplateProcessing(
            single="[CLS] $A [SEP]",
            special_tokens=[
                ("[CLS]", bin_vocab["[CLS]"]),
                ("[SEP]", bin_vocab["[SEP]"]),
            ],
        )

        return tokenizer

    def _build_expr_level_tokenizer(self) -> Tokenizer:


        level_vocab = {}


        special_tokens = ["[PAD]", "[CLS]", "[SEP]", "[UNK]", "[MASK]"]
        for i, token in enumerate(special_tokens):
            level_vocab[token] = i


        for i in range(4):
            level_vocab[f"LEVEL_{i}"] = len(level_vocab)

        tokenizer = Tokenizer(WordLevel(level_vocab, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()

        tokenizer.post_processor = TemplateProcessing(
            single="[CLS] $A [SEP]",
            special_tokens=[
                ("[CLS]", level_vocab["[CLS]"]),
                ("[SEP]", level_vocab["[SEP]"]),
            ],
        )

        return tokenizer

    def get_gene_locations(self, gene_ids: List[int]) -> List[str]:

        gene_locations = []
        for gene_id in gene_ids:
            if gene_id < len(self.gene_location_list):
                gene_locations.append(self.gene_location_list[gene_id])
            else:
                gene_locations.append("unknown_location")
        return gene_locations

    def encode_genes(self, gene_ids: List[int], max_length: Optional[int] = None) -> torch.Tensor:

        gene_locations = self.get_gene_locations(gene_ids)
        gene_tokens = []

        for i, location in enumerate(gene_locations):
            gene_token = f"GENE_{gene_ids[i]}_{location}"
            gene_tokens.append(gene_token)


        gene_text = " ".join(gene_tokens)


        encoding = self.gene_tokenizer.encode(gene_text)
        tokens = encoding.ids


        if max_length:
            if len(tokens) > max_length:
                tokens = tokens[:max_length]
            else:
                tokens.extend([self.gene_tokenizer.token_to_id("[PAD]")] * (max_length - len(tokens)))

        return torch.tensor(tokens, dtype=torch.long)

    def encode_expr_bins(self, expr_bins: List[int], max_length: Optional[int] = None) -> torch.Tensor:

        bin_tokens = [f"BIN_{bin_val}" for bin_val in expr_bins]
        bin_text = " ".join(bin_tokens)

        encoding = self.expr_bin_tokenizer.encode(bin_text)
        tokens = encoding.ids

        if max_length:
            if len(tokens) > max_length:
                tokens = tokens[:max_length]
            else:
                tokens.extend([self.expr_bin_tokenizer.token_to_id("[PAD]")] * (max_length - len(tokens)))

        return torch.tensor(tokens, dtype=torch.long)

    def encode_expr_levels(self, expr_levels: List[int], max_length: Optional[int] = None) -> torch.Tensor:

        level_tokens = [f"LEVEL_{level_val}" for level_val in expr_levels]
        level_text = " ".join(level_tokens)

        encoding = self.expr_level_tokenizer.encode(level_text)
        tokens = encoding.ids

        if max_length:
            if len(tokens) > max_length:
                tokens = tokens[:max_length]
            else:
                tokens.extend([self.expr_level_tokenizer.token_to_id("[PAD]")] * (max_length - len(tokens)))

        return torch.tensor(tokens, dtype=torch.long)

    def decode_genes(self, tokens: torch.Tensor) -> List[str]:

        token_ids = tokens.tolist() if isinstance(tokens, torch.Tensor) else tokens
        decoded = self.gene_tokenizer.decode(token_ids)
        return decoded.split()

    def decode_expr_bins(self, tokens: torch.Tensor) -> List[str]:

        token_ids = tokens.tolist() if isinstance(tokens, torch.Tensor) else tokens
        decoded = self.expr_bin_tokenizer.decode(token_ids)
        return decoded.split()

    def decode_expr_levels(self, tokens: torch.Tensor) -> List[str]:

        token_ids = tokens.tolist() if isinstance(tokens, torch.Tensor) else tokens
        decoded = self.expr_level_tokenizer.decode(token_ids)
        return decoded.split()

    def get_vocab_sizes(self) -> Dict[str, int]:

        return {
            'gene_vocab_size': self.gene_tokenizer.get_vocab_size(),
            'expr_bin_vocab_size': self.expr_bin_tokenizer.get_vocab_size(),
            'expr_level_vocab_size': self.expr_level_tokenizer.get_vocab_size()
        }

    def save_tokenizers(self, save_dir: str):

        import os
        os.makedirs(save_dir, exist_ok=True)

        self.gene_tokenizer.save(os.path.join(save_dir, "gene_tokenizer.json"))
        self.expr_bin_tokenizer.save(os.path.join(save_dir, "expr_bin_tokenizer.json"))
        self.expr_level_tokenizer.save(os.path.join(save_dir, "expr_level_tokenizer.json"))

    @classmethod
    def load_tokenizers(cls, save_dir: str, gene_location_list_path: str):

        import os
        instance = cls.__new__(cls)

        with open(gene_location_list_path, 'r') as f:
            instance.gene_location_list = json.load(f)

        instance.gene_tokenizer = Tokenizer.from_file(os.path.join(save_dir, "gene_tokenizer.json"))
        instance.expr_bin_tokenizer = Tokenizer.from_file(os.path.join(save_dir, "expr_bin_tokenizer.json"))
        instance.expr_level_tokenizer = Tokenizer.from_file(os.path.join(save_dir, "expr_level_tokenizer.json"))

        return instance