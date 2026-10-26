


import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional, Any

from .pipeline_model import pack_pipeline_input


class PipelineDataset(Dataset):


    def __init__(
        self,
        base_dataset: Dataset,
        n_pathways: int,
    ):

        self.base_dataset = base_dataset
        self.n_pathways = n_pathways

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:

        sample = self.base_dataset[idx]


        pathway_embedding = sample['pathway_embedding']
        gene_tokens = sample['gene_tokens']
        expr_bin_tokens = sample['expr_bin_tokens']
        expr_level_tokens = sample['expr_level_tokens']
        attention_mask = sample['attention_mask']


        if not isinstance(pathway_embedding, torch.Tensor):
            pathway_embedding = torch.tensor(pathway_embedding, dtype=torch.float)
        if not isinstance(gene_tokens, torch.Tensor):
            gene_tokens = torch.tensor(gene_tokens, dtype=torch.long)
        if not isinstance(expr_bin_tokens, torch.Tensor):
            expr_bin_tokens = torch.tensor(expr_bin_tokens, dtype=torch.long)
        if not isinstance(expr_level_tokens, torch.Tensor):
            expr_level_tokens = torch.tensor(expr_level_tokens, dtype=torch.long)
        if not isinstance(attention_mask, torch.Tensor):
            attention_mask = torch.tensor(attention_mask, dtype=torch.float)


        pathway_embedding = pathway_embedding.unsqueeze(0)
        gene_tokens = gene_tokens.unsqueeze(0)
        expr_bin_tokens = expr_bin_tokens.unsqueeze(0)
        expr_level_tokens = expr_level_tokens.unsqueeze(0)
        attention_mask = attention_mask.unsqueeze(0)


        packed_input = pack_pipeline_input(
            pathway_embedding,
            gene_tokens,
            expr_bin_tokens,
            expr_level_tokens,
            attention_mask,
        ).squeeze(0)



        L = gene_tokens.shape[1]
        labels = torch.stack([
            gene_tokens.squeeze(0)[1:],
            expr_bin_tokens.squeeze(0)[1:],
            expr_level_tokens.squeeze(0)[1:],
        ], dim=-1)

        return packed_input, labels


def pipeline_collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor]],
    pad_value: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:

    packed_inputs, labels_list = zip(*batch)


    max_input_len = max(inp.shape[0] for inp in packed_inputs)
    max_label_len = max(lab.shape[0] for lab in labels_list)

    B = len(batch)


    batched_inputs = torch.full((B, max_input_len), pad_value, dtype=packed_inputs[0].dtype)
    batched_labels = torch.zeros((B, max_label_len, 3), dtype=labels_list[0].dtype)

    for i, (inp, lab) in enumerate(zip(packed_inputs, labels_list)):
        batched_inputs[i, :inp.shape[0]] = inp
        batched_labels[i, :lab.shape[0], :] = lab

    return batched_inputs, batched_labels


class PipelineDataCollator:


    def __init__(
        self,
        n_pathways: int,
        pad_token_id: int = 0,
    ):
        self.n_pathways = n_pathways
        self.pad_token_id = pad_token_id

    def __call__(
        self,
        batch: List[Dict[str, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:


        pathway_embeddings = []
        gene_tokens_list = []
        expr_bin_tokens_list = []
        expr_level_tokens_list = []
        attention_masks = []

        for sample in batch:
            pathway_embeddings.append(sample['pathway_embedding'])
            gene_tokens_list.append(sample['gene_tokens'])
            expr_bin_tokens_list.append(sample['expr_bin_tokens'])
            expr_level_tokens_list.append(sample['expr_level_tokens'])
            attention_masks.append(sample['attention_mask'])


        max_len = max(t.shape[0] for t in gene_tokens_list)
        B = len(batch)


        padded_gene = torch.zeros(B, max_len, dtype=torch.long)
        padded_expr_bin = torch.zeros(B, max_len, dtype=torch.long)
        padded_expr_level = torch.zeros(B, max_len, dtype=torch.long)
        padded_mask = torch.zeros(B, max_len, dtype=torch.float)

        for i in range(B):
            L = gene_tokens_list[i].shape[0]
            padded_gene[i, :L] = gene_tokens_list[i]
            padded_expr_bin[i, :L] = expr_bin_tokens_list[i]
            padded_expr_level[i, :L] = expr_level_tokens_list[i]
            padded_mask[i, :L] = attention_masks[i]


        pathway_emb = torch.stack(pathway_embeddings, dim=0)


        packed_inputs = pack_pipeline_input(
            pathway_emb,
            padded_gene,
            padded_expr_bin,
            padded_expr_level,
            padded_mask,
        )


        labels = torch.stack([
            padded_gene[:, 1:],
            padded_expr_bin[:, 1:],
            padded_expr_level[:, 1:],
        ], dim=-1)

        return packed_inputs, labels


def create_pipeline_dataloader(
    dataset: Dataset,
    n_pathways: int,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = True,
) -> torch.utils.data.DataLoader:

    collator = PipelineDataCollator(n_pathways=n_pathways)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collator,
    )

    return dataloader






class RepeatingLoader:


    def __init__(self, dataloader: torch.utils.data.DataLoader):
        self.dataloader = dataloader
        self._iterator = None

    def __iter__(self):
        return self

    def __next__(self):
        if self._iterator is None:
            self._iterator = iter(self.dataloader)

        try:
            batch = next(self._iterator)
        except StopIteration:
            self._iterator = iter(self.dataloader)
            batch = next(self._iterator)

        return batch






if __name__ == '__main__':
    print("=" * 80)
    print(" TESTING PIPELINE DATA UTILS")
    print("=" * 80)


    class MockDataset(Dataset):
        def __init__(self, n_samples: int = 100, n_pathways: int = 2000, seq_len: int = 100):
            self.n_samples = n_samples
            self.n_pathways = n_pathways
            self.seq_len = seq_len

        def __len__(self):
            return self.n_samples

        def __getitem__(self, idx):
            return {
                'pathway_embedding': torch.randn(self.n_pathways),
                'gene_tokens': torch.randint(5, 60000, (self.seq_len,)),
                'expr_bin_tokens': torch.randint(0, 16, (self.seq_len,)),
                'expr_level_tokens': torch.randint(0, 9, (self.seq_len,)),
                'attention_mask': torch.ones(self.seq_len),
            }


    print("\n  Testing PipelineDataset...")
    base_dataset = MockDataset(n_samples=10)
    pipeline_dataset = PipelineDataset(base_dataset, n_pathways=2000)

    packed_input, labels = pipeline_dataset[0]
    print(f"   Packed input shape: {packed_input.shape}")
    print(f"   Labels shape: {labels.shape}")


    print("\n  Testing create_pipeline_dataloader...")
    dataloader = create_pipeline_dataloader(
        MockDataset(n_samples=16),
        n_pathways=2000,
        batch_size=4,
        num_workers=0,
    )

    batch_inputs, batch_labels = next(iter(dataloader))
    print(f"   Batch inputs shape: {batch_inputs.shape}")
    print(f"   Batch labels shape: {batch_labels.shape}")


    print("\n  Testing RepeatingLoader...")
    repeating_loader = RepeatingLoader(dataloader)

    for i, (inputs, labels) in enumerate(repeating_loader):
        if i >= 3:
            break
        print(f"   Iteration {i}: inputs {inputs.shape}, labels {labels.shape}")

    print("\n All tests completed!")
