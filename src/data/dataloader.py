import torch
from torch.utils.data import DataLoader, DistributedSampler
from typing import Dict, List, Optional, Any
import numpy as np






class ResumableDistributedSampler(DistributedSampler):


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._start_index: int = 0

    def set_start_index(self, start_index: int):

        self._start_index = start_index

    def __iter__(self):
        full_indices = list(super().__iter__())
        if self._start_index > 0 and self._start_index < len(full_indices):
            return iter(full_indices[self._start_index:])
        return iter(full_indices)

    def __len__(self):
        full_len = super().__len__()
        if self._start_index > 0:
            return max(0, full_len - self._start_index)
        return full_len


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:


    batch = [sample for sample in batch if sample['seq_length'] > 0]

    if len(batch) == 0:

        return {
            'gene_tokens': torch.zeros(1, 1, dtype=torch.long),
            'expr_bin_tokens': torch.zeros(1, 1, dtype=torch.long),
            'expr_level_tokens': torch.zeros(1, 1, dtype=torch.long),
            'pathway_embedding': torch.zeros(1, 1, dtype=torch.long),
            'attention_mask': torch.zeros(1, 1, dtype=torch.bool),
            'seq_lengths': torch.tensor([0], dtype=torch.long),
            'cell_types': [None],
        }

    batch_size = len(batch)


    max_seq_len = max(sample['seq_length'] for sample in batch)


    n_pathways = max(sample['pathway_embedding'].shape[0] for sample in batch)

    has_pathway_scores = any(('pathway_scores' in sample and sample['pathway_scores'] is not None) for sample in batch)


    gene_tokens = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
    expr_bin_tokens = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
    expr_level_tokens = torch.zeros(batch_size, max_seq_len, dtype=torch.long)

    pathway_embeddings = torch.zeros(batch_size, n_pathways, dtype=torch.long)
    pathway_scores = torch.zeros(batch_size, n_pathways, dtype=torch.float32) if has_pathway_scores else None
    attention_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.bool)
    seq_lengths = torch.zeros(batch_size, dtype=torch.long)
    cell_types = []


    for i, sample in enumerate(batch):
        seq_len = sample['seq_length']
        seq_lengths[i] = seq_len


        gene_tokens[i, :seq_len] = sample['gene_tokens'][:seq_len]
        expr_bin_tokens[i, :seq_len] = sample['expr_bin_tokens'][:seq_len]
        expr_level_tokens[i, :seq_len] = sample['expr_level_tokens'][:seq_len]


        attention_mask[i, :seq_len] = True


        pw_dim = sample['pathway_embedding'].shape[0]
        if pw_dim == n_pathways:
            pathway_embeddings[i] = sample['pathway_embedding']
        else:

            pathway_embeddings[i, :min(pw_dim, n_pathways)] = sample['pathway_embedding'][:min(pw_dim, n_pathways)]

        if has_pathway_scores and ('pathway_scores' in sample) and (sample['pathway_scores'] is not None):
            pw_scores = sample['pathway_scores']
            score_dim = pw_scores.shape[0]
            pathway_scores[i, :min(score_dim, n_pathways)] = pw_scores[:min(score_dim, n_pathways)]


        cell_types.append(sample['cell_type'])

    result = {
        'gene_tokens': gene_tokens,
        'expr_bin_tokens': expr_bin_tokens,
        'expr_level_tokens': expr_level_tokens,
        'pathway_embedding': pathway_embeddings,
        'attention_mask': attention_mask,
        'seq_lengths': seq_lengths,
        'cell_types': cell_types,
    }

    if has_pathway_scores:
        result['pathway_scores'] = pathway_scores

    return result


def collate_fn_fixed_length(batch: List[Dict[str, Any]], fixed_length: int = 2048) -> Dict[str, torch.Tensor]:


    batch = [sample for sample in batch if sample['seq_length'] > 0]

    if len(batch) == 0:

        return {
            'gene_tokens': torch.zeros(1, fixed_length, dtype=torch.long),
            'expr_bin_tokens': torch.zeros(1, fixed_length, dtype=torch.long),
            'expr_level_tokens': torch.zeros(1, fixed_length, dtype=torch.long),
            'pathway_embedding': torch.zeros(1, 1, dtype=torch.float32),
            'attention_mask': torch.zeros(1, fixed_length, dtype=torch.bool),
            'seq_lengths': torch.tensor([0], dtype=torch.long),
            'cell_types': [None],
        }

    batch_size = len(batch)


    n_pathways = batch[0]['pathway_embedding'].shape[0]

    has_pathway_scores = any(('pathway_scores' in sample and sample['pathway_scores'] is not None) for sample in batch)


    gene_tokens = torch.zeros(batch_size, fixed_length, dtype=torch.long)
    expr_bin_tokens = torch.zeros(batch_size, fixed_length, dtype=torch.long)
    expr_level_tokens = torch.zeros(batch_size, fixed_length, dtype=torch.long)
    pathway_embeddings = torch.zeros(batch_size, n_pathways, dtype=torch.float32)
    pathway_scores = torch.zeros(batch_size, n_pathways, dtype=torch.float32) if has_pathway_scores else None
    attention_mask = torch.zeros(batch_size, fixed_length, dtype=torch.bool)
    seq_lengths = torch.zeros(batch_size, dtype=torch.long)
    cell_types = []


    for i, sample in enumerate(batch):
        seq_len = min(sample['seq_length'], fixed_length)
        seq_lengths[i] = seq_len


        gene_tokens[i, :seq_len] = sample['gene_tokens'][:seq_len]
        expr_bin_tokens[i, :seq_len] = sample['expr_bin_tokens'][:seq_len]
        expr_level_tokens[i, :seq_len] = sample['expr_level_tokens'][:seq_len]


        attention_mask[i, :seq_len] = True


        pathway_embeddings[i] = sample['pathway_embedding']

        if has_pathway_scores and ('pathway_scores' in sample) and (sample['pathway_scores'] is not None):
            pw_scores = sample['pathway_scores']
            score_dim = pw_scores.shape[0]
            pathway_scores[i, :min(score_dim, n_pathways)] = pw_scores[:min(score_dim, n_pathways)]


        cell_types.append(sample['cell_type'])


    assert gene_tokens.shape[1] == fixed_length, f"gene_tokens shape mismatch: {gene_tokens.shape[1]} != {fixed_length}"

    result = {
        'gene_tokens': gene_tokens,
        'expr_bin_tokens': expr_bin_tokens,
        'expr_level_tokens': expr_level_tokens,
        'pathway_embedding': pathway_embeddings,
        'attention_mask': attention_mask,
        'seq_lengths': seq_lengths,
        'cell_types': cell_types,
    }

    if has_pathway_scores:
        result['pathway_scores'] = pathway_scores

    return result


def collate_fn_with_labels(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:


    collated = collate_fn(batch)

    if collated['seq_lengths'].sum() == 0:
        collated['mlm_labels'] = torch.full_like(collated['gene_tokens'], -100)
        return collated



    collated['mlm_labels'] = torch.full_like(collated['gene_tokens'], -100)

    return collated


class CellDataLoader:


    def __init__(
        self,
        dataset,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
        drop_last: bool = False,
        distributed: bool = False,
        world_size: int = 1,
        rank: int = 0,
        use_mlm_labels: bool = False,
        prefetch_factor: int = 4,
        fixed_seq_length: int = None,
    ):

        self.dataset = dataset
        self.batch_size = batch_size
        self.distributed = distributed
        self.fixed_seq_length = fixed_seq_length


        if fixed_seq_length is not None:

            from functools import partial
            self.collate_fn = partial(collate_fn_fixed_length, fixed_length=fixed_seq_length)

            import os
            rank = int(os.environ.get('RANK', 0))
            print(f"[Rank {rank}] CellDataLoader: using fixed_seq_length = {fixed_seq_length}")
        elif use_mlm_labels:
            self.collate_fn = collate_fn_with_labels
        else:
            self.collate_fn = collate_fn


        if distributed:
            self.sampler = ResumableDistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=shuffle,
                drop_last=drop_last
            )
            shuffle = False
        else:
            self.sampler = None






        self._dl_kwargs = dict(
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=num_workers > 0,
        )
        self._shuffle = shuffle if self.sampler is None else False
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=self._shuffle,
            collate_fn=self.collate_fn,
            sampler=self.sampler,
            **self._dl_kwargs,
        )

    def __iter__(self):
        return iter(self.dataloader)

    def __len__(self):
        return len(self.dataloader)

    def set_epoch(self, epoch: int):

        if self.sampler is not None and hasattr(self.sampler, 'set_epoch'):
            self.sampler.set_epoch(epoch)

    def set_resume_step(self, batch_step: int):

        if isinstance(self.sampler, ResumableDistributedSampler) and batch_step > 0:
            sample_offset = batch_step * self.batch_size
            self.sampler.set_start_index(sample_offset)

    def reset_resume(self):

        if isinstance(self.sampler, ResumableDistributedSampler):
            self.sampler.set_start_index(0)

    def restart_workers(self):

        import gc
        old_dl = self.dataloader
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=self._shuffle,
            collate_fn=self.collate_fn,
            sampler=self.sampler,
            **self._dl_kwargs,
        )
        del old_dl
        gc.collect()


def create_dataloaders(
    train_dataset,
    val_dataset=None,
    test_dataset=None,
    batch_size: int = 32,
    num_workers: int = 4,
    distributed: bool = False,
    world_size: int = 1,
    rank: int = 0,
    use_mlm_labels: bool = False,
    fixed_seq_length: int = None,
) -> Dict[str, 'CellDataLoader']:

    dataloaders = {}


    dataloaders['train'] = CellDataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        distributed=distributed,
        world_size=world_size,
        rank=rank,
        use_mlm_labels=use_mlm_labels,
        fixed_seq_length=fixed_seq_length,
    )





    val_batch_size = batch_size * 2
    if val_dataset is not None:
        dataloaders['val'] = CellDataLoader(
            dataset=val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            distributed=distributed,
            world_size=world_size,
            rank=rank,
            use_mlm_labels=use_mlm_labels,
            fixed_seq_length=fixed_seq_length,
        )


        dataloaders['val']._dl_kwargs['persistent_workers'] = False
        dataloaders['val'].dataloader = DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            collate_fn=dataloaders['val'].collate_fn,
            sampler=dataloaders['val'].sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            prefetch_factor=4,
            persistent_workers=False,
        )


    if test_dataset is not None:
        dataloaders['test'] = CellDataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            distributed=distributed,
            world_size=world_size,
            rank=rank,
            use_mlm_labels=use_mlm_labels,
            fixed_seq_length=fixed_seq_length,
        )

    return dataloaders


if __name__ == '__main__':
    import time
    from dataset import SparseCellDataset

    print("=" * 80)
    print(" TESTING DATALOADER")
    print("=" * 80)


    dataset = SparseCellDataset(
        index_folder='data/sample_csv_index',
        csv_filepath='data/sample_csv',
        global_index='data/line_count.json',
        cell_description='data/raw/cell_descriptions_anonymous.json',
        gene_location_list_path='data/gene_location_list.json',
        max_len=8192,
        anonymous=True,
        pathway_db_path='./pathway_data/pathway_database_ensembl.pkl',
        pathway_method='combined'
    )

    print(f"\n  Dataset size: {len(dataset)}")


    dataloader = CellDataLoader(
        dataset=dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        use_mlm_labels=True,
    )

    print(f"  Number of batches: {len(dataloader)}")


    print("\n  Testing batch iteration...")
    start_time = time.time()

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx == 0:
            print(f"\n  First batch info:")
            print(f"  gene_tokens shape: {batch['gene_tokens'].shape}")
            print(f"  expr_bin_tokens shape: {batch['expr_bin_tokens'].shape}")
            print(f"  expr_level_tokens shape: {batch['expr_level_tokens'].shape}")
            print(f"  pathway_embedding shape: {batch['pathway_embedding'].shape}")
            print(f"  attention_mask shape: {batch['attention_mask'].shape}")
            print(f"  seq_lengths: {batch['seq_lengths']}")
            print(f"  cell_types: {batch['cell_types'][:3]}...")

            if 'mlm_labels' in batch:
                print(f"  mlm_labels shape: {batch['mlm_labels'].shape}")

        if batch_idx >= 4:
            break

    elapsed_time = time.time() - start_time
    print(f"\n    Time for 5 batches: {elapsed_time:.2f}s")
    print(f"  Throughput: {5 * 16 / elapsed_time:.1f} samples/s")

    print("\n DataLoader test completed!")