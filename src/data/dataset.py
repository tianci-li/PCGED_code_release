import torch
from torch.utils.data import Dataset
import json
from pathlib import Path
import sys
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.csv_global_index import create_global_index, find_file_and_local_index, get_global_len, read_csv_by_local_index, get_line_by_local_index
import numpy as np
from typing import List, Tuple, Optional, Dict
from src.data.tokenizer import GeneTokenizers
from src.pathway.pathway_activation import PathwayActivationAnalyzer












PATHWAY_NUM_BINS = 5



class SparseCellDataset(Dataset):
    def __init__(self, index_folder: str, csv_filepath: str, global_index: str, cell_description: str,
                 gene_location_list_path: str, max_len: int = 8192, anonymous: bool = False,
                 pathway_db_path: str = './pathway_data/pathway_database_ensembl.pkl',
                 pathway_method: str = 'combined',
                 min_pathway_genes: int = 5,
                 max_pathway_genes: int = 500,
                 n_expr_bins: int = 20,
                 use_global_binning: bool = True,
                 pathway_num_bins: int = PATHWAY_NUM_BINS,
                 pathway_bin_edges: Optional[np.ndarray] = None,
                 return_pathway_scores: bool = False):

        self.max_expressed_genes = max_len
        self.index_folder = index_folder
        self.csv_filepath = csv_filepath
        self.global_index = create_global_index(global_index)
        self.anonymous = anonymous
        self.pathway_method = pathway_method
        self.n_expr_bins = n_expr_bins
        self.use_global_binning = use_global_binning
        self.pathway_num_bins = pathway_num_bins
        self.return_pathway_scores = return_pathway_scores




        if pathway_bin_edges is not None:
            self.pathway_bin_edges = np.array(pathway_bin_edges, dtype=np.float32)
        else:

            self.pathway_bin_edges = np.linspace(0, 1, pathway_num_bins + 1)[1:-1].astype(np.float32)

        print(f"  Pathway discretization: {pathway_num_bins} bins, edges={self.pathway_bin_edges}")


        self.tokenizers = GeneTokenizers(gene_location_list_path)
        self.use_gene_importance = False


        with open(gene_location_list_path, 'r', encoding='utf-8') as f:
            self.gene_list = json.load(f)


        self._init_pathway_analyzer(pathway_db_path, min_pathway_genes, max_pathway_genes)

    def _init_pathway_analyzer(self, pathway_db_path: str, min_genes: int, max_genes: int):

        try:
            self.pathway_analyzer = PathwayActivationAnalyzer(
                pathway_db_path=pathway_db_path,
                min_pathway_genes=min_genes,
                max_pathway_genes=max_genes
            )
            self.n_pathways = self.pathway_analyzer.n_pathways
            print(f"OK PathwayActivationAnalyzer initialized with {self.n_pathways} pathways")
        except FileNotFoundError as e:
            print(f"WARN  Failed to initialize PathwayActivationAnalyzer: {e}")
            self.pathway_analyzer = None
            self.n_pathways = 0

    def _discretize_pathway_scores(self, continuous_scores: np.ndarray) -> np.ndarray:



        bin_ids = np.digitize(continuous_scores, self.pathway_bin_edges)
        bin_ids = np.clip(bin_ids, 0, self.pathway_num_bins - 1)
        return bin_ids.astype(np.int64)

    def _compute_pathway_embedding(
        self,
        gene_ids: np.ndarray,
        expression_values: np.ndarray,
        return_scores: bool = False,
    ):


        n_pw = max(self.n_pathways, 1)

        if self.pathway_analyzer is None or len(gene_ids) == 0:

            empty_bins = torch.zeros(n_pw, dtype=torch.long)
            if return_scores:
                return empty_bins, torch.zeros(n_pw, dtype=torch.float32)
            return empty_bins

        try:

            ensembl_ids = [self.gene_list[idx] for idx in gene_ids if idx < len(self.gene_list)]
            expr_values = expression_values[:len(ensembl_ids)].tolist()

            if len(ensembl_ids) == 0:
                empty_bins = torch.zeros(n_pw, dtype=torch.long)
                if return_scores:
                    return empty_bins, torch.zeros(n_pw, dtype=torch.float32)
                return empty_bins


            pathway_scores = self.pathway_analyzer.compute_activation_scores(
                gene_ids=ensembl_ids,
                expression_values=expr_values,
                method=self.pathway_method
            )


            pathway_scores = np.nan_to_num(pathway_scores, nan=0.0, posinf=1.0, neginf=0.0)


            bin_ids = self._discretize_pathway_scores(pathway_scores)


            pathway_tensor = torch.tensor(bin_ids, dtype=torch.long)
            if pathway_tensor.shape[0] != n_pw:
                empty_bins = torch.zeros(n_pw, dtype=torch.long)
                if return_scores:
                    return empty_bins, torch.zeros(n_pw, dtype=torch.float32)
                return empty_bins

            if return_scores:
                scores_tensor = torch.tensor(pathway_scores, dtype=torch.float32)
                return pathway_tensor, scores_tensor

            return pathway_tensor
        except Exception:

            empty_bins = torch.zeros(n_pw, dtype=torch.long)
            if return_scores:
                return empty_bins, torch.zeros(n_pw, dtype=torch.float32)
            return empty_bins

    def get_raw_by_index(self, idx: int):

        file_path, local_idx = find_file_and_local_index(self.global_index, idx)
        label_raw, expr_bin_raw = read_csv_by_local_index(self.index_folder, file_path, local_idx)
        return label_raw, expr_bin_raw

    def get_raw_sample(self, idx: int):
        file_path, local_idx = find_file_and_local_index(self.global_index, idx)
        return get_line_by_local_index(self.index_folder,file_path, local_idx)


    def encode_expression_values(self, expression_list: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        max_expr_bin = self.n_expr_bins


        expression_array = np.array([float(x) for x in expression_list], dtype=np.float32)


        nonzero_mask = expression_array > 0.0
        nonzero_indices = np.where(nonzero_mask)[0]
        nonzero_values = expression_array[nonzero_mask]

        if len(nonzero_indices) == 0:

            return np.array([]), np.array([]), np.array([])


        max_val = nonzero_values.max()

        if max_val > 1.0:

            expr_bins_raw = nonzero_values.astype(int)
            expr_bins_raw = np.clip(expr_bins_raw, 1, max_expr_bin)
            sorted_indices = np.argsort(expr_bins_raw)[::-1]

            normalized_values = nonzero_values / max_expr_bin
        else:

            expr_bins_raw = np.clip(np.round(nonzero_values * max_expr_bin), 1, max_expr_bin).astype(int)
            sorted_indices = np.argsort(nonzero_values)[::-1]
            normalized_values = nonzero_values


        max_genes = min(len(sorted_indices), self.max_expressed_genes - 2)
        selected_indices = sorted_indices[:max_genes]

        gene_ids = nonzero_indices[selected_indices]
        expr_bins = expr_bins_raw[selected_indices]
        expression_values = normalized_values[selected_indices]



        expr_levels = np.zeros(len(expr_bins), dtype=int)

        low_threshold = max_expr_bin * 0.33
        mid_threshold = max_expr_bin * 0.66

        expr_levels[expr_bins <= low_threshold] = 1
        expr_levels[(expr_bins > low_threshold) & (expr_bins <= mid_threshold)] = 2
        expr_levels[expr_bins > mid_threshold] = 3

        return gene_ids, expr_bins, expr_levels

    def metadata_to_text(self, label_raw: list) -> Tuple[str, Optional[str]]:


        fields = label_raw
        text_parts = []
        cell_type = None


        for field in fields[:-3]:
            field = field.strip()

            if field.startswith('suspension_type_'):
                suspension_type = field.replace('suspension_type_', '').replace('_', ' ')
                text_parts.append(f"This is a {suspension_type} suspension")

            elif field.startswith('cell_type_'):
                cell_type = field.replace('cell_type_', '').replace('_', ' ')

                if self.anonymous:
                    text_parts.append("This is a cell")
                else:
                    text_parts.append(f"cell type is {cell_type}")

            elif field.startswith('assay_'):
                assay_type = field.replace('assay_', '').replace('_', ' ')
                text_parts.append(f"analyzed using {assay_type} assay")

            elif field.startswith('disease_'):
                disease = field.replace('disease_', '').replace('_', ' ')
                if disease.lower() == 'normal':
                    text_parts.append("from healthy tissue")
                else:
                    text_parts.append(f"associated with {disease}")

            elif field.startswith('sex_'):
                sex = field.replace('sex_', '')
                text_parts.append(f"from {sex} donor")

            elif field.startswith('tissue_'):
                tissue = field.replace('tissue_', '').replace('_', ' ')
                text_parts.append(f"from {tissue} tissue")

            elif field.startswith('development_stage_'):
                dev_stage = field.replace('development_stage_', '').replace('_', ' ')
                text_parts.append(f"at {dev_stage} development stage")


        if len(fields) >= 3:
            age_fields = fields[-3:]

            if age_fields[-1].strip():
                age_info = age_fields[-1].strip()
                if age_info != age_fields[1].strip():
                    text_parts.append(f"age group {age_info}")


        if text_parts:

            result = text_parts[0].capitalize()
            if len(text_parts) > 1:
                result += ", " + ", ".join(text_parts[1:])
            return result + ".", cell_type
        else:
            return "Cell metadata information not available.", None

    def __len__(self):
        return get_global_len(self.global_index)

    def get_cell_type(self,idx):
        label_raw, _ = self.get_raw_by_index(idx)
        for label in label_raw:
            if label.startswith('cell_type_'):
                cell_type = label.replace('cell_type_', '').replace('_', ' ')
                return cell_type

    def __getitem__(self, idx):

        label_raw, expr_bin_raw = self.get_raw_by_index(idx)
        gene_ids, expr_bins, expr_levels = self.encode_expression_values(expr_bin_raw)


        expression_array = np.array([float(x) for x in expr_bin_raw], dtype=np.float32)

        max_val = expression_array.max()
        if max_val > 1.0:

            expression_array_normalized = expression_array / self.n_expr_bins
        else:
            expression_array_normalized = expression_array

        if len(gene_ids) == 0:

            n_pw = max(self.n_pathways, 1)
            empty_tensor = torch.zeros(self.max_expressed_genes, dtype=torch.long)

            empty_pathway_embedding = torch.zeros(n_pw, dtype=torch.long)
            empty_result = {
                'gene_tokens': empty_tensor,
                'expr_bin_tokens': empty_tensor,
                'expr_level_tokens': empty_tensor,
                'pathway_embedding': empty_pathway_embedding,
                'gene_locations': [],
                'cell_type': None,
                'seq_length': 0,
                'attention_mask': torch.zeros(self.max_expressed_genes, dtype=torch.bool),
            }
            if self.return_pathway_scores:
                empty_result['pathway_scores'] = torch.zeros(n_pw, dtype=torch.float32)
            return empty_result


        gene_locations = self.tokenizers.get_gene_locations(gene_ids.tolist())


        gene_tokens = self.tokenizers.encode_genes(gene_ids.tolist(), max_length=self.max_expressed_genes)
        expr_bin_tokens = self.tokenizers.encode_expr_bins(expr_bins.tolist(), max_length=self.max_expressed_genes)
        expr_level_tokens = self.tokenizers.encode_expr_levels(expr_levels.tolist(), max_length=self.max_expressed_genes)


        _, cell_type = self.metadata_to_text(label_raw)


        pathway_scores = None
        if self.return_pathway_scores:
            pathway_embedding, pathway_scores = self._compute_pathway_embedding(
                gene_ids,
                expression_array_normalized[gene_ids],
                return_scores=True,
            )
        else:
            pathway_embedding = self._compute_pathway_embedding(gene_ids, expression_array_normalized[gene_ids])


        seq_len = len(gene_ids)
        attention_mask = torch.zeros(self.max_expressed_genes, dtype=torch.bool)
        attention_mask[:seq_len] = True

        result = {
            'gene_tokens': gene_tokens,
            'expr_bin_tokens': expr_bin_tokens,
            'expr_level_tokens': expr_level_tokens,
            'pathway_embedding': pathway_embedding,
            'gene_locations': gene_locations,
            'cell_type': cell_type,
            'seq_length': len(gene_ids),
            'attention_mask': attention_mask,
        }

        if pathway_scores is not None:
            result['pathway_scores'] = pathway_scores

        return result

    def get_vocab_sizes(self):

        return self.tokenizers.get_vocab_sizes()

    def get_pathway_info(self) -> Dict:

        if self.pathway_analyzer is None:
            return {
                'pathway_enabled': False,
                'n_pathways': 0,
                'pathway_names': []
            }

        return {
            'pathway_enabled': True,
            'n_pathways': self.n_pathways,
            'pathway_method': self.pathway_method,
            'pathway_names': self.pathway_analyzer.pathway_names[:10]
        }

    def print_pathway_info(self):

        info = self.get_pathway_info()
        print("\n  Pathway Analysis Info:")
        print(f"   Pathway enabled: {info['pathway_enabled']}")
        if info['pathway_enabled']:
            print(f"   Number of pathways: {info['n_pathways']:,}")
            print(f"   Scoring method: {info['pathway_method']}")
            print(f"   Sample pathways: {info['pathway_names'][:5]}")

if __name__ == '__main__':
    import time



    print("=" * 80)
    print(" TESTING PATHWAY ACTIVATION EMBEDDINGS")
    print("=" * 80)


    print("Initializing dataset with pathway analysis:")
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


    dataset.print_pathway_info()



    print("\nTesting multiple samples...")
    total_time = 0
    num_samples = 10

    for i in range(num_samples):
        start_time = time.time()
        x = dataset[i * 10]
        sample_time = time.time() - start_time
        total_time += sample_time

        if i == 0:
            print(f"Sample {i}:")
            print(f"  Gene tokens shape: {x['gene_tokens'].shape}")
            print(f"  Expr bin tokens shape: {x['expr_bin_tokens'].shape}")
            print(f"  Expr level tokens shape: {x['expr_level_tokens'].shape}")
            print(f"  Pathway embedding shape: {x['pathway_embedding'].shape}")
            print(f"  Pathway embedding stats: min={x['pathway_embedding'].min():.4f}, max={x['pathway_embedding'].max():.4f}, mean={x['pathway_embedding'].mean():.4f}")
            print(f"  Sequence length: {x['seq_length']}")
            print(f"  Cell type: {x['cell_type']}")
            print(f"  Processing time: {sample_time:.4f}s")

    avg_time = total_time / num_samples
    print(f"\nAverage processing time: {avg_time:.4f}s per sample")
    print(f"Estimated throughput: {1/avg_time:.1f} samples/s")

    print("\n Testing completed!")
    print("  Vocab sizes:", dataset.get_vocab_sizes())