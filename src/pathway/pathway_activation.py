

import os
import pickle
import json
import numpy as np
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


class PathwayActivationAnalyzer:


    def __init__(self,
                 pathway_db_path: str = "./pathway_data/pathway_database_ensembl.pkl",
                 min_pathway_genes: int = 5,
                 max_pathway_genes: int = 500):

        self.pathway_dict = self._load_pathway_database(pathway_db_path)
        self.min_genes = min_pathway_genes
        self.max_genes = max_pathway_genes


        self._filter_pathways()


        self._build_gene_index()


        self.pathway_names = sorted(self.pathway_dict.keys())
        self.n_pathways = len(self.pathway_names)

        print(f"    {self.n_pathways}    ")

    def _load_pathway_database(self, path: str) -> Dict[str, Set[str]]:

        if not os.path.exists(path):
            raise FileNotFoundError(f"        : {path}\n     download_pathway_db.py")

        with open(path, 'rb') as f:
            loaded_dict = pickle.load(f)

        return {k: set(v) for k, v in loaded_dict.items()}

    def _filter_pathways(self):

        original_count = len(self.pathway_dict)

        self.pathway_dict = {
            k: v for k, v in self.pathway_dict.items()
            if self.min_genes <= len(v) <= self.max_genes
        }

        filtered_count = original_count - len(self.pathway_dict)
        if filtered_count > 0:
            print(f"    {filtered_count}    (      {self.min_genes}-{self.max_genes}    )")

    def _build_gene_index(self):

        self.gene_to_pathways = defaultdict(list)

        for pathway_name, genes in self.pathway_dict.items():
            for gene in genes:
                self.gene_to_pathways[gene].append(pathway_name)

    def compute_activation_scores(self,
                                   gene_ids: List[str],
                                   expression_values: List[float],
                                   method: str = "combined") -> np.ndarray:


        expressed_genes = set(gene_ids)
        gene_expr_dict = dict(zip(gene_ids, [float(v) for v in expression_values]))


        if method == "coverage":
            return self._compute_coverage_score(expressed_genes)
        elif method == "mean_expr":
            return self._compute_mean_expression(expressed_genes, gene_expr_dict)
        elif method == "aucell":
            return self._compute_aucell_score(gene_ids, expressed_genes)
        elif method == "zscore":
            return self._compute_zscore(expressed_genes, gene_expr_dict)
        elif method == "combined":
            return self._compute_combined_score(gene_ids, expressed_genes, gene_expr_dict)
        else:
            raise ValueError(f"       : {method}")

    def _compute_coverage_score(self, expressed_genes: Set[str]) -> np.ndarray:

        scores = np.zeros(self.n_pathways)

        for i, pathway_name in enumerate(self.pathway_names):
            pathway_genes = self.pathway_dict[pathway_name]
            overlap = len(expressed_genes & pathway_genes)
            scores[i] = overlap / len(pathway_genes)

        return scores

    def _compute_mean_expression(self,
                                  expressed_genes: Set[str],
                                  gene_expr_dict: Dict[str, float]) -> np.ndarray:

        scores = np.zeros(self.n_pathways)

        for i, pathway_name in enumerate(self.pathway_names):
            pathway_genes = self.pathway_dict[pathway_name]
            overlap = expressed_genes & pathway_genes

            if overlap:
                expr_values = [gene_expr_dict[g] for g in overlap]
                scores[i] = np.mean(expr_values)

        return scores

    def _compute_aucell_score(self,
                               ranked_genes: List[str],
                               expressed_genes: Set[str]) -> np.ndarray:

        scores = np.zeros(self.n_pathways)
        n_genes = len(ranked_genes)


        threshold = max(int(n_genes * 0.05), 50)
        top_genes = set(ranked_genes[:threshold])

        for i, pathway_name in enumerate(self.pathway_names):
            pathway_genes = self.pathway_dict[pathway_name]


            ranks = []
            for j, gene in enumerate(ranked_genes):
                if gene in pathway_genes:
                    ranks.append(j)

            if ranks:


                auc = sum(1 - r/n_genes for r in ranks) / len(pathway_genes)
                scores[i] = auc


                top_overlap = len(top_genes & pathway_genes) / len(pathway_genes)
                scores[i] = 0.7 * scores[i] + 0.3 * top_overlap

        return scores

    def _compute_zscore(self,
                        expressed_genes: Set[str],
                        gene_expr_dict: Dict[str, float]) -> np.ndarray:


        all_expr = list(gene_expr_dict.values())
        global_mean = np.mean(all_expr)
        global_std = np.std(all_expr) + 1e-8

        scores = np.zeros(self.n_pathways)

        for i, pathway_name in enumerate(self.pathway_names):
            pathway_genes = self.pathway_dict[pathway_name]
            overlap = expressed_genes & pathway_genes

            if overlap:
                expr_values = [gene_expr_dict[g] for g in overlap]
                pathway_mean = np.mean(expr_values)

                scores[i] = (pathway_mean - global_mean) / global_std

        return scores

    def _compute_combined_score(self,
                                 ranked_genes: List[str],
                                 expressed_genes: Set[str],
                                 gene_expr_dict: Dict[str, float]) -> np.ndarray:

        coverage = self._compute_coverage_score(expressed_genes)
        mean_expr = self._compute_mean_expression(expressed_genes, gene_expr_dict)
        aucell = self._compute_aucell_score(ranked_genes, expressed_genes)


        def normalize(arr):
            arr_min = arr.min()
            arr_max = arr.max()
            if arr_max - arr_min > 1e-8:
                return (arr - arr_min) / (arr_max - arr_min)

            return np.zeros_like(arr) if arr_max < 1e-8 else np.full_like(arr, 0.5)

        coverage_norm = normalize(coverage)
        mean_expr_norm = normalize(mean_expr)
        aucell_norm = normalize(aucell)


        combined = (0.3 * coverage_norm +
                   0.3 * mean_expr_norm +
                   0.4 * aucell_norm)


        combined = np.nan_to_num(combined, nan=0.0, posinf=1.0, neginf=0.0)

        return combined

    def get_pathway_features(self,
                              gene_ids: List[str],
                              expression_values: List[float],
                              method: str = "combined",
                              top_k: Optional[int] = None) -> Dict:

        scores = self.compute_activation_scores(gene_ids, expression_values, method)

        result = {
            'scores': scores,
            'pathway_names': self.pathway_names,
            'n_pathways': self.n_pathways
        }

        if top_k:
            top_indices = np.argsort(scores)[-top_k:][::-1]
            result['top_pathways'] = [
                {'name': self.pathway_names[i],
                 'score': float(scores[i]),
                 'rank': rank + 1}
                for rank, i in enumerate(top_indices)
            ]

        return result


class BatchPathwayAnalyzer:


    def __init__(self, analyzer: PathwayActivationAnalyzer):
        self.analyzer = analyzer

    def process_batch(self,
                      batch_gene_ids: List[List[str]],
                      batch_expression: List[List[float]],
                      method: str = "combined") -> np.ndarray:

        batch_size = len(batch_gene_ids)
        scores_matrix = np.zeros((batch_size, self.analyzer.n_pathways))

        for i in range(batch_size):
            scores_matrix[i] = self.analyzer.compute_activation_scores(
                batch_gene_ids[i],
                batch_expression[i],
                method
            )

        return scores_matrix



def get_pathway_activation(gene_ids: List[str],
                           expression_values: List[float],
                           pathway_db_path: str = "./pathway_data/pathway_database_ensembl.pkl",
                           method: str = "combined") -> np.ndarray:

    analyzer = PathwayActivationAnalyzer(pathway_db_path)
    return analyzer.compute_activation_scores(gene_ids, expression_values, method)


def demo(gene_ids: List[str] = None, expression_values: List[float] = None):
    import time


    if gene_ids is None or expression_values is None:
        gene_ids = ['ENSG00000141510', 'ENSG00000171862', 'ENSG00000141736',
                    'ENSG00000198691', 'ENSG00000115414']
        expression_values = ['5.0', '4.2', '3.8', '2.5', '1.2']


    try:
        analyzer = PathwayActivationAnalyzer(
            pathway_db_path="./pathway_data/pathway_database_ensembl.pkl"
        )


        combined_start = time.time()
        features = analyzer.get_pathway_features(
            gene_ids=gene_ids,
            expression_values=expression_values,
            method="combined",
            top_k=200
        )
        combined_end = time.time()

        print(f"    : {features['n_pathways']}")
        print(f"    : {features['scores'].shape}")
        print("\n       :")
        for pw in features['top_pathways']:
            print(f"  {pw['rank']}. {pw['name']}: {pw['score']:.4f}")
        print(f"\n        : {combined_end - combined_start:.4f}  ")
        print("\n      :\n"+"-"*60)
        for key in features.keys():
            if key == 'pathway_names':
                continue
            print(f"{key}: {features[key]}")

            print("-"*60)
        print("\n"+"="*60)
        print(features.keys())



    except FileNotFoundError as e:
        print(e)
        print("\n    : python download_pathway_db.py")


if __name__ == "__main__":
    demo()