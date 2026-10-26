



import sys
import argparse
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from bin_patch.inference import PCGEDInference


def _torch_load_compat(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--gpu_id", type=int, required=True)
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    gpu_id = args.gpu_id
    device = "cuda:0"

    torch.manual_seed(args.seed + gpu_id)

    shard = _torch_load_compat(args.input_file, map_location="cpu")
    pathway_bins = shard["pathway_bins"]
    pathway_scores = shard.get("pathway_scores")
    shard_indices = shard["indices"]
    n_cells = pathway_bins.shape[0]
    print(f"[GPU {gpu_id}] Loaded shard: {n_cells} cells", flush=True)

    t0 = time.time()
    engine = PCGEDInference(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        device=device,
        multi_gpu=False,
    )
    print(f"[GPU {gpu_id}] Model loaded in {time.time()-t0:.1f}s", flush=True)

    t1 = time.time()
    results = []
    N = pathway_bins.shape[0]
    bs = args.batch_size
    save_interval = 50

    for start in range(0, N, bs):
        end = min(start + bs, N)
        batch_pw = pathway_bins[start:end]
        batch_scores = None
        if pathway_scores is not None:
            batch_scores = pathway_scores[start:end]
        chunk_results = engine.generate(
            pathway_bins=batch_pw,
            pathway_scores=batch_scores,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            do_sample=True,
            batch_size=bs,
        )
        results.extend(chunk_results)
        done = len(results)
        elapsed = time.time() - t1
        speed = elapsed / done if done > 0 else 0
        eta = speed * (N - done)
        print(f"[GPU {gpu_id}] {done}/{N} cells  "
              f"({speed:.2f}s/cell, ETA {eta:.0f}s)", flush=True)

        if done % save_interval < bs or done == N:
            _save_results(args.output_file, shard_indices[:done], results, time.time() - t1)

    gen_time = time.time() - t1
    print(f"[GPU {gpu_id}] Done! {len(results)} cells in {gen_time:.1f}s "
          f"({gen_time/max(len(results),1):.2f}s/cell)", flush=True)

    _save_results(args.output_file, shard_indices, results, gen_time)
    print(f"[GPU {gpu_id}] Saved -> {args.output_file}", flush=True)


def _save_results(output_file, indices, results, gen_time):
    save_data = {
        "indices": indices[:len(results)],
        "gen_vectors": np.array([r["expression_vector"] for r in results]),
        "details": [{
            "gene_indices": r["gene_indices"],
            "gene_names": r["gene_names"],
            "expr_bins": r["expr_bins"],
            "expr_levels": r["expr_levels"],
            "n_genes": r["n_genes"],
        } for r in results],
        "gen_time": gen_time,
        "is_complete": len(results) == len(indices),
    }
    torch.save(save_data, output_file)


if __name__ == "__main__":
    main()
