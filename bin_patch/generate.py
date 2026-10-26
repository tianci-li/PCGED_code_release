



import sys
import json
import argparse
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from bin_patch.inference import PCGEDInference

DEFAULT_CONFIG = "configs/train.yaml"


def load_dataset_split(config_path: str, split: str = "test", seed: int = 42):
    import yaml
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    from src.data.dataset import SparseCellDataset
    data_config = config["data"]

    full_dataset = SparseCellDataset(
        index_folder=data_config["index_folder"],
        csv_filepath=data_config["csv_filepath"],
        global_index=data_config["global_index"],
        cell_description=data_config["cell_description"],
        gene_location_list_path=data_config["gene_location_list_path"],
        max_len=data_config.get("max_len", 8192),
        anonymous=data_config.get("anonymous", True),
        pathway_db_path=data_config.get("pathway_db_path", "./pathway_data/pathway_database_ensembl.pkl"),
        pathway_method=data_config.get("pathway_method", "combined"),
        min_pathway_genes=data_config.get("min_pathway_genes", 5),
        max_pathway_genes=data_config.get("max_pathway_genes", 500),
        n_expr_bins=data_config.get("n_expr_bins", 20),
        use_global_binning=data_config.get("use_global_binning", True),
    )

    total_len = len(full_dataset)
    train_len = int(total_len * 0.8)
    val_len = int(total_len * 0.1)
    test_len = total_len - train_len - val_len

    data_seed = config.get("seed", seed)
    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        full_dataset, [train_len, val_len, test_len],
        generator=torch.Generator().manual_seed(data_seed),
    )

    splits = {"train": train_ds, "val": val_ds, "test": test_ds}
    chosen = splits[split]

    print(f"Dataset split (seed={data_seed}):")
    print(f"   Total: {total_len:,}  Train: {train_len:,}  Val: {val_len:,}  Test: {test_len:,}")
    print(f"   Using: '{split}' split ({len(chosen):,} samples)")
    return chosen, full_dataset


def _make_cache_key(split, n_samples, seed, min_nonzero):
    return f"sampled_cells_{split}_n{n_samples}_s{seed}_min{min_nonzero}.pt"


def sample_real_cells(dataset, n_samples, seed=42, min_nonzero=50,
                      cache_dir=None, split="test"):
    cache_path = None
    if cache_dir is not None:
        cache_path = Path(cache_dir) / _make_cache_key(split, n_samples, seed, min_nonzero)
        if cache_path.exists():
            print(f"Loading cached samples from {cache_path} ...")
            cached = torch.load(cache_path, map_location="cpu", weights_only=False)
            return (cached["samples"], cached["pathway_bins"],
                    cached["cell_types"], cached["real_vectors"])

    rng = np.random.RandomState(seed)
    total = len(dataset)
    indices = rng.choice(total, size=min(n_samples * 2, total), replace=False)

    samples, pathway_bins_list, cell_types, expression_vectors = [], [], [], []
    skipped_low = 0

    import os
    from torch.utils.data import DataLoader, Subset
    NUM_WORKERS = min(16, os.cpu_count() or 4)

    def _no_collate(batch):
        return batch[0]

    random_subset = Subset(dataset, indices.tolist())
    loader = DataLoader(
        random_subset, batch_size=1, num_workers=NUM_WORKERS,
        prefetch_factor=4, collate_fn=_no_collate, persistent_workers=False,
    )

    print(f"Sampling real cells (total={total}, target={n_samples}, "
          f"min_nonzero={min_nonzero}, workers={NUM_WORKERS}) ...")
    scanned = 0
    for sample in loader:
        scanned += 1
        if len(samples) >= n_samples:
            break
        if sample["seq_length"] == 0:
            continue
        vec = PCGEDInference.real_sample_to_expression_vector(sample)
        if np.count_nonzero(vec) < min_nonzero:
            skipped_low += 1
            continue
        samples.append(sample)
        pathway_bins_list.append(sample["pathway_embedding"])
        cell_types.append(sample.get("cell_type", "unknown"))
        expression_vectors.append(vec)
        if len(samples) % 500 == 0:
            print(f"   Collected {len(samples)}/{n_samples} "
                  f"(scanned {scanned}, skipped {skipped_low}) ...")
    del loader

    pathway_bins = torch.stack(pathway_bins_list, dim=0)
    real_vectors = np.array(expression_vectors)
    print(f"Sampled {len(samples)} valid cells (skipped {skipped_low})")

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "samples": samples,
            "pathway_bins": pathway_bins,
            "cell_types": cell_types,
            "real_vectors": real_vectors,
        }, cache_path)
        print(f"Cached -> {cache_path}")
    return samples, pathway_bins, cell_types, real_vectors


def compute_metrics(real_vectors, gen_vectors):
    from numpy.linalg import norm
    metrics = {}

    real_nnz = np.mean(np.count_nonzero(real_vectors, axis=1))
    gen_nnz = np.mean(np.count_nonzero(gen_vectors, axis=1))
    metrics["real_avg_nonzero_genes"] = float(real_nnz)
    metrics["gen_avg_nonzero_genes"] = float(gen_nnz)
    metrics["nonzero_ratio"] = float(gen_nnz / max(real_nnz, 1))

    cos_sims = []
    for r, g in zip(real_vectors, gen_vectors):
        rn, gn = norm(r), norm(g)
        if rn > 0 and gn > 0:
            cos_sims.append(float(np.dot(r, g) / (rn * gn)))
    if cos_sims:
        metrics["cosine_similarity_mean"] = float(np.mean(cos_sims))
        metrics["cosine_similarity_std"] = float(np.std(cos_sims))
        metrics["cosine_similarity_median"] = float(np.median(cos_sims))

    overlaps = []
    for r, g in zip(real_vectors, gen_vectors):
        real_genes = set(np.nonzero(r)[0])
        gen_genes = set(np.nonzero(g)[0])
        if real_genes:
            overlaps.append(len(real_genes & gen_genes) / len(real_genes | gen_genes))
    if overlaps:
        metrics["gene_jaccard_mean"] = float(np.mean(overlaps))
        metrics["gene_jaccard_std"] = float(np.std(overlaps))

    rm, gm = real_vectors.mean(axis=0), gen_vectors.mean(axis=0)
    rn, gn = norm(rm), norm(gm)
    if rn > 0 and gn > 0:
        metrics["mean_profile_cosine"] = float(np.dot(rm, gm) / (rn * gn))


    real_high = float(np.mean((real_vectors > 10).any(axis=1)))
    gen_high = float(np.mean((gen_vectors > 10).any(axis=1)))
    metrics["real_pct_cells_with_bin_gt_10"] = real_high
    metrics["gen_pct_cells_with_bin_gt_10"] = gen_high
    metrics["real_max_bin"] = int(real_vectors.max()) if real_vectors.size else 0
    metrics["gen_max_bin"] = int(gen_vectors.max()) if gen_vectors.size else 0
    return metrics


def main():
    parser = argparse.ArgumentParser(description="PCGED Cell Generation")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG)
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--output_dir", type=str, default="outputs/inference")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_length", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--split", type=str, default="test", choices=["test", "val", "train"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_nonzero", type=int, default=50)
    parser.add_argument("--multi_gpu", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)


    print("=" * 60)
    print("Step 1: Loading PCGED inference engine")
    print("=" * 60)
    engine = PCGEDInference(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        device=args.device,
        multi_gpu=False,
    )


    print("\n" + "=" * 60)
    print(f"Step 2: Sampling real cells from '{args.split}' split")
    print("=" * 60)
    split_dataset, full_dataset = load_dataset_split(
        args.config, split=args.split, seed=args.seed
    )
    if full_dataset.n_pathways != engine.pcged_config.n_pathways:
        print(f"ERR n_pathways mismatch! Dataset={full_dataset.n_pathways} "
              f"Model={engine.pcged_config.n_pathways}")
        sys.exit(1)

    samples, pathway_bins, cell_types, real_vectors = sample_real_cells(
        split_dataset, args.n_samples, seed=args.seed,
        min_nonzero=args.min_nonzero, cache_dir=str(output_dir), split=args.split,
    )
    total_cells = len(samples)


    print("\n" + "=" * 60)
    print(f"Step 3: Generating {total_cells} cells")
    print("=" * 60)
    n_gpus = torch.cuda.device_count() if (torch.cuda.is_available() and args.multi_gpu) else 1
    shard_dir = output_dir / "_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    if n_gpus > 1:
        import subprocess
        chunks = list(pathway_bins.chunk(n_gpus))
        offset = 0
        shard_inputs, shard_outputs = [], []
        for gpu_id, chunk in enumerate(chunks):
            inds = list(range(offset, offset + chunk.shape[0]))
            offset += chunk.shape[0]
            inp = shard_dir / f"shard_input_{gpu_id}.pt"
            out = shard_dir / f"shard_output_{gpu_id}.pt"
            torch.save({"pathway_bins": chunk, "indices": inds}, inp)
            shard_inputs.append(inp)
            shard_outputs.append(out)
        print(f"   Launching {n_gpus} independent GPU workers ...")

        worker_script = str(Path(__file__).parent / "worker.py")
        procs = []
        t0 = time.time()
        for gpu_id in range(n_gpus):
            cmd = [
                sys.executable, worker_script,
                "--checkpoint", args.checkpoint,
                "--config", args.config,
                "--gpu_id", str(gpu_id),
                "--input_file", str(shard_inputs[gpu_id]),
                "--output_file", str(shard_outputs[gpu_id]),
                "--max_length", str(args.max_length),
                "--temperature", str(args.temperature),
                "--top_k", str(args.top_k),
                "--top_p", str(args.top_p),
                "--batch_size", str(args.batch_size),
                "--seed", str(args.seed),
            ]
            import os
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            p = subprocess.Popen(cmd, env=env, stdout=sys.stdout, stderr=sys.stderr)
            procs.append((gpu_id, p))

        failed = []
        for gpu_id, p in procs:
            rc = p.wait()
            if rc != 0:
                failed.append(gpu_id)
                print(f"   ERR GPU {gpu_id} worker exited with code {rc}")
        if failed:
            print(f"FATAL: Workers on GPU {failed} failed.")
            sys.exit(1)
        gen_time = time.time() - t0

        all_gen_vectors = [None] * total_cells
        all_details = [None] * total_cells
        for gpu_id in range(n_gpus):
            shard = torch.load(shard_outputs[gpu_id], map_location="cpu", weights_only=False)
            for i, orig_idx in enumerate(shard["indices"]):
                all_gen_vectors[orig_idx] = shard["gen_vectors"][i]
                d = shard["details"][i]
                d["expression_vector"] = shard["gen_vectors"][i]
                all_details[orig_idx] = d
        gen_vectors = np.array(all_gen_vectors)
        generated = all_details
        import shutil
        shutil.rmtree(shard_dir, ignore_errors=True)
    else:
        t0 = time.time()
        generated, gen_vectors_list = [], []
        for chunk_start in range(0, total_cells, 20):
            chunk_end = min(chunk_start + 20, total_cells)
            chunk_pw = pathway_bins[chunk_start:chunk_end]
            print(f"   Generating cells {chunk_start+1}~{chunk_end} ...")
            chunk_results = engine.generate(
                pathway_bins=chunk_pw, max_length=args.max_length,
                temperature=args.temperature, top_k=args.top_k,
                top_p=args.top_p, do_sample=True, batch_size=args.batch_size,
            )
            for g in chunk_results:
                generated.append(g)
                gen_vectors_list.append(g["expression_vector"])
        gen_time = time.time() - t0
        gen_vectors = np.array(gen_vectors_list)

    print(f"\nGeneration done in {gen_time:.1f}s "
          f"({gen_time / max(total_cells,1):.2f}s/cell)")


    print("\n" + "=" * 60)
    print("Step 4: Computing quality metrics")
    print("=" * 60)
    metrics = compute_metrics(real_vectors, gen_vectors)
    metrics.update({
        "n_samples": total_cells,
        "n_gpus": n_gpus,
        "split": args.split,
        "gen_time_seconds": gen_time,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "checkpoint": args.checkpoint,
    })
    print(f"\n{'Metric':<35} {'Value':>12}")
    print("-" * 50)
    for k, v in sorted(metrics.items()):
        if isinstance(v, float):
            print(f"  {k:<33} {v:>12.4f}")
        else:
            print(f"  {k:<33} {str(v):>12}")

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved: {output_dir / 'metrics.json'}")

    details_out = [{
        "gene_indices": g["gene_indices"],
        "gene_names": g["gene_names"],
        "expr_bins": g["expr_bins"],
        "expr_levels": g["expr_levels"],
        "n_genes": g["n_genes"],
    } for g in generated]
    with open(output_dir / "generated_details.json", "w") as f:
        json.dump(details_out, f)


    print("\n" + "=" * 60)
    print("Step 5: Extracting PCGED cell embeddings")
    print("=" * 60)
    print("  Real cell embeddings ...")
    real_embeddings = engine.extract_embeddings(samples, batch_size=args.batch_size)
    print(f"  -> {real_embeddings.shape}")
    print("  Generated cell embeddings ...")
    gen_samples = engine._build_generated_samples(generated, pathway_bins)
    gen_embeddings = engine.extract_embeddings(gen_samples, batch_size=args.batch_size)
    print(f"  -> {gen_embeddings.shape}")

    np.savez_compressed(
        output_dir / "generated_samples.npz",
        real_vectors=real_vectors,
        gen_vectors=gen_vectors,
        cell_types=np.array(cell_types, dtype=object),
        real_embeddings=real_embeddings,
        gen_embeddings=gen_embeddings,
    )
    print("\n" + "=" * 60)
    print("Done! Files for local UMAP analysis:")
    print(f"  {output_dir / 'generated_samples.npz'}")
    print(f"  {output_dir / 'metrics.json'}")
    print(f"  {output_dir / 'generated_details.json'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
