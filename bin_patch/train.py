

from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path
from typing import Iterable, List

import torch




REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))




def _install_tokenizer() -> None:
    from bin_patch.tokenizer import GeneTokenizers as FixedGeneTokenizers


    orig_mod = importlib.import_module("src.data.tokenizer")
    orig_mod.GeneTokenizers = FixedGeneTokenizers
    print("[bin_patch] Installed GeneTokenizers")


_install_tokenizer()




import deepspeed

_DS_INIT_ORIG = deepspeed.initialize


def _patched_ds_initialize(*args, **kwargs):
    model = kwargs.get("model") or (args[0] if args else None)
    pg = getattr(model, "_bin_patch_param_groups", None)
    if pg is not None:
        n0 = sum(p.numel() for p in pg[0]["params"])
        n1 = sum(p.numel() for p in pg[1]["params"]) if len(pg) > 1 else 0
        print(
            f"[bin_patch] Injecting {len(pg)} parameter groups into deepspeed.initialize: "
            f"group0(n={n0:,}, lr={pg[0].get('lr')}) | "
            f"group1(n={n1:,}, lr={pg[1].get('lr') if len(pg) > 1 else None})"
        )
        kwargs["model_parameters"] = pg
    return _DS_INIT_ORIG(*args, **kwargs)


deepspeed.initialize = _patched_ds_initialize




import training as base





def _build_param_groups(
    model: torch.nn.Module,
    bin_patterns: List[str],
    lr_bin: float,
    lr_backbone: float,
    weight_decay: float,
):

    compiled = [re.compile(p) for p in bin_patterns]
    bin_params, other_params = [], []
    bin_names, other_names = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(c.search(name) for c in compiled):
            bin_params.append(param)
            bin_names.append(name)
        else:
            other_params.append(param)
            other_names.append(name)


    if base.is_main_process():
        print("[bin_patch] Parameter group split:")
        print(f"   group0 (bin, lr={lr_bin}): {len(bin_params)} tensors, "
              f"{sum(p.numel() for p in bin_params):,} params")
        for n in bin_names:
            print(f"      + {n}")
        print(f"   group1 (backbone, lr={lr_backbone}): {len(other_params)} tensors, "
              f"{sum(p.numel() for p in other_params):,} params")

    return [
        {"params": bin_params, "lr": lr_bin, "weight_decay": weight_decay},
        {"params": other_params, "lr": lr_backbone, "weight_decay": weight_decay},
    ]





def main():
    if not base.DEEPSPEED_AVAILABLE:
        print("ERR DeepSpeed is required.")
        sys.exit(1)

    args = base.parse_args()
    base.set_seed(args.seed)

    config = base.load_yaml_config(args.config)
    ds_config = base.load_json_config(args.deepspeed_config)
    config["seed"] = args.seed


    lrg = (config.get("training", {}) or {}).get("lr_groups", {})
    if not lrg:
        raise ValueError(
            "training.lr_groups requires bin_patterns, lr_bin, and lr_backbone"
        )
    bin_patterns = lrg.get("bin_patterns", [
        r"^gene_expr_embedding\.expr_bin_embedding\.",
        r"^expr_bin_head\.",
    ])
    lr_bin = float(lrg["lr_bin"])
    lr_backbone = float(lrg["lr_backbone"])
    weight_decay = float(lrg.get("weight_decay", 0.1))


    sched_params = (ds_config.get("scheduler") or {}).get("params", {}) or {}
    for k in ("warmup_max_lr", "warmup_min_lr"):
        v = sched_params.get(k)
        if isinstance(v, list) and len(v) != 2:
            raise ValueError(
                f"ds_config scheduler.params.{k}   list        2 (bin, backbone),"
                f"    ={len(v)}"
            )


    base.print_rank_0("=" * 80)
    base.print_rank_0(" PCGED Training")
    base.print_rank_0("=" * 80)
    base.print_rank_0(f"  Config: {args.config}")
    base.print_rank_0(f"  DeepSpeed config: {args.deepspeed_config}")
    base.print_rank_0(f"  Output dir: {args.output_dir}")
    base.print_rank_0(f"  Resume mode: {args.resume_mode}")
    base.print_rank_0(f"  Seed: {args.seed}")
    base.print_rank_0(f" lr_bin={lr_bin}  lr_backbone={lr_backbone}  weight_decay={weight_decay}")
    base.print_rank_0(f" bin_patterns={bin_patterns}")
    base.print_rank_0("=" * 80)


    base.print_rank_0("\n  Creating datasets...")
    train_dataset, val_dataset, test_dataset, n_pathways = base.create_datasets(config)
    if config["model"].get("n_pathways", 2000) != n_pathways:
        base.print_rank_0(
            f"WARN  Updating n_pathways from {config['model'].get('n_pathways', 2000)} to {n_pathways}"
        )
        config["model"]["n_pathways"] = n_pathways

    data_config = config.get("data", {})
    batch_size = ds_config.get("train_micro_batch_size_per_gpu", data_config.get("batch_size", 16))

    dataloaders = base.create_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=batch_size,
        num_workers=data_config.get("num_workers", 4),
        distributed=base.get_world_size() > 1,
        world_size=base.get_world_size(),
        rank=base.get_rank(),
        fixed_seq_length=None,
    )
    train_dl = dataloaders["train"]
    val_dl = dataloaders.get("val")
    test_dl = dataloaders.get("test")


    base.print_rank_0("\n  Creating model...")
    from src.model.model_pathway_aux import __MODEL_VERSION__
    base.print_rank_0(f"       Model code version: {__MODEL_VERSION__}")
    model = base.create_model_from_config(config, n_pathways)
    base.print_rank_0(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    base.print_rank_0(f"   n_expr_bins (must be 26): {model.config.n_expr_bins}")
    if model.config.n_expr_bins != 26:
        raise RuntimeError(
            f"model.config.n_expr_bins={model.config.n_expr_bins}   26,"
            "   yaml model.n_expr_bins    26        tokenizer"
        )


    model._bin_patch_param_groups = _build_param_groups(
        model, bin_patterns, lr_bin, lr_backbone, weight_decay
    )


    loss_config = config.get("training", {}).get("loss_weights", {})
    loss_fn = base.PCGEDLoss(
        gene_loss_weight=loss_config.get("gene_loss", loss_config.get("gene", 1.0)),
        expr_bin_loss_weight=loss_config.get("expr_bin_loss", loss_config.get("expr_bin", 0.5)),
        expr_level_loss_weight=loss_config.get("expr_level_loss", loss_config.get("expr_level", 0.3)),
        pathway_loss_weight=loss_config.get("pathway_loss", loss_config.get("pathway", 0.0)),
        pathway_loss_type=loss_config.get("pathway_loss_type", "smooth_l1"),
        pathway_loss_use_sigmoid=loss_config.get("pathway_loss_use_sigmoid", True),
        pathway_num_bins=config.get("data", {}).get("pathway_num_bins", 5),
        eos_loss_weight=loss_config.get("eos_loss", 0.0),
        eos_pos_weight=loss_config.get("eos_pos_weight", 16.0),
        eos_token_id=config.get("model", {}).get("eos_token_id", 2),
    )


    base.print_rank_0("\n Creating DeepSpeed trainer...")
    trainer = base.DeepSpeedTrainer(
        model=model,
        train_dataloader=train_dl,
        val_dataloader=val_dl,
        test_dataloader=test_dl,
        args=args,
        ds_config=ds_config,
        loss_fn=loss_fn,
    )


    base.print_rank_0("\n Starting fine-tune...")
    import signal
    def _sigterm(signum, frame):
        base.print_rank_0(f"\n  Received signal {signum}, requesting graceful shutdown...")
        trainer._stop_requested = True
    signal.signal(signal.SIGTERM, _sigterm)
    signal.signal(signal.SIGUSR1, _sigterm)

    try:
        best_val_loss = trainer.train()
        base.print_rank_0(f"\nOK Fine-tune completed. Best val loss: {best_val_loss:.4f}")
        if test_dl is not None:
            base.print_rank_0("\n Final test evaluation...")
            test_metrics = trainer._test()
            if test_metrics:
                trainer._log_metrics(test_metrics)
    except KeyboardInterrupt:
        base.print_rank_0("\nWARN Interrupted by user")
        trainer._save_checkpoint("interrupted")
    except Exception:
        base.print_rank_0("\nERR Training failed")
        raise


if __name__ == "__main__":
    main()
