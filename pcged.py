import sys


HELP = """Usage:
  python pcged.py train [training options]
  python pcged.py infer [inference options]

Default checkpoint:
  checkpoints/pcged/universal_model.pt
"""


def main():
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print(HELP)
        return
    if sys.argv[1] not in {"train", "infer"}:
        raise SystemExit(HELP)

    command = sys.argv.pop(1)
    if "-h" in sys.argv or "--help" in sys.argv:
        print(HELP)
        return
    if command == "train":
        if "--config" not in sys.argv:
            sys.argv.extend(["--config", "configs/train.yaml"])
        if "--deepspeed_config" not in sys.argv and "--deepspeed" not in sys.argv:
            sys.argv.extend(["--deepspeed_config", "configs/deepspeed.json"])
        from bin_patch.train import main as run
    else:
        if "--checkpoint" not in sys.argv:
            sys.argv.extend(["--checkpoint", "checkpoints/pcged/universal_model.pt"])
        if "--config" not in sys.argv:
            sys.argv.extend(["--config", "configs/train.yaml"])
        from bin_patch.generate import main as run

    run()


if __name__ == "__main__":
    main()
