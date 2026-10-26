# PCGED

This package contains the PCGED training and inference code with the expression-bin patch.

## Availability

- GitHub: https://github.com/tianci-li/PCGED_code_release
- Zenodo archive: https://doi.org/10.5281/zenodo.23670427
- Zenodo record: https://zenodo.org/records/23670427

The trained model weights are not distributed through GitHub or Zenodo. They remain hosted server-side and are available only for fixed-checkpoint inference through the designated GPU platform.

## Structure

- `pcged.py`: command-line entry point
- `bin_patch/`: training and inference integration
- `src/`: model, data, pathway, and training code
- `configs/`: model and DeepSpeed configuration
- `checkpoints/pcged/`: model checkpoint location

## Installation

```bash
pip install -r requirements.txt
```

## Model weights

Place the model checkpoint at:

```text
checkpoints/pcged/universal_model.pt
```

## Training

Update the dataset paths in `configs/train.yaml`, then run:

```bash
torchrun --standalone --nproc_per_node=8 pcged.py train --output_dir outputs/train
```

## Inference

After placing the checkpoint and updating the dataset paths in `configs/train.yaml`, run:

```bash
python pcged.py infer --n_samples 100 --output_dir outputs/inference
```
