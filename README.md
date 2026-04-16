# Over-the-Air Neural Group Testing (OTA-NGT)

Official implementation of **[Over-the-Air Neural Group Testing](https://ieeexplore.ieee.org/abstract/document/10624979)**.

OTA-NGT leverages neural networks for efficient group testing in over-the-air wireless settings. The framework supports multiple transmission and testing algorithms, realistic wireless channel effects (noise, fading, random phase delays), and dynamic signal power computation for robust performance.

---

## Project Structure

```
.
├── main.py                 # Training and evaluation entry point
├── constants.py            # Firearm image filenames for validation
├── resnet_design2/         # Primary model architecture (with phase shift support)
├── resnet_design3/         # Alternative model architecture
├── data_scripts/           # Dataset creation from ImageNet
├── analysis/               # Evaluation and plotting scripts
├── requirements.txt        # Python dependencies
└── requirements.yml        # Conda environment
```

---

## Configuration Options

### Model Design Selection

Toggle the model design by editing the import in `main.py`:

```python
import resnet_design2 as models  # Default: supports phase shifts
# import resnet_design3 as models  # Alternative design
```

### Algorithm Selection (`--GT-alg`)

| Flag | Algorithm | Description |
|------|-----------|-------------|
| 1    | ITIT      | Individual Transmission, Individual Testing |
| 2    | GTGT-FM   | Group Transmission, Group Testing with Feature Merge |
| 3    | ITGT-FM   | Individual Transmission, Group Testing with Feature Merge |
| 4    | GTGT-PM   | Group Transmission, Group Testing with Pixel Merge |

### Noise Configuration (`--SNR`)

Set the Signal-to-Noise Ratio (SNR) in dB. Omit to train without noise.

### SNR Schedule (`--snr-schedule`)

- `1` (default): SNR updated per epoch
- `2`: SNR updated per batch

### SNR Type (`--snr-type`)

- `1` (default): SNR-specific — constant SNR throughout training/validation
- `2`: SNR-agnostic — SNR randomly selected and updated per schedule

### Random Phase Shift (`--phase`)

Enable random phase shift [-pi, pi] on intermediate feature maps to emulate realistic wireless channel phase delays.

### Group Size (`--background-K`)

The `--background-K` flag defines group size minus one:
- `0`: Group size = 1
- `3`: Group size = 4
- `7`: Group size = 8

---

## Example Usage

**AWGN Mode (SNR-specific):**
```bash
CUDA_VISIBLE_DEVICES=0,1 python -u main.py \
    --background-K 7 --SNR -8 --GT-alg 4 \
    --data data/GroupTestingDataset --pretrained \
    --lr 0.001 --batch-size 32 -a resnext101_32x8d \
    --task-num 2 --log-name output.log \
    --output_dir Trained_Models/ResNeXt_K7_A4 \
    --dist-url 'tcp://127.0.0.1:7184' --dist-backend 'nccl' \
    --multiprocessing-distributed --epochs 200 \
    --world-size 1 --rank 0
```

**AWGN Mode (SNR-agnostic, batch-wise):**
```bash
CUDA_VISIBLE_DEVICES=0,1 python -u main.py \
    --background-K 7 --snr-type 2 --snr-schedule 2 --GT-alg 4 \
    --data data/GroupTestingDataset --pretrained \
    --lr 0.001 --batch-size 32 -a resnext101_32x8d \
    --task-num 2 --log-name output.log \
    --output_dir Trained_Models/ResNeXt_K7_A4 \
    --dist-url 'tcp://127.0.0.1:7184' --dist-backend 'nccl' \
    --multiprocessing-distributed --epochs 200 \
    --world-size 1 --rank 0
```

**Random Phase Mode (SNR-specific):**
```bash
CUDA_VISIBLE_DEVICES=0,1 python -u main.py \
    --background-K 7 --SNR -8 --phase --GT-alg 4 \
    --data data/GroupTestingDataset --pretrained \
    --lr 0.001 --batch-size 32 -a resnext101_32x8d \
    --task-num 2 --log-name output.log \
    --output_dir Trained_Models/ResNeXt_K7_A4 \
    --dist-url 'tcp://127.0.0.1:7184' --dist-backend 'nccl' \
    --multiprocessing-distributed --epochs 200 \
    --world-size 1 --rank 0
```

**Random Phase Mode (SNR-agnostic, batch-wise):**
```bash
CUDA_VISIBLE_DEVICES=0,1 python -u main.py \
    --background-K 7 --snr-type 2 --snr-schedule 2 --phase --GT-alg 4 \
    --data data/GroupTestingDataset --pretrained \
    --lr 0.001 --batch-size 32 -a resnext101_32x8d \
    --task-num 2 --log-name output.log \
    --output_dir Trained_Models/ResNeXt_K7_A4 \
    --dist-url 'tcp://127.0.0.1:7184' --dist-backend 'nccl' \
    --multiprocessing-distributed --epochs 200 \
    --world-size 1 --rank 0
```

> For SNR-specific cases, omit the `--SNR` flag to train without noise.

---

## Dataset Preparation

See `data_scripts/` for scripts to prepare the GroupTestingDataset from ImageNet (ILSVRC2012):

1. `extract_ILSVRC.sh` — Extract ImageNet train/val from tarballs
2. `create_dataset_from_imagenet.py` — Reorganize into group testing format

```bash
cd data_scripts
python create_dataset_from_imagenet.py > create_dataset_from_imagenet.sh
sh create_dataset_from_imagenet.sh
```

---

## Citation

If you use this code, please cite:

```
@article{valmeekam2024over,
  title={Over-the-Air Neural Group Testing},
  journal={IEEE},
  year={2024},
  url={https://ieeexplore.ieee.org/abstract/document/10624979}
}
```
