# Over-the-Air Neural Group Testing (OTA-NGT)

This repository contains the official implementation of the paper **[Over-the-Air Neural Group Testing](https://ieeexplore.ieee.org/abstract/document/10624979)**.

OTA-NGT is an innovative approach that leverages neural networks for efficient group testing in over-the-air settings. The framework supports multiple transmission and testing algorithms, along with customizable configurations for different use cases.

---

## Main Script

The primary script for running the OTA-NGT algorithm is:

**`main_dist.py`**

---

## Configuration Options

### 1. Model Design Selection

The import statement in the `main_dist.py` file determines the model design. You can toggle between two available designs:

- Design 2:
  ```python
  import resnet_design2 as models
  ```

- Design 3:
  ```python
  import resnet_design3 as models
  ```

---

### 2. Algorithm Selection (--GT-alg)

Specify the desired testing algorithm using the --GT-alg flag. Options include:
1. ITIT: Individual Transmission Individual Testing
2. GTGT-FM: Group Transmission Group Testing with Feature Merge
3. ITGT-FM: Individual Transmission Group Testing with Feature Merge
4. GTGT-PM: Group Transmission Group Testing with Pixel Merge

---

### 3. Noise Configuration (--SNR)

Set the Signal-to-Noise Ratio (SNR) in dB for noisy transmission scenarios.

---

### 4. Group Size (--background-K)

The --background-K flag defines the group size minus one. For instance:
* 0: Group size = 1
* 3: Group size = 4

---

### 4. Example Code 

```bash
CUDA_VISIBLE_DEVICES=0,1 python -u main_dist.py --background-K 7 --GT-alg 4 --data data/GroupTestingDataset --pretrained --lr 0.001  --batch-size 32 -a resnext101_32x8d --task-num 2 --log-name DEBUG.log --output_dir Trained_Models/ResNeXt_K7_A4 --dist-url 'tcp://127.0.0.1:7184' --dist-backend 'nccl' --multiprocessing-distributed --epochs 200 --world-size 1 --rank 0 > Log_files/log_ResNeXt_K7_A4.txt 2>&1 || true
```

---

### Notes

* Detailed Documentation: A more comprehensive README with in-depth instructions, examples, and use cases is currently under development.
* Citation: If you use this code, please cite our paper **[Over-the-Air Neural Group Testing](https://ieeexplore.ieee.org/abstract/document/10624979)**.

---
Feel free to reach out with issues or questions by creating an issue on this repository.

