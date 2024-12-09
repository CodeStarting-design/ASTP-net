# ASTP-net

**Adaptive Spatio-temporal Partition Transformer for Video Dehazing**

## Introduction

This repository contains the official implementation of our paper: *Adaptive Spatio-temporal Partition Transformer for Video Dehazing*.

## Requirements

To set up the environment, use the following commands:

```bash
conda create -n ASTP python=3.10
conda activate ASTP
pip install -r requirements.txt
```

## Getting Started

### Model Training

Run the following command to train the model:

```bash
python train.py --model ASTP-s --gpu 0
```

### Model Testing

Run the following command to test the model:

```bash
python test.py --model ASTP-s --gpu 0
```

### Dataset

We trained and evaluated our model using the **HazeWorld** dataset. You can find details on dataset preparation [here](https://github.com/jiaqixuac/MAP-Net/blob/main/docs/dataset_prepare.md).

## Acknowledgements

This code is built upon the work of [DehazeFormer](https://github.com/IDKiro/DehazeFormer) and [NATTEN](https://github.com/SHI-Labs/NATTEN). We sincerely appreciate their contributions.
