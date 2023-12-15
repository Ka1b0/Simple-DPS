# A Simple implementation for DPS

## Diffusion Posterior Sampling for General Noisy Inverse Problems

Noisy (non)linear inverse problems solvers via approximation of the posterior sampling. [Details](https://openreview.net/forum?id=OnD9zGAGT0k)

This is a simple reimplementation based on [official code](https://github.com/DPS2022/diffusion-posterior-sampling)

## Main difference
1. Fix a severe bug in its implementation of MGS, which may lead to failure of generation.
2. Rewrite operators via [Kornia](https://github.com/kornia/kornia).
3. Add two non-linear operators (gamma adjustment and sobel filter) for low light enhancement and recoloring.
4. Rewrite some code in diffusion model to make it easy to follow, but only DDPM implemented.
5. Rewrite the cmd in inference for easy experiment.

## Prerequisites
- python 3.8

- pytorch 1.11.0

- CUDA 11.3.1

- nvidia-docker (if you use GPU in docker container)

It is okay to use lower version of CUDA with proper pytorch version.


<br />

## Getting started 

### 1) Clone the repository

```
git clone https://github.com/DPS2022/diffusion-posterior-sampling

cd diffusion-posterior-sampling
```

<br />

### 2) Download pretrained checkpoint
From the [link](https://drive.google.com/drive/folders/1jElnRoFv7b31fG0v6pTSQkelbSX3xGZh?usp=sharing), download the checkpoint "ffhq_10m.pt" or "imagenet256.pt" and paste it to ./models/
```
mkdir models
mv {DOWNLOAD_DIR}/ffqh_10m.pt ./models/
```
{DOWNLOAD_DIR} is the directory that you downloaded checkpoint to.


### 3) Set environment

Install dependencies

```
conda create -n DPS python=3.8

conda activate DPS

pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

pip install -r requirements.txt
```


### 4) Inference

```
python3 main.py \
--model_config configs/model_config.yaml \
--task_config configs/base_config.yaml \
--data {DATA_PATH} \
--task_name box-inpaint --noise guassian \
--conditioner ps --scale 0.3
```

For imagenet, use configs/imagenet_model_config.yaml

### 5) Quick start
It may take 3.5 min per task (7 in total).
```
bash scripts/example.sh
```


