# Reducing the Error Propagation of Diffusion Models

Due to the chain structure, current diffusion models suffer from error propagation. [The paper](https://openreview.net/forum?id=RtAct1E2zS) in ICLR-2024 theoretically analyzes this problem and addresses it with efficient regularization. This repository contains a Pytorch implementation of the diffusion model with the introduced regularization method.

## Setup

Firstly, create a folder called "dataset", containing a set of fix-sized images. For example, 32 x 32 images from [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html). Image formats of many kinds (e.g., jpg, png, and tiff) are supported.

Secondly, fork the repository and build a virtual environment with some necessary packages

```
$ conda create --name tmp_env python=3.8
$ conda activate tmp_env
$ pip install -r requirements.txt
```

## Run Scripts

Train a regularized diffusion model with $1000$ denoising iterations and $128$ hidden units:

```
bash cases/reg_train.sh dataset 1000 128
```

Train an ordinary diffusion model with similar hyper-parameters:

```
bash cases/vanilla_train.sh dataset 1000 128
```

## Citation

```
@inproceedings{
li2024on,
title={On Error Propagation of Diffusion Models},
author={Yangming Li and Mihaela van der Schaar},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=RtAct1E2zS}
}
```
