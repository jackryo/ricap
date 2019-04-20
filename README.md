
# RICAP: Data Augmentation using Random Image Cropping and Patching for Deep CNNs

PyTorch implementation of data augmentation method RICAP for deep CNNs proposed by "[Data Augmentation using Random Image Cropping and Patching for Deep CNNs](https://arxiv.org/abs/1811.09030)."

## Prerequisites

* Python 3.5
* PyTorch 1.0
* GPU (recommended)

## Datasets

* CIFAR-10/100: automatically downloaded by PyTorch scripts to `data` folder
* ImageNet: manually downloaded from [ImageNet](http://www.image-net.org/) (ILSVRC2012 version) and moved to `train` and `val` folders in your `dataroot` path (e.g., `./imagenet/`)

## Results

### CIFAR

|                         |        CIFAR-10        |        CIFAR-100        |
|:------------------------|:----------------------:|:-----------------------:|
| WideResNet28-10         |          3.89          |          18.85          |
| WideResNet28-10 + RICAP | **2.85** &plusmn; 0.06 | **17.22** &plusmn; 0.20 |

### ImageNet

|                        | Epochs |   top-1   |  top-5   |
|:-----------------------|:------:|:---------:|:--------:|
| WideResNet50-2         |  100   |   21.90   |   6.03   |
| WideResNet50-2 + RICAP |  100   |   21.08   |   5.66   |
| WideResNet50-2         |  200   |   21.84   |   6.03   |
| WideResNet50-2 + RICAP |  200   | **20.33** | **5.26** |

* Details are in our [paper](https://arxiv.org/abs/1811.09030).

## How to Train

Our script occupies all available GPUs. Please set environment `CUDA_VISIBLE_DEVICES`.

### CIFAR-10 and WideResNet28-10

with RICAP

```bash
python main.py --dataset cifar10 --model WideResNetDropout --depth 28 --params 10 --beta_of_ricap 0.3 --postfix ricap0.3
```

without RICAP

```bash
python main.py --dataset cifar10 --model WideResNetDropout --depth 28 --params 10
```

We trained these models on a single GPU (GeForce GTX 1080).

### CIFAR-100 and WideResNet28-10

with RICAP

```bash
python main.py --dataset cifar100 --model WideResNetDropout --depth 28 --params 10 --beta_of_ricap 0.3 --postfix ricap0.3
```

without RICAP

```bash
python main.py --dataset cifar100 --model WideResNetDropout --depth 28 --params 10
```

We trained these models on a single GPU (GeForce GTX 1080).


### ImageNet and WideResNetBottleneck50-2 for 100 epochs

with RICAP

```bash
python main.py --dataset ImageNet --dataroot [your imagenet folder path(like ./imagenet)] --model WideResNetBottleneck --depth 50 --epoch 100 --adlr 30,60,90 --droplr 0.1 --wd 1e-4 --batch 256 --params 2 --beta_of_ricap 0.3 --postfix ricap0.3
```

without RICAP

```bash
python main.py --dataset ImageNet --dataroot [your imagenet folder path(like ./imagenet)] --model WideResNetBottleneck --depth 50 --epoch 100 --adlr 30,60,90 --droplr 0.1 --wd 1e-4 --batch 256 --params 2
```

We trained these models on four GPUs (GeForce GTX 1080).

## References

```bibtex
@inproceedings{RICAP2018ACML,
  title = {RICAP: Random Image Cropping and Patching Data Augmentation for Deep CNNs},
  author = {Takahashi, Ryo and Matsubara, Takashi and Uehara, Kuniaki},
  booktitle = {Asian Conference on Machine Learning (ACML)},
  url={http://proceedings.mlr.press/v95/takahashi18a.html},
  year = {2018}
}
```

```bibtex
@article{RICAP2018arXiv,
  title={Data Augmentation using Random Image Cropping and Patching for Deep CNNs},
  author={Takahashi, Ryo and Matsubara, Takashi and Uehara, Kuniaki},
  journal={arXiv},
  url={https://arxiv.org/abs/1811.09030},
  year={2018}
}
```
