<div align="center">

# 🔍 ViT Augmentation Study

### Understanding the Impact of Data Augmentation Strategies on Vision Transformers
**A Multi-Dataset Analysis of Performance, Robustness, and Interpretability**

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![timm](https://img.shields.io/badge/timm-ViT-blue)](https://github.com/huggingface/pytorch-image-models)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## 📌 Overview

This project empirically studies how **Mixup**, **CutMix**, and **RandAugment** — alone
and combined — affect Vision Transformers, using **ViT-Tiny** (`vit_tiny_patch16_224`,
ImageNet-pretrained) fine-tuned on three datasets of increasing visual complexity:
**USPS → MNIST → CIFAR-100**.

Beyond accuracy, the study evaluates:

- 🛡️ **Robustness** under Gaussian noise, blur, brightness, and occlusion
- 👁️ **Interpretability** via attention rollout / attention map visualization
- 📈 **Scaling** — a ViT-Small vs. ViT-Tiny comparison on CIFAR-100

## 📂 Repository Structure

```
.
├── VIT_TINY_USPS_FINAL.ipynb        # USPS: 8 augmentation configs + robustness + attention
├── VIT_TINY_MNIST_FINAL.ipynb       # MNIST: 8 augmentation configs + robustness + attention
├── VIT_TINY_CIFAR100_FINAL.ipynb    # CIFAR-100: 8 augmentation configs + robustness + attention
├── report/                          # Final written report (PDF)
└── README.md
```

## 🗂️ Datasets

| Dataset | Classes | Image Type | Complexity |
|---|---|---|---|
| USPS | 10 | Grayscale digits | Low |
| MNIST | 10 | Grayscale digits | Medium |
| CIFAR-100 | 100 | RGB images | High |

## ⚙️ Training Configuration

| Parameter | Value |
|---|---|
| Architecture | ViT-Tiny Patch16-224 |
| Pretraining | ImageNet |
| Optimizer | AdamW |
| Learning Rate | 1e-4 |
| Weight Decay | 0.05 |
| Scheduler | Cosine Annealing |
| Loss | Cross-Entropy |
| Epochs | 20 |
| Batch Size | 64 |
| Input Resolution | 224 × 224 |

Eight augmentation configurations were evaluated per dataset: **Baseline, Mixup, CutMix,
RandAugment, Mixup+CutMix, Mixup+RandAugment, CutMix+RandAugment,
Mixup+CutMix+RandAugment.**

## 🚀 Getting Started

```bash
git clone https://github.com/<your-username>/vit-augmentation-study.git
cd vit-augmentation-study
pip install torch torchvision timm tqdm opencv-python pandas matplotlib scikit-learn
```

Each notebook is self-contained — open it in Jupyter/Colab, point the dataset/checkpoint
paths to your own storage, and run top-to-bottom. Sections are organized by augmentation
config, so you can also run a single section once the setup cells above it have executed.

## 📊 Results

### Accuracy by Augmentation Strategy

| Method | USPS | MNIST | CIFAR-100 |
|---|---|---|---|
| Baseline | 99.25 | 99.66 | 86.96 |
| Mixup | 99.30 | 99.67 | 86.02 |
| CutMix | 99.25 | 99.68 | 86.91 |
| RandAugment | 99.41 | 99.62 | 86.78 |
| Mixup+CutMix | 80.05 ⚠️ | 99.65 | 87.07 |
| Mixup+RandAugment | 99.41 | **99.73** | 86.20 |
| CutMix+RandAugment | 99.41 | 99.64 | **87.41** |
| Mixup+CutMix+RandAugment | **99.46** | 99.69 | 87.16 |

> ⚠️ On USPS, Mixup+CutMix alone destabilized training (80.05%) — RandAugment-inclusive
> configs were consistently reliable.

### Robustness (Accuracy % under Corruption)

| Condition | USPS | MNIST | CIFAR-100 |
|---|---|---|---|
| Clean | 99.46 | 99.61 | 87.41 |
| Gaussian Noise | 92.96 | 99.58 | 58.80 |
| Brightness | 84.89 | 99.56 | 86.59 |
| Blur | 99.46 | 99.64 | 87.20 |
| Occlusion/Contrast | 99.30 | 99.58 | 86.83 |

MNIST stays >99.5% under every corruption; CIFAR-100 is by far the most sensitive,
especially to Gaussian noise.

### ViT-Tiny vs. ViT-Small (CIFAR-100)

| Model | Parameters | Best Accuracy |
|---|---|---|
| ViT-Tiny | 5.5M | 87.41% |
| ViT-Small | 21.7M | **89.42%** |

ViT-Small (only 5 training epochs) outperforms the best ViT-Tiny result, with its best
config being Mixup+CutMix rather than CutMix+RandAugment — augmentation effectiveness
shifts with model capacity.

## 🔑 Key Findings

- **Dataset complexity drives augmentation impact** — USPS/MNIST are near-saturated;
  CIFAR-100 sees meaningful gains.
- **Augmentation effects aren't strictly additive** — the 3-way combo isn't always best.
- **Robustness scales with augmentation strength**, but complex datasets stay more fragile.
- **Attention rollout** shows augmentation drives more distributed, generalizable attention.
- **Model capacity interacts with augmentation choice** — bigger models, different optimal recipe.

## 🧰 Tech Stack

`PyTorch` · `timm` · `torchvision` · `scikit-learn` · `OpenCV` · `matplotlib` · `pandas`

## 🔮 Future Work

- Larger transformer architectures
- Self-supervised / contrastive augmentation
- Adversarial robustness evaluation
- Augmentation-aware transformer optimization



## 👤 Author

**Satvik Pandey**

## 📜 License

MIT — see [LICENSE](LICENSE) for details.
