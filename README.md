# Understanding the Impact of Data Augmentation Strategies on Vision Transformers

A multi-dataset analysis of performance, robustness, and interpretability

## 1. Overview

This repository contains a research-style empirical study of how data augmentation changes the behavior of Vision Transformers (ViTs). The project evaluates three augmentation families — Mixup, CutMix, and RandAugment — both individually and in combination across three benchmarks with increasing visual complexity:

- USPS (10-class grayscale digit classification)
- MNIST (10-class grayscale digit classification)
- CIFAR-100 (100-class RGB object classification)

The central question is not only which augmentation helps most, but also how augmentation changes learning dynamics, robustness to distribution shift, and the spatial attention patterns learned by the transformer.

The study uses ViT-Tiny models based on `vit_tiny_patch16_224` and compares them against larger ViT-Small variants on CIFAR-100. The notebook-driven experiments are organized per dataset and evaluate the eight augmentation settings used in the paper:

1. Baseline
2. Mixup
3. CutMix
4. RandAugment
5. Mixup + CutMix
6. Mixup + RandAugment
7. CutMix + RandAugment
8. Mixup + CutMix + RandAugment

---

## 2. Research Motivation

Vision Transformers are powerful visual learners, but unlike CNNs they have weaker built-in inductive bias. This makes them more sensitive to overfitting and more dependent on regularization when data is limited or visually complex.

Data augmentation acts as a practical regularizer by:

- increasing the diversity of training examples,
- smoothing decision boundaries,
- encouraging invariance to geometric and photometric transformations,
- improving robustness to perturbations such as noise, blur, and brightness shifts.

However, augmentation effects are not universal. In this project, the best strategy changes with dataset complexity, model capacity, and corruption type, which is exactly the phenomenon the notebooks and experiments aim to uncover.

---

## 3. Project Structure

```text
IIIT_ALLAHABAD_RESEARCH_PROJECT-main/
├── README.md
├── VIT_TINY_USPS_FINAL (1).ipynb
├── VIT_TINY_MNIST_FINAL (1).ipynb
├── VIT_TINY_CIFAR100_FINAL (1).ipynb
└── supporting checkpoints / weights (generated during training)
```

### Notebook analysis

- `VIT_TINY_USPS_FINAL (1).ipynb`
  - loads USPS via OpenML / torchvision-style preprocessing,
  - resizes images to 224×224,
  - compares 8 augmentation configurations,
  - evaluates clean accuracy and perturbation robustness,
  - visualizes attention maps and decision focus.

- `VIT_TINY_MNIST_FINAL (1).ipynb`
  - uses MNIST with grayscale-to-RGB conversion to fit the ViT patch pipeline,
  - tests the same augmentation combinations,
  - records best validation and test performance,
  - confirms near-saturation performance on an easy-to-learn dataset.

- `VIT_TINY_CIFAR100_FINAL (1).ipynb`
  - evaluates the most challenging task in the study,
  - follows the same training recipe across augmentation settings,
  - focuses on the larger performance gaps and corruption sensitivity,
  - compares ViT-Tiny and ViT-Small behavior.

Each notebook is designed as a research experiment rather than a generic tutorial: it tracks best checkpoints, logs validation accuracy each epoch, saves model state, and measures robustness across multiple corruption conditions.

---

## 4. Model and Training Setup

The study keeps the architecture and optimization recipe fixed across experiments, changing only the augmentation policy.

| Component | Configuration |
|---|---|
| Backbone | `vit_tiny_patch16_224` |
| Pretraining | ImageNet-pretrained |
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 0.05 |
| Scheduler | Cosine annealing |
| Loss | Cross-entropy / soft target cross-entropy for Mixup-style augmentation |
| Epochs | 20 |
| Batch size | 64 |
| Input resolution | 224×224 |

The notebooks also contain checkpointing logic that saves the best-performing validation model and periodic snapshots for each augmentation regime.

---

## 5. Augmentation Methods Used

### 5.1 Mixup

Mixup creates virtual examples by blending features and labels:

$$
\tilde{x} = \lambda x_i + (1-\lambda)x_j
$$

$$
\tilde{y} = \lambda y_i + (1-\lambda)y_j
$$

This encourages smooth interpolation between examples and reduces overconfident predictions in low-data settings.

### 5.2 CutMix

CutMix replaces a region of one image with a region from another image while also mixing labels proportionally to the area replaced. It combines local feature dropout with region-level contrast and tends to improve localization and robustness.

### 5.3 RandAugment

RandAugment applies a random sequence of augmentations with controlled magnitude. It is useful because it does not require policy search and is computationally light while still creating strong image diversity.

### 5.4 Combined strategies

The study tests combinations such as:

- Mixup + CutMix
- Mixup + RandAugment
- CutMix + RandAugment
- Mixup + CutMix + RandAugment

This is important because augmentation interactions are often non-additive: the best multi-augmentation configuration may not simply be the sum of the best individual components.

---

## 6. Dataset Details

| Dataset | Classes | Data type | Complexity | Main challenge |
|---|---:|---|---|---|
| USPS | 10 | Grayscale digits | Low | Clean structure, saturated performance ceiling |
| MNIST | 10 | Grayscale digits | Medium | Simple digit shapes, strong baseline performance |
| CIFAR-100 | 100 | RGB natural images | High | Intra-class variability, object diversity, harder generalization |

The three datasets form a progressive difficulty ladder: USPS and MNIST are relatively easy, while CIFAR-100 is substantially harder and benefits more clearly from regularization and augmentation diversity.

---

## 7. Quantitative Results

### 7.1 Best accuracy by dataset

The empirical results from the project notebooks and paper text are summarized below.

| Method | USPS | MNIST | CIFAR-100 |
|---|---:|---:|---:|
| Baseline | 99.25% | 99.66% | 86.96% |
| Mixup | 99.30% | 99.67% | 86.02% |
| CutMix | 99.25% | 99.68% | 86.91% |
| RandAugment | 99.41% | 99.62% | 86.78% |
| Mixup + CutMix | 80.05% | 99.65% | 87.07% |
| Mixup + RandAugment | 99.41% | 99.73% | 86.20% |
| CutMix + RandAugment | 99.41% | 99.64% | 87.41% |
| Mixup + CutMix + RandAugment | 99.46% | 99.69% | 87.16% |

### Best-performing setup by dataset

| Dataset | Best method | Best accuracy |
|---|---|---:|
| USPS | Mixup + CutMix + RandAugment | 99.46% |
| MNIST | Mixup + RandAugment | 99.73% |
| CIFAR-100 | CutMix + RandAugment | 87.41% |

### 7.2 Key interpretation of the numbers

- On USPS, performance is already near the ceiling; augmentation still helps, but the gain is small.
- On MNIST, the model is highly robust and all augmentation setups remain above 99.6%.
- On CIFAR-100, the gap between strategies is much larger, showing that augmentation is decisive when visual complexity rises.
- A strong warning sign appears in USPS with Mixup + CutMix alone, which falls to 80.05%, showing that aggressive mixing can be harmful if not combined with controlled transforms.

---

## 8. Robustness Analysis

The notebooks evaluate model reliability under common perturbations: Gaussian noise, brightness changes, blur, and occlusion/contrast variation.

| Condition | USPS | MNIST | CIFAR-100 |
|---|---:|---:|---:|
| Clean | 99.46% | 99.61% | 87.41% |
| Gaussian Noise | 92.96% | 99.58% | 58.80% |
| Brightness | 84.89% | 99.56% | 86.59% |
| Blur | 99.46% | 99.64% | 87.20% |
| Occlusion / Contrast | 99.30% | 99.58% | 86.83% |

### Robustness interpretation

- USPS remains stable under blur and occlusion, but brightness changes are destructive.
- MNIST is extremely stable, with accuracy above 99.5% even under corruption.
- CIFAR-100 shows a major drop under Gaussian noise, from 87.41% to 58.80%, identifying noise as the hardest corruption mode for complex image recognition.

This finding supports a central conclusion: augmentation improves not just accuracy, but also representation stability under realistic shifts.

---

## 9. Attention / Image Interpretation

The paper and figure annotations in the research material emphasize the interpretability component of the study.

### What the attention maps show

- For USPS and MNIST, attention is strongly concentrated around digit contours and structural strokes.
- For CIFAR-100, attention becomes broader and more spatially distributed, reflecting the complexity of object scenes and multiple object parts.
- Augmentation appears to encourage more distributed, semantically meaningful attention, which is associated with better generalization and reduced overfitting.

### Image-based interpretation

The visual evidence indicates that the ViT learns to allocate attention to feature-rich image regions rather than noise or background artifacts. This explains why augmentation strategies improve both performance and reliability: they encourage the model to rely on invariant, high-signal features instead of brittle shortcuts.

---

## 10. Scaling Study: ViT-Tiny vs. ViT-Small

To assess model-capacity effects, a CIFAR-100 scaling experiment was added using ViT-Small.

| Model | Parameters | Best accuracy |
|---|---:|---:|
| ViT-Tiny | 5.5M | 87.41% |
| ViT-Small | 21.7M | 89.42% |

### Scaling conclusion

The larger model improves by around 2.01 percentage points over the best ViT-Tiny configuration. This demonstrates that stronger transformer capacity can exploit augmentation diversity more effectively.

A notable result is that the optimal augmentation recipe changes with model size:

- ViT-Tiny best: CutMix + RandAugment
- ViT-Small best: Mixup + CutMix

This suggests that augmentation choice is not independent of model size; the best curriculum depends on the inductive and representational capacity of the network.

---

## 11. Core Findings

1. Augmentation helps more on complex datasets than on simple ones.
2. The effect of augmentation is dataset-dependent and often non-linear.
3. The three-way combination is not always optimal.
4. RandAugment is particularly important for robust, controlled regularization.
5. Larger models can exploit augmentation diversity more effectively.
6. Attention maps reveal that augmentation improves the semantic focus of the model.

---

## 12. Why this project matters

This project is valuable because it connects four important dimensions in one pipeline:

- classification performance,
- robustness under corruption,
- interpretability through attention,
- capacity scaling across architectures.

Most augmentation studies examine only accuracy. This project asks a deeper question: how do augmentation strategies reshape a transformer’s internal representation and reliability? That is why it is an important empirical study in modern computer vision.

---

## 13. Reproducibility Notes

To reproduce the project:

```bash
pip install torch torchvision timm matplotlib pandas scikit-learn tqdm
```

Then open the relevant notebook and run each cell sequentially. Each notebook is designed to be self-contained and includes:

- dataset setup,
- image preprocessing,
- model initialization,
- augmentation configuration,
- training loop,
- evaluation metrics,
- checkpoint saving,
- robustness and visualization cells.

---

## 14. Practical Research Summary

This project demonstrates that augmentation is not a minor engineering detail — it is a key factor controlling the training behavior of Vision Transformers. The empirical results show that:

- small datasets can saturate quickly,
- medium-complexity datasets still benefit from augmentation,
- complex visual tasks benefit the most from strategically chosen augmentation policies,
- and larger ViTs can exploit augmentation more effectively than smaller ones.

The main practical takeaway is that augmentation policy design should be tuned with both dataset complexity and model capacity in mind.

---

## 15. Conclusion

This repository presents a complete empirical study of augmentation in Vision Transformers across three datasets and multiple robustness scenarios. The results show that augmentation improves performance and representation quality, but its effectiveness is highly conditional on the dataset and architecture.

The strongest general message is simple:

> The best augmentation strategy is not universal; the best strategy is task-aware, dataset-aware, and model-aware.

---

## 16. Citation / academic framing

This work fits into the broader literature on ViT training, regularization, robustness, and transformer interpretability. It aligns with the growing recognition that data augmentation is a crucial component of ViT design, especially when training on limited or moderately complex datasets.

---

## 17. Author

Satvik Pandey

---

## 18. License

This project is distributed for academic and research use.

