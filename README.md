# ViT-Tiny: Augmentation Study on CIFAR-100, MNIST & USPS

Fine-tuning **`vit_tiny_patch16_224`** (ImageNet-pretrained, via `timm`) on three image
classification datasets, comparing **Mixup**, **CutMix**, and **RandAugment** — alone and
combined — against a plain fine-tuning baseline. All experiments were run in Google Colab
with data/checkpoints stored on Google Drive.

## Notebooks

| Notebook | Dataset | Classes | Image size (resized) |
|---|---|---|---|
| `VIT_TINY_CIFAR100_FINAL.ipynb` | CIFAR-100 | 100 | 224×224 |
| `VIT_TINY_MNIST_FINAL.ipynb` | MNIST | 10 | 224×224 |
| `VIT_TINY_USPS_FINAL.ipynb` | USPS | 10 | 224×224 |

## Common pipeline (each notebook)

1. **Setup** — mount Google Drive, load dataset from Drive (`download=False`), build
   `train`/`test` `DataLoader`s (batch size 64).
2. **Model** — `timm.create_model("vit_tiny_patch16_224", pretrained=True, num_classes=N)`.
3. **Training loop** — `AdamW` (lr `1e-4`, weight decay `0.05`) + `CosineAnnealingLR`
   (`T_max=20`), 20 epochs, checkpoint + best-model saving each epoch, resumable via
   saved `epoch`/`optimizer`/`scheduler` state.
4. **Experiments run per dataset** (each a separate re-training of the ViT-Tiny model):
   - Baseline (no augmentation)
   - Mixup
   - CutMix
   - RandAugment
   - Mixup + CutMix
   - Mixup + RandAugment
   - CutMix + RandAugment
   - Mixup + CutMix + RandAugment
   Mixup/CutMix use `timm.data.Mixup` with `SoftTargetCrossEntropy`; RandAugment is
   applied as a torchvision transform on the training set.
5. **Evaluation** — test accuracy, precision/recall/F1, classification report and
   confusion matrix (`sklearn`) for the best checkpoint of each variant; results
   collated into a comparison `DataFrame` + bar chart.
6. **AV — Attention Visualization** — forward hooks on each transformer block's
   attention module, **attention rollout** across layers, upsampled with OpenCV and
   overlaid on the input image (saved as PNG).
7. **RA — Robustness Analysis** — evaluates the best model under corruptions (Gaussian
   noise, Gaussian blur, brightness jitter, contrast jitter) and plots accuracy drop
   vs. the clean-test baseline.

## Results (test accuracy, best checkpoint per variant)

| Variant | CIFAR-100 | MNIST | USPS |
|---|---|---|---|
| Baseline | 86.96% | 99.65% | 99.25% |
| Mixup | 86.02% | 99.66% | 99.30% |
| CutMix | 86.91% | 99.68% | 99.25% |
| RandAugment | 86.78% | 99.62%¹ | 99.41% |
| Mixup + CutMix | 87.00% | 99.65%¹ | 80.05%² |
| Mixup + RandAugment | 86.20% | — | 99.41% |
| CutMix + RandAugment | **87.41%** | 99.65% | 99.41% |
| Mixup + CutMix + RandAugment | 87.16% | — | **99.46%** |

¹ Some MNIST run logs are incomplete/overwritten across cells — treat as approximate.
² USPS "Mixup+CutMix" combined run collapsed to ~80% accuracy (likely a training
instability/config issue) — worth re-running before trusting this number.

**Takeaway:** CutMix + RandAugment is the strongest single combo on CIFAR-100; on MNIST
and USPS (already near-ceiling ~99%), augmentation choice matters far less, with the
full Mixup+CutMix+RandAugment combo giving a small edge on USPS.

## Requirements

```
torch, torchvision, timm, tqdm, opencv-python (cv2), pandas, matplotlib, scikit-learn
```

## Reproducing

1. Open a notebook in Google Colab.
2. Update `data_path` / checkpoint paths to your own Google Drive locations.
3. Run cells top-to-bottom per section; each augmentation experiment re-initializes
   the model/optimizer/scheduler, so sections can be run independently once the model
   and data cells above them have executed.

## Notes / things to clean up if reusing this code

- Training/eval loops are re-defined in nearly every section (copy-pasted, not
  refactored into a shared module) — safe to consolidate into a single utils cell.
- Some checkpoint filenames have redundant `(1)`/`(3)` suffixes from repeated Drive
  uploads — verify you're loading the intended file.
- The USPS Mixup+CutMix run's ~80% result and a couple of MNIST epoch logs look like
  logging/training artifacts rather than final numbers; re-run those cells if you need
  clean figures.
