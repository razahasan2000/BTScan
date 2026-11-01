# BTScan Methodology Documentation

## Overview

BTScan (BrainTechScan) is a hybrid tri-modal deep learning framework for brain tumor MRI classification. The model integrates three backbone architectures with a novel SF-1 Spatial Fusion layer to preserve spatial topology for Explainable AI (XAI) analysis.

## Architecture

### Backbone Networks

| Backbone | Features | Output Channels | Spatial Resolution | Role |
|----------|----------|-----------------|-------------------|------|
| Swin Transformer (tiny) | Shifted-window self-attention | 768 | 7×7 | Global context |
| ResNet-50 | Residual connections | 2,048 | 7×7 | Hierarchical spatial features |
| EfficientNet-B3 | Compound scaling | 384 | 7×7 | Fine-grained textural features |

### SF-1 Spatial Fusion Layer

The SF-1 layer preserves spatial topology during feature fusion:

1. **Feature Extraction**: Extract features from final stages of each backbone
2. **Spatial Alignment**: Bilinear interpolation to 7×7 resolution
3. **Channel Concatenation**: Concatenate to form 3,200-channel tensor
4. **1×1 Convolution**: Learnable fusion to 512 channels
5. **Classification**: Adaptive pooling + Linear layers

### Mathematical Formulation

Spatial fusion operation:

$$F_{fused}(i, j) = \text{Conv}_{1\times1}\left(\text{Concat}\left(\uparrow F_{Swin}(i,j), \uparrow F_{ResNet}(i,j), \uparrow F_{Eff}(i,j)\right)\right)$$

Where $\uparrow$ denotes bilinear interpolation to 7×7 resolution.

XAI Consensus Score:

$$C = \alpha \cdot \bar{R} + (1-\alpha) \cdot \bar{S}$$

Where $\alpha=0.5$, $\bar{R}$ = mean pairwise Pearson correlation, $\bar{S}$ = mean pairwise SSIM.

Inter-Method Region-Based Concordance (primary measure):

Each attribution map is normalised to [0,1] and binarised at 20% of its maximum. Pairwise IoU and Dice are computed across all 6 method pairs.

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning Rate | 1e-4 |
| Weight Decay | 1e-4 |
| Batch Size | 32 (training), 16 (evaluation) |
| Epochs | 50 (main), 25 (ablation) |
| Scheduler | ReduceLROnPlateau (factor=0.1, patience=5) |
| Loss | Cross-Entropy |
| Backbone Strategy | Frozen (pretrained weights fixed) |

## Data Preprocessing

| Step | Parameter |
|------|-----------|
| Resize | 224×224 |
| Normalization | ImageNet mean/std (0.485, 0.456, 0.406 / 0.229, 0.224, 0.225) |
| Augmentation | Rotation ±10°, horizontal flip, color jitter |
| Skull Stripping | Not applied |
| Deduplication | pHash + SSIM > 0.92 threshold |

## Evaluation Metrics

- Weighted Accuracy
- Weighted F1-Score
- Per-class Precision, Recall
- MCC, Cohen's Kappa
- ECE (Expected Calibration Error)
- 95% Bootstrap CIs (1000 iterations)
- **XAI Consensus Score (C)**
- **Inter-Method Region-Based IoU / Dice** (primary XAI concordance)

## Ablation Study Results (n=100 tumour-class test subset, seed 42)

| Configuration | Accuracy | Inter-Method IoU | Parameters |
|---------------|----------|------------------|------------|
| Full Model (SF-1, deployed) | 91.0% | 0.192 | 148.2M |
| Flattened (concat_flatten) | 90.0% | 0.203 | 145.0M |
| Concat+GAP | 92.0% | 0.215 | 145.0M |
| Concat+1×1 | 91.0% | 0.221 | 145.0M |
| Attention Fusion | 96.0% | 0.235 | 145.0M |
| w/o Swin | 86.0% | 0.238 | 80.4M |
| w/o ResNet | 88.0% | 0.219 | 71.0M |
| w/o EfficientNet | 91.0% | 0.208 | 110M |
| Swin Only | 84.0% | 0.215 | 28M |
| ResNet Only | 83.0% | 0.262 | 25M |
| EfficientNet Only | 80.0% | 0.237 | 12M |

Full test set accuracy (1,311 images): **93.44%** (95% CI: [92.1%, 94.7%])

### Key Findings

1. **SF-1 Spatial Fusion does NOT improve inter-method concordance**: The deployed model (0.192 IoU) has the lowest concordance of all 11 configurations.
2. **Fusion design has no measurable effect on concordance**: Inter-method IoU varies only 0.203–0.262 (SD = 0.017) across the 10 variants.
3. **Concordance is not a proxy for accuracy**: Accuracy vs. IoU across configurations shows r = −0.46 (p = 0.16).
4. **Between-run variation exceeds between-configuration variation**: Re-training the identical architecture moves IoU by 0.027, larger than the between-configuration SD (0.017).
5. **Attention fusion achieves highest accuracy** (96.0% on evaluation subset) and also preserves spatial structure.

## XAI Consensus Analysis (n=100 tumour-class test images, seed 42)

### Consensus Metrics (REAL model attributions)

1. **XAI Consensus Score (C)**: C = 0.5·R̄ + 0.5·S̄
   - **Measured value: C = 0.116 (95% CI: 0.111–0.122, bootstrap 1000 iterations)**
   - R̄ = 0.0479 ± 0.0553, S̄ = 0.1848 ± 0.0271 (unrounded)
   - C was 0.117 for correctly classified cases (n=91) and 0.108 for incorrect (n=9) — no association with decision correctness.

2. **Inter-Method Region-Based Overlap** (primary concordance measure):
   - **Mean IoU = 0.192 ± 0.061**, Dice = 0.258 ± 0.070
   - **Grad-CAM vs. LIME IoU = 0.665 ± 0.154** (strongest pair)
   - Occlusion is intermediate (~0.21 against each)
   - Pixel-gradient Saliency is spatially uncorrelated with region-based methods (~0.025)

3. **XAI Agreement Rate** (thresholded): **Retired**
   - Threshold: Pearson r > 0.85 AND SSIM > 0.60
   - On real attributions: **0 of 100 images** met this criterion under every reading tested
   - The metric is retired rather than re-thresholded.

### Pairwise Correlations (whole-map, real attributions)

| Method | Grad-CAM | Saliency | Occlusion | LIME |
|--------|----------|----------|-----------|------|
| Grad-CAM | 1.000 | 0.051 | 0.211 | 0.032 |
| Saliency | 0.051 | 1.000 | 0.019 | −0.040 |
| Occlusion | 0.211 | 0.019 | 1.000 | 0.014 |
| LIME | 0.032 | −0.040 | 0.014 | 1.000 |

> **Correction**: Previous versions reported C = 0.7458 and agreement rate = 96.4%. Those values were produced by a synthetic heatmap generator and are **withdrawn**. All values above are recomputed from real Grad-CAM, Saliency, Occlusion, and LIME gradients on real test images.

## Limitations

1. Patient-level data leakage cannot be guaranteed (no patient IDs in source dataset)
2. Test set class distribution differs from full dataset (No Tumor: 31% vs 15%)
3. XAI consistency (inter-method agreement) does **not** establish clinical validity
4. No ground-truth segmentation masks available in Kaggle dataset
5. Model operates on 2D slices, not 3D volumes
6. Statistical significance not achieved vs. baselines (p > 0.05, McNemar's test)

## References

See the paper references for detailed citations.