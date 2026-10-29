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

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning Rate | 1e-4 |
| Weight Decay | 1e-4 |
| Batch Size | 32 |
| Epochs | 50 |
| Scheduler | ReduceLROnPlateau (factor=0.1, patience=5) |
| Loss | Cross-Entropy |
| Backbone Strategy | Frozen |

## Data Preprocessing

| Step | Parameter |
|------|-----------|
| Resize | 224×224 |
| Normalization | Z-score per image |
| Augmentation | Rotation ±15°, flips, zoom 0.9-1.1, brightness/contrast |
| Skull Stripping | Not applied |

## Evaluation Metrics

- Weighted Accuracy
- Weighted F1-Score
- Precision, Recall
- MCC, Cohen's Kappa
- ECE (Expected Calibration Error)
- XAI Consensus Score
- XAI Agreement Rate

## Ablation Study Results

| Configuration | Accuracy | XAI Consensus | XAI Agreement |
|---------------|----------|---------------|---------------|
| Full Model (SF-1) | 93.44% | 0.7458 | 96.4% |
| w/o SF-1 (Flattened) | 91.50% | 0.512 | 78.0% |
| w/o Swin Transformer | 91.20% | 0.698 | 92.0% |
| w/o ResNet-50 | 91.80% | 0.712 | 93.0% |
| w/o EfficientNet-B3 | 94.66% | — | — |
| Attention Fusion | 97.18% | — | — |

## Limitations

1. Patient-level data leakage cannot be guaranteed (no patient IDs in source dataset)
2. Test set class distribution differs from full dataset (No Tumor: 31% vs 15%)
3. XAI consistency does not establish clinical validity
4. Model operates on 2D slices, not 3D volumes
5. Statistical significance not achieved vs. baselines (p > 0.05)

## References

See the paper references for detailed citations.