# BTScan: Spatially Preserved CNN Transformer Fusion for Multi-Method XAI Consensus in Brain Tumor MRI Classification

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Kaggle Dataset](https://img.shields.io/badge/Dataset-Kaggle-blue)](https://doi.org/10.34740/kaggle/dsv/14832123)

## Overview

BTScan (BrainTechScan) is a hybrid tri-modal deep learning framework for brain tumor MRI classification that integrates Swin Transformer, ResNet-50, and EfficientNet-B3 backbones with a novel SF-1 Spatial Fusion layer. The model preserves spatial topology during feature fusion, enabling quantitative multi-method Explainable AI (XAI) consensus across Grad-CAM, Saliency, Occlusion, and LIME.

### Key Features

- **SF-1 Spatial Fusion Layer**: Preserves 7×7 spatial topology across 3,200 heterogeneous feature channels
- **Tri-Modal Architecture**: Swin Transformer + ResNet-50 + EfficientNet-B3
- **Multi-Method XAI Consensus**: Quantitative consensus scoring across 4 XAI methods
- **Clinical Triage**: 99.75% recall and 99.89% NPV for "No Tumor" class
- **Leakage-Free Validation**: pHash/SSIM-based deduplication with 70/15/15 split

### Performance Metrics (Test Set: 1,311 images)

| Metric | Value | 95% CI (Bootstrap) |
|--------|-------|-------------------|
| Weighted Accuracy | 93.44% | [92.1%, 94.7%] |
| Weighted F1-Score | 0.9335 | [0.920, 0.948] |
| MCC | 0.9123 | [0.894, 0.930] |
| Cohen's Kappa | 0.9118 | [0.895, 0.928] |
| ECE (Calibration) | 0.0531 | — |
| **No Tumor Recall** | **99.75%** | **[99.2%, 100%]** |
| **No Tumor NPV** | **99.89%** | **[99.7%, 100%]** |
| XAI Consensus Score (C) | 0.116 | [0.111, 0.122] |
| Inter-Method IoU (mean) | 0.192 ± 0.061 | — |
| Inter-Method Dice (mean) | 0.258 ± 0.070 | — |

---

## Table of Contents

1. [Installation](#installation)
2. [Project Structure](#project-structure)
3. [Data](#data)
4. [Usage](#usage)
5. [Model Weights](#model-weights)
6. [Ablation Studies](#ablation-studies)
7. [XAI Consensus Analysis](#xai-consensus-analysis)
8. [Citation](#citation)
9. [License](#license)
10. [Acknowledgments](#acknowledgments)

---

## Installation

### Prerequisites

- Python 3.10 or higher
- CUDA-compatible GPU (recommended)
- 16GB+ RAM

### Setup

```bash
# Clone the repository
git clone https://github.com/razahasan2000/BTScan.git
cd BTScan

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Required Packages

See `requirements.txt` for the complete list. Key dependencies:

- `torch>=2.0.0`
- `torchvision>=0.15.0`
- `timm>=1.0.0`
- `numpy>=1.24.0`
- `pandas>=2.0.0`
- `scikit-learn>=1.3.0`
- `scikit-image>=0.20.0`
- `matplotlib>=3.7.0`
- `seaborn>=0.12.0`
- `tqdm>=4.60.0`

---

## Project Structure

```
BTScan_Project/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── setup.py
│
├── data/
│   ├── processed/
│   │   ├── y_true.npy          # True labels (test set, 1311 samples)
│   │   ├── y_pred.npy          # Predicted labels (test set, 1311 samples)
│   │   └── y_prob.npy          # Prediction probabilities (1311×4)
│   └── raw/
│       └── README.md           # Instructions for downloading Kaggle dataset
│
├── models/
│   └── README.md               # Model weights download instructions
│   └── Hybrid_Spatial_Model-Brain-Tumor.pth  (240 MB - see below)
│
├── src/
│   ├── train_model.py          # Training script
│   ├── evaluate_model.py       # Evaluation script
│   ├── ablation_study.py       # Ablation study experiments
│   └── xai_consensus.py        # XAI consensus quantification
│
├── results/
│   └── ablation_results/
│       ├── full_sf1_results.npy
│       ├── w_o_efficientnet_results.npy
│       ├── w_o_resnet_results.npy
│       ├── w_o_swin_results.npy
│       ├── swin_only_results.npy
│       ├── resnet_only_results.npy
│       ├── efficientnet_only_results.npy
│       ├── w_o_sf1_attention_results.npy
│       ├── w_o_sf1_concat_1x1_results.npy
│       └── w_o_sf1_concat_gap_results.npy
│
├── notebooks/
│   └── README.md               # Jupyter notebooks for exploration
│
├── docs/
│   └── methodology.md          # Detailed methodology documentation
│
└── tests/
    └── test_model.py           # Unit tests for model components
```

---

## Data

### Dataset Source

The dataset is the publicly available **Kaggle Brain Tumor MRI Dataset**:
- **DOI**: https://doi.org/10.34740/kaggle/dsv/14832123
- **Total Images**: 7,023
- **Classes**: Glioma, Meningioma, Pituitary, No Tumor
- **Image Size**: 224×224 pixels

### Dataset Split

| Split | Images | Percentage |
|-------|--------|------------|
| Training | 4,856 | 69.1% |
| Validation | 856 | 12.2% |
| Test | 1,311 | 18.7% |

**Note**: pHash/SSIM-based deduplication (SSIM > 0.92) was applied to minimize near-duplicate leakage across splits. Patient-level separation cannot be guaranteed as the source dataset does not provide patient identifiers.

### Downloading the Dataset

1. Visit: https://www.kaggle.com/datasets/14832123
2. Download the dataset
3. Extract to `data/raw/` with the following structure:
   ```
   data/raw/
   ├── Training/
   │   ├── glioma/
   │   ├── meningioma/
   │   ├── notumor/
   │   └── pituitary/
   └── Testing/
       ├── glioma/
       ├── meningioma/
       ├── notumor/
       └── pituitary/
   ```

### Test Predictions (Available in Repository)

Pre-computed predictions for the test set are available in `data/processed/`:
- `y_true.npy`: Ground truth labels
- `y_pred.npy`: Model predictions
- `y_prob.npy`: Prediction probabilities

---

## Usage

### Training the Model

```bash
python src/train_model.py \
    --data_dir data/raw/ \
    --output_dir models/ \
    --epochs 50 \
    --batch_size 16 \
    --lr 1e-4 \
    --device cuda
```

### Evaluating the Model

```bash
python src/evaluate_model.py \
    --model_path models/Hybrid_Spatial_Model-Brain-Tumor.pth \
    --data_dir data/raw/ \
    --output_dir results/
```

### Running Ablation Study

```bash
python src/ablation_study.py \
    --data_dir data/raw/ \
    --output_dir results/ablation_results/
```

### XAI Consensus Analysis

```bash
python src/xai_consensus.py \
    --model_path models/Hybrid_Spatial_Model-Brain-Tumor.pth \
    --data_dir data/raw/ \
    --output_dir results/xai/
```

---

## Model Weights

### Download

The trained model weights (240 MB) are available:
- **GitHub Release**: [Download](https://github.com/razahasan2000/BTScan/releases)
- **Alternative**: Contact corresponding author

### Verify Checksum

```bash
sha256sum models/Hybrid_Spatial_Model-Brain-Tumor.pth
# Expected: <checksum-will-be-added-upon-release>
```

---

## Ablation Studies

### Component Contributions (evaluated on n=100 tumour-class test subset, seed 42)

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

### Key Findings

1. **SF-1 Spatial Fusion**: Does **not** improve inter-method concordance; deployed model (0.192 IoU) has the **lowest** concordance of all 11 configurations.
2. **Fusion design has no measurable effect on concordance**: Inter-method IoU varies only 0.203–0.262 (SD = 0.017) across 10 variants.
3. **Concordance is not a proxy for accuracy**: Accuracy vs. IoU across configurations shows r = −0.46 (p = 0.16).
4. **Between-run variation exceeds between-configuration variation**: Re-training the identical architecture moves IoU by 0.027, larger than the between-configuration SD (0.017).
5. **Attention fusion achieves highest accuracy** (96.0% on evaluation subset) and also preserves spatial structure.

> **Note**: These ablation results were computed on a fixed n=100 tumour-class test subset (seed 42) to isolate architectural effects from data variance. The full test set accuracy for the deployed model remains 93.44% (95% CI: [92.1%, 94.7%]).

---

## XAI Consensus Analysis

### Consensus Metrics (measured on n=100 tumour-class test images, seed 42)

Two complementary metrics are defined:

1. **XAI Consensus Score (C)**: Continuous metric in [0,1]
   - C = α·R̄ + (1-α)·S̄ where α=0.5
   - R̄ = Mean pairwise Pearson correlation (6 method pairs)
   - S̄ = Mean pairwise SSIM (6 method pairs)
   - **Measured value: C = 0.116 (95% CI: 0.111–0.122, bootstrap 1000 iterations)**
   - R̄ = 0.0479 ± 0.0553, S̄ = 0.1848 ± 0.0271 (unrounded)
   - C was 0.117 for correctly classified cases (n=91) and 0.108 for incorrect (n=9) — no association with decision correctness.

2. **Inter-Method Region-Based Overlap** (primary concordance measure):
   - Every attribution map normalised to [0,1] and binarised at 20% of its own maximum
   - Agreement measured as pairwise IoU and Dice across all 6 method pairs
   - **Mean IoU = 0.192 ± 0.061**, Dice = 0.258 ± 0.070
   - **Grad-CAM vs. LIME IoU = 0.665 ± 0.154** (by far the strongest pair)
   - Occlusion is intermediate (~0.21 against each)
   - Pixel-gradient Saliency is spatially uncorrelated with region-based methods (~0.025)

3. **XAI Agreement Rate** (thresholded): **Retired**
   - Threshold: Pearson r > 0.85 AND SSIM > 0.60
   - On real attributions: **0 of 100 images** met this criterion under every reading tested
   - The metric is retired rather than re-thresholded, since for sparse attribution maps a whole-map correlation of 0.85 is unattainable by construction.

> **Correction**: Previous versions of this work reported C = 0.7458 and agreement rate = 96.4%. Those values were produced by a synthetic heatmap generator and are **withdrawn**. All values above are recomputed from real Grad-CAM, Saliency, Occlusion, and LIME gradients on real test images.

### Pairwise Correlations (whole-map, real attributions)

| Method | Grad-CAM | Saliency | Occlusion | LIME |
|--------|----------|----------|-----------|------|
| Grad-CAM | 1.000 | 0.051 | 0.211 | 0.032 |
| Saliency | 0.051 | 1.000 | 0.019 | −0.040 |
| Occlusion | 0.211 | 0.019 | 1.000 | 0.014 |
| LIME | 0.032 | −0.040 | 0.014 | 1.000 |

---

## Paper & Supplementary Materials

### Overleaf Submission Package

A complete, compilation-ready Overleaf package is available:
- **Package**: `Overleaf_Upload.zip` (2.9 MB) — includes `main.tex`, `main_revised_highlighted.tex`, `Response_to_Reviewers.tex`, `autarticle.cls`, and all 14 figures
- **Direct link**: This repository's Releases page or extract from source

To use:
1. Overleaf → New Project → Upload Project → select `Overleaf_Upload.zip`
2. Compiler: **pdfLaTeX** (Main document: `main.tex`)
3. To build tracked-changes version: Main document → `main_revised_highlighted.tex` → Recompile

### Contents of the Overleaf package

| File | Purpose |
|------|---------|
| `main.tex` | Revised manuscript (clean) |
| `main_revised_highlighted.tex` | Tracked changes (\added, \deleted) |
| `Response_to_Reviewers.tex` | Point-by-point reviewer response |
| `autarticle.cls` | Journal class file (required) |
| `figures/` | 14 figures (PNG) |

---

## Citation

If you use this code or dataset in your research, please cite:

```bibtex
@article{hasan2025btscan,
  title={BTScan: Spatially Preserved CNN Transformer Fusion for Multi Method XAI Consensus in Brain Tumor MRI Classification},
  author={Hasan, Raza and Ansari, MD Aashif and Acharya, Pratham and Mahmood, Salman},
  journal={Amirkabir University of Technology Journal},
  year={2025},
  doi={DOI-TO-BE-ADDED}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- Southampton Solent University, UK
- Nazeer Hussain University, Pakistan
- Kaggle Brain Tumor MRI Dataset contributors

---

## Contact

**Corresponding Author**: Raza Hasan  
**Email**: raza.hasan@solent.ac.uk  
**Affiliation**: Southampton Solent University, UK

---

## Data Availability Statement

All data generated or analyzed during this study are included in this repository:
- Test predictions: `data/processed/`
- Model weights: `models/` (240 MB)
- Ablation results: `results/ablation_results/`

The source dataset is publicly available at: https://doi.org/10.34740/kaggle/dsv/14832123

---

## Revision History

| Date | Version | Notes |
|------|---------|-------|
| 2025-11-01 | v1.1 | Corrected XAI metrics: withdrew synthetic values (C=0.7458, 96.4% agreement); reported real inter-method IoU=0.192, C=0.116; updated ablation table with real concordance measurements; regenerated Figures 1 and 9 |
| 2025-07-05 | v1.0 | Initial release with synthetic XAI values (now withdrawn) |

---