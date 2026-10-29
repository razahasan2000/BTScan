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

| Metric | Value | 95% CI |
|--------|-------|--------|
| Weighted Accuracy | 93.44% | [92.6%, 94.3%] |
| Weighted F1-Score | 0.933 | [0.920, 0.948] |
| MCC | 0.912 | [0.894, 0.930] |
| Cohen's Kappa | 0.912 | [0.894, 0.928] |
| ECE (Calibration) | 0.0531 | — |
| **No Tumor Recall** | **99.75%** | **[99.2%, 100%]** |
| **No Tumor NPV** | **99.89%** | **[99.7%, 100%]** |
| XAI Consensus Score | 0.7458 | [0.7413, 0.7504] |
| XAI Agreement Rate | 96.4% | — |

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
git clone https://github.com/your-username/BTScan.git
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
- **GitHub Release**: [Download](https://github.com/your-username/BTScan/releases)
- **Alternative**: Contact corresponding author

### Verify Checksum

```bash
sha256sum models/Hybrid_Spatial_Model-Brain-Tumor.pth
# Expected: <checksum-will-be-added-upon-release>
```

---

## Ablation Studies

### Component Contributions

| Configuration | Accuracy | F1-Weighted | XAI Consensus | XAI Agreement | Parameters |
|---------------|----------|-------------|---------------|---------------|------------|
| Full Model (SF-1) | 93.44% | 0.933 | 0.7458 | 96.4% | 148.2M |
| w/o SF-1 (Flattened) | 91.50% | 0.900 | 0.512 | 78.0% | 145.0M |
| w/o SF-1 (Concat+GAP) | 94.13% | 0.941 | — | — | — |
| w/o SF-1 (Concat+1×1) | 93.82% | 0.938 | — | — | — |
| w/o SF-1 (Attention) | 97.18% | 0.972 | — | — | — |
| w/o Swin Transformer | 91.20% | 0.870 | 0.698 | 92.0% | 80.4M |
| w/o ResNet-50 | 91.80% | 0.880 | 0.712 | 93.0% | 71.0M |
| w/o EfficientNet-B3 | 94.66% | 0.946 | — | — | 110M |
| Swin Only | 89.32% | 0.892 | — | — | 28M |
| ResNet-50 Only | 88.56% | 0.885 | — | — | 25M |
| EfficientNet-B3 Only | 84.29% | 0.841 | — | — | 12M |

### Key Findings

1. **SF-1 Spatial Fusion**: Critical for XAI consensus (96.4% vs 78.0% without)
2. **Swin Transformer**: Most important backbone (drop of 2.98% when removed)
3. **EfficientNet-B3**: Marginal contribution (removal improves accuracy to 94.66%)
4. **Attention Fusion**: Higher accuracy (97.18%) but no spatial preservation

---

## XAI Consensus Analysis

### Consensus Metrics

Two complementary metrics are defined:

1. **XAI Consensus Score (C)**: Continuous metric in [0,1]
   - C = α·R̄ + (1-α)·S̄ where α=0.5
   - R̄ = Mean pairwise Pearson correlation
   - S̄ = Mean pairwise SSIM
   - Reported value: 0.7458 (95% CI: 0.7413–0.7504)

2. **XAI Agreement Rate**: Percentage of cases with anatomical agreement
   - Threshold: Pearson > 0.85 AND SSIM > 0.60
   - Reported value: 96.4% of correctly classified cases

### Pairwise Correlations

| Method | Grad-CAM | Saliency | Occlusion | LIME |
|--------|----------|----------|-----------|------|
| Grad-CAM | 1.000 | 0.942 | 0.967 | 0.918 |
| Saliency | 0.942 | 1.000 | 0.931 | 0.889 |
| Occlusion | 0.967 | 0.931 | 1.000 | 0.924 |
| LIME | 0.918 | 0.889 | 0.924 | 1.000 |

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
- OpenAI ChatGPT-5.5 for content writing assistance

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
