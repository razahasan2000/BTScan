# Raw Dataset

## Source

The dataset used in this study is the publicly available **Kaggle Brain Tumor MRI Dataset**:
- **DOI**: https://doi.org/10.34740/kaggle/dsv/14832123
- **Total Images**: 7,023
- **Classes**: Glioma, Meningioma, Pituitary Tumor, No Tumor
- **Image Size**: Varies (resized to 224×224 during preprocessing)

## Download Instructions

1. Visit the Kaggle dataset page: https://www.kaggle.com/datasets/14832123
2. Download the dataset files
3. Extract to this directory (`data/raw/`) with the following structure:

```
data/raw/
├── Training/
│   ├── glioma/
│   │   ├── Tr-glTr_0000.jpg
│   │   └── ...
│   ├── meningioma/
│   │   ├── Tr-meTr_0000.jpg
│   │   └── ...
│   ├── notumor/
│   │   ├── Tr-noTr_0000.jpg
│   │   └── ...
│   └── pituitary/
│       ├── Tr-piTr_0000.jpg
│       └── ...
└── Testing/
    ├── glioma/
    │   ├── Te-glTr_0000.jpg
    │   └── ...
    ├── meningioma/
    │   ├── Te-meTr_0000.jpg
    │   └── ...
    ├── notumor/
    │   ├── Te-noTr_0000.jpg
    │   └── ...
    └── pituitary/
        ├── Te-piTr_0000.jpg
        └── ...
```

## Class Distribution

| Class | Train Images | Test Images | Total |
|-------|-------------|-------------|-------|
| Glioma | ~1,457 | 300 | ~2,000 |
| Meningioma | ~1,339 | 306 | ~2,000 |
| Pituitary | ~1,457 | 300 | ~2,000 |
| No Tumor | ~1,595 | 405 | ~2,000 |
| **Total** | ~5,846 | 1,311 | ~7,023 |

## Data Integrity

- pHash/SSIM-based deduplication (SSIM > 0.92) was applied
- Near-duplicate images were forced into the same data split
- Patient-level separation cannot be guaranteed (no patient IDs in source dataset)

## Preprocessing

All images undergo:
1. Resizing to 224×224 pixels
2. Z-score intensity normalization (mean=0, std=1) per image
3. Data augmentation during training (rotations, flips, zoom, brightness/contrast)

No skull stripping is applied to preserve full cranial context.