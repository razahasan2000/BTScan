# Processed Data

This directory contains pre-computed model predictions on the test set (1,311 images).

## Files

| File | Size | Description |
|------|------|-------------|
| `y_true.npy` | ~10 KB | Ground truth labels (1311 samples) |
| `y_pred.npy` | ~10 KB | Model predictions (1311 samples) |
| `y_prob.npy` | ~21 KB | Prediction probabilities (1311×4) |

## Class Mapping

| Index | Class |
|-------|-------|
| 0 | Glioma |
| 1 | Meningioma |
| 2 | No Tumor |
| 3 | Pituitary |

## Usage Example

```python
import numpy as np

y_true = np.load('data/processed/y_true.npy')
y_pred = np.load('data/processed/y_pred.npy')
y_prob = np.load('data/processed/y_prob.npy')

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_true, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")
```

## Dataset Statistics

| Metric | Value |
|--------|-------|
| Test Samples | 1,311 |
| Glioma | 300 |
| Meningioma | 306 |
| No Tumor | 405 |
| Pituitary | 300 |
| Test Accuracy | 93.44% |

## Generated From

These predictions were generated using the BTScan model with SF-1 spatial fusion layer, trained on the Kaggle Brain Tumor MRI Dataset (7,023 images) with 70/15/15 train/validation/test split.