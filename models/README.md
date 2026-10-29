# Model Weights

## Available Models

| File | Size | Description |
|------|------|-------------|
| `Hybrid_Spatial_Model-Brain-Tumor.pth` | ~240 MB | Full BTScan model with SF-1 fusion layer |

## Download

The model weights are too large for GitHub's default file size limit. Download from:

- **GitHub Release**: [Releases page](https://github.com/razahasan2000/BTScan/releases/tag/BTScan)
- **Alternative**: Contact corresponding author (raza.hasan@solent.ac.uk)

## Verification

```bash
sha256sum models/Hybrid_Spatial_Model-Brain-Tumor.pth
```

Checksum will be provided upon release.

## Model Architecture

The BTScan model consists of:
- **Swin Transformer** (tiny variant, frozen): Global contextual features
- **ResNet-50** (frozen): Hierarchical spatial features
- **EfficientNet-B3** (frozen): Fine-grained textural features
- **SF-1 Fusion Layer**: Channel-wise concatenation + 1×1 convolution
- **Classifier**: Adaptive pooling + Linear layers

Total parameters: ~148M (trainable: ~1.6M)

## Usage

```python
import torch
from src.train_model import SpatialHybridModel

model = SpatialHybridModel(num_classes=4)
model.load_state_dict(torch.load('models/Hybrid_Spatial_Model-Brain-Tumor.pth'))
model.eval()
```

## Citation

If you use these model weights, please cite the BTScan paper (DOI to be added upon publication).
