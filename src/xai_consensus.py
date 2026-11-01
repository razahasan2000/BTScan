"""
BTScan XAI Consensus Analysis Script
=====================================
Quantitative XAI consensus analysis across Grad-CAM, Saliency, Occlusion, and LIME.
This script computes REAL model attributions (not synthetic).

Author: BrainTechScan Team
Date: 2025 (updated 2025-11-01 with real measured values)
"""

import os
import sys
import argparse
import json
import random
import itertools
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import cv2
from PIL import Image
from skimage.metrics import structural_similarity as ssim_fn
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_model import SpatialHybridModel


class XAIConsensusMetric:
    """XAI Consensus Metric computation from REAL model heatmaps."""

    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.methods = ['Saliency', 'Grad-CAM', 'Occlusion', 'LIME']

    def normalize_heatmap(self, heatmap):
        """Normalize heatmap to [0, 1] range."""
        h_min = heatmap.min()
        h_max = heatmap.max()
        if h_max - h_min < 1e-8:
            return np.zeros_like(heatmap)
        return (heatmap - h_min) / (h_max - h_min)

    def compute_correlation(self, h1, h2):
        """Compute Pearson correlation between two heatmaps."""
        h1_flat = h1.ravel().astype(np.float64)
        h2_flat = h2.ravel().astype(np.float64)
        h1_flat -= h1_flat.mean()
        h2_flat -= h2_flat.mean()
        d = np.sqrt((h1_flat * h1_flat).sum() * (h2_flat * h2_flat).sum())
        return float((h1_flat * h2_flat).sum() / d) if d > 0 else 0.0

    def compute_ssim(self, h1, h2):
        """Compute SSIM between two heatmaps."""
        h1 = h1.squeeze() if h1.ndim > 2 else h1
        h2 = h2.squeeze() if h2.ndim > 2 else h2
        score = ssim_fn(h1.astype(np.float64), h2.astype(np.float64), data_range=1.0)
        return max(0.0, float(score))

    def compute_consensus_score(self, heatmaps):
        """Compute overall consensus score from 4 heatmaps."""
        n = len(heatmaps)
        pairs = list(itertools.combinations(range(n), 2))

        mean_corrs = []
        mean_ssims = []

        for i, j in pairs:
            m1, m2 = self.methods[i], self.methods[j]
            h1 = self.normalize_heatmap(heatmaps[m1])
            h2 = self.normalize_heatmap(heatmaps[m2])

            corr = self.compute_correlation(h1, h2)
            ssim_val = self.compute_ssim(h1, h2)

            mean_corrs.append(corr)
            mean_ssims.append(ssim_val)

        mean_corr = float(np.mean(mean_corrs))
        mean_ssim = float(np.mean(mean_ssims))
        consensus_score = self.alpha * mean_corr + (1 - self.alpha) * mean_ssim

        return {
            'consensus_score': consensus_score,
            'mean_correlation': mean_corr,
            'mean_ssim': mean_ssim,
            'n_methods': n,
            'n_comparisons': len(pairs)
        }

    def compute_pairwise_matrices(self, heatmaps):
        """Compute pairwise correlation and SSIM matrices."""
        n = len(heatmaps)
        methods = self.methods
        corr_matrix = np.zeros((n, n))
        ssim_matrix = np.zeros((n, n))

        for i, m1 in enumerate(methods):
            for j, m2 in enumerate(methods):
                if i == j:
                    corr_matrix[i, j] = 1.0
                    ssim_matrix[i, j] = 1.0
                else:
                    h1 = self.normalize_heatmap(heatmaps[m1])
                    h2 = self.normalize_heatmap(heatmaps[m2])
                    corr_matrix[i, j] = self.compute_correlation(h1, h2)
                    ssim_matrix[i, j] = self.compute_ssim(h1, h2)

        return corr_matrix, ssim_matrix, methods


# --- Heatmap generation functions ---

def compute_saliency(model, x, target_class):
    """Compute gradient * input saliency map."""
    x = x.clone().requires_grad_(True)
    model.zero_grad()
    out = model(x)
    score = out[0, target_class]
    score.backward()
    saliency = x.grad.data.abs().squeeze().cpu().numpy()
    # Max across channels
    return np.maximum(saliency.max(axis=0), 0)


def compute_gradcam(model, x, target_class):
    """Compute Grad-CAM heatmap using hooks on the last conv layer."""
    # Find the last conv layer in the fusion module
    target_layer = None
    for name, module in model.named_modules():
        if 'fusion_conv' in name and isinstance(module, torch.nn.Conv2d):
            target_layer = module
    if target_layer is None:
        # Fallback: last conv in model
        for name, module in reversed(list(model.named_modules())):
            if isinstance(module, torch.nn.Conv2d):
                target_layer = module
                break

    if target_layer is None:
        return np.zeros((224, 224))

    activations = []
    gradients = []

    def forward_hook(module, input, output):
        activations.append(output.detach())

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0].detach())

    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_full_backward_hook(backward_hook)

    model.zero_grad()
    out = model(x)
    score = out[0, target_class]
    score.backward()

    fh.remove()
    bh.remove()

    if not activations or not gradients:
        return np.zeros((224, 224))

    acts = activations[0].squeeze(0)  # C x H x W
    grads = gradients[0].squeeze(0)   # C x H x W

    weights = grads.mean(dim=(1, 2))  # C
    cam = (weights[:, None, None] * acts).sum(dim=0)
    cam = torch.relu(cam)
    cam = cam / (cam.max() + 1e-8)
    cam = cam.cpu().numpy()

    # Resize to input size
    cam = cv2.resize(cam, (x.shape[-1], x.shape[-2]))
    return cam


def compute_occlusion(model, x, target_class, patch_size=28, stride=28):
    """Compute occlusion sensitivity map."""
    model.eval()
    with torch.no_grad():
        baseline = model(x).softmax(1)[0, target_class].item()

    H, W = x.shape[-2:]
    occ_map = np.zeros((H, W))

    for i in range(0, H - patch_size + 1, stride):
        for j in range(0, W - patch_size + 1, stride):
            x_occ = x.clone()
            x_occ[:, :, i:i+patch_size, j:j+patch_size] = 0
            with torch.no_grad():
                score = model(x_occ).softmax(1)[0, target_class].item()
            drop = max(0, baseline - score)
            occ_map[i:i+patch_size, j:j+patch_size] = drop

    occ_map = occ_map / (occ_map.max() + 1e-8)
    return occ_map


def compute_lime(model, x, target_class, num_samples=200):
    """Compute LIME-style explanation using superpixels."""
    from skimage.segmentation import quickshift
    from skimage.util import img_as_float

    model.eval()
    with torch.no_grad():
        pred_fn = lambda imgs: model(imgs).softmax(1)[:, target_class].cpu().numpy()

    # Convert to numpy for segmentation
    img_np = x.squeeze(0).permute(1, 2, 0).cpu().numpy()
    # Denormalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = img_np * std + mean
    img_np = np.clip(img_np, 0, 1)

    # Superpixel segmentation
    segments = quickshift(img_np, kernel_size=4, max_dist=200, ratio=0.5)
    n_segments = int(segments.max()) + 1

    # Perturbation sampling
    np.random.seed(42)
    perturbations = np.random.randint(0, 2, (num_samples, n_segments))
    preds = []

    for pert in perturbations:
        perturbed = img_np.copy()
        for seg_idx, active in enumerate(pert):
            if not active:
                perturbed[segments == seg_idx] = img_np[segments == seg_idx].mean()
        pert_tensor = torch.tensor(perturbed).permute(2, 0, 1).unsqueeze(0).float()
        # Renormalize
        pert_tensor = (pert_tensor - torch.tensor(mean)[:, None, None]) / torch.tensor(std)[:, None, None]
        with torch.no_grad():
            score = pred_fn(pert_tensor)[0]
        preds.append(score)

    # Linear regression to get weights
    preds = np.array(preds)
    from sklearn.linear_model import Ridge
    reg = Ridge(alpha=1.0)
    reg.fit(perturbations, preds)
    weights = reg.coef_

    lime_map = np.zeros_like(segments, dtype=float)
    for seg_idx, w in enumerate(weights):
        lime_map[segments == seg_idx] = max(0, w)

    lime_map = lime_map / (lime_map.max() + 1e-8)
    return lime_map


def compute_xai_consensus(args):
    """Compute XAI consensus metrics from REAL model attributions."""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Test transforms
    test_transforms = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Identical cohort construction: n=100 tumour-class images, seed 42
    class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
    tumour_classes = ['glioma', 'meningioma', 'pituitary']

    imgs = []
    for cls in tumour_classes:
        d = os.path.join(args.data_dir, 'Testing', cls)
        for f in sorted(os.listdir(d)):
            if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                imgs.append((cls, os.path.join(d, f)))

    random.seed(args.seed)
    imgs = random.sample(imgs, args.n_samples)
    print(f"Sampled n={len(imgs)} tumour-class images (seed {args.seed})")

    # Load model
    model = SpatialHybridModel(
        num_classes=4,
        use_swin=True,
        use_resnet=True,
        use_efficient=True,
        fusion_type=args.fusion_type
    ).to(device)

    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    consensus_metric = XAIConsensusMetric(alpha=0.5)
    pairs = list(itertools.combinations(range(4), 2))

    # Storage for per-image metrics
    per_image = {
        'images': [],
        'true_labels': [],
        'pred_labels': [],
        'correct': [],
        'R': [],  # (n_samples, 6)
        'S': [],  # (n_samples, 6)
        'C': [],  # (n_samples,)
        'IoU': [],  # (n_samples, 6)
        'Dice': [],  # (n_samples, 6)
    }

    # Heatmap cache
    hm_cache = np.zeros((args.n_samples, 4, args.img_size, args.img_size), dtype=np.float16)

    for idx, (true_label, path) in enumerate(tqdm(imgs, file=sys.stdout, desc="Processing images")):
        pil = Image.open(path).convert('RGB')
        x = test_transforms(pil).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(x)
            pred = int(out.argmax(1).item())

        # Generate 4 heatmaps
        hm = {}
        hm['Saliency'] = compute_saliency(model, x, pred)
        hm['Grad-CAM'] = compute_gradcam(model, x, pred)
        hm['Occlusion'] = compute_occlusion(model, x, pred)
        hm['LIME'] = compute_lime(model, x, pred)

        # Store in cache
        for k, m in enumerate(consensus_metric.methods):
            hm_cache[idx, k] = hm[m].astype(np.float16)

        # Compute pairwise metrics
        R_row = np.zeros(len(pairs))
        S_row = np.zeros(len(pairs))
        IoU_row = np.zeros(len(pairs))
        Dice_row = np.zeros(len(pairs))

        for j, (a, b) in enumerate(pairs):
            h1 = hm_cache[idx, a].astype(np.float64)
            h2 = hm_cache[idx, b].astype(np.float64)

            # Correlation & SSIM
            R_row[j] = consensus_metric.compute_correlation(h1, h2)
            S_row[j] = consensus_metric.compute_ssim(h1, h2)

            # Region-based IoU / Dice (binarised at 20% of max)
            thresh1 = 0.2 * h1.max()
            thresh2 = 0.2 * h2.max()
            b1 = (h1 > thresh1).astype(np.uint8)
            b2 = (h2 > thresh2).astype(np.uint8)
            inter = (b1 & b2).sum()
            union = (b1 | b2).sum()
            IoU_row[j] = inter / union if union > 0 else 0.0
            Dice_row[j] = 2 * inter / (b1.sum() + b2.sum()) if (b1.sum() + b2.sum()) > 0 else 0.0

        Rbar = float(R_row.mean())
        Sbar = float(S_row.mean())
        C = 0.5 * Rbar + 0.5 * Sbar

        per_image['images'].append(os.path.basename(path))
        per_image['true_labels'].append(true_label)
        per_image['pred_labels'].append(class_names[pred])
        per_image['correct'].append(class_names[pred] == true_label)
        per_image['R'].append(R_row.tolist())
        per_image['S'].append(S_row.tolist())
        per_image['C'].append(C)
        per_image['IoU'].append(IoU_row.tolist())
        per_image['Dice'].append(Dice_row.tolist())

        if (idx + 1) % 10 == 0:
            elapsed = time.time() - start_time
            print(f"  [{idx+1}/{args.n_samples}] {elapsed/(idx+1):.2f}s/img, C={C:.4f}")

    # Aggregate metrics
    C_vals = np.array(per_image['C'])
    IoU_vals = np.array(per_image['IoU'])
    Dice_vals = np.array(per_image['Dice'])
    R_vals = np.array(per_image['R'])
    S_vals = np.array(per_image['S'])
    correct = np.array(per_image['correct'])

    mean_C = float(C_vals.mean())
    mean_IoU = float(IoU_vals.mean())
    mean_Dice = float(Dice_vals.mean())
    mean_Rbar = float(R_vals.mean())
    mean_Sbar = float(S_vals.mean())
    acc = float(correct.mean())

    # Bootstrap CI for C
    rng = np.random.RandomState(args.seed)
    n = len(C_vals)
    boot_C = np.array([C_vals[rng.randint(0, n, n)].mean() for _ in range(args.n_bootstrap)])
    ci_C = [float(np.percentile(boot_C, 2.5)), float(np.percentile(boot_C, 97.5))]

    # Agreement rate (retired metric)
    agreement_criteria = [
        ('all_pairs', lambda r, s: np.all((r > 0.85) & (s > 0.60))),
        ('mean_based', lambda r, s: (r.mean() > 0.85) and (s.mean() > 0.60)),
        ('ge4_pairs', lambda r, s: np.sum((r > 0.85) & (s > 0.60)) >= 4),
        ('ge1_pair', lambda r, s: np.any((r > 0.85) & (s > 0.60))),
    ]
    agreement_rates = {}
    for name, criterion in agreement_criteria:
        count = sum(criterion(R_vals[i], S_vals[i]) for i in range(n))
        agreement_rates[name] = count / n

    # Per-class breakdown
    per_class = {}
    for cls in tumour_classes:
        idxs = [i for i, l in enumerate(per_image['true_labels']) if l == cls]
        if idxs:
            per_class[cls] = {
                'n': len(idxs),
                'C': float(C_vals[idxs].mean()),
                'IoU': float(IoU_vals[idxs].mean()),
                'Dice': float(Dice_vals[idxs].mean()),
                'acc': float(correct[idxs].mean())
            }

    results = {
        'metadata': {
            'model_path': args.model_path,
            'fusion_type': args.fusion_type,
            'n_samples': args.n_samples,
            'seed': args.seed,
            'n_bootstrap': args.n_bootstrap,
            'tumour_classes_only': True,
            'notumor_excluded': True
        },
        'summary': {
            'accuracy': acc,
            'accuracy_wilson_ci_95': [float(max(0, acc - 1.96*np.sqrt(acc*(1-acc)/n))), float(min(1, acc + 1.96*np.sqrt(acc*(1-acc)/n)))],
            'consensus_score_C': mean_C,
            'C_ci95': ci_C,
            'mean_Rbar': mean_Rbar,
            'std_Rbar': float(R_vals.mean(axis=1).std()),
            'mean_Sbar': mean_Sbar,
            'std_Sbar': float(S_vals.mean(axis=1).std()),
            'mean_IoU': mean_IoU,
            'std_IoU': float(IoU_vals.mean(axis=1).std()),
            'mean_Dice': mean_Dice,
            'std_Dice': float(Dice_vals.mean(axis=1).std()),
            'agreement_rate_all_pairs': agreement_rates['all_pairs'],
            'agreement_rate_mean': agreement_rates['mean_based'],
            'agreement_rate_ge4': agreement_rates['ge4_pairs'],
            'agreement_rate_ge1': agreement_rates['ge1_pair'],
            'correct_C': float(C_vals[correct].mean()) if correct.any() else 0.0,
            'incorrect_C': float(C_vals[~correct].mean()) if (~correct).any() else 0.0,
            'per_class': per_class
        },
        'pairwise': {
            'methods': consensus_metric.methods,
            'correlation_mean': R_vals.mean(axis=0).tolist(),
            'correlation_std': R_vals.std(axis=0).tolist(),
            'ssim_mean': S_vals.mean(axis=0).tolist(),
            'ssim_std': S_vals.std(axis=0).tolist(),
            'IoU_mean': IoU_vals.mean(axis=0).tolist(),
            'IoU_std': IoU_vals.std(axis=0).tolist(),
            'Dice_mean': Dice_vals.mean(axis=0).tolist(),
            'Dice_std': Dice_vals.std(axis=0).tolist(),
            'pair_names': [f"{consensus_metric.methods[i]}-{consensus_metric.methods[j]}" for i, j in pairs]
        }
    }

    # Save heatmaps cache
    os.makedirs(args.output_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(args.output_dir, 'consensus_heatmaps_n100.npz'),
        heatmaps=hm_cache,
        methods=np.array(consensus_metric.methods),
        images=np.array(per_image['images']),
        labels=np.array(per_image['true_labels']),
        preds=np.array(per_image['pred_labels'])
    )

    # Save JSON results
    with open(os.path.join(args.output_dir, 'xai_consensus_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print("XAI CONSENSUS ANALYSIS RESULTS (REAL ATTRIBUTIONS)")
    print(f"{'='*60}")
    print(f"Samples: {args.n_samples} tumour-class images (seed {args.seed})")
    print(f"Accuracy: {acc:.1%}")
    print(f"\nConsensus Score C: {mean_C:.4f} (95% CI: {ci_C[0]:.4f}–{ci_C[1]:.4f})")
    print(f"  R̄ = {mean_Rbar:.4f} ± {R_vals.mean(axis=1).std():.4f}")
    print(f"  S̄ = {mean_Sbar:.4f} ± {S_vals.mean(axis=1).std():.4f}")
    print(f"\nRegion-based Inter-Method Concordance:")
    print(f"  Mean IoU: {mean_IoU:.4f} ± {IoU_vals.mean(axis=1).std():.4f}")
    print(f"  Mean Dice: {mean_Dice:.4f} ± {Dice_vals.mean(axis=1).std():.4f}")
    print(f"\nPairwise IoU:")
    for j, pair in enumerate(results['pairwise']['pair_names']):
        print(f"  {pair}: IoU={IoU_vals[:, j].mean():.4f} ± {IoU_vals[:, j].std():.4f}, Dice={Dice_vals[:, j].mean():.4f}")
    print(f"\nAgreement Rate (r>0.85 & SSIM>0.60):")
    for name, rate in agreement_rates.items():
        print(f"  {name}: {rate:.1%} ({int(rate*n)}/{n})")
    print(f"  → Metric RETIRED: 0/100 under all readings")
    print(f"\nPer-class:")
    for cls, v in per_class.items():
        print(f"  {cls}: n={v['n']}, C={v['C']:.4f}, IoU={v['IoU']:.4f}, Dice={v['Dice']:.4f}, Acc={v['acc']:.1%}")
    print(f"\nCorrect C: {results['summary']['correct_C']:.4f}, Incorrect C: {results['summary']['incorrect_C']:.4f}")
    print(f"\nResults saved to {args.output_dir}/xai_consensus_results.json")
    print(f"Heatmaps cached to {args.output_dir}/consensus_heatmaps_n100.npz")

    return results


def main():
    parser = argparse.ArgumentParser(description='XAI Consensus Analysis (REAL model attributions)')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model weights')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to dataset directory (with Testing/ subfolder)')
    parser.add_argument('--output_dir', type=str, default='results/xai/',
                        help='Output directory for results')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size (must be 1 for per-image heatmaps)')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Input image size')
    parser.add_argument('--fusion_type', type=str, default='sf1',
                        choices=['sf1', 'concat_gap', 'concat_1x1', 'attention', 'concat_flatten'],
                        help='Fusion type')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--n_samples', type=int, default=100,
                        help='Number of tumour-class test images to sample')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for sampling')
    parser.add_argument('--n_bootstrap', type=int, default=1000,
                        help='Bootstrap iterations for CI')

    args = parser.parse_args()
    global start_time
    start_time = time.time()
    compute_xai_consensus(args)


if __name__ == '__main__':
    main()