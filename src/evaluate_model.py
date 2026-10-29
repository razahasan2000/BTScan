"""
BTScan Model Evaluation Script
==============================
Evaluates the trained BTScan model on test data and generates performance metrics.

Author: BrainTechScan Team
Date: 2025
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    matthews_corrcoef, cohen_kappa_score, confusion_matrix,
    roc_auc_score, classification_report
)
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_model import SpatialHybridModel


def evaluate_model(args):
    """Main evaluation function."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load test data
    test_transforms = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    test_dataset = datasets.ImageFolder(
        root=os.path.join(args.data_dir, 'Testing'),
        transform=test_transforms
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    
    class_names = test_dataset.classes
    num_classes = len(class_names)
    
    print(f"Classes: {class_names}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Load model
    model = SpatialHybridModel(
        num_classes=num_classes,
        use_swin=True,
        use_resnet=True,
        use_efficient=True,
        fusion_type=args.fusion_type
    ).to(device)
    
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    
    # Evaluate
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc='Testing'):
            imgs = imgs.to(device)
            outputs = model(imgs)
            probs = F.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Compute metrics
    acc = accuracy_score(all_labels, all_preds)
    f1_w = f1_score(all_labels, all_preds, average='weighted')
    f1_m = f1_score(all_labels, all_preds, average='macro')
    prec_w = precision_score(all_labels, all_preds, average='weighted')
    rec_w = recall_score(all_labels, all_preds, average='weighted')
    mcc = matthews_corrcoef(all_labels, all_preds)
    kappa = cohen_kappa_score(all_labels, all_preds)
    
    # Per-class AUC
    auc_scores = {}
    for i, class_name in enumerate(class_names):
        y_true_binary = (all_labels == i).astype(int)
        y_score = all_probs[:, i]
        try:
            auc_scores[class_name] = roc_auc_score(y_true_binary, y_score)
        except ValueError:
            auc_scores[class_name] = float('nan')
    
    cm = confusion_matrix(all_labels, all_preds)
    
    # Print results
    print('\n' + '='*60)
    print('EVALUATION RESULTS')
    print('='*60)
    print(f"Accuracy:  {acc:.4f}")
    print(f"F1-W:      {f1_w:.4f}")
    print(f"F1-M:      {f1_m:.4f}")
    print(f"Precision: {prec_w:.4f}")
    print(f"Recall:    {rec_w:.4f}")
    print(f"MCC:       {mcc:.4f}")
    print(f"Kappa:     {kappa:.4f}")
    print()
    print("Per-class AUC:")
    for class_name, auc in auc_scores.items():
        print(f"  {class_name}: {auc:.4f}")
    print()
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, 
                                target_names=class_names, digits=4))
    print()
    print("Confusion Matrix:")
    print(cm)
    print('='*60)
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    results = {
        'accuracy': acc,
        'f1_weighted': f1_w,
        'f1_macro': f1_m,
        'precision_weighted': prec_w,
        'recall_weighted': rec_w,
        'mcc': mcc,
        'kappa': kappa,
        'auc_scores': auc_scores,
        'confusion_matrix': cm.tolist(),
        'predictions': all_preds.tolist(),
        'labels': all_labels.tolist(),
        'probabilities': all_probs.tolist(),
        'class_names': class_names
    }
    
    np.save(os.path.join(args.output_dir, 'evaluation_results.npy'), results)
    print(f'\nResults saved to {args.output_dir}/evaluation_results.npy')
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate BTScan model')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model weights')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='results/',
                        help='Output directory for results')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Input image size')
    parser.add_argument('--fusion_type', type=str, default='sf1',
                        choices=['sf1', 'concat_gap', 'concat_1x1', 'attention', 'concat_flatten'],
                        help='Fusion type')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    evaluate_model(args)


if __name__ == '__main__':
    from tqdm import tqdm
    main()