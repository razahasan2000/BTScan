"""
BTScan Ablation Study Script
============================
Runs controlled ablation experiments to isolate the contribution of each
component: SF-1 fusion layer, individual backbones, and fusion strategies.

Author: BrainTechScan Team
Date: 2025
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import timm
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_model import SpatialHybridModel


def train_and_evaluate(args, config_name, model_config):
    """Train and evaluate a single ablation configuration."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    train_transforms = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    test_transforms = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    full_train_dataset = datasets.ImageFolder(
        root=os.path.join(args.data_dir, 'Training'),
        transform=train_transforms
    )
    
    test_dataset = datasets.ImageFolder(
        root=os.path.join(args.data_dir, 'Testing'),
        transform=test_transforms
    )
    
    class_names = full_train_dataset.classes
    num_classes = len(class_names)
    
    val_size = int(len(full_train_dataset) * 0.15)
    train_size = len(full_train_dataset) - val_size
    train_subset, val_subset = torch.utils.data.random_split(
        full_train_dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_subset, batch_size=args.batch_size, 
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=4, pin_memory=True)
    
    # Create model
    model = SpatialHybridModel(
        num_classes=num_classes,
        **model_config
    ).to(device)
    
    # Train
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=5
    )
    
    best_val_acc = 0.0
    best_epoch = 0
    
    model_dir = os.path.join(args.output_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f'{config_name}_best.pth')
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for imgs, labels in tqdm(train_loader, 
                                 desc=f'{config_name} Epoch {epoch+1}/{args.epochs} [Train]',
                                 leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            preds = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
        
        train_acc = train_correct / train_total
        
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        
        val_acc = val_correct / val_total
        scheduler.step(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), model_path)
    
    # Evaluate on test set
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    acc = accuracy_score(all_labels, all_preds)
    f1_w = f1_score(all_labels, all_preds, average='weighted')
    f1_m = f1_score(all_labels, all_preds, average='macro')
    
    return {
        'config': config_name,
        'best_val_acc': best_val_acc,
        'test_acc': acc,
        'f1_weighted': f1_w,
        'f1_macro': f1_m,
        'best_epoch': best_epoch
    }


def run_ablation_study(args):
    """Run all ablation experiments."""
    
    configs = {
        'full_sf1': {
            'use_swin': True, 'use_resnet': True, 'use_efficient': True,
            'fusion_type': 'sf1'
        },
        'w/o_sf1_flattened': {
            'use_swin': True, 'use_resnet': True, 'use_efficient': True,
            'fusion_type': 'concat_flatten'
        },
        'w/o_sf1_concat_gap': {
            'use_swin': True, 'use_resnet': True, 'use_efficient': True,
            'fusion_type': 'concat_gap'
        },
        'w/o_sf1_concat_1x1': {
            'use_swin': True, 'use_resnet': True, 'use_efficient': True,
            'fusion_type': 'concat_1x1'
        },
        'w/o_sf1_attention': {
            'use_swin': True, 'use_resnet': True, 'use_efficient': True,
            'fusion_type': 'attention'
        },
        'w/o_swin': {
            'use_swin': False, 'use_resnet': True, 'use_efficient': True,
            'fusion_type': 'sf1'
        },
        'w/o_resnet': {
            'use_swin': True, 'use_resnet': False, 'use_efficient': True,
            'fusion_type': 'sf1'
        },
        'w/o_efficientnet': {
            'use_swin': True, 'use_resnet': True, 'use_efficient': False,
            'fusion_type': 'sf1'
        },
        'swin_only': {
            'use_swin': True, 'use_resnet': False, 'use_efficient': False,
            'fusion_type': 'sf1'
        },
        'resnet_only': {
            'use_swin': False, 'use_resnet': True, 'use_efficient': False,
            'fusion_type': 'sf1'
        },
        'efficientnet_only': {
            'use_swin': False, 'use_resnet': False, 'use_efficient': True,
            'fusion_type': 'sf1'
        }
    }
    
    results = {}
    
    for config_name, model_config in configs.items():
        print('\n' + '='*60)
        print(f'Running ablation: {config_name}')
        print('='*60)
        
        result = train_and_evaluate(args, config_name, model_config)
        results[config_name] = result
        
        print(f"\n  Test Accuracy: {result['test_acc']:.4f}")
        print(f"  F1-Weighted:   {result['f1_weighted']:.4f}")
        print(f"  F1-Macro:      {result['f1_macro']:.4f}")
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    np.save(os.path.join(args.output_dir, 'ablation_results.npy'), results)
    
    # Print summary
    print('\n' + '='*70)
    print('ABLATION STUDY SUMMARY')
    print('='*70)
    print(f"{'Configuration':<30} {'Val Acc':>10} {'Test Acc':>10} {'F1-W':>10} {'F1-M':>10}")
    print('-'*70)
    
    for config_name, result in results.items():
        print(f"{config_name:<30} {result['best_val_acc']:>10.4f} "
              f"{result['test_acc']:>10.4f} {result['f1_weighted']:>10.4f} "
              f"{result['f1_macro']:>10.4f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Run BTScan ablation study')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='results/ablation_results/',
                        help='Output directory for results')
    parser.add_argument('--epochs', type=int, default=25,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Input image size')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    run_ablation_study(args)


if __name__ == '__main__':
    from sklearn.metrics import accuracy_score, f1_score
    main()