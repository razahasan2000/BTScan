"""
BTScan Model Training Script
============================
Trains the BTScan hybrid spatial fusion model on brain tumor MRI dataset.

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

# Add src directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class SpatialHybridModel(nn.Module):
    """BTScan Hybrid Spatial Fusion Model with SF-1 layer."""
    
    def __init__(self, num_classes=4, use_swin=True, use_resnet=True, 
                 use_efficient=True, fusion_type='sf1'):
        super().__init__()
        self.use_swin = use_swin
        self.use_resnet = use_resnet
        self.use_efficient = use_efficient
        self.fusion_type = fusion_type
        
        # Backbone networks
        if use_swin:
            self.swin = timm.create_model(
                'swin_tiny_patch4_window7_224',
                pretrained=True,
                features_only=True
            )
            for p in self.swin.parameters():
                p.requires_grad = False
            C_swin = 768
        else:
            C_swin = 0
            
        if use_resnet:
            resnet = models.resnet50(weights='IMAGENET1K_V1')
            self.resnet = nn.Sequential(*list(resnet.children())[:-2])
            for p in self.resnet.parameters():
                p.requires_grad = False
            C_res = 2048
        else:
            C_res = 0
            
        if use_efficient:
            self.efficient = timm.create_model(
                'efficientnet_b3',
                pretrained=True,
                features_only=True
            )
            for p in self.efficient.parameters():
                p.requires_grad = False
            C_eff = 384
        else:
            C_eff = 0
        
        total_channels = C_swin + C_res + C_eff
        
        # SF-1 Spatial Fusion layer
        if fusion_type == 'sf1':
            self.fusion_conv = nn.Conv2d(total_channels, 512, kernel_size=1)
        elif fusion_type == 'concat_gap':
            self.fusion_conv = None
        elif fusion_type == 'concat_1x1':
            self.fusion_conv = nn.Conv2d(total_channels, 512, kernel_size=1)
        elif fusion_type == 'attention':
            self.fusion_conv = nn.MultiheadAttention(
                embed_dim=total_channels, num_heads=8, batch_first=True
            )
        
        # Classification head
        if fusion_type in ['sf1', 'concat_1x1']:
            clf_in = 512
        elif fusion_type == 'concat_gap':
            clf_in = total_channels
        elif fusion_type == 'concat_flatten':
            clf_in = total_channels * 7 * 7
        elif fusion_type == 'attention':
            clf_in = total_channels
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(clf_in, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )
        
        if fusion_type == 'concat_flatten':
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(total_channels * 7 * 7, 256),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(256, num_classes)
            )
    
    def forward(self, x):
        features = []
        
        if self.use_swin:
            f_swin = self.swin(x)[-1].permute(0, 3, 1, 2)
            features.append(f_swin)
            
        if self.use_resnet:
            f_res = self.resnet(x)
            features.append(f_res)
            
        if self.use_efficient:
            f_eff = self.efficient(x)[-1]
            if features:
                target_size = features[0].shape[-2:]
                f_eff = nn.functional.interpolate(
                    f_eff, size=target_size, mode='bilinear', align_corners=False
                )
            features.append(f_eff)
        
        # Fusion
        if self.fusion_type in ['sf1', 'concat_gap', 'concat_1x1']:
            fused = torch.cat(features, dim=1)
            if self.fusion_type in ['sf1', 'concat_1x1']:
                fused = self.fusion_conv(fused)
        elif self.fusion_type == 'concat_flatten':
            fused = torch.cat(features, dim=1)
        elif self.fusion_type == 'attention':
            fused = torch.cat(features, dim=1)
            B, C, H, W = fused.shape
            fused = fused.flatten(2).transpose(1, 2)
            fused, _ = self.fusion_conv(fused, fused, fused)
            fused = fused.transpose(1, 2).reshape(B, C, H, W)
        
        return self.classifier(fused)


def create_data_loaders(data_dir, batch_size=16, img_size=224):
    """Create train, validation, and test data loaders."""
    
    train_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    test_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    train_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, 'Training'),
        transform=train_transforms
    )
    
    test_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, 'Testing'),
        transform=test_transforms
    )
    
    # Train/Validation split (85/15)
    val_size = int(len(train_dataset) * 0.15)
    train_size = len(train_dataset) - val_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, train_dataset.classes


def train_model(args):
    """Main training function."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loaders
    train_loader, val_loader, test_loader, class_names = create_data_loaders(
        args.data_dir, batch_size=args.batch_size, img_size=args.img_size
    )
    
    print(f"Classes: {class_names}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    model = SpatialHybridModel(
        num_classes=len(class_names),
        use_swin=True,
        use_resnet=True,
        use_efficient=True,
        fusion_type=args.fusion_type
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params / 1e6:.1f}M")
    print(f"Trainable parameters: {trainable_params / 1e6:.1f}M")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW([
        {'params': [p for n, p in model.named_parameters() 
                   if 'swin' in n and p.requires_grad], 'lr': 1e-5},
        {'params': [p for n, p in model.named_parameters() 
                   if 'swin' not in n and p.requires_grad], 'lr': args.lr},
    ], weight_decay=1e-4)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=5
    )
    
    # Training loop
    best_val_acc = 0.0
    best_epoch = 0
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    for epoch in range(args.epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for imgs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs} [Train]'):
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
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        
        val_acc = val_correct / val_total
        val_loss /= len(val_loader)
        
        scheduler.step(val_loss)
        
        print(f'Epoch {epoch+1}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}, '
              f'LR={optimizer.param_groups[0]["lr"]:.2e}')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), 
                       os.path.join(args.output_dir, 'best_model.pth'))
            print(f'  -> Best model saved (Val Acc={val_acc:.4f})')
    
    print(f'\nBest Validation Accuracy: {best_val_acc:.4f} at epoch {best_epoch}')
    
    # Evaluate on test set
    model.load_state_dict(torch.load(
        os.path.join(args.output_dir, 'best_model.pth'),
        map_location=device
    ))
    model.eval()
    
    test_correct = 0
    test_total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc='Testing'):
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            test_correct += (preds == labels).sum().item()
            test_total += labels.size(0)
    
    test_acc = test_correct / test_total
    print(f'\nTest Accuracy: {test_acc:.4f}')
    
    # Save predictions
    os.makedirs(os.path.join(args.output_dir, '..', 'data', 'processed'), exist_ok=True)
    np.save(os.path.join(args.output_dir, '..', 'data', 'processed', 'y_true.npy'),
            np.array(all_labels))
    np.save(os.path.join(args.output_dir, '..', 'data', 'processed', 'y_pred.npy'),
            np.array(all_preds))
    
    return test_acc


def main():
    parser = argparse.ArgumentParser(description='Train BTScan model')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='models/',
                        help='Output directory for model weights')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Input image size')
    parser.add_argument('--fusion_type', type=str, default='sf1',
                        choices=['sf1', 'concat_gap', 'concat_1x1', 'attention', 'concat_flatten'],
                        help='Fusion type')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    train_model(args)


if __name__ == '__main__':
    main()