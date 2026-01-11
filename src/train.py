"""
Training Script for Disaster Damage Detection Model
Supports local training (M2 Mac with MPS) and Kaggle GPU
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent))

from dataset import get_dataloaders
from model import DamageDetector, count_parameters


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    Focuses on hard examples by down-weighting easy ones.
    """

    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # Class weights
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(
            inputs, targets, weight=self.alpha, reduction='none'
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class DiceLoss(nn.Module):
    """Dice Loss for segmentation."""

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # Convert inputs to probabilities
        inputs = torch.softmax(inputs, dim=1)

        # One-hot encode targets
        num_classes = inputs.shape[1]
        targets_one_hot = torch.zeros_like(inputs)
        targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)

        # Calculate Dice for each class
        intersection = (inputs * targets_one_hot).sum(dim=(2, 3))
        union = inputs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))

        dice = (2 * intersection + self.smooth) / (union + self.smooth)

        return 1 - dice.mean()


class CombinedLoss(nn.Module):
    """Combined Focal + Dice Loss."""

    def __init__(self, class_weights=None, focal_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.focal = FocalLoss(alpha=class_weights, gamma=2.0)
        self.dice = DiceLoss()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight

    def forward(self, inputs, targets):
        return (
            self.focal_weight * self.focal(inputs, targets) +
            self.dice_weight * self.dice(inputs, targets)
        )


def calculate_metrics(preds, targets, num_classes=4):
    """
    Calculate segmentation metrics.

    Returns:
        Dictionary with IoU, F1, accuracy per class and overall
    """
    metrics = {}

    # Overall accuracy
    correct = (preds == targets).sum().item()
    total = targets.numel()
    metrics['accuracy'] = correct / total

    # Per-class metrics
    ious = []
    f1s = []

    for c in range(num_classes):
        pred_c = (preds == c)
        target_c = (targets == c)

        intersection = (pred_c & target_c).sum().item()
        union = (pred_c | target_c).sum().item()

        if union > 0:
            iou = intersection / union
        else:
            iou = 0.0

        # F1 / Dice
        tp = intersection
        fp = pred_c.sum().item() - tp
        fn = target_c.sum().item() - tp

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        ious.append(iou)
        f1s.append(f1)

        metrics[f'iou_class_{c}'] = iou
        metrics[f'f1_class_{c}'] = f1

    metrics['mean_iou'] = np.mean(ious)
    metrics['mean_f1'] = np.mean(f1s)

    # Damage-specific metrics (classes 1-3)
    damage_ious = ious[1:]  # Exclude no-damage class
    metrics['mean_damage_iou'] = np.mean(damage_ious) if damage_ious else 0.0

    return metrics


def train_epoch(model, dataloader, criterion, optimizer, device, scaler=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_targets = []

    pbar = tqdm(dataloader, desc='Training')
    for batch in pbar:
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)

        optimizer.zero_grad()

        # Mixed precision training
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

        # Collect predictions for metrics
        preds = torch.argmax(outputs, dim=1)
        all_preds.append(preds.cpu())
        all_targets.append(masks.cpu())

        pbar.set_postfix({'loss': loss.item()})

    # Calculate epoch metrics
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    metrics = calculate_metrics(all_preds, all_targets)
    metrics['loss'] = total_loss / len(dataloader)

    return metrics


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []

    for batch in tqdm(dataloader, desc='Validating'):
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)

        outputs = model(images)
        loss = criterion(outputs, masks)

        total_loss += loss.item()

        preds = torch.argmax(outputs, dim=1)
        all_preds.append(preds.cpu())
        all_targets.append(masks.cpu())

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    metrics = calculate_metrics(all_preds, all_targets)
    metrics['loss'] = total_loss / len(dataloader)

    return metrics


def get_device():
    """Get best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def train(
    data_dir: str,
    output_dir: str,
    architecture: str = 'DeepLabV3Plus',
    encoder: str = 'resnet50',
    batch_size: int = 8,
    num_epochs: int = 30,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    num_workers: int = 4,
    use_amp: bool = True,
    resume_from: str = None
):
    """
    Main training function.

    Args:
        data_dir: Path to processed data
        output_dir: Path to save checkpoints and logs
        architecture: Model architecture
        encoder: Encoder backbone
        batch_size: Training batch size
        num_epochs: Number of epochs
        learning_rate: Initial learning rate
        weight_decay: Weight decay for optimizer
        num_workers: Data loading workers
        use_amp: Use automatic mixed precision
        resume_from: Path to checkpoint to resume from
    """
    # Setup
    device = get_device()
    print(f"\nUsing device: {device}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Data
    print("\nLoading data...")
    train_loader, val_loader = get_dataloaders(
        data_dir, batch_size=batch_size, num_workers=num_workers
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Model
    print(f"\nCreating model: {architecture} with {encoder}")
    model = DamageDetector(
        architecture=architecture,
        encoder=encoder,
        num_classes=4,
        pretrained=True
    )
    model = model.to(device)
    print(f"Parameters: {count_parameters(model):,}")

    # Class weights (typical for xView2 - damage classes are rare)
    class_weights = torch.tensor([0.5, 1.5, 2.0, 2.5], device=device)
    criterion = CombinedLoss(class_weights=class_weights)

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    # Scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if use_amp and device.type == 'cuda' else None

    # Resume from checkpoint
    start_epoch = 0
    best_val_iou = 0

    if resume_from:
        print(f"\nResuming from {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_iou = checkpoint.get('best_val_iou', 0)

    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_iou': [], 'val_iou': [],
        'train_f1': [], 'val_f1': []
    }

    # Training loop
    print(f"\nStarting training for {num_epochs} epochs...")
    for epoch in range(start_epoch, num_epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"{'='*50}")

        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )

        # Validate
        val_metrics = validate(model, val_loader, criterion, device)

        # Update scheduler
        scheduler.step()

        # Log metrics
        print(f"\nTrain - Loss: {train_metrics['loss']:.4f}, "
              f"IoU: {train_metrics['mean_iou']:.4f}, "
              f"F1: {train_metrics['mean_f1']:.4f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, "
              f"IoU: {val_metrics['mean_iou']:.4f}, "
              f"F1: {val_metrics['mean_f1']:.4f}")

        # Damage-specific IoU
        print(f"Damage IoU - Train: {train_metrics['mean_damage_iou']:.4f}, "
              f"Val: {val_metrics['mean_damage_iou']:.4f}")

        # Update history
        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['train_iou'].append(train_metrics['mean_iou'])
        history['val_iou'].append(val_metrics['mean_iou'])
        history['train_f1'].append(train_metrics['mean_f1'])
        history['val_f1'].append(val_metrics['mean_f1'])

        # Save checkpoint
        is_best = val_metrics['mean_iou'] > best_val_iou
        if is_best:
            best_val_iou = val_metrics['mean_iou']
            print(f"New best validation IoU: {best_val_iou:.4f}")

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'best_val_iou': best_val_iou,
            'config': {
                'architecture': architecture,
                'encoder': encoder,
                'num_classes': 4
            }
        }

        # Save latest
        torch.save(checkpoint, output_dir / 'latest.pth')

        # Save best
        if is_best:
            torch.save(checkpoint, output_dir / 'best.pth')

        # Save periodic
        if (epoch + 1) % 5 == 0:
            torch.save(checkpoint, output_dir / f'epoch_{epoch+1}.pth')

    # Save training history
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f)

    print(f"\nTraining complete!")
    print(f"Best validation IoU: {best_val_iou:.4f}")
    print(f"Checkpoints saved to: {output_dir}")

    return history


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train damage detection model')
    parser.add_argument('--data-dir', type=str, default='/Users/mac/iitmhack/data/xview2_processed',
                        help='Path to processed data')
    parser.add_argument('--output-dir', type=str, default='/Users/mac/iitmhack/models',
                        help='Path to save checkpoints')
    parser.add_argument('--architecture', type=str, default='DeepLabV3Plus',
                        choices=['DeepLabV3Plus', 'UNet', 'FPN', 'PSPNet'])
    parser.add_argument('--encoder', type=str, default='resnet50')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--resume', type=str, default=None)

    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        architecture=args.architecture,
        encoder=args.encoder,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        num_workers=args.num_workers,
        resume_from=args.resume
    )
