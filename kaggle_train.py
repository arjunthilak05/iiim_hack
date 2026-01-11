"""
RapidEye - Kaggle Training Script
All-in-one script for training on Kaggle GPU

Instructions:
1. Create new Kaggle notebook
2. Add xView2 dataset: https://www.kaggle.com/datasets/residentmario/xview2-train
3. Enable GPU accelerator (Settings > Accelerator > GPU)
4. Upload this file and run cells
"""

# ============================================================
# CELL 1: Install dependencies and setup
# ============================================================

# !pip install segmentation-models-pytorch albumentations -q

import os
import sys
import json
import time
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import albumentations as A
from albumentations.pytorch import ToTensorV2

import segmentation_models_pytorch as smp

import cv2
from shapely import wkt

import warnings
warnings.filterwarnings('ignore')

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ============================================================
# CELL 2: Configuration
# ============================================================

class Config:
    # Paths - Update these for Kaggle
    RAW_DATA_DIR = '/kaggle/input/xview2-challenge-dataset-train-and-test/train'  # Kaggle xView2 path
    OUTPUT_DIR = '/kaggle/working/processed_data'
    MODEL_DIR = '/kaggle/working/models'

    # Preprocessing
    IMG_SIZE = 512
    MAX_SAMPLES = None  # None for all, or set limit like 2000

    # Training
    BATCH_SIZE = 8
    NUM_WORKERS = 2
    EPOCHS = 30
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4

    # Model
    ARCHITECTURE = 'DeepLabV3Plus'
    ENCODER = 'resnet50'
    NUM_CLASSES = 4

config = Config()

# Create directories
os.makedirs(config.OUTPUT_DIR, exist_ok=True)
os.makedirs(config.MODEL_DIR, exist_ok=True)

# ============================================================
# CELL 3: Preprocessing Functions
# ============================================================

def parse_damage_label(label_path):
    """Parse xView2 JSON label file."""
    with open(label_path) as f:
        data = json.load(f)

    buildings = []
    damage_mapping = {
        'no-damage': 0,
        'minor-damage': 1,
        'major-damage': 2,
        'destroyed': 3,
        'un-classified': 0
    }

    for feature in data.get('features', {}).get('xy', []):
        props = feature.get('properties', {})
        damage_type = props.get('subtype', 'no-damage')
        damage_class = damage_mapping.get(damage_type, 0)

        wkt_str = feature.get('wkt', '')
        if wkt_str:
            buildings.append({
                'wkt': wkt_str,
                'damage_class': damage_class
            })

    return buildings

def create_mask_from_buildings(buildings, height=1024, width=1024):
    """Create segmentation mask from building polygons."""
    mask = np.zeros((height, width), dtype=np.uint8)

    for building in buildings:
        try:
            poly = wkt.loads(building['wkt'])
            coords = np.array(poly.exterior.coords).astype(np.int32)
            cv2.fillPoly(mask, [coords], building['damage_class'])
        except:
            continue

    return mask

def preprocess_xview2(raw_dir, output_dir, img_size=512, max_samples=None):
    """Preprocess xView2 dataset."""
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)

    images_dir = raw_dir / 'images'
    labels_dir = raw_dir / 'labels'

    # Get all post-disaster labels
    label_files = sorted(labels_dir.glob('*_post_disaster.json'))

    if max_samples:
        label_files = label_files[:max_samples]

    print(f"Processing {len(label_files)} samples...")

    # Split 80/20
    split_idx = int(len(label_files) * 0.8)
    train_files = label_files[:split_idx]
    val_files = label_files[split_idx:]

    for split, files in [('train', train_files), ('val', val_files)]:
        split_img_dir = output_dir / split / 'images'
        split_mask_dir = output_dir / split / 'masks'
        split_img_dir.mkdir(parents=True, exist_ok=True)
        split_mask_dir.mkdir(parents=True, exist_ok=True)

        for label_path in tqdm(files, desc=f'Processing {split}'):
            sample_id = label_path.stem.replace('_post_disaster', '')

            pre_path = images_dir / f'{sample_id}_pre_disaster.png'
            post_path = images_dir / f'{sample_id}_post_disaster.png'

            if not pre_path.exists() or not post_path.exists():
                continue

            # Load images
            pre_img = np.array(Image.open(pre_path).convert('RGB'))
            post_img = np.array(Image.open(post_path).convert('RGB'))

            # Parse labels and create mask
            buildings = parse_damage_label(label_path)
            mask = create_mask_from_buildings(buildings)

            # Resize
            pre_img = cv2.resize(pre_img, (img_size, img_size))
            post_img = cv2.resize(post_img, (img_size, img_size))
            mask = cv2.resize(mask, (img_size, img_size), interpolation=cv2.INTER_NEAREST)

            # Combine pre+post into 6-channel
            combined = np.concatenate([pre_img, post_img], axis=-1)

            # Save
            np.save(split_img_dir / f'{sample_id}.npy', combined)
            Image.fromarray(mask).save(split_mask_dir / f'{sample_id}.png')

    print("Preprocessing complete!")
    return output_dir

# ============================================================
# CELL 4: Dataset Class
# ============================================================

class DisasterDataset(Dataset):
    def __init__(self, data_dir, split='train', use_augmentation=True):
        self.data_dir = Path(data_dir)
        self.split = split
        self.use_augmentation = use_augmentation and split == 'train'

        self.images_dir = self.data_dir / split / 'images'
        self.masks_dir = self.data_dir / split / 'masks'

        self.sample_ids = [f.stem for f in self.images_dir.glob('*.npy')]
        print(f"Loaded {len(self.sample_ids)} samples for {split}")

        if self.use_augmentation:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Affine(scale=(0.9, 1.1), translate_percent=(-0.1, 0.1), rotate=(-15, 15), p=0.5),
                A.GaussNoise(p=0.2),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406, 0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225, 0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                A.Normalize(
                    mean=[0.485, 0.456, 0.406, 0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225, 0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]

        image = np.load(self.images_dir / f'{sample_id}.npy')
        mask = np.array(Image.open(self.masks_dir / f'{sample_id}.png'))

        transformed = self.transform(image=image, mask=mask)
        image = transformed['image']
        mask = transformed['mask'].long()

        return {'image': image, 'mask': mask, 'sample_id': sample_id}

# ============================================================
# CELL 5: Model
# ============================================================

class DamageDetector(nn.Module):
    def __init__(self, architecture='DeepLabV3Plus', encoder='resnet50', num_classes=4, pretrained=True):
        super().__init__()

        encoder_weights = 'imagenet' if pretrained else None

        if architecture == 'DeepLabV3Plus':
            self.model = smp.DeepLabV3Plus(
                encoder_name=encoder,
                encoder_weights=encoder_weights,
                in_channels=6,
                classes=num_classes
            )
        elif architecture == 'UNet':
            self.model = smp.Unet(
                encoder_name=encoder,
                encoder_weights=encoder_weights,
                in_channels=6,
                classes=num_classes
            )
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

    def forward(self, x):
        return self.model(x)

# ============================================================
# CELL 6: Loss Functions
# ============================================================

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        num_classes = inputs.shape[1]
        inputs = F.softmax(inputs, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()

        intersection = (inputs * targets_one_hot).sum(dim=(2, 3))
        union = inputs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))

        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

class CombinedLoss(nn.Module):
    def __init__(self, focal_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.focal = FocalLoss()
        self.dice = DiceLoss()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight

    def forward(self, inputs, targets):
        return self.focal_weight * self.focal(inputs, targets) + self.dice_weight * self.dice(inputs, targets)

# ============================================================
# CELL 7: Metrics
# ============================================================

def calculate_iou(preds, targets, num_classes=4):
    """Calculate mean IoU."""
    ious = []
    preds = preds.view(-1)
    targets = targets.view(-1)

    for cls in range(num_classes):
        pred_mask = preds == cls
        target_mask = targets == cls

        intersection = (pred_mask & target_mask).sum().float()
        union = (pred_mask | target_mask).sum().float()

        if union > 0:
            ious.append((intersection / union).item())

    return np.mean(ious) if ious else 0.0

def calculate_f1(preds, targets, num_classes=4):
    """Calculate mean F1 score."""
    f1_scores = []
    preds = preds.view(-1)
    targets = targets.view(-1)

    for cls in range(num_classes):
        pred_mask = preds == cls
        target_mask = targets == cls

        tp = (pred_mask & target_mask).sum().float()
        fp = (pred_mask & ~target_mask).sum().float()
        fn = (~pred_mask & target_mask).sum().float()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        f1_scores.append(f1.item())

    return np.mean(f1_scores)

# ============================================================
# CELL 8: Training Function
# ============================================================

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_targets = []

    pbar = tqdm(loader, desc='Training')
    for batch in pbar:
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        preds = torch.argmax(outputs, dim=1)
        all_preds.append(preds.cpu())
        all_targets.append(masks.cpu())

        pbar.set_postfix(loss=loss.item())

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    return {
        'loss': total_loss / len(loader),
        'iou': calculate_iou(all_preds, all_targets),
        'f1': calculate_f1(all_preds, all_targets)
    }

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(loader, desc='Validation'):
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

    return {
        'loss': total_loss / len(loader),
        'iou': calculate_iou(all_preds, all_targets),
        'f1': calculate_f1(all_preds, all_targets)
    }

# ============================================================
# CELL 9: Main Training Loop
# ============================================================

def main():
    print("=" * 60)
    print("RapidEye Training")
    print("=" * 60)

    # Step 1: Preprocess data (if not already done)
    processed_dir = Path(config.OUTPUT_DIR)
    if not (processed_dir / 'train' / 'images').exists():
        print("\n[1/4] Preprocessing data...")
        preprocess_xview2(
            config.RAW_DATA_DIR,
            config.OUTPUT_DIR,
            img_size=config.IMG_SIZE,
            max_samples=config.MAX_SAMPLES
        )
    else:
        print("\n[1/4] Using existing preprocessed data")

    # Step 2: Create datasets
    print("\n[2/4] Creating datasets...")
    train_dataset = DisasterDataset(config.OUTPUT_DIR, split='train', use_augmentation=True)
    val_dataset = DisasterDataset(config.OUTPUT_DIR, split='val', use_augmentation=False)

    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=True
    )

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Step 3: Create model
    print("\n[3/4] Creating model...")
    model = DamageDetector(
        architecture=config.ARCHITECTURE,
        encoder=config.ENCODER,
        num_classes=config.NUM_CLASSES,
        pretrained=True
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {config.ARCHITECTURE} with {config.ENCODER}")
    print(f"Parameters: {total_params:,}")

    # Setup training
    criterion = CombinedLoss()
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.EPOCHS)

    # Step 4: Training loop
    print(f"\n[4/4] Training for {config.EPOCHS} epochs...")

    history = {'train_loss': [], 'val_loss': [], 'train_iou': [], 'val_iou': [], 'train_f1': [], 'val_f1': []}
    best_iou = 0

    for epoch in range(config.EPOCHS):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{config.EPOCHS}")
        print(f"{'='*50}")

        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_metrics = validate(model, val_loader, criterion, device)

        # Update scheduler
        scheduler.step()

        # Log metrics
        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['train_iou'].append(train_metrics['iou'])
        history['val_iou'].append(val_metrics['iou'])
        history['train_f1'].append(train_metrics['f1'])
        history['val_f1'].append(val_metrics['f1'])

        print(f"\nTrain - Loss: {train_metrics['loss']:.4f}, IoU: {train_metrics['iou']:.4f}, F1: {train_metrics['f1']:.4f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, IoU: {val_metrics['iou']:.4f}, F1: {val_metrics['f1']:.4f}")

        # Save best model
        if val_metrics['iou'] > best_iou:
            best_iou = val_metrics['iou']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_iou': best_iou,
                'config': vars(config)
            }, f'{config.MODEL_DIR}/best.pth')
            print(f"Saved best model (IoU: {best_iou:.4f})")

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history
            }, f'{config.MODEL_DIR}/checkpoint_epoch{epoch+1}.pth')

    # Save final model and history
    torch.save(model.state_dict(), f'{config.MODEL_DIR}/final.pth')

    with open(f'{config.MODEL_DIR}/history.json', 'w') as f:
        json.dump(history, f)

    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Best Validation IoU: {best_iou:.4f}")
    print(f"Models saved to: {config.MODEL_DIR}")
    print("=" * 60)

    return model, history

# Run training
if __name__ == '__main__':
    model, history = main()
