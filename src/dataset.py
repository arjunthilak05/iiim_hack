"""
PyTorch Dataset for Disaster Damage Detection
Loads preprocessed xView2 data with augmentations
"""

import os
import numpy as np
from PIL import Image
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2


class DisasterDataset(Dataset):
    """
    Dataset for disaster damage segmentation.

    Loads 6-channel images (pre + post disaster) and corresponding damage masks.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        transform=None,
        use_augmentation: bool = True
    ):
        """
        Args:
            data_dir: Path to processed data directory
            split: 'train' or 'val'
            transform: Optional custom transform
            use_augmentation: Whether to apply augmentations (only for train)
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.use_augmentation = use_augmentation and split == 'train'

        # Get all sample IDs
        self.images_dir = self.data_dir / split / 'images'
        self.masks_dir = self.data_dir / split / 'masks'

        self.sample_ids = [
            f.stem for f in self.images_dir.glob('*.npy')
        ]

        print(f"Loaded {len(self.sample_ids)} samples for {split}")

        # Default augmentations
        if self.transform is None and self.use_augmentation:
            self.transform = self._get_train_augmentations()
        elif self.transform is None:
            self.transform = self._get_val_augmentations()

    def _get_train_augmentations(self):
        """Training augmentations for 6-channel images."""
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Affine(
                scale=(0.9, 1.1),
                translate_percent=(-0.1, 0.1),
                rotate=(-15, 15),
                p=0.5
            ),
            A.GaussNoise(p=0.2),
            A.Normalize(
                mean=[0.485, 0.456, 0.406, 0.485, 0.456, 0.406],  # ImageNet mean for 6 channels
                std=[0.229, 0.224, 0.225, 0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])

    def _get_val_augmentations(self):
        """Validation augmentations (just normalization)."""
        return A.Compose([
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

        # Load 6-channel image (H, W, 6)
        img_path = self.images_dir / f'{sample_id}.npy'
        image = np.load(img_path)

        # Load mask (H, W)
        mask_path = self.masks_dir / f'{sample_id}.png'
        mask = np.array(Image.open(mask_path))

        # Apply augmentations
        if self.transform:
            # Albumentations expects image as uint8 or float
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']  # (6, H, W) tensor
            mask = transformed['mask']    # (H, W) tensor

        # Convert mask to long tensor for CrossEntropyLoss
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(np.array(mask)).long()
        else:
            mask = mask.clone().detach().long()

        return {
            'image': image,
            'mask': mask,
            'sample_id': sample_id
        }


def get_dataloaders(
    data_dir: str,
    batch_size: int = 8,
    num_workers: int = 4,
    img_size: int = 512
) -> tuple:
    """
    Create train and validation dataloaders.

    Args:
        data_dir: Path to processed data
        batch_size: Batch size
        num_workers: Number of data loading workers
        img_size: Image size (not used here, preprocessing handles it)

    Returns:
        train_loader, val_loader
    """
    train_dataset = DisasterDataset(data_dir, split='train', use_augmentation=True)
    val_dataset = DisasterDataset(data_dir, split='val', use_augmentation=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


class DisasterDatasetRaw(Dataset):
    """
    Dataset that loads raw images directly (for inference/testing).
    No preprocessing required.
    """

    def __init__(self, image_pairs: list, img_size: int = 512):
        """
        Args:
            image_pairs: List of dicts with 'pre_img' and 'post_img' paths
            img_size: Target size for resizing
        """
        self.image_pairs = image_pairs
        self.img_size = img_size
        self.transform = A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406, 0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225, 0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        pair = self.image_pairs[idx]

        # Load images
        pre_img = np.array(Image.open(pair['pre_img']).convert('RGB'))
        post_img = np.array(Image.open(pair['post_img']).convert('RGB'))

        # Combine
        combined = np.concatenate([pre_img, post_img], axis=-1)

        # Transform
        transformed = self.transform(image=combined)
        image = transformed['image']

        return {
            'image': image,
            'sample_id': pair.get('sample_id', f'sample_{idx}')
        }


if __name__ == '__main__':
    # Test dataset
    data_dir = '/Users/mac/iitmhack/data/xview2_processed'

    if Path(data_dir).exists():
        train_loader, val_loader = get_dataloaders(data_dir, batch_size=4)

        print(f"\nTrain batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")

        # Test one batch
        batch = next(iter(train_loader))
        print(f"\nBatch shapes:")
        print(f"  Image: {batch['image'].shape}")  # Should be (B, 6, H, W)
        print(f"  Mask: {batch['mask'].shape}")    # Should be (B, H, W)
        print(f"  Mask unique values: {torch.unique(batch['mask'])}")
    else:
        print(f"Data directory not found: {data_dir}")
        print("Run preprocessing.py first")
