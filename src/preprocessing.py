"""
xView2 Dataset Preprocessing Pipeline
Converts raw images + JSON labels into training-ready segmentation masks
"""

import os
import json
import numpy as np
from PIL import Image
from pathlib import Path
from shapely import wkt
from shapely.geometry import Polygon
import cv2
from tqdm import tqdm
import random
from concurrent.futures import ProcessPoolExecutor, as_completed

# Damage level mapping
DAMAGE_CLASSES = {
    'no-damage': 0,
    'minor-damage': 1,
    'major-damage': 2,
    'destroyed': 3,
    'un-classified': 0  # Treat unclassified as background
}

class XView2Preprocessor:
    def __init__(
        self,
        raw_data_dir: str,
        output_dir: str,
        img_size: int = 512,
        val_split: float = 0.2
    ):
        """
        Initialize the preprocessor.

        Args:
            raw_data_dir: Path to xView2 raw data (contains train/train/images and labels)
            output_dir: Path to save processed data
            img_size: Target image size (will resize to img_size x img_size)
            val_split: Fraction of data to use for validation
        """
        self.raw_data_dir = Path(raw_data_dir)
        self.output_dir = Path(output_dir)
        self.img_size = img_size
        self.val_split = val_split

        # Setup paths
        self.images_dir = self.raw_data_dir / 'train' / 'train' / 'images'
        self.labels_dir = self.raw_data_dir / 'train' / 'train' / 'labels'

        # Create output directories
        for split in ['train', 'val']:
            (self.output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.output_dir / split / 'masks').mkdir(parents=True, exist_ok=True)

    def parse_polygon(self, wkt_str: str) -> np.ndarray:
        """Parse WKT polygon string into numpy array of coordinates."""
        try:
            poly = wkt.loads(wkt_str)
            if poly.is_valid:
                coords = np.array(poly.exterior.coords)
                return coords
        except Exception as e:
            pass
        return None

    def create_mask(self, label_path: str, img_size: tuple = (1024, 1024)) -> np.ndarray:
        """
        Create segmentation mask from JSON label file.

        Args:
            label_path: Path to JSON label file
            img_size: Original image size (height, width)

        Returns:
            mask: numpy array of shape (H, W) with damage class values
        """
        mask = np.zeros(img_size, dtype=np.uint8)

        with open(label_path, 'r') as f:
            data = json.load(f)

        # Use xy coordinates (pixel space)
        buildings = data.get('features', {}).get('xy', [])

        for building in buildings:
            damage_type = building.get('properties', {}).get('subtype', 'no-damage')
            damage_class = DAMAGE_CLASSES.get(damage_type, 0)

            wkt_str = building.get('wkt', '')
            coords = self.parse_polygon(wkt_str)

            if coords is not None:
                # Convert to integer pixel coordinates
                pts = coords.astype(np.int32)
                # Fill polygon on mask
                cv2.fillPoly(mask, [pts], damage_class)

        return mask

    def get_sample_pairs(self) -> list:
        """
        Get all pre/post disaster image pairs.

        Returns:
            List of tuples: (pre_img_path, post_img_path, pre_label_path, post_label_path, sample_id)
        """
        pairs = []

        # Get all post-disaster images (they have the damage labels)
        post_images = list(self.images_dir.glob('*_post_disaster.png'))

        for post_img in post_images:
            # Construct corresponding pre-disaster path
            sample_id = post_img.stem.replace('_post_disaster', '')
            pre_img = self.images_dir / f'{sample_id}_pre_disaster.png'

            # Construct label paths
            post_label = self.labels_dir / f'{sample_id}_post_disaster.json'
            pre_label = self.labels_dir / f'{sample_id}_pre_disaster.json'

            # Verify all files exist
            if pre_img.exists() and post_label.exists():
                pairs.append({
                    'pre_img': str(pre_img),
                    'post_img': str(post_img),
                    'pre_label': str(pre_label) if pre_label.exists() else None,
                    'post_label': str(post_label),
                    'sample_id': sample_id
                })

        return pairs

    def process_sample(self, sample: dict) -> dict:
        """
        Process a single sample pair.

        Args:
            sample: Dictionary with paths to pre/post images and labels

        Returns:
            Dictionary with processing results
        """
        try:
            # Load images
            pre_img = Image.open(sample['pre_img']).convert('RGB')
            post_img = Image.open(sample['post_img']).convert('RGB')

            # Create damage mask from post-disaster label
            mask = self.create_mask(sample['post_label'], img_size=(1024, 1024))

            # Resize images and mask
            pre_img = pre_img.resize((self.img_size, self.img_size), Image.BILINEAR)
            post_img = post_img.resize((self.img_size, self.img_size), Image.BILINEAR)
            mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

            # Concatenate pre and post images (6 channels: RGB + RGB)
            pre_arr = np.array(pre_img)
            post_arr = np.array(post_img)
            combined = np.concatenate([pre_arr, post_arr], axis=-1)  # Shape: (H, W, 6)

            return {
                'sample_id': sample['sample_id'],
                'combined_img': combined,
                'mask': mask,
                'success': True
            }

        except Exception as e:
            return {
                'sample_id': sample['sample_id'],
                'error': str(e),
                'success': False
            }

    def save_sample(self, result: dict, split: str):
        """Save processed sample to disk."""
        if not result['success']:
            return

        sample_id = result['sample_id']

        # Save combined image as numpy file (6 channels)
        img_path = self.output_dir / split / 'images' / f'{sample_id}.npy'
        np.save(img_path, result['combined_img'])

        # Save mask as PNG (single channel)
        mask_path = self.output_dir / split / 'masks' / f'{sample_id}.png'
        Image.fromarray(result['mask']).save(mask_path)

    def run(self, max_samples: int = None, num_workers: int = 4):
        """
        Run the full preprocessing pipeline.

        Args:
            max_samples: Maximum number of samples to process (None for all)
            num_workers: Number of parallel workers
        """
        print("Getting sample pairs...")
        pairs = self.get_sample_pairs()

        if max_samples:
            pairs = pairs[:max_samples]

        print(f"Found {len(pairs)} sample pairs")

        # Shuffle and split
        random.seed(42)
        random.shuffle(pairs)

        n_val = int(len(pairs) * self.val_split)
        val_pairs = pairs[:n_val]
        train_pairs = pairs[n_val:]

        print(f"Train: {len(train_pairs)}, Val: {len(val_pairs)}")

        # Process train set
        print("\nProcessing training set...")
        for sample in tqdm(train_pairs, desc="Train"):
            result = self.process_sample(sample)
            self.save_sample(result, 'train')

        # Process validation set
        print("\nProcessing validation set...")
        for sample in tqdm(val_pairs, desc="Val"):
            result = self.process_sample(sample)
            self.save_sample(result, 'val')

        # Print statistics
        self.print_stats()

    def print_stats(self):
        """Print dataset statistics."""
        for split in ['train', 'val']:
            img_count = len(list((self.output_dir / split / 'images').glob('*.npy')))
            print(f"\n{split.upper()} set: {img_count} samples")

            # Count damage distribution
            damage_counts = {0: 0, 1: 0, 2: 0, 3: 0}
            mask_dir = self.output_dir / split / 'masks'

            for mask_path in list(mask_dir.glob('*.png'))[:100]:  # Sample 100
                mask = np.array(Image.open(mask_path))
                for i in range(4):
                    damage_counts[i] += np.sum(mask == i)

            total_pixels = sum(damage_counts.values())
            if total_pixels > 0:
                print("Damage distribution (sampled):")
                for level, count in damage_counts.items():
                    pct = count / total_pixels * 100
                    print(f"  Class {level}: {pct:.2f}%")


def preprocess_xview2(
    raw_dir: str = '/Users/mac/iitmhack/archive',
    output_dir: str = '/Users/mac/iitmhack/data/xview2_processed',
    img_size: int = 512,
    max_samples: int = None
):
    """
    Convenience function to run preprocessing.

    Args:
        raw_dir: Path to raw xView2 data
        output_dir: Path to save processed data
        img_size: Target image size
        max_samples: Maximum samples to process (None for all)
    """
    preprocessor = XView2Preprocessor(
        raw_data_dir=raw_dir,
        output_dir=output_dir,
        img_size=img_size
    )
    preprocessor.run(max_samples=max_samples)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Preprocess xView2 dataset')
    parser.add_argument('--raw-dir', type=str, default='/Users/mac/iitmhack/archive',
                        help='Path to raw xView2 data')
    parser.add_argument('--output-dir', type=str, default='/Users/mac/iitmhack/data/xview2_processed',
                        help='Path to save processed data')
    parser.add_argument('--img-size', type=int, default=512,
                        help='Target image size')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum samples to process')

    args = parser.parse_args()

    preprocess_xview2(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        img_size=args.img_size,
        max_samples=args.max_samples
    )
