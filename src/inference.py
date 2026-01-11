"""
Inference Pipeline for Disaster Damage Detection
Run predictions on new satellite imagery
"""

import os
import sys
import time
from pathlib import Path
from typing import Union, Tuple, List

import numpy as np
import torch
from PIL import Image
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

sys.path.append(str(Path(__file__).parent))

from model import DamageDetector, load_checkpoint


# Damage class names and colors
DAMAGE_CLASSES = {
    0: {'name': 'No Damage', 'color': (0, 255, 0)},      # Green
    1: {'name': 'Minor Damage', 'color': (255, 255, 0)},  # Yellow
    2: {'name': 'Major Damage', 'color': (255, 165, 0)},  # Orange
    3: {'name': 'Destroyed', 'color': (255, 0, 0)}        # Red
}


class DamagePredictor:
    """
    Predictor class for running inference on satellite imagery.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = None,
        img_size: int = 512,
        architecture: str = 'DeepLabV3Plus',
        encoder: str = 'resnet50'
    ):
        """
        Initialize predictor.

        Args:
            checkpoint_path: Path to model checkpoint
            device: Device to run on ('cuda', 'mps', 'cpu', or None for auto)
            img_size: Input image size
            architecture: Model architecture
            encoder: Encoder backbone
        """
        self.img_size = img_size

        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")

        # Load model
        self.model = DamageDetector(
            architecture=architecture,
            encoder=encoder,
            num_classes=4,
            pretrained=False
        )

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)

        self.model = self.model.to(self.device)
        self.model.eval()

        # Preprocessing transform
        self.transform = A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406, 0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225, 0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])

    def load_image(self, path: str) -> np.ndarray:
        """Load image from path."""
        img = Image.open(path).convert('RGB')
        return np.array(img)

    def preprocess(
        self,
        pre_img: Union[str, np.ndarray],
        post_img: Union[str, np.ndarray]
    ) -> torch.Tensor:
        """
        Preprocess image pair.

        Args:
            pre_img: Pre-disaster image (path or numpy array)
            post_img: Post-disaster image (path or numpy array)

        Returns:
            Preprocessed tensor of shape (1, 6, H, W)
        """
        # Load if paths
        if isinstance(pre_img, str):
            pre_img = self.load_image(pre_img)
        if isinstance(post_img, str):
            post_img = self.load_image(post_img)

        # Store original size
        self.original_size = pre_img.shape[:2]  # (H, W)

        # Combine images
        combined = np.concatenate([pre_img, post_img], axis=-1)

        # Transform
        transformed = self.transform(image=combined)
        tensor = transformed['image'].unsqueeze(0)  # Add batch dim

        return tensor.to(self.device)

    @torch.no_grad()
    def predict(
        self,
        pre_img: Union[str, np.ndarray],
        post_img: Union[str, np.ndarray],
        return_probs: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Run prediction on image pair.

        Args:
            pre_img: Pre-disaster image
            post_img: Post-disaster image
            return_probs: Whether to return class probabilities

        Returns:
            damage_map: Numpy array of shape (H, W) with class indices
            probs (optional): Numpy array of shape (4, H, W) with probabilities
        """
        start_time = time.time()

        # Preprocess
        tensor = self.preprocess(pre_img, post_img)

        # Inference
        logits = self.model(tensor)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        # Convert to numpy
        damage_map = preds[0].cpu().numpy()
        probs_np = probs[0].cpu().numpy()

        # Resize back to original size
        if hasattr(self, 'original_size'):
            damage_map = cv2.resize(
                damage_map.astype(np.uint8),
                (self.original_size[1], self.original_size[0]),
                interpolation=cv2.INTER_NEAREST
            )
            # Resize probabilities
            probs_resized = []
            for c in range(4):
                p = cv2.resize(
                    probs_np[c],
                    (self.original_size[1], self.original_size[0]),
                    interpolation=cv2.INTER_LINEAR
                )
                probs_resized.append(p)
            probs_np = np.stack(probs_resized)

        inference_time = time.time() - start_time
        print(f"Inference time: {inference_time:.3f}s")

        if return_probs:
            return damage_map, probs_np
        return damage_map

    def predict_batch(
        self,
        image_pairs: List[dict],
        batch_size: int = 4
    ) -> List[np.ndarray]:
        """
        Run predictions on multiple image pairs.

        Args:
            image_pairs: List of dicts with 'pre_img' and 'post_img' keys
            batch_size: Batch size for inference

        Returns:
            List of damage maps
        """
        results = []

        for i in range(0, len(image_pairs), batch_size):
            batch = image_pairs[i:i+batch_size]

            # Preprocess batch
            tensors = []
            sizes = []
            for pair in batch:
                pre = self.load_image(pair['pre_img'])
                post = self.load_image(pair['post_img'])
                sizes.append(pre.shape[:2])

                combined = np.concatenate([pre, post], axis=-1)
                transformed = self.transform(image=combined)
                tensors.append(transformed['image'])

            # Stack and predict
            batch_tensor = torch.stack(tensors).to(self.device)

            with torch.no_grad():
                logits = self.model(batch_tensor)
                preds = torch.argmax(logits, dim=1)

            # Process results
            for j, pred in enumerate(preds):
                damage_map = pred.cpu().numpy()
                # Resize to original
                damage_map = cv2.resize(
                    damage_map.astype(np.uint8),
                    (sizes[j][1], sizes[j][0]),
                    interpolation=cv2.INTER_NEAREST
                )
                results.append(damage_map)

        return results


def create_damage_overlay(
    post_img: np.ndarray,
    damage_map: np.ndarray,
    alpha: float = 0.5
) -> np.ndarray:
    """
    Create visualization overlay showing damage areas.

    Args:
        post_img: Post-disaster image (H, W, 3)
        damage_map: Damage predictions (H, W)
        alpha: Transparency of overlay

    Returns:
        Overlay image (H, W, 3)
    """
    # Create colored mask
    overlay = np.zeros_like(post_img)

    for class_id, info in DAMAGE_CLASSES.items():
        mask = damage_map == class_id
        overlay[mask] = info['color']

    # Blend with original
    result = cv2.addWeighted(post_img, 1 - alpha, overlay, alpha, 0)

    return result


def calculate_damage_stats(damage_map: np.ndarray, pixel_size_m: float = 0.5) -> dict:
    """
    Calculate damage statistics from prediction.

    Args:
        damage_map: Damage predictions (H, W)
        pixel_size_m: Size of each pixel in meters

    Returns:
        Dictionary with statistics
    """
    total_pixels = damage_map.size
    pixel_area_m2 = pixel_size_m ** 2

    stats = {
        'total_area_km2': total_pixels * pixel_area_m2 / 1e6,
        'damage_counts': {},
        'damage_areas_km2': {},
        'damage_percentages': {}
    }

    for class_id, info in DAMAGE_CLASSES.items():
        count = np.sum(damage_map == class_id)
        area_km2 = count * pixel_area_m2 / 1e6
        pct = count / total_pixels * 100

        stats['damage_counts'][info['name']] = int(count)
        stats['damage_areas_km2'][info['name']] = round(area_km2, 4)
        stats['damage_percentages'][info['name']] = round(pct, 2)

    # Calculate damaged area (classes 1-3)
    damaged_pixels = np.sum(damage_map > 0)
    stats['total_damaged_area_km2'] = round(damaged_pixels * pixel_area_m2 / 1e6, 4)
    stats['total_damaged_percentage'] = round(damaged_pixels / total_pixels * 100, 2)

    return stats


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run damage detection inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--pre-img', type=str, required=True,
                        help='Path to pre-disaster image')
    parser.add_argument('--post-img', type=str, required=True,
                        help='Path to post-disaster image')
    parser.add_argument('--output', type=str, default='output.png',
                        help='Output path for overlay image')

    args = parser.parse_args()

    # Run inference
    predictor = DamagePredictor(args.checkpoint)
    damage_map = predictor.predict(args.pre_img, args.post_img)

    # Create overlay
    post_img = np.array(Image.open(args.post_img).convert('RGB'))
    overlay = create_damage_overlay(post_img, damage_map)

    # Save
    Image.fromarray(overlay).save(args.output)
    print(f"Saved overlay to {args.output}")

    # Print stats
    stats = calculate_damage_stats(damage_map)
    print("\nDamage Statistics:")
    for name, pct in stats['damage_percentages'].items():
        print(f"  {name}: {pct}%")
    print(f"\nTotal damaged area: {stats['total_damaged_percentage']}%")
