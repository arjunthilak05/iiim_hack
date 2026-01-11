"""
Model Architecture for Disaster Damage Detection
Uses segmentation-models-pytorch with modified input channels
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


def get_model(
    architecture: str = 'DeepLabV3Plus',
    encoder: str = 'resnet50',
    in_channels: int = 6,
    num_classes: int = 4,
    pretrained: bool = True
) -> nn.Module:
    """
    Create a segmentation model for damage detection.

    Args:
        architecture: Model architecture ('DeepLabV3Plus', 'UNet', 'FPN', 'PSPNet')
        encoder: Encoder backbone ('resnet50', 'resnet34', 'efficientnet-b3', etc.)
        in_channels: Number of input channels (6 for pre+post RGB)
        num_classes: Number of output classes (4 damage levels)
        pretrained: Use ImageNet pretrained weights

    Returns:
        PyTorch model
    """
    encoder_weights = 'imagenet' if pretrained else None

    model_map = {
        'DeepLabV3Plus': smp.DeepLabV3Plus,
        'UNet': smp.Unet,
        'UNetPlusPlus': smp.UnetPlusPlus,
        'FPN': smp.FPN,
        'PSPNet': smp.PSPNet,
        'Linknet': smp.Linknet,
        'MAnet': smp.MAnet,
    }

    if architecture not in model_map:
        raise ValueError(f"Architecture {architecture} not supported. Choose from {list(model_map.keys())}")

    model = model_map[architecture](
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
        activation=None  # We'll apply softmax in loss function
    )

    return model


class DamageDetector(nn.Module):
    """
    Wrapper class for damage detection model with additional functionality.
    """

    def __init__(
        self,
        architecture: str = 'DeepLabV3Plus',
        encoder: str = 'resnet50',
        num_classes: int = 4,
        pretrained: bool = True
    ):
        super().__init__()

        self.model = get_model(
            architecture=architecture,
            encoder=encoder,
            in_channels=6,
            num_classes=num_classes,
            pretrained=pretrained
        )

        self.num_classes = num_classes

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, 6, H, W)
               Channels 0-2: Pre-disaster RGB
               Channels 3-5: Post-disaster RGB

        Returns:
            Logits of shape (B, num_classes, H, W)
        """
        return self.model(x)

    def predict(self, x):
        """
        Get predictions (class indices) from model.

        Args:
            x: Input tensor

        Returns:
            Predictions of shape (B, H, W) with class indices
        """
        with torch.no_grad():
            logits = self.forward(x)
            preds = torch.argmax(logits, dim=1)
        return preds

    def predict_proba(self, x):
        """
        Get class probabilities.

        Args:
            x: Input tensor

        Returns:
            Probabilities of shape (B, num_classes, H, W)
        """
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=1)
        return probs


class SiameseUNet(nn.Module):
    """
    Siamese architecture that processes pre and post images separately
    then fuses features for damage detection.

    This is an alternative architecture that might perform better
    for change detection tasks.
    """

    def __init__(
        self,
        encoder: str = 'resnet34',
        num_classes: int = 4,
        pretrained: bool = True
    ):
        super().__init__()

        encoder_weights = 'imagenet' if pretrained else None

        # Shared encoder for pre and post images
        self.encoder = smp.Unet(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=num_classes,
        )

        # Get encoder
        self.shared_encoder = self.encoder.encoder

        # Decoder that takes concatenated features
        self.decoder = smp.Unet(
            encoder_name=encoder,
            encoder_weights=None,
            in_channels=6,  # Will be modified
            classes=num_classes,
        ).decoder

        self.segmentation_head = smp.Unet(
            encoder_name=encoder,
            encoder_weights=None,
            in_channels=6,
            classes=num_classes,
        ).segmentation_head

        self.num_classes = num_classes

    def forward(self, x):
        """
        Args:
            x: Input tensor (B, 6, H, W)
               First 3 channels: pre-disaster
               Last 3 channels: post-disaster
        """
        # Split input
        pre = x[:, :3, :, :]
        post = x[:, 3:, :, :]

        # Extract features
        pre_features = self.shared_encoder(pre)
        post_features = self.shared_encoder(post)

        # Concatenate features at each level
        fused_features = []
        for pre_f, post_f in zip(pre_features, post_features):
            # Concatenate along channel dimension
            diff = torch.abs(post_f - pre_f)  # Change features
            fused = torch.cat([pre_f, post_f, diff], dim=1)
            fused_features.append(fused)

        # Note: This would require modifying decoder to handle different channel sizes
        # For simplicity, we use the standard approach in get_model()
        # This class is provided as a reference for potential improvements

        return self.encoder(x)


def load_checkpoint(model: nn.Module, checkpoint_path: str, device: str = 'cpu'):
    """
    Load model weights from checkpoint.

    Args:
        model: Model instance
        checkpoint_path: Path to checkpoint file
        device: Device to load weights to

    Returns:
        Model with loaded weights
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    return model


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Test model creation
    print("Testing model architectures...\n")

    for arch in ['DeepLabV3Plus', 'UNet', 'FPN']:
        model = get_model(architecture=arch, encoder='resnet34')
        params = count_parameters(model)
        print(f"{arch} with ResNet34: {params:,} parameters")

        # Test forward pass
        x = torch.randn(2, 6, 512, 512)
        with torch.no_grad():
            y = model(x)
        print(f"  Input: {x.shape} -> Output: {y.shape}\n")

    # Test DamageDetector wrapper
    print("\nTesting DamageDetector wrapper...")
    detector = DamageDetector(architecture='DeepLabV3Plus', encoder='resnet50')

    x = torch.randn(2, 6, 512, 512)
    preds = detector.predict(x)
    probs = detector.predict_proba(x)

    print(f"Predictions shape: {preds.shape}")
    print(f"Probabilities shape: {probs.shape}")
    print(f"Unique predictions: {torch.unique(preds)}")
