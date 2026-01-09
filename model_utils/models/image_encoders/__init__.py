"""
Image encoder wrappers for different computer vision models.

Each wrapper provides a consistent interface for encoding PIL Images:
- encode_image: Encode a single image
- encode_images: Encode a batch of images  
- embedding_dim: Output embedding dimension
- to(device): Move to device
"""

from .base_image_encoder import BaseImageEncoder
from .vit_encoder import ViTImageEncoder
from .resnet_encoder import ResNetImageEncoder
from .convnext_encoder import ConvNextV2Encoder
from .vitmae_encoder import ViTMAEEncoder
from .siglip_image_encoder import SigLIPImageEncoder


__all__ = [
    'BaseImageEncoder',
    'ViTImageEncoder',
    'ResNetImageEncoder',
    'ConvNextV2Encoder',
    'ViTMAEEncoder',
    'SigLIPImageEncoder',
]
