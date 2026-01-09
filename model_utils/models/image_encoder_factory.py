"""
Factory for creating image encoder instances.

Provides a centralized way to instantiate any image encoder with consistent configuration.
"""

from model_utils.models.image_encoders import (
    ViTImageEncoder,
    ResNetImageEncoder,
    ConvNextV2Encoder,
    ViTMAEEncoder,
    SigLIPImageEncoder,
)


class ImageEncoderFactory:
    """Factory class for creating different image encoder wrappers."""
    
    MODEL_CONFIGS = {
        'vit': {
            'class': ViTImageEncoder,
            'default_name': 'google/vit-base-patch16-224',
            'description': 'Vision Transformer',
            'year': 2020,
        },
        'resnet': {
            'class': ResNetImageEncoder,
            'default_name': 'microsoft/resnet-50',
            'description': 'ResNet CNN',
            'year': 2015,
        },
        'convnext': {
            'class': ConvNextV2Encoder,
            'default_name': 'facebook/convnextv2-base-1k-224',
            'description': 'ConvNeXt V2 CNN',
            'year': 2023,
        },
        'vitmae': {
            'class': ViTMAEEncoder,
            'default_name': 'facebook/vit-mae-base',
            'description': 'ViT-MAE (Self-supervised)',
            'year': 2022,
        },
        'siglip': {
            'class': SigLIPImageEncoder,
            'default_name': 'google/siglip-base-patch16-224',
            'description': 'SigLIP VLM (Image component)',
            'year': 2023,
        },
    }
    
    @classmethod
    def create_model(cls, model_type, model_name=None, device=None):
        """
        Create an image encoder of the specified type.
        
        Args:
            model_type: One of 'vit', 'resnet', 'convnext', 'vitmae', 'siglip'
            model_name: Specific model name (optional, uses default if not provided)
            device: Device to run on (auto-detected if None)
            
        Returns:
            Image encoder instance
            
        Raises:
            ValueError: If model_type is not supported
            
        Example:
            >>> encoder = ImageEncoderFactory.create_model('vit', device='cuda')
            >>> images = [img1, img2, img3]  # PIL Images
            >>> embeddings = encoder.encode_images(images)
        """
        if model_type not in cls.MODEL_CONFIGS:
            supported = ", ".join(cls.MODEL_CONFIGS.keys())
            raise ValueError(
                f"Unsupported model type: {model_type}. "
                f"Supported types: {supported}"
            )
        
        config = cls.MODEL_CONFIGS[model_type]
        model_class = config['class']
        default_name = config['default_name']
        
        # Use provided model_name or default
        model_name = model_name or default_name
        
        try:
            return model_class(model_name, device)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load {model_type} model '{model_name}': {str(e)}"
            )
    
    @classmethod
    def list_models(cls):
        """
        List all available image encoder models.
        
        Returns:
            dict with model info
        """
        return {
            model_type: {
                'description': config['description'],
                'default_name': config['default_name'],
                'year': config['year'],
            }
            for model_type, config in cls.MODEL_CONFIGS.items()
        }
