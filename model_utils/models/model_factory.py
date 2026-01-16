from .wrappers import (
    CLIPModelWrapper,
    CoCaModelWrapper,
    FLAVAModelWrapper,
    SigLIPModelWrapper,
    InternVLModelWrapper,
    SAILVModelWrapper,
)

class ModelFactory:
    """Factory class for creating different model wrappers."""
    
    MODEL_CONFIGS = {
        'clip': {
            'class': CLIPModelWrapper,
            'default_name': 'openai/clip-vit-base-patch32',
            'year': 2021,
        },
        'coca': {
            'class': CoCaModelWrapper,
            'default_name': 'microsoft/git-base-coco',
            'year': 2022,
        },
        'flava': {
            'class': FLAVAModelWrapper,
            'default_name': 'facebook/flava-full',
            'year': 2021,
        },
        'siglip': {
            'class': SigLIPModelWrapper,
            'default_name': 'google/siglip-base-patch16-224',
            'year': 2023,
        },
        'internvl': {
            'class': InternVLModelWrapper,
            'default_name': 'OpenGVLab/InternVL2-2B',
            'year': 2024,
        },
        'sailv': {
            'class': SAILVModelWrapper,
            'default_name': 'BytedanceDouyinContent/SAIL-VL-1d5-2B',
            'year': 2024,
        },
    }
    
    @classmethod
    def create_model(cls, model_type, model_name=None, device=None):
        """
        Create a model wrapper of the specified type.
        
        Args:
            model_type: One of 'clip', 'coca', 'flava', 'siglip', 'internvl'
            model_name: Specific model name (optional, uses default if not provided)
            device: Device to run on (auto-detected if None)
            
        Returns:
            Model wrapper instance
            
        Raises:
            ValueError: If model_type is not supported
            RuntimeError: If model cannot be created
        """
        if model_type not in cls.MODEL_CONFIGS:
            raise ValueError(f"Unsupported model type: {model_type}. "
                           f"Supported types: {list(cls.MODEL_CONFIGS.keys())}")
        
        config = cls.MODEL_CONFIGS[model_type]
        model_class = config['class']
        default_name = config['default_name']
        
        # Use provided model_name or default
        model_name = model_name or default_name
        
        try:
            return model_class(model_name, device)
        except ImportError as e:
            # Provide helpful error message for missing dependencies
            if model_type == 'siglip':
                raise ImportError(
                    f"Failed to create {model_type} model: {str(e)}\n"
                    "Please install required dependencies: pip install sentencepiece==0.2.0"
                )
            else:
                raise ImportError(f"Failed to create {model_type} model: {str(e)}")
        except OSError as e:
            # Handle gated models and network errors
            error_msg = str(e)
            if 'gated repo' in error_msg.lower() or '403' in error_msg:
                raise RuntimeError(
                    f"Failed to create {model_type} model: This is a gated model.\n"
                    f"You need to authenticate with HuggingFace. Visit: https://huggingface.co/{model_name}\n"
                    f"Then run: huggingface-cli login"
                )
            else:
                raise RuntimeError(f"Failed to create {model_type} model: {str(e)[:200]}")
        except Exception as e:
            # For CogVLM and QwenVLM, provide special error messages
            if model_type in ['cogvlm', 'qwenvlm']:
                error_msg = str(e)
                if 'Unrecognized configuration class' in error_msg:
                    raise RuntimeError(
                        f"Failed to create {model_type} model: {model_type.upper()} requires a newer or custom "
                        f"version of the transformers library. The model config is not registered in this environment.\n"
                        f"To fix this, try: pip install --upgrade transformers\n"
                        f"Error details: {error_msg[:200]}"
                    )
            raise RuntimeError(f"Failed to create {model_type} model: {str(e)}")
    
    @classmethod
    def get_available_models(cls):
        """Get list of available model types."""
        return list(cls.MODEL_CONFIGS.keys())
    
    @classmethod
    def get_default_model_name(cls, model_type):
        """Get the default model name for a given model type."""
        if model_type not in cls.MODEL_CONFIGS:
            raise ValueError(f"Unsupported model type: {model_type}")
        return cls.MODEL_CONFIGS[model_type]['default_name'] 