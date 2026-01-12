"""
ResNet image encoder wrapper.
"""

import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel
from .base_image_encoder import BaseImageEncoder


class ResNetImageEncoder(BaseImageEncoder):
    """
    Wrapper for Microsoft ResNet image encoders.
    
    ResNet models use the pooled representation from the final layer.
    """
    
    def __init__(self, model_name="microsoft/resnet-50", device=None):
        """
        Initialize ResNet image encoder.
        
        Args:
            model_name: Model identifier on Hugging Face Hub
                       (default: microsoft/resnet-50)
            device: Device to run on (cuda/cpu). Auto-detected if None.
        """
        self.model_name = model_name
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
        self._load_model()
    
    def _load_model(self):
        """Load ResNet model and image processor from Hugging Face."""
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()
    
    def encode_image(self, image):
        """
        Encode a single PIL Image to embedding.
        
        Args:
            image: PIL.Image object
            
        Returns:
            torch.FloatTensor of shape (1, embedding_dim)
        """
        return self.encode_images([image])
    
    def encode_images(self, images):
        """
        Encode a batch of PIL Images to embeddings.
        
        ResNet outputs a 1D pooled representation.
        
        Args:
            images: List of PIL.Image objects
            
        Returns:
            torch.FloatTensor of shape (batch_size, embedding_dim)
        """
        # Preprocess images
        pixel_values = self.processor(images=images, return_tensors="pt").pixel_values.to(self.device)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(pixel_values=pixel_values)
            # ResNet returns (last_hidden_state,) or (pooler_output,)
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                embeddings = outputs.pooler_output
            else:
                # Fallback: use last hidden state
                embeddings = outputs.last_hidden_state
                if len(embeddings.shape) > 2:
                    embeddings = embeddings.mean(dim=[2, 3])  # Pool spatial dimensions
        
        return F.normalize(embeddings, dim=1)
    
    @property
    def embedding_dim(self):
        """Return embedding dimension."""
        return self.model.config.hidden_size
