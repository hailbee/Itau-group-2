"""
ViT-MAE (Vision Transformer - Masked AutoEncoder) image encoder wrapper.
"""

import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel
from .base_image_encoder import BaseImageEncoder


class ViTMAEEncoder(BaseImageEncoder):
    """
    Wrapper for Meta Vision Transformer - Masked AutoEncoder (ViT-MAE).
    
    ViT-MAE is a self-supervised vision model trained with masked image modeling.
    Uses the pooled [CLS] token representation from the encoder.
    """
    
    def __init__(self, model_name="facebook/vit-mae-base", device=None):
        """
        Initialize ViT-MAE encoder.
        
        Args:
            model_name: Model identifier on Hugging Face Hub
                       (default: facebook/vit-mae-base)
            device: Device to run on (cuda/cpu). Auto-detected if None.
        """
        self.model_name = model_name
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
        self._load_model()
    
    def _load_model(self):
        """Load ViT-MAE model and image processor from Hugging Face."""
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
        
        Uses the masked autoencoder encoder output (last hidden state of [CLS] token).
        
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
            
            # ViT-MAE returns last_hidden_state
            embeddings = outputs.last_hidden_state
            
            # Extract [CLS] token (first token)
            if len(embeddings.shape) == 3:
                embeddings = embeddings[:, 0, :]  # [batch_size, hidden_size]
        
        return F.normalize(embeddings, dim=1)
    
    @property
    def embedding_dim(self):
        """Return embedding dimension."""
        return self.model.config.hidden_size
