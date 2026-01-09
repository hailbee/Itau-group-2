"""
Vision Transformer (ViT) image encoder wrapper.
"""

import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel
from .base_image_encoder import BaseImageEncoder


class ViTImageEncoder(BaseImageEncoder):
    """
    Wrapper for Google Vision Transformer (ViT) image encoders.
    
    Uses the pooled [CLS] token as the image representation.
    """
    
    def __init__(self, model_name="google/vit-base-patch16-224", device=None):
        """
        Initialize ViT image encoder.
        
        Args:
            model_name: Model identifier on Hugging Face Hub
                       (default: google/vit-base-patch16-224)
            device: Device to run on (cuda/cpu). Auto-detected if None.
        """
        self.model_name = model_name
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
        self._load_model()
    
    def _load_model(self):
        """Load ViT model and image processor from Hugging Face."""
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
        
        Uses the pooled output (from the [CLS] token).
        
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
            # ViT returns (last_hidden_state, pooler_output)
            # pooler_output is already the [CLS] token representation
            embeddings = outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs.last_hidden_state[:, 0, :]
        
        return F.normalize(embeddings, dim=1)
    
    @property
    def embedding_dim(self):
        """Return embedding dimension (hidden size of the model)."""
        return self.model.config.hidden_size
