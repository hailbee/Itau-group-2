"""
SigLIP Image encoder wrapper.
"""

import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel
from .base_image_encoder import BaseImageEncoder


class SigLIPImageEncoder(BaseImageEncoder):
    """
    Wrapper for Google SigLIP vision-language model - image encoder component.
    
    Extracts the image encoder from the SigLIP VLM to use independently
    for image representation learning.
    """
    
    def __init__(self, model_name="google/siglip-base-patch16-224", device=None):
        """
        Initialize SigLIP image encoder.
        
        Args:
            model_name: Model identifier on Hugging Face Hub
                       (default: google/siglip-base-patch16-224)
            device: Device to run on (cuda/cpu). Auto-detected if None.
        """
        self.model_name = model_name
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.image_encoder = None
        self.processor = None
        self._load_model()
    
    def _load_model(self):
        """Load SigLIP model and extract image encoder."""
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        
        # Load full SigLIP model
        full_model = AutoModel.from_pretrained(self.model_name).to(self.device)
        
        # Extract the vision encoder component
        if hasattr(full_model, 'vision_model'):
            self.image_encoder = full_model.vision_model
        else:
            # Fallback: use full model
            self.image_encoder = full_model
        
        self.model = self.image_encoder
        self.image_encoder.eval()
    
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
        
        Uses the pooled representation from the vision encoder.
        
        Args:
            images: List of PIL.Image objects
            
        Returns:
            torch.FloatTensor of shape (batch_size, embedding_dim)
        """
        # Preprocess images
        pixel_values = self.processor(images=images, return_tensors="pt").pixel_values.to(self.device)
        
        # Forward pass through image encoder
        with torch.no_grad():
            outputs = self.image_encoder(pixel_values=pixel_values)
            
            # SigLIP vision encoder returns (last_hidden_state, pooler_output)
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                embeddings = outputs.pooler_output
            else:
                # Fallback: use last hidden state with mean pooling
                embeddings = outputs.last_hidden_state
                if len(embeddings.shape) == 3:
                    embeddings = embeddings.mean(dim=1)
        
        return F.normalize(embeddings, dim=1)
    
    @property
    def embedding_dim(self):
        """Return embedding dimension."""
        return self.image_encoder.config.hidden_size
