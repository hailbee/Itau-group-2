"""
Base class for image encoder wrappers.

Provides consistent interface for all image encoding models.
"""

import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod


class BaseImageEncoder(ABC):
    """
    Abstract base class for image encoder wrappers.
    
    All image encoders should inherit from this class and implement:
    - encode_image: Method to encode a single PIL Image to embedding
    - encode_images: Method to encode batch of PIL Images to embeddings
    - embedding_dim: Property returning output embedding dimension
    - to: Method to move model to different device
    """
    
    def __init__(self, model_name=None, device=None):
        """
        Initialize the image encoder.
        
        Args:
            model_name: Name/path of the model to load
            device: Device to run on (cuda/cpu). Auto-detected if None.
        """
        self.model_name = model_name
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
        self._load_model()
    
    @abstractmethod
    def _load_model(self):
        """Load the model and processor. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def encode_image(self, image):
        """
        Encode a single PIL Image to an embedding.
        
        Args:
            image: PIL.Image object
            
        Returns:
            torch.FloatTensor of shape (embedding_dim,) or (1, embedding_dim)
        """
        pass
    
    @abstractmethod
    def encode_images(self, images):
        """
        Encode a batch of PIL Images to embeddings.
        
        Args:
            images: List of PIL.Image objects
            
        Returns:
            torch.FloatTensor of shape (batch_size, embedding_dim)
        """
        pass
    
    @property
    @abstractmethod
    def embedding_dim(self):
        """Return the output embedding dimension."""
        pass
    
    def to(self, device):
        """
        Move model to specified device.
        
        Args:
            device: torch.device or string ('cuda'/'cpu')
            
        Returns:
            self
        """
        self.device = device
        if self.model:
            self.model = self.model.to(device)
        return self
    
    def eval(self):
        """Set model to evaluation mode."""
        if self.model:
            self.model.eval()
        return self
    
    def train(self):
        """Set model to training mode."""
        if self.model:
            self.model.train()
        return self
