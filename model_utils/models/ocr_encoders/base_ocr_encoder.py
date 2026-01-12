"""
Base class for OCR encoder wrappers.
"""

from abc import ABC, abstractmethod


class BaseOCREncoder(ABC):
    """
    Abstract base class for OCR encoders.
    
    OCR encoders extract text from images and optionally perform fuzzy matching.
    Different from image encoders - these return text rather than embeddings.
    """
    
    def __init__(self):
        """Initialize the OCR encoder."""
        pass
    
    @abstractmethod
    def extract_text(self, image):
        """
        Extract text from a single PIL Image.
        
        Args:
            image: PIL.Image object
            
        Returns:
            str: Extracted text
        """
        pass
    
    @abstractmethod
    def extract_texts(self, images):
        """
        Extract text from multiple PIL Images.
        
        Args:
            images: List of PIL.Image objects
            
        Returns:
            List of strings
        """
        pass
