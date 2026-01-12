"""
Pytesseract OCR encoder with optional fuzzy matching.
"""

import pytesseract
from fuzzywuzzy import fuzz
from .base_ocr_encoder import BaseOCREncoder


class PytesseractOCREncoder(BaseOCREncoder):
    """
    OCR encoder using pytesseract (Tesseract OCR engine).
    
    Supports optional fuzzy matching against a reference set of texts.
    Useful for validating OCR output against known business names.
    """
    
    def __init__(self, fuzzy_threshold=80, reference_texts=None):
        """
        Initialize pytesseract OCR encoder.
        
        Args:
            fuzzy_threshold: Fuzzy matching score threshold (0-100)
                           (default: 80, means 80% similarity required)
            reference_texts: Optional list of reference texts for fuzzy matching
                           If provided, OCR output will be matched against these.
        """
        super().__init__()
        self.fuzzy_threshold = fuzzy_threshold
        self.reference_texts = reference_texts or []
    
    def extract_text(self, image):
        """
        Extract text from a single PIL Image using OCR.
        
        Args:
            image: PIL.Image object
            
        Returns:
            str: Extracted text
        """
        return pytesseract.image_to_string(image)
    
    def extract_texts(self, images):
        """
        Extract text from multiple PIL Images using OCR.
        
        Args:
            images: List of PIL.Image objects
            
        Returns:
            List of strings
        """
        return [self.extract_text(img) for img in images]
    
    def fuzzy_match(self, extracted_text):
        """
        Find the best fuzzy match for extracted text in reference texts.
        
        Uses token_set_ratio for more robust matching.
        
        Args:
            extracted_text: Text extracted via OCR
            
        Returns:
            dict with keys:
                - 'best_match': Best matching reference text (or None)
                - 'score': Matching score (0-100)
                - 'is_match': Boolean indicating if score >= threshold
        """
        if not self.reference_texts:
            return {
                'best_match': None,
                'score': 0,
                'is_match': False
            }
        
        best_score = 0
        best_match = None
        
        for ref_text in self.reference_texts:
            score = fuzz.token_set_ratio(extracted_text.lower(), ref_text.lower())
            if score > best_score:
                best_score = score
                best_match = ref_text
        
        return {
            'best_match': best_match,
            'score': best_score,
            'is_match': best_score >= self.fuzzy_threshold
        }
    
    def extract_and_match(self, image):
        """
        Extract text from image and perform fuzzy matching.
        
        Args:
            image: PIL.Image object
            
        Returns:
            dict with keys:
                - 'extracted_text': Text extracted via OCR
                - 'best_match': Best matching reference text (or None)
                - 'match_score': Fuzzy matching score (0-100)
                - 'is_match': Boolean indicating if matched above threshold
        """
        extracted_text = self.extract_text(image)
        match_result = self.fuzzy_match(extracted_text)
        
        return {
            'extracted_text': extracted_text,
            'best_match': match_result['best_match'],
            'match_score': match_result['score'],
            'is_match': match_result['is_match']
        }
    
    def extract_batch_and_match(self, images):
        """
        Extract text from multiple images and perform fuzzy matching.
        
        Args:
            images: List of PIL.Image objects
            
        Returns:
            List of dicts with results (see extract_and_match)
        """
        return [self.extract_and_match(img) for img in images]
