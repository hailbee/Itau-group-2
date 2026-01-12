"""
OCR encoder wrappers for text extraction from images.
"""

from .base_ocr_encoder import BaseOCREncoder
from .pytesseract_encoder import PytesseractOCREncoder


__all__ = [
    'BaseOCREncoder',
    'PytesseractOCREncoder',
]
