"""
Text-to-Glyph conversion utility.

Converts text strings to visual glyph representations (PIL Images).
Used for image-based encoding models.
"""

import unicodedata
from PIL import Image, ImageDraw, ImageFont
from matplotlib import font_manager


def _get_unicode_font(font_size=14):
    """
    Load a Unicode-compatible font for rendering text glyphs.
    
    Falls back to default font if no suitable font is found.
    
    Args:
        font_size: Font size in pixels (default: 14)
        
    Returns:
        PIL.ImageFont object
    """
    try:
        path = font_manager.findfont("DejaVu Sans", fallback_to_default=True)
        return ImageFont.truetype(path, font_size)
    except Exception:
        return ImageFont.load_default()


def text_to_glyph(text, image_size=(224, 224), font_size=14):
    """
    Converts a given text string into a glyph (image representation).
    
    The text is centered on a black background with white text.
    This representation can be used as input for image encoder models.
    
    Args:
        text: Input text string to convert
        image_size: Tuple of (width, height) for the output image (default: 224x224)
        font_size: Font size in pixels (default: 14)
        
    Returns:
        PIL.Image: RGB image with white text on black background
        
    Example:
        >>> glyph = text_to_glyph("ACME Corp", image_size=(224, 224))
        >>> glyph.save("company_name.png")
    """
    # Normalize unicode to ensure consistent rendering
    text = unicodedata.normalize('NFC', text)
    
    # Create black background image
    image = Image.new("RGB", image_size, color=(0, 0, 0))
    draw = ImageDraw.Draw(image)
    font = _get_unicode_font(font_size=font_size)
    
    # Get text bounding box to center it
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Center the text
    x = (image_size[0] - text_width) // 2
    y = (image_size[1] - text_height) // 2
    
    # Draw white text on black background
    draw.text((x, y), text, font=font, fill=(255, 255, 255))
    
    return image


def text_to_glyphs_batch(texts, image_size=(224, 224), font_size=14):
    """
    Convert multiple texts to glyphs in batch.
    
    Args:
        texts: List of text strings
        image_size: Tuple of (width, height) for output images
        font_size: Font size in pixels
        
    Returns:
        List of PIL.Image objects
    """
    return [text_to_glyph(text, image_size=image_size, font_size=font_size) for text in texts]
