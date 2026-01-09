"""
Example usage of image encoders and text-to-glyph conversion.

This module demonstrates how to:
1. Convert text to glyphs (images)
2. Encode glyphs with image encoders
3. Extract text from glyphs using OCR
"""

import torch
from model_utils.utils.text_to_glyph import text_to_glyph, text_to_glyphs_batch
from model_utils.models.image_encoder_factory import ImageEncoderFactory
from model_utils.models.ocr_encoders import PytesseractOCREncoder


def example_basic_glyph_encoding():
    """Example: Convert text to glyph and encode with ViT."""
    print("=" * 60)
    print("Example 1: Basic Glyph Encoding with ViT")
    print("=" * 60)
    
    # Convert text to glyph
    text = "ACME Corporation"
    glyph = text_to_glyph(text, image_size=(224, 224))
    print(f"Created glyph for: '{text}'")
    print(f"Glyph shape: {glyph.size}")
    
    # Load ViT encoder
    encoder = ImageEncoderFactory.create_model('vit', device='cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loaded ViT encoder with embedding_dim: {encoder.embedding_dim}")
    
    # Encode glyph
    embedding = encoder.encode_image(glyph)
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding (first 10 dims): {embedding[0, :10]}")
    print()


def example_batch_encoding():
    """Example: Batch encode multiple glyphs with different encoders."""
    print("=" * 60)
    print("Example 2: Batch Encoding with Multiple Models")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create multiple text samples
    texts = [
        "ACME Corporation",
        "TechCorp Inc",
        "Global Finance Ltd",
        "Data Systems Company"
    ]
    
    # Convert all to glyphs
    glyphs = text_to_glyphs_batch(texts, image_size=(224, 224))
    print(f"Created {len(glyphs)} glyphs")
    
    # Test different encoders
    encoders_to_test = ['vit', 'resnet', 'convnext', 'siglip']
    
    for encoder_type in encoders_to_test:
        try:
            encoder = ImageEncoderFactory.create_model(encoder_type, device=device)
            embeddings = encoder.encode_images(glyphs)
            print(f"{encoder_type.upper():12} | Embedding dim: {encoder.embedding_dim:4} | Shape: {embeddings.shape}")
        except Exception as e:
            print(f"{encoder_type.upper():12} | ERROR: {str(e)[:50]}")
    
    print()


def example_ocr_extraction():
    """Example: Extract text from glyphs using OCR."""
    print("=" * 60)
    print("Example 3: OCR Text Extraction")
    print("=" * 60)
    
    # Create glyphs
    texts = ["ACME Corp", "TechCorp Inc"]
    glyphs = text_to_glyphs_batch(texts, image_size=(224, 224))
    
    # Create OCR encoder with optional fuzzy matching
    reference_texts = ["ACME Corp", "TechCorp Inc", "Global Finance"]
    ocr = PytesseractOCREncoder(fuzzy_threshold=75, reference_texts=reference_texts)
    
    print(f"Created OCR encoder with {len(reference_texts)} reference texts")
    print()
    
    # Extract text from each glyph
    for i, glyph in enumerate(glyphs):
        result = ocr.extract_and_match(glyph)
        print(f"Glyph {i+1}:")
        print(f"  Original:     {texts[i]}")
        print(f"  Extracted:    {result['extracted_text']}")
        print(f"  Best Match:   {result['best_match']}")
        print(f"  Match Score:  {result['match_score']:.1f}%")
        print(f"  Is Match:     {result['is_match']}")
        print()


def example_vitmae_comparison():
    """Example: Compare self-supervised models (ViT-MAE vs ImageGPT)."""
    print("=" * 60)
    print("Example 4: Self-Supervised Models")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    text = "Self-Supervised Learning"
    glyph = text_to_glyph(text, image_size=(224, 224))
    
    models = ['vitmae', 'imagegpt']
    
    for model_type in models:
        try:
            encoder = ImageEncoderFactory.create_model(model_type, device=device)
            embedding = encoder.encode_image(glyph)
            print(f"{model_type.upper():12} | Dim: {encoder.embedding_dim:4} | L2-norm: {embedding.norm(dim=1).item():.4f}")
        except Exception as e:
            print(f"{model_type.upper():12} | ERROR: {str(e)[:40]}")
    
    print()


def example_similarity_matching():
    """Example: Use embeddings for similarity matching."""
    print("=" * 60)
    print("Example 5: Similarity Matching")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create glyph pairs
    text_pairs = [
        ("ACME Corporation", "ACME Corp"),
        ("TechCorp Inc", "Tech Corp Inc"),
        ("Global Finance", "Global Banking"),
    ]
    
    # Flatten for encoding
    all_texts = [text for pair in text_pairs for text in pair]
    all_glyphs = text_to_glyphs_batch(all_texts, image_size=(224, 224))
    
    # Use SigLIP for comparison
    encoder = ImageEncoderFactory.create_model('siglip', device=device)
    embeddings = encoder.encode_images(all_glyphs)
    
    print(f"Using SigLIP encoder (dim: {encoder.embedding_dim})\n")
    
    # Compute cosine similarity for pairs
    cos_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    
    for i, (text1, text2) in enumerate(text_pairs):
        emb1 = embeddings[2*i].unsqueeze(0)
        emb2 = embeddings[2*i + 1].unsqueeze(0)
        similarity = cos_sim(emb1, emb2).item()
        
        print(f"Pair {i+1}:")
        print(f"  '{text1}' vs '{text2}'")
        print(f"  Cosine Similarity: {similarity:.4f}")
        print()


if __name__ == "__main__":
    # Run examples (comment out as needed to avoid long load times)
    
    # example_basic_glyph_encoding()
    # example_batch_encoding()
    # example_ocr_extraction()
    # example_vitmae_comparison()
    # example_similarity_matching()
    
    print("Examples available:")
    print("  - example_basic_glyph_encoding()")
    print("  - example_batch_encoding()")
    print("  - example_ocr_extraction()")
    print("  - example_vitmae_comparison()")
    print("  - example_similarity_matching()")
