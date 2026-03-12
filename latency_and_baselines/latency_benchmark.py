"""
Main latency and resource utilization benchmark script.

Benchmarks various models for inference latency, throughput, and resource usage.

Usage:
    python latency_benchmark.py --model metrics --batch_size 32 --num_samples 1000 --warmup_samples 10

Models:
    - metrics: Text metrics only (fast baseline)
    - text: Text encoder + metrics
    - image: Image encoder + metrics
    - siglip_text, clip_text, coca_text, flava_text: Individual text encoders
    - siglip_image, clip_image, coca_image, flava_image: Individual image encoders
    - vit, resnet, vitmae: Individual vision encoders
    - pytesseract: OCR baseline
"""

import argparse
import time
import json
import traceback
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch

from latency_benchmark_utils import (
    load_test_data,
    create_mini_dataset,
    generate_glyph_image,
    ResourceMonitor,
    TimingContext,
    create_batches,
)


# ============================================================================
# Text Metrics (Fast Baseline)
# ============================================================================

class MetricsModel:
    """Compute text string metrics (Levenshtein, token_set_ratio, partial_ratio)."""
    
    def __init__(self):
        try:
            from rapidfuzz import fuzz as rf_fuzz
            from rapidfuzz.distance import Levenshtein as rf_lev
            self.rf_fuzz = rf_fuzz
            self.rf_lev = rf_lev
            self.use_rapid = True
        except ImportError:
            self.use_rapid = False
    
    def process_pair(self, fraud_name: str, real_name: str) -> np.ndarray:
        """Compute metrics for a single pair."""
        if self.use_rapid:
            lev = self.rf_lev.distance(fraud_name, real_name)
            tsr = self.rf_fuzz.token_set_ratio(fraud_name, real_name) / 100.0
            pr = self.rf_fuzz.partial_ratio(fraud_name, real_name) / 100.0
        else:
            lev = self._lev_distance(fraud_name, real_name)
            tsr = 0.0
            pr = 0.0
        
        return np.array([float(-lev), float(tsr), float(pr)], dtype=np.float32)
    
    @staticmethod
    def _lev_distance(a: str, b: str) -> int:
        """Pure Python Levenshtein distance."""
        if a == b:
            return 0
        la, lb = len(a), len(b)
        if la == 0:
            return lb
        if lb == 0:
            return la
        
        prev = list(range(lb + 1))
        for i, ca in enumerate(a, start=1):
            cur = [i]
            for j, cb in enumerate(b, start=1):
                ins = cur[j - 1] + 1
                dele = prev[j] + 1
                sub = prev[j - 1] + (0 if ca == cb else 1)
                cur.append(min(ins, dele, sub))
            prev = cur
        return int(prev[-1])
    
    def process_batch(self, fraud_names: List[str], real_names: List[str]) -> np.ndarray:
        """Process a batch of name pairs."""
        results = []
        for fraud, real in zip(fraud_names, real_names):
            results.append(self.process_pair(fraud, real))
        return np.vstack(results)


# ============================================================================
# String Metrics for Ensemble Models
# ============================================================================

class StringMetrics:
    """Compute string similarity metrics for ensemble models."""
    
    def __init__(self):
        try:
            from rapidfuzz import fuzz as rf_fuzz
            from rapidfuzz.distance import Levenshtein as rf_lev
            self.rf_fuzz = rf_fuzz
            self.rf_lev = rf_lev
            self.use_rapid = True
        except ImportError:
            self.use_rapid = False
    
    def compute_metrics(self, fraud: str, real: str) -> Tuple[float, float, float]:
        """Compute (lev_score, tsr, pr) for a pair."""
        if self.use_rapid:
            lev = self.rf_lev.distance(fraud, real)
            tsr = self.rf_fuzz.token_set_ratio(fraud, real) / 100.0
            pr = self.rf_fuzz.partial_ratio(fraud, real) / 100.0
        else:
            lev = self._lev_distance(fraud, real)
            tsr = 0.0
            pr = 0.0
        
        return float(-lev), float(tsr), float(pr)
    
    @staticmethod
    def _lev_distance(a: str, b: str) -> int:
        """Pure Python Levenshtein distance."""
        if a == b:
            return 0
        la, lb = len(a), len(b)
        if la == 0:
            return lb
        if lb == 0:
            return la
        prev = list(range(lb + 1))
        for i, ca in enumerate(a, start=1):
            cur = [i]
            for j, cb in enumerate(b, start=1):
                ins = cur[j - 1] + 1
                dele = prev[j] + 1
                sub = prev[j - 1] + (0 if ca == cb else 1)
                cur.append(min(ins, dele, sub))
            prev = cur
        return prev[-1]


# ============================================================================
# Ensemble Feature Configuration
# ============================================================================

ENSEMBLE_FEATURE_CONFIGS = {
    "metrics": {
        "description": "Text metrics only (3 features)",
        "features": ["token_set_ratio", "levenshtein_distance_score", "partial_ratio"],
        "model_file": "metrics_model.joblib"
    },
    "small": {
        "description": "Downloads + Deja cosines (2 features)",
        "features": ["cosine_downloads", "cosine_deja"],
        "model_file": "small_model.joblib"
    },
    "medium": {
        "description": "4 image cosines (4 features)",
        "features": ["cosine_downloads", "cosine_deja", "cosine_unifont", "cosine_gentium"],
        "model_file": "medium_model.joblib"
    },
    "large": {
        "description": "7 image cosines (7 features)",
        "features": ["cosine_downloads", "cosine_deja", "cosine_unifont", "cosine_gentium", 
                    "cosine_libre", "cosine_exo2", "cosine_doulos"],
        "model_file": "large_model.joblib"
    },
    "total_1f": {
        "description": "Text + 1 image + metrics (5 features)",
        "features": ["text_cosine", "token_set_ratio", "levenshtein_distance_score", 
                    "partial_ratio", "cosine_deja"],
        "model_file": "total_1f_model.joblib"
    },
    "total_3f": {
        "description": "Text + 3 images + metrics (7 features)",
        "features": ["text_cosine", "token_set_ratio", "levenshtein_distance_score", 
                    "partial_ratio", "cosine_deja", "cosine_unifont", "cosine_gentium"],
        "model_file": "total_3f_model.joblib"
    },
    "total_5f": {
        "description": "Text + 6 images + metrics (10 features)",
        "features": ["text_cosine", "token_set_ratio", "levenshtein_distance_score", 
                    "partial_ratio", "cosine_deja", "cosine_unifont", "cosine_libre",
                    "cosine_exo2", "cosine_doulos", "cosine_cousine"],
        "model_file": "total_5f_model.joblib"
    },
    "total_5f_img": {
        "description": "5 images + metrics (8 features, no text)",
        "features": ["token_set_ratio", "levenshtein_distance_score", "partial_ratio",
                    "cosine_deja", "cosine_unifont", "cosine_libre", "cosine_exo2", "cosine_doulos"],
        "model_file": "total_5f_img_model.joblib"
    },
}


# ============================================================================
# Ensemble Feature Computer
# ============================================================================

class EnsembleFeatureComputer:
    """Computes features for ensemble models by running underlying models."""
    
    def __init__(self, ensemble_name: str, device: str = None):
        if ensemble_name not in ENSEMBLE_FEATURE_CONFIGS:
            raise ValueError(f"Unknown ensemble: {ensemble_name}")
        
        self.ensemble_name = ensemble_name
        self.config = ENSEMBLE_FEATURE_CONFIGS[ensemble_name]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.metrics = StringMetrics()
        
        # Initialize encoders based on what features this ensemble needs
        self.text_encoder = None
        self.image_encoder = None
        self._init_encoders()
    
    def _init_encoders(self):
        """Initialize text and image encoders based on required features."""
        features = self.config["features"]
        
        # Check if we need text embedding
        if "text_cosine" in features or "text_embedding" in features:
            # Use SigLIP as default text encoder
            self.text_encoder = SigLIPTextEncoder("google/siglip-base-patch16-224")
            if self.text_encoder.device != self.device:
                self.text_encoder.to_device(self.device)
        
        # Check if we need image cosines
        image_cosine_features = [f for f in features if f.startswith("cosine_") and f != "text_cosine"]
        if image_cosine_features:
            # Use SigLIP as default image encoder
            self.image_encoder = SigLIPImageEncoder("google/siglip-base-patch16-224")
            if self.image_encoder.device != self.device:
                self.image_encoder.to_device(self.device)
    
    def compute_features_batch(self, fraud_names: List[str], real_names: List[str]) -> np.ndarray:
        """
        Compute features for ensemble model by running underlying models.
        
        This includes:
        1. Text metrics (fast)
        2. Text encoder embeddings (if needed)
        3. Image rendering + encoding (if needed)
        4. Cosine similarities (if needed)
        """
        features = []
        config_features = self.config["features"]
        
        # Pre-compute text embeddings if needed
        text_embeddings_fraud = None
        text_embeddings_real = None
        if "text_cosine" in config_features and self.text_encoder:
            text_embeddings_fraud = self.text_encoder.encode_batch(fraud_names)
            text_embeddings_real = self.text_encoder.encode_batch(real_names)
        
        # Pre-compute image embeddings if needed
        image_embeddings_fraud = None
        image_embeddings_real = None
        image_cosine_features = [f for f in config_features if f.startswith("cosine_") and f != "text_cosine"]
        if image_cosine_features and self.image_encoder:
            fraud_images = [generate_glyph_image(name) for name in fraud_names]
            real_images = [generate_glyph_image(name) for name in real_names]
            image_embeddings_fraud = self.image_encoder.encode_batch(fraud_images)
            image_embeddings_real = self.image_encoder.encode_batch(real_images)
        
        # Compute text metrics (for all ensembles)
        if any(f in config_features for f in ["token_set_ratio", "levenshtein_distance_score", "partial_ratio"]):
            pass
        
        for i, (fraud, real) in enumerate(zip(fraud_names, real_names)):
            feature_row = []
            
            # Text metrics (always computed)
            if any(f in config_features for f in ["token_set_ratio", "levenshtein_distance_score", "partial_ratio"]):
                lev_score, tsr, pr = self.metrics.compute_metrics(fraud, real)
            
            # Add feature columns based on ensemble config, in the correct order
            for feature_name in config_features:
                if feature_name == "token_set_ratio":
                    feature_row.append(float(tsr))
                elif feature_name == "levenshtein_distance_score":
                    feature_row.append(float(lev_score))
                elif feature_name == "partial_ratio":
                    feature_row.append(float(pr))
                elif feature_name == "text_cosine" or feature_name == "cosine_sim" or feature_name == "text_embedding":
                    # Compute cosine similarity between text embeddings
                    if text_embeddings_fraud is not None:
                        cosine = float(np.dot(text_embeddings_fraud[i], text_embeddings_real[i]) / 
                                     (np.linalg.norm(text_embeddings_fraud[i]) * np.linalg.norm(text_embeddings_real[i]) + 1e-8))
                        feature_row.append(cosine)
                    else:
                        feature_row.append(0.5)  # Fallback
                elif feature_name in ["cosine_downloads", "cosine_deja", "cosine_unifont", 
                                     "cosine_libre", "cosine_exo2", "cosine_doulos", 
                                     "cosine_cousine", "cosine_gentium"]:
                    # Compute cosine similarity between image embeddings
                    if image_embeddings_fraud is not None:
                        cosine = float(np.dot(image_embeddings_fraud[i], image_embeddings_real[i]) / 
                                     (np.linalg.norm(image_embeddings_fraud[i]) * np.linalg.norm(image_embeddings_real[i]) + 1e-8))
                        feature_row.append(cosine)
                    else:
                        feature_row.append(0.5)  # Fallback
                else:
                    feature_row.append(0.0)
            
            features.append(feature_row)
        
        return np.array(features, dtype=np.float32)


# ============================================================================
# Ensemble Model Wrapper
# ============================================================================

class EnsembleModel:
    """Generic ensemble model loaded from joblib."""
    
    def __init__(self, model_path: str):
        try:
            import joblib
            self.model_data = joblib.load(model_path)
            self.clf = self.model_data.get('model')
            self.feature_names = self.model_data.get('feature_names', [])
            if self.clf is None:
                raise ValueError(f"No 'model' key found in {model_path}")
        except ImportError:
            raise ImportError("joblib not installed. Install with: pip install joblib")
    
    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if hasattr(self.clf, 'predict_proba'):
            return self.clf.predict_proba(X)[:, 1].astype(np.float32)
        elif hasattr(self.clf, 'decision_function'):
            scores = self.clf.decision_function(X)
            return (1.0 / (1.0 + np.exp(-scores))).astype(np.float32)
        else:
            raise RuntimeError("Model does not support probability predictions")


# ============================================================================
# Vision-Language Model Wrappers
# ============================================================================

class TextEncoderBase:
    """Base class for text encoders."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        raise NotImplementedError
    
    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """Encode a batch of texts to embeddings."""
        raise NotImplementedError
    
    def to_device(self, device):
        """Move model to device."""
        self.device = device
        if self.model:
            self.model.to(device)
        return self


class CLIPTextEncoder(TextEncoderBase):
    """CLIP text encoder."""
    
    def _load_model(self):
        from transformers import CLIPModel, CLIPTokenizer
        self.model = CLIPModel.from_pretrained(self.model_name, trust_remote_code=True).to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model.eval()
    
    def encode_batch(self, texts: List[str]) -> np.ndarray:
        with torch.inference_mode():
            inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model.get_text_features(**inputs)
            return outputs.cpu().numpy().astype(np.float32)


class SigLIPTextEncoder(TextEncoderBase):
    """SigLIP text encoder."""
    
    def _load_model(self):
        from transformers import AutoTokenizer, AutoModel
        self.model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model.eval()
    
    def encode_batch(self, texts: List[str]) -> np.ndarray:
        with torch.inference_mode():
            inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model.get_text_features(**inputs)
            if hasattr(outputs, 'pooler_output'):
                embeddings = outputs.pooler_output
            elif hasattr(outputs, 'last_hidden_state'):
                embeddings = outputs.last_hidden_state.mean(dim=1)
            else:
                embeddings = outputs[0].mean(dim=1) if isinstance(outputs, tuple) else outputs.mean(dim=1)
            return embeddings.cpu().numpy().astype(np.float32)


class CoCaTextEncoder(TextEncoderBase):
    """CoCa text encoder."""
    
    def _load_model(self):
        from transformers import AutoTokenizer, AutoModel
        self.model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model.eval()
    
    def encode_batch(self, texts: List[str]) -> np.ndarray:
        with torch.inference_mode():
            inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            if hasattr(outputs, 'last_hidden_state'):
                embeddings = outputs.last_hidden_state.mean(dim=1)
            else:
                embeddings = outputs[0].mean(dim=1) if isinstance(outputs, tuple) else outputs.mean(dim=1)
            return embeddings.cpu().numpy().astype(np.float32)


class FLAVATextEncoder(TextEncoderBase):
    """FLAVA text encoder."""
    
    def _load_model(self):
        from transformers import AutoTokenizer, AutoModel
        self.model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model.eval()
    
    def encode_batch(self, texts: List[str]) -> np.ndarray:
        with torch.inference_mode():
            inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            # FLAVA returns special output object, handle gracefully
            if hasattr(outputs, 'text_embeddings'):
                embeddings = outputs.text_embeddings
            elif hasattr(outputs, 'last_hidden_state'):
                embeddings = outputs.last_hidden_state.mean(dim=1)
            else:
                embeddings = outputs[0].mean(dim=1) if isinstance(outputs, tuple) else outputs.mean(dim=1)
            return embeddings.cpu().numpy().astype(np.float32)


# ============================================================================
# Image Encoder Wrappers
# ============================================================================

class ImageEncoderBase:
    """Base class for image encoders."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
        self._load_model()
    
    def _load_model(self):
        raise NotImplementedError
    
    def encode_batch(self, images: List) -> np.ndarray:
        """Encode a batch of PIL Images to embeddings."""
        raise NotImplementedError
    
    def to_device(self, device):
        """Move model to device."""
        self.device = device
        if self.model:
            self.model.to(device)
        return self


class ViTImageEncoder(ImageEncoderBase):
    """Vision Transformer image encoder."""
    
    def _load_model(self):
        from transformers import AutoImageProcessor, AutoModel
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()
    
    def encode_batch(self, images: List) -> np.ndarray:
        with torch.inference_mode():
            inputs = self.processor(images, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :]
            return embeddings.cpu().numpy().astype(np.float32)


class CLIPImageEncoder(ImageEncoderBase):
    """CLIP image encoder."""
    
    def _load_model(self):
        from transformers import CLIPProcessor, CLIPModel
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self.model = CLIPModel.from_pretrained(self.model_name, trust_remote_code=True).to(self.device)
        self.model.eval()
    
    def encode_batch(self, images: List) -> np.ndarray:
        with torch.inference_mode():
            inputs = self.processor(images=images, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model.get_image_features(**inputs)
            return outputs.cpu().numpy().astype(np.float32)


class SigLIPImageEncoder(ImageEncoderBase):
    """SigLIP image encoder."""
    
    def _load_model(self):
        from transformers import AutoImageProcessor, AutoModel
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True).to(self.device)
        self.model.eval()
    
    def encode_batch(self, images: List) -> np.ndarray:
        with torch.inference_mode():
            inputs = self.processor(images=images, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model.get_image_features(**inputs)
            return outputs.cpu().numpy().astype(np.float32)


class CoCaImageEncoder(ImageEncoderBase):
    """CoCa image encoder - requires both image and text input."""
    
    def _load_model(self):
        from transformers import AutoImageProcessor, AutoModel, AutoTokenizer
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True).to(self.device)
        self.model.eval()
    
    def encode_batch(self, images: List) -> np.ndarray:
        """Encode images using CoCa model.
        
        CoCa requires both image and text input. We provide a dummy text prompt.
        """
        with torch.inference_mode():
            # Process images
            image_inputs = self.processor(images=images, return_tensors="pt")
            image_inputs = {k: v.to(self.device) for k, v in image_inputs.items()}
            
            # Create dummy text input - CoCa needs text tokens even for image encoding
            # Use a simple prompt like "a picture of"
            batch_size = len(images)
            dummy_text = ["a picture of"] * batch_size
            text_inputs = self.tokenizer(dummy_text, return_tensors="pt", padding=True, truncation=True)
            text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
            
            # Combine inputs: both pixel_values and input_ids are required
            combined_inputs = {**image_inputs, **text_inputs}
            
            # Forward pass
            outputs = self.model(**combined_inputs)
            
            # Extract image features from output
            if hasattr(outputs, 'image_features'):
                embeddings = outputs.image_features
            elif hasattr(outputs, 'last_hidden_state'):
                embeddings = outputs.last_hidden_state
                if embeddings.dim() == 3:
                    embeddings = embeddings.mean(dim=1)  # Pool over sequence
            elif hasattr(outputs, 'pooler_output'):
                embeddings = outputs.pooler_output
            elif isinstance(outputs, torch.Tensor):
                embeddings = outputs
                if embeddings.dim() == 3:
                    embeddings = embeddings.mean(dim=1)
            elif isinstance(outputs, tuple) and len(outputs) > 0:
                embeddings = outputs[0]
                if isinstance(embeddings, torch.Tensor) and embeddings.dim() == 3:
                    embeddings = embeddings.mean(dim=1)
            else:
                raise ValueError(f"Cannot extract embeddings from output type: {type(outputs)}")
            
            return embeddings.cpu().numpy().astype(np.float32)


class FLAVAImageEncoder(ImageEncoderBase):
    """FLAVA image encoder."""
    
    def _load_model(self):
        from transformers import AutoImageProcessor, AutoModel
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True).to(self.device)
        self.model.eval()
    
    def encode_batch(self, images: List) -> np.ndarray:
        with torch.inference_mode():
            inputs = self.processor(images=images, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model.get_image_features(**inputs)
            if hasattr(outputs, 'image_embeddings'):
                embeddings = outputs.image_embeddings
            elif isinstance(outputs, tuple):
                embeddings = outputs[0] if outputs[0] is not None else outputs
            else:
                embeddings = outputs
            return embeddings.cpu().numpy().astype(np.float32)


class ResNetImageEncoder(ImageEncoderBase):
    """ResNet image encoder."""
    
    def _load_model(self):
        from transformers import AutoImageProcessor, AutoModel
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()
    
    def encode_batch(self, images: List) -> np.ndarray:
        with torch.inference_mode():
            inputs = self.processor(images, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            embeddings = outputs.pooler_output
            return embeddings.cpu().numpy().astype(np.float32)


class ViTMAEImageEncoder(ImageEncoderBase):
    """ViT-MAE image encoder."""
    
    def _load_model(self):
        from transformers import AutoImageProcessor, AutoModel
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()
    
    def encode_batch(self, images: List) -> np.ndarray:
        with torch.inference_mode():
            inputs = self.processor(images, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
            return embeddings.cpu().numpy().astype(np.float32)


# ============================================================================
# OCR Encoder
# ============================================================================

class PytesseractOCREncoder:
    """Pytesseract OCR encoder."""
    
    def __init__(self):
        try:
            import pytesseract
            self.pytesseract = pytesseract
            self.available = True
        except ImportError:
            self.pytesseract = None
            self.available = False
    
    def encode_batch(self, images: List) -> List[str]:
        """Extract text from images using OCR."""
        if not self.available:
            # Return dummy results if pytesseract not available
            return [""] * len(images)
        
        results = []
        for img in images:
            try:
                text = self.pytesseract.image_to_string(img)
                results.append(text)
            except Exception:
                results.append("")
        return results


# ============================================================================
# Benchmark Runner
# ============================================================================

class BenchmarkRunner:
    """Run benchmarks for a single model."""
    
    def __init__(
        self,
        model_name: str,
        model_instance,
        batch_size: int = 32,
        warmup_samples: int = 10,
        num_samples: int = 100,
        device: str = None,
        is_image_model: bool = False,
        feature_computer = None,  # For ensemble models
    ):
        self.model_name = model_name
        self.model = model_instance
        self.feature_computer = feature_computer
        self.batch_size = batch_size
        self.warmup_samples = warmup_samples
        self.num_samples = num_samples
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.is_image_model = is_image_model
    
    def run(self, df: pd.DataFrame) -> Dict:
        """
        Run benchmark on test data.
        
        Args:
            df: DataFrame with fraudulent_name, real_name, label columns
            
        Returns:
            Dict with benchmark results
        """
        fraud_names = df['fraudulent_name'].tolist()
        real_names = df['real_name'].tolist()
        
        # Create data for encoding
        if self.is_image_model:
            # Render text to images
            fraud_images = [generate_glyph_image(name) for name in fraud_names]
            real_images = [generate_glyph_image(name) for name in real_names]
            warmup_data = (fraud_images[:self.warmup_samples], real_images[:self.warmup_samples])
            bench_data = (fraud_images[self.warmup_samples:], real_images[self.warmup_samples:])
        else:
            warmup_data = (fraud_names[:self.warmup_samples], real_names[:self.warmup_samples])
            bench_data = (fraud_names[self.warmup_samples:], real_names[self.warmup_samples:])
        
        monitor = ResourceMonitor()
        
        # Warmup phase
        warmup_fraud, warmup_real = warmup_data
        for i in range(0, len(warmup_fraud), self.batch_size):
            batch_f = warmup_fraud[i:i+self.batch_size]
            batch_r = warmup_real[i:i+self.batch_size]
            try:
                if isinstance(self.model, MetricsModel):
                    self.model.process_batch(batch_f, batch_r)
                elif isinstance(self.model, EnsembleModel):
                    # Warmup with feature computation
                    if self.feature_computer:
                        X_warmup = self.feature_computer.compute_features_batch(batch_f, batch_r)
                        self.model.predict_batch(X_warmup)
                    else:
                        X_warmup = np.zeros((len(batch_f), len(self.model.feature_names)), dtype=np.float32)
                        self.model.predict_batch(X_warmup)
                else:
                    self.model.encode_batch(batch_f)
                    self.model.encode_batch(batch_r)
            except Exception:
                pass  # Some models may not support encoding
        
        # Benchmark phase
        bench_fraud, bench_real = bench_data
        num_pairs = min(len(bench_fraud), self.num_samples - self.warmup_samples)
        bench_fraud = bench_fraud[:num_pairs]
        bench_real = bench_real[:num_pairs]
        
        times = []
        monitor.start_monitoring()
        
        for i in range(0, len(bench_fraud), self.batch_size):
            batch_f = bench_fraud[i:i+self.batch_size]
            batch_r = bench_real[i:i+self.batch_size]
            
            start = time.perf_counter()
            try:
                if isinstance(self.model, MetricsModel):
                    self.model.process_batch(batch_f, batch_r)
                elif isinstance(self.model, EnsembleModel):
                    if self.feature_computer:
                        X_batch = self.feature_computer.compute_features_batch(batch_f, batch_r)
                    else:
                        X_batch = np.zeros((len(batch_f), len(self.model.feature_names)), dtype=np.float32)
                    self.model.predict_batch(X_batch)
                else:
                    self.model.encode_batch(batch_f)
                    self.model.encode_batch(batch_r)
            except Exception as e:
                print(f"  Warning: {e}")
                continue
            elapsed = time.perf_counter() - start
            
            times.append(elapsed)
            monitor.sample()
        
        if not times:
            return {
                'model': self.model_name,
                'batch_size': self.batch_size,
                'num_samples': 0,
                'total_pairs': 0,
                'latency_avg_ms': 0.0,
                'latency_min_ms': 0.0,
                'latency_max_ms': 0.0,
                'latency_std_ms': 0.0,
                'throughput_pairs_per_sec': 0.0,
                'cpu_avg_percent': 0.0,
                'gpu_memory_avg_gb': 0.0,
                'gpu_memory_max_gb': 0.0,
            }
        
        times_ms = [t * 1000 for t in times]
        resource_stats = monitor.get_stats()
        
        total_pairs = num_pairs * 2  # fraud + real
        total_time = sum(times)
        throughput = total_pairs / total_time if total_time > 0 else 0
        
        return {
            'model': self.model_name,
            'batch_size': self.batch_size,
            'num_samples': num_pairs,
            'total_pairs': total_pairs,
            'num_batches': len(times),
            'latency_avg_ms': float(np.mean(times_ms)),
            'latency_min_ms': float(np.min(times_ms)),
            'latency_max_ms': float(np.max(times_ms)),
            'latency_std_ms': float(np.std(times_ms)),
            'throughput_pairs_per_sec': float(throughput),
            'cpu_avg_percent': resource_stats.get('cpu_avg_percent', 0.0),
            'gpu_memory_avg_gb': resource_stats.get('gpu_memory_avg_gb', 0.0),
            'gpu_memory_max_gb': resource_stats.get('gpu_memory_max_gb', 0.0),
        }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark latency and resource utilization of all models"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Specific model to benchmark (if omitted, benchmarks all models)"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to test")
    parser.add_argument("--warmup_samples", type=int, default=10, help="Number of samples for warmup")
    parser.add_argument("--data_path", type=str, default="test_pairs_all.parquet", help="Path to test data")
    parser.add_argument("--output_csv", type=str, default="latency_results.csv", help="Output CSV file")
    parser.add_argument("--device", type=str, default=None, help="Device: cuda, cpu, or mps")
    parser.add_argument("--models_dir", type=str, default="saved_models", help="Directory with saved ensemble models")
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.data_path}...")
    if Path(args.data_path).exists():
        df = load_test_data(args.data_path, args.num_samples)
    else:
        print(f"Data file not found, using synthetic data")
        df = create_mini_dataset(args.num_samples)
    
    print(f"Loaded {len(df)} samples\n")
    
    # Define all available models
    all_baseline_models = [
        "metrics",
        "siglip_text", "clip_text", "coca_text", "flava_text",
        "siglip_image", "clip_image", "coca_image", "flava_image",
        "vit", "resnet", "vitmae",
        "pytesseract",
    ]
    
    all_ensemble_models = [
        "small", "medium", "large",
        "total_1f", "total_3f", "total_5f", "total_5f_img",
    ]
    
    all_models = all_baseline_models + all_ensemble_models
    
    # Determine which models to run
    if args.model:
        models_to_run = [args.model]
    else:
        models_to_run = all_models
    
    # Model configurations for baselines
    baseline_configs = {
        "metrics": (MetricsModel, False),
        "siglip_text": (lambda: SigLIPTextEncoder("google/siglip-base-patch16-224"), False),
        "clip_text": (lambda: CLIPTextEncoder("openai/clip-vit-base-patch32"), False),
        "coca_text": (lambda: CoCaTextEncoder("microsoft/git-base-coco"), False),
        "flava_text": (lambda: FLAVATextEncoder("facebook/flava-full"), False),
        "siglip_image": (lambda: SigLIPImageEncoder("google/siglip-base-patch16-224"), True),
        "clip_image": (lambda: CLIPImageEncoder("openai/clip-vit-base-patch32"), True),
        "coca_image": (lambda: CoCaImageEncoder("microsoft/git-base-coco"), True),
        "flava_image": (lambda: FLAVAImageEncoder("facebook/flava-full"), True),
        "vit": (lambda: ViTImageEncoder("google/vit-base-patch16-224"), True),
        "resnet": (lambda: ResNetImageEncoder("microsoft/resnet-50"), True),
        "vitmae": (lambda: ViTMAEImageEncoder("facebook/vit-mae-base"), True),
        "pytesseract": (lambda: PytesseractOCREncoder(), True),
    }
    
    all_results = []
    
    for model_name in models_to_run:
        print(f"{'='*70}")
        print(f"Benchmarking: {model_name.upper()}")
        print(f"{'='*70}")
        
        try:
            feature_computer = None
            is_image = False
            
            # Check if it's a baseline or ensemble model
            if model_name in baseline_configs:
                model_class, is_image = baseline_configs[model_name]
                model = model_class() if callable(model_class) else model_class()
                print(f"  Baseline model")
            elif model_name in all_ensemble_models:
                # Load ensemble model
                model_path = Path(args.models_dir) / ENSEMBLE_FEATURE_CONFIGS[model_name]["model_file"]
                if not model_path.exists():
                    print(f"  ERROR: Model not found: {model_path}")
                    continue
                
                model = EnsembleModel(str(model_path))
                feature_computer = EnsembleFeatureComputer(model_name, device=args.device)
                print(f"  {ENSEMBLE_FEATURE_CONFIGS[model_name]['description']}")
            else:
                print(f"  ERROR: Unknown model: {model_name}")
                continue
            
            # Run benchmark
            runner = BenchmarkRunner(
                model_name=model_name,
                model_instance=model,
                batch_size=args.batch_size,
                warmup_samples=args.warmup_samples,
                num_samples=args.num_samples,
                device=args.device,
                is_image_model=is_image,
                feature_computer=feature_computer,
            )
            
            results = runner.run(df)
            all_results.append(results)
            
            # Print summary
            print(f"  Latency (avg):     {results['latency_avg_ms']:.4f} ms")
            print(f"  Latency (min/max): {results['latency_min_ms']:.4f} / {results['latency_max_ms']:.4f} ms")
            print(f"  Throughput:        {results['throughput_pairs_per_sec']:.2f} pairs/sec")
            print(f"  CPU (avg):         {results['cpu_avg_percent']:.1f}%")
            if results['gpu_memory_max_gb'] > 0:
                print(f"  GPU Memory (max):  {results['gpu_memory_max_gb']:.2f} GB")
            print()
        
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            print()
            continue
    
    # Save all results to CSV
    if all_results:
        results_df = pd.DataFrame(all_results)
        print(f"Saving results to {args.output_csv}...")
        results_df.to_csv(args.output_csv, index=False)
        
        # Print summary table
        print("\n" + "="*70)
        print("SUMMARY TABLE")
        print("="*70)
        print(results_df[['model', 'latency_avg_ms', 'latency_std_ms', 'throughput_pairs_per_sec', 'cpu_avg_percent']].to_string(index=False))
        print("="*70)
    else:
        print("No results to save.")


if __name__ == "__main__":
    main()
