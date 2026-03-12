"""
Evaluate missing models on the test set.

This script evaluates vision-language models that don't have results yet:
- siglip_text: SigLIP text encoder
- siglip_image: SigLIP image encoder
- clip_text: CLIP text encoder
- clip_image: CLIP image encoder
- coca_text: CoCa text encoder
- coca_image: CoCa image encoder
- flava_text: FLAVA text encoder
- flava_image: FLAVA image encoder

Computes the following metrics for each model:
- ROC AUC
- Best threshold (Youden index)
- Threshold for best accuracy
- Best accuracy

And at Youden threshold:
- Accuracy
- Precision
- Recall
"""

import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.metrics import (
    roc_curve, auc, accuracy_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix
)

from latency_benchmark_utils import (
    load_test_data,
    create_mini_dataset,
    generate_glyph_image,
)


# ============================================================================
# Text Encoders (from latency_benchmark.py)
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
            return outputs.cpu().numpy().astype(np.float32)


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
    """FLAVA text encoder - uses mean pooling for consistent embedding extraction."""
    
    def _load_model(self):
        from transformers import AutoTokenizer, AutoModel, AutoProcessor
        self.model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        try:
            self.processor = AutoProcessor.from_pretrained(self.model_name)
        except Exception:
            self.processor = None
        self.model.eval()
    
    def encode_batch(self, texts: List[str]) -> np.ndarray:
        with torch.inference_mode():
            # Use tokenizer directly for consistent results
            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Forward pass - FLAVA requires text input
            outputs = self.model.get_text_features(**inputs) if hasattr(self.model, 'get_text_features') else self.model(**inputs)
            
            # Extract embeddings
            if isinstance(outputs, torch.Tensor):
                embeddings = outputs
            elif hasattr(outputs, 'text_features'):
                embeddings = outputs.text_features
            elif hasattr(outputs, 'last_hidden_state'):
                # Use mean pooling over sequence dimension
                embeddings = outputs.last_hidden_state.mean(dim=1)
            elif isinstance(outputs, tuple) and len(outputs) > 0:
                embeddings = outputs[0]
                if isinstance(embeddings, torch.Tensor):
                    if embeddings.dim() == 3:
                        # Mean pooling over sequence dimension
                        embeddings = embeddings.mean(dim=1)
            else:
                embeddings = outputs
            
            # Ensure tensor and consistent shape
            if not isinstance(embeddings, torch.Tensor):
                embeddings = torch.from_numpy(embeddings).to(self.device)
            
            # Apply mean pooling if 3D to get consistent 2D output
            if embeddings.dim() == 3:
                embeddings = embeddings.mean(dim=1)
            elif embeddings.dim() == 1:
                embeddings = embeddings.unsqueeze(0)
            
            return embeddings.cpu().numpy().astype(np.float32)


# ============================================================================
# Image Encoders
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
    """FLAVA image encoder - uses mean pooling for consistent embedding extraction."""
    
    def _load_model(self):
        from transformers import AutoImageProcessor, AutoModel
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True).to(self.device)
        self.model.eval()
    
    def encode_batch(self, images: List) -> np.ndarray:
        with torch.inference_mode():
            inputs = self.processor(images=images, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Forward pass
            outputs = self.model.get_image_features(**inputs) if hasattr(self.model, 'get_image_features') else self.model(**inputs)
            
            # Extract embeddings
            if isinstance(outputs, torch.Tensor):
                embeddings = outputs
            elif hasattr(outputs, 'image_features'):
                embeddings = outputs.image_features
            elif hasattr(outputs, 'last_hidden_state'):
                # Use mean pooling over sequence dimension
                embeddings = outputs.last_hidden_state.mean(dim=1)
            elif hasattr(outputs, 'pooler_output'):
                embeddings = outputs.pooler_output
            elif isinstance(outputs, tuple) and len(outputs) > 0:
                embeddings = outputs[0]
                if isinstance(embeddings, torch.Tensor):
                    if embeddings.dim() == 3:
                        # Mean pooling over sequence dimension
                        embeddings = embeddings.mean(dim=1)
            else:
                embeddings = outputs
            
            # Ensure tensor and consistent shape
            if not isinstance(embeddings, torch.Tensor):
                embeddings = torch.from_numpy(embeddings).to(self.device)
            
            # Apply mean pooling if 3D to get consistent 2D output
            if embeddings.dim() == 3:
                embeddings = embeddings.mean(dim=1)
            elif embeddings.dim() == 1:
                embeddings = embeddings.unsqueeze(0)
            
            return embeddings.cpu().numpy().astype(np.float32)


# ============================================================================
# Cosine Similarity Computation
# ============================================================================

def compute_cosine_similarities(embeddings_1: np.ndarray, embeddings_2: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between pairs of embeddings.
    
    Args:
        embeddings_1: Shape (N, D)
        embeddings_2: Shape (N, D)
    
    Returns:
        Similarity scores, shape (N,)
    """
    # Normalize embeddings
    norms_1 = np.linalg.norm(embeddings_1, axis=1, keepdims=True)
    norms_2 = np.linalg.norm(embeddings_2, axis=1, keepdims=True)
    
    embeddings_1_norm = embeddings_1 / (norms_1 + 1e-8)
    embeddings_2_norm = embeddings_2 / (norms_2 + 1e-8)
    
    # Compute cosine similarity
    similarity = np.sum(embeddings_1_norm * embeddings_2_norm, axis=1)
    return similarity


# ============================================================================
# Evaluation Utilities
# ============================================================================

def compute_metrics_at_threshold(y_true: np.ndarray, y_scores: np.ndarray, threshold: float) -> Dict:
    """Compute metrics at a specific threshold."""
    y_pred = (y_scores >= threshold).astype(int)
    
    return {
        'threshold': threshold,
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
    }


def find_youden_threshold(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[float, Dict]:
    """
    Find the threshold that maximizes the Youden index (J = TPR + TNR - 1).
    
    Returns:
        (threshold, metrics_dict)
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    
    # Youden index = TPR - FPR = TPR + TNR - 1
    youden = tpr - fpr
    best_idx = np.argmax(youden)
    best_threshold = thresholds[best_idx]
    
    metrics = compute_metrics_at_threshold(y_true, y_scores, best_threshold)
    return best_threshold, metrics


def find_best_accuracy_threshold(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[float, Dict]:
    """Find the threshold that gives the best accuracy."""
    # Test a range of thresholds
    thresholds_to_test = np.arange(0, 1.01, 0.01)
    best_accuracy = -1
    best_threshold = 0.5
    best_metrics = None
    
    for threshold in thresholds_to_test:
        metrics = compute_metrics_at_threshold(y_true, y_scores, threshold)
        if metrics['accuracy'] > best_accuracy:
            best_accuracy = metrics['accuracy']
            best_threshold = threshold
            best_metrics = metrics
    
    return best_threshold, best_metrics


def evaluate_model(
    model_name: str,
    encoder,
    df: pd.DataFrame,
    is_image: bool = False,
    batch_size: int = 32,
) -> Dict:
    """
    Evaluate a single model.
    
    Args:
        model_name: Name of the model
        encoder: Encoder instance (text or image)
        df: Test DataFrame with 'fraudulent_name', 'real_name', 'label'
        is_image: Whether to render text as images first
        batch_size: Batch size for encoding
    
    Returns:
        Dictionary with evaluation results
    """
    # For image models, use a smaller sample to avoid OOM
    if is_image and len(df) > 50000:
        print(f"  Warning: Large dataset detected. Using 50000 samples for image model to avoid OOM.")
        df = df.sample(n=50000, random_state=42).reset_index(drop=True)
    
    fraud_names = df['fraudulent_name'].tolist()
    real_names = df['real_name'].tolist()
    y_true = df['label'].astype(int).values
    
    print(f"\n{'='*70}")
    print(f"Evaluating: {model_name.upper()}")
    print(f"{'='*70}")
    print(f"  Number of samples: {len(df)}")
    print(f"  Positive (spoof) samples: {np.sum(y_true)}")
    print(f"  Negative (non-spoof) samples: {len(y_true) - np.sum(y_true)}")
    
    # Prepare data
    if is_image:
        print(f"  Rendering text to images...")
        fraud_data = [generate_glyph_image(name) for name in fraud_names]
        real_data = [generate_glyph_image(name) for name in real_names]
    else:
        fraud_data = fraud_names
        real_data = real_names
    
    # Encode in batches
    print(f"  Encoding fraud names...")
    embeddings_fraud = []
    for i in range(0, len(fraud_data), batch_size):
        batch = fraud_data[i:i+batch_size]
        try:
            batch_emb = encoder.encode_batch(batch)
            # Debug: print shape
            if i == 0:
                print(f"    Batch 0 shape: {batch_emb.shape}")
            # Ensure consistent shape
            if batch_emb.ndim == 3 and batch_emb.shape[1] == 1:
                batch_emb = batch_emb.squeeze(1)
            elif batch_emb.ndim != 2:
                raise ValueError(f"Expected 2D embeddings, got shape {batch_emb.shape}")
            embeddings_fraud.append(batch_emb)
        except Exception as e:
            print(f"    ERROR in batch {i}: {e}")
            raise
    
    print(f"    Combining {len(embeddings_fraud)} batches...")
    try:
        embeddings_fraud = np.vstack(embeddings_fraud)
    except ValueError as e:
        print(f"    ERROR during vstack:")
        for j, emb in enumerate(embeddings_fraud):
            print(f"      Batch {j} shape: {emb.shape}")
        raise
    
    print(f"  Encoding real names...")
    embeddings_real = []
    for i in range(0, len(real_data), batch_size):
        batch = real_data[i:i+batch_size]
        try:
            batch_emb = encoder.encode_batch(batch)
            # Ensure consistent shape
            if batch_emb.ndim == 3 and batch_emb.shape[1] == 1:
                batch_emb = batch_emb.squeeze(1)
            elif batch_emb.ndim != 2:
                raise ValueError(f"Expected 2D embeddings, got shape {batch_emb.shape}")
            embeddings_real.append(batch_emb)
        except Exception as e:
            print(f"    ERROR in batch {i}: {e}")
            raise
    
    print(f"    Combining {len(embeddings_real)} batches...")
    try:
        embeddings_real = np.vstack(embeddings_real)
    except ValueError as e:
        print(f"    ERROR during vstack:")
        for j, emb in enumerate(embeddings_real):
            print(f"      Batch {j} shape: {emb.shape}")
        raise
    
    # Compute cosine similarities
    print(f"  Computing cosine similarities...")
    y_scores = compute_cosine_similarities(embeddings_fraud, embeddings_real)
    
    # Compute ROC AUC
    roc_auc = roc_auc_score(y_true, y_scores)
    print(f"  ROC AUC: {roc_auc:.4f}")
    
    # Find Youden threshold
    youden_threshold, youden_metrics = find_youden_threshold(y_true, y_scores)
    print(f"\n  Youden Threshold: {youden_threshold:.4f}")
    print(f"    Accuracy:  {youden_metrics['accuracy']:.4f}")
    print(f"    Precision: {youden_metrics['precision']:.4f}")
    print(f"    Recall:    {youden_metrics['recall']:.4f}")
    
    # Find best accuracy threshold
    best_acc_threshold, best_acc_metrics = find_best_accuracy_threshold(y_true, y_scores)
    print(f"\n  Best Accuracy Threshold: {best_acc_threshold:.4f}")
    print(f"    Accuracy:  {best_acc_metrics['accuracy']:.4f}")
    
    return {
        'model': model_name,
        'num_samples': len(df),
        'num_spoofs': int(np.sum(y_true)),
        'num_non_spoofs': int(len(y_true) - np.sum(y_true)),
        'roc_auc': roc_auc,
        'youden_threshold': youden_threshold,
        'youden_accuracy': youden_metrics['accuracy'],
        'youden_precision': youden_metrics['precision'],
        'youden_recall': youden_metrics['recall'],
        'best_acc_threshold': best_acc_threshold,
        'best_accuracy': best_acc_metrics['accuracy'],
    }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate missing models on test set"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Specific model to evaluate (siglip_text, siglip_image, clip_text, clip_image, coca_text, coca_image, flava_text, flava_image)"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="test_pairs_all.parquet",
        help="Path to test data"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to use (if None, use all)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for encoding"
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="missing_models_evaluation.csv",
        help="Output CSV file"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device: cuda, cpu"
    )
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading test data from {args.data_path}...")
    if Path(args.data_path).exists():
        df = load_test_data(args.data_path, args.num_samples)
    else:
        print(f"Data file not found, using synthetic data")
        df = create_mini_dataset(args.num_samples or 100)
    
    print(f"Loaded {len(df)} samples\n")
    
    # Define missing models
    missing_models = {
        'siglip_text': {
            'encoder_class': SigLIPTextEncoder,
            'model_name': 'google/siglip-base-patch16-224',
            'is_image': False,
        },
        'siglip_image': {
            'encoder_class': SigLIPImageEncoder,
            'model_name': 'google/siglip-base-patch16-224',
            'is_image': True,
        },
        'clip_text': {
            'encoder_class': CLIPTextEncoder,
            'model_name': 'openai/clip-vit-base-patch32',
            'is_image': False,
        },
        'clip_image': {
            'encoder_class': CLIPImageEncoder,
            'model_name': 'openai/clip-vit-base-patch32',
            'is_image': True,
        },
        'coca_text': {
            'encoder_class': CoCaTextEncoder,
            'model_name': 'microsoft/git-base-coco',
            'is_image': False,
        },
        'coca_image': {
            'encoder_class': CoCaImageEncoder,
            'model_name': 'microsoft/git-base-coco',
            'is_image': True,
        },
        'flava_text': {
            'encoder_class': FLAVATextEncoder,
            'model_name': 'facebook/flava-full',
            'is_image': False,
        },
        'flava_image': {
            'encoder_class': FLAVAImageEncoder,
            'model_name': 'facebook/flava-full',
            'is_image': True,
        },
    }
    
    # Determine which models to evaluate
    if args.model:
        models_to_eval = {args.model: missing_models[args.model]}
    else:
        models_to_eval = missing_models
    
    all_results = []
    
    for model_name, config in models_to_eval.items():
        try:
            print(f"\nInitializing {model_name}...")
            encoder = config['encoder_class'](config['model_name'])
            if args.device:
                encoder.to_device(args.device)
            
            results = evaluate_model(
                model_name=model_name,
                encoder=encoder,
                df=df,
                is_image=config['is_image'],
                batch_size=args.batch_size,
            )
            all_results.append(results)
            
            # Clean up to save memory
            del encoder
            torch.cuda.empty_cache()
        
        except Exception as e:
            print(f"\nERROR evaluating {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save results
    if all_results:
        results_df = pd.DataFrame(all_results)
        print(f"\n{'='*70}")
        print(f"Saving results to {args.output_csv}...")
        results_df.to_csv(args.output_csv, index=False)
        
        # Print summary table
        print(f"\n{'='*70}")
        print("SUMMARY TABLE")
        print(f"{'='*70}")
        print(results_df[[
            'model', 'roc_auc', 'youden_threshold', 'youden_accuracy',
            'youden_precision', 'youden_recall', 'best_acc_threshold', 'best_accuracy'
        ]].to_string(index=False))
        print(f"{'='*70}")
    else:
        print("No results to save.")


if __name__ == "__main__":
    main()
