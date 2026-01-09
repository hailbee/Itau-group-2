"""
Image Encoder Evaluator - evaluates image encoders on glyph-based text similarity.

Similar to the text evaluator but works with image encoders and glyphs.
"""

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, precision_score, recall_score, accuracy_score, roc_auc_score, auc

from model_utils.utils.text_to_glyph import text_to_glyphs_batch
from utils.evals import find_best_threshold_youden, plot_roc_curve, plot_confusion_matrix, find_best_threshold_accuracy


class ImageEncoderEvaluator:
    """
    Unified evaluation interface for image encoders on glyph-based text similarity.
    
    Converts text pairs to glyphs, encodes them, and evaluates similarity.
    """
    
    def __init__(self, image_encoder, batch_size=32, glyph_size=(224, 224)):
        """
        Initialize image encoder evaluator.
        
        Args:
            image_encoder: Image encoder instance with encode_images() method
            batch_size: Batch size for processing
            glyph_size: Size of generated glyphs (width, height)
        """
        self.image_encoder = image_encoder
        self.batch_size = batch_size
        self.glyph_size = glyph_size
    
    def encode_texts_to_embeddings(self, texts):
        """
        Convert texts to glyphs and encode them.
        
        Args:
            texts: List of text strings
            
        Returns:
            torch.Tensor: Embeddings of shape (len(texts), embedding_dim)
        """
        # Convert texts to glyphs
        glyphs = text_to_glyphs_batch(texts, image_size=self.glyph_size)
        
        # Process in batches
        all_embeddings = []
        for i in range(0, len(glyphs), self.batch_size):
            batch_glyphs = glyphs[i:i + self.batch_size]
            batch_embeddings = self.image_encoder.encode_images(batch_glyphs)
            all_embeddings.append(batch_embeddings)
        
        # Concatenate all embeddings
        embeddings = torch.cat(all_embeddings, dim=0)
        return embeddings
    
    def compute_metrics(self, results_df, plot=False):
        """
        Compute evaluation metrics from results.
        
        Args:
            results_df: DataFrame with columns: fraudulent_name, real_name, label, similarity
            plot: Whether to plot ROC curve and confusion matrices
            
        Returns:
            dict: Dictionary of metrics
        """
        y_true = results_df['label']
        y_scores = results_df['similarity']
        
        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        print(f"ROC AUC: {roc_auc:.4f}")
        
        # Find thresholds
        youden_thresh = find_best_threshold_youden(fpr, tpr, thresholds)
        best_acc, best_acc_threshold = find_best_threshold_accuracy(y_true, y_scores, thresholds)
        
        # Compute predictions at Youden threshold
        y_pred = (y_scores > youden_thresh).astype(int)
        
        # Build metrics dict
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'threshold': youden_thresh,
            'roc_curve': (fpr, tpr, thresholds),
            'roc_auc': roc_auc,
            'best_accuracy': best_acc,
            'best_accuracy_threshold': best_acc_threshold
        }
        
        # Plot if requested
        if plot:
            print(f"Plotting confusion matrix at Youden's threshold: {youden_thresh:.3f}")
            plot_confusion_matrix(y_true, y_scores, youden_thresh)
            print(f"Best Accuracy: {best_acc:.4f} at Threshold: {best_acc_threshold:.3f}")
            plot_confusion_matrix(y_true, y_scores, best_acc_threshold)
        
        return metrics
    
    def evaluate(self, test_filepath, plot=False):
        """
        Evaluate image encoder on a file of text pairs.
        
        Args:
            test_filepath: Path to test data (CSV or Parquet with fraudulent_name, real_name, label)
            plot: Whether to plot ROC and confusion matrices
            
        Returns:
            tuple: (results_df, metrics)
        """
        return self.test_pairs(test_filepath, plot=plot)
    
    def test_pairs(self, test_filepath, plot=False):
        """
        Test image encoder on text pairs by converting to glyphs.
        
        Args:
            test_filepath: Path to test data
            plot: Whether to plot results
            
        Returns:
            tuple: (results_df, metrics)
        """
        # Load data
        if test_filepath.endswith('.csv'):
            df = pd.read_csv(test_filepath)
        else:
            df = pd.read_parquet(test_filepath)
        
        # take only first 32
        df = df.head(32)
        fraud_names = df['fraudulent_name'].astype(str).tolist()
        real_names = df['real_name'].astype(str).tolist()
        labels = df['label'].astype(float).tolist()
        
        print(f"Encoding {len(fraud_names)} fraudulent names...")
        fraud_embs = self.encode_texts_to_embeddings(fraud_names)
        
        print(f"Encoding {len(real_names)} real names...")
        real_embs = self.encode_texts_to_embeddings(real_names)
        
        print(f"Computing similarities...")
        similarities = F.cosine_similarity(fraud_embs, real_embs, dim=1).detach().cpu().numpy()
        
        # Ensure similarities is 1D
        if similarities.ndim > 1:
            similarities = similarities.flatten()
        
        # Ensure labels is a list or array
        labels = np.array(labels).flatten().tolist()
        
        print(f"fraud_names: {len(fraud_names)}, real_names: {len(real_names)}, labels: {len(labels)}, similarities: {similarities.shape}")
        
        results_df = pd.DataFrame({
            'fraudulent_name': fraud_names,
            'real_name': real_names,
            'label': labels,
            'similarity': similarities
        })
        
        metrics = self.compute_metrics(results_df, plot=plot)
        return results_df, metrics
