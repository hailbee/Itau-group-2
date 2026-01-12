"""
OCR Tester - tests OCR encoders on glyph-based text recognition tasks.

Converts text to glyphs, extracts text via OCR, and evaluates accuracy.
Mirrors BaselineTester and ImageEncoderTester structure.
"""

import pandas as pd
import torch
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt

from model_utils.models.ocr_encoders import PytesseractOCREncoder
from model_utils.utils.text_to_glyph import text_to_glyph, text_to_glyphs_batch


class OCRTester:
    """
    Test OCR encoders on glyph-based text recognition tasks.
    
    Converts text to glyphs, extracts text via OCR with fuzzy matching,
    and evaluates recognition accuracy on test data.
    """
    
    def __init__(self, ocr_encoder=None, batch_size=32, glyph_size=(224, 224), fuzzy_threshold=80):
        """
        Initialize OCR tester.
        
        Args:
            ocr_encoder: OCR encoder instance (default: PytesseractOCREncoder)
            batch_size: Batch size for OCR processing
            glyph_size: Size of generated glyphs (width, height)
            fuzzy_threshold: Fuzzy matching threshold (0-100)
        """
        self.batch_size = batch_size
        self.glyph_size = glyph_size
        self.fuzzy_threshold = fuzzy_threshold
        
        # Create OCR encoder
        if ocr_encoder is None:
            self.ocr_encoder = PytesseractOCREncoder(fuzzy_threshold=fuzzy_threshold)
        else:
            self.ocr_encoder = ocr_encoder
    
    def test(self, test_filepath, plot_roc=False):
        """
        Test OCR encoder on test data.
        
        Args:
            test_filepath: Path to test data (CSV or Parquet)
                          Expected columns: fraudulent_name, real_name, label
            plot_roc: Whether to plot ROC curve and confusion matrix
            
        Returns:
            tuple: (results_df, metrics) - Results dataframe and evaluation metrics
        """
        # Load test data
        if test_filepath.endswith('.csv'):
            df = pd.read_csv(test_filepath)
        else:
            df = pd.read_parquet(test_filepath)
        
        df = df.head(32)
        # Generate glyphs for both texts
        print(f"Generating glyphs for {len(df)} text pairs...")
        fraudulent_glyphs = text_to_glyphs_batch(df['fraudulent_name'].tolist(), image_size=self.glyph_size)
        real_glyphs = text_to_glyphs_batch(df['real_name'].tolist(), image_size=self.glyph_size)
        
        # Extract text via OCR
        print("Extracting text via OCR...")
        fraudulent_extracted = self._batch_extract_text(fraudulent_glyphs)
        real_extracted = self._batch_extract_text(real_glyphs)
        
        # Calculate similarity using fuzzy matching
        print("Calculating text similarity via fuzzy matching...")
        similarities = []
        for faud_extracted, real_extracted in zip(fraudulent_extracted, real_extracted):
            # Use token_set_ratio for robust comparison
            from fuzzywuzzy import fuzz
            score = fuzz.token_set_ratio(faud_extracted.lower(), real_extracted.lower())
            similarities.append(score / 100.0)  # Normalize to 0-1
        
        similarities = np.array(similarities)
        labels = df['label'].values
        
        # Evaluate
        print("Evaluating OCR accuracy...")
        metrics = self._evaluate(similarities, labels, plot_roc=plot_roc)
        
        # Create results dataframe
        results_df = pd.DataFrame({
            'fraudulent_name': df['fraudulent_name'],
            'real_name': df['real_name'],
            'fraudulent_extracted': fraudulent_extracted,
            'real_extracted': real_extracted,
            'ocr_similarity': similarities,
            'label': labels,
            'predicted_label': (similarities >= metrics['threshold']).astype(int)
        })
        
        return results_df, metrics
    
    def test_all_models(self, test_filepath, thresholds=None, plot_roc=False):
        """
        Test all available OCR configurations.
        
        Currently only supports Pytesseract with different fuzzy thresholds.
        
        Args:
            test_filepath: Path to test data
            thresholds: List of threshold values to test (default: [10, 30, 50, 60, 70, 80, 90])
            plot_roc: Whether to plot ROC curves
            
        Returns:
            dict: Results keyed by model configuration name
        """
        if thresholds is None:
            thresholds = [10, 30, 50, 60, 70, 80, 90]
        results = {}
        
        for threshold in thresholds:
            try:
                print(f"\n{'='*60}")
                print(f"Testing Pytesseract OCR with fuzzy threshold={threshold}...")
                print(f"{'='*60}")
                
                # Create new tester with different threshold
                tester = OCRTester(
                    ocr_encoder=PytesseractOCREncoder(fuzzy_threshold=threshold),
                    batch_size=self.batch_size,
                    glyph_size=self.glyph_size,
                    fuzzy_threshold=threshold
                )
                
                results_df, metrics = tester.test(test_filepath, plot_roc=plot_roc)
                results[f'pytesseract_{threshold}'] = (results_df, metrics)
                
            except Exception as e:
                print(f"Error testing pytesseract with threshold {threshold}: {str(e)}")
                results[f'pytesseract_{threshold}'] = None
        
        return results
    
    def _batch_extract_text(self, images, batch_size=None):
        """
        Extract text from images in batches.
        
        Args:
            images: List of PIL Image objects
            batch_size: Batch size (uses self.batch_size if None)
            
        Returns:
            List of extracted text strings
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        extracted_texts = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            batch_extracted = self.ocr_encoder.extract_texts(batch)
            extracted_texts.extend(batch_extracted)
        
        return extracted_texts
    
    def _evaluate(self, similarities, labels, plot_roc=False):
        """
        Evaluate OCR performance.
        
        Args:
            similarities: Array of similarity scores (0-1)
            labels: Array of ground truth labels (0 or 1)
            plot_roc: Whether to plot ROC curve and confusion matrix
            
        Returns:
            dict: Evaluation metrics
        """
        # Find optimal threshold using ROC curve
        fpr, tpr, thresholds = roc_curve(labels, similarities)
        roc_auc = auc(fpr, tpr)
        
        # Find optimal threshold (Youden's J statistic)
        youden_j = tpr - fpr
        optimal_threshold = thresholds[np.argmax(youden_j)]
        
        # Predictions using optimal threshold
        predictions = (similarities >= optimal_threshold).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions, zero_division=0)
        recall = recall_score(labels, predictions, zero_division=0)
        f1 = f1_score(labels, predictions, zero_division=0)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'threshold': optimal_threshold,
            'roc_curve': (fpr, tpr)
        }
        
        if plot_roc:
            self._plot_results(similarities, labels, predictions, roc_auc, optimal_threshold, fpr, tpr)
        
        return metrics
    
    def _plot_results(self, similarities, labels, predictions, roc_auc, threshold, fpr, tpr):
        """
        Plot ROC curve and confusion matrix.
        
        Args:
            similarities: Array of similarity scores
            labels: Array of ground truth labels
            predictions: Array of predicted labels
            roc_auc: AUC score
            threshold: Optimal threshold used
            fpr: False positive rates for ROC
            tpr: True positive rates for ROC
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # ROC Curve
        axes[0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        axes[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        axes[0].set_xlim([0.0, 1.0])
        axes[0].set_ylim([0.0, 1.05])
        axes[0].set_xlabel('False Positive Rate')
        axes[0].set_ylabel('True Positive Rate')
        axes[0].set_title('OCR Text Similarity - ROC Curve')
        axes[0].legend(loc="lower right")
        axes[0].grid(alpha=0.3)
        
        # Confusion Matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(labels, predictions)
        
        im = axes[1].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        axes[1].figure.colorbar(im, ax=axes[1])
        
        # Plot confusion matrix values
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                axes[1].text(j, i, format(cm[i, j], 'd'),
                            ha="center", va="center",
                            color="white" if cm[i, j] > thresh else "black")
        
        axes[1].set_ylabel('True Label')
        axes[1].set_xlabel('Predicted Label')
        axes[1].set_title('Confusion Matrix')
        axes[1].set_xticks([0, 1])
        axes[1].set_yticks([0, 1])
        
        plt.tight_layout()
        plt.savefig('ocr_evaluation_results.png', dpi=100, bbox_inches='tight')
        print("Evaluation plots saved to 'ocr_evaluation_results.png'")
        plt.show()
