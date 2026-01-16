import pandas as pd
import torch
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
from fuzzywuzzy import fuzz
import Levenshtein

class StringMethodTester:
    """_summary_
    """
    def __init__(self, type):
        self.type = type
        if self.type == 'token-set':
            self.is_dist = False
        elif self.type == 'levenshtein':
            self.is_dist = True
        else:
            raise ValueError(f"Unknown string method type: {self.type}")
        
    def test(self, test_filepath, plot_roc = False):
        
        # load test data
        df = pd.read_parquet(test_filepath)
        pairs = list(df.itertuples(index=False, name=None))
        
        y_scores = []
        y_true = []
        
        # similarity scoring

        if self.type == 'token-set':
            print('Calculating token set ratio similarity...')
            for fraud, real, label in pairs:
                ratio = fuzz.token_set_ratio(fraud, real) / 100.0
                y_scores.append(ratio)
                y_true.append(label)
                    
        elif self.type == 'levenshtein':
            print('Calculating levenshtein distances...')
            for fraud, real, score in pairs:
                dist = Levenshtein.distance(fraud, real)
                max_len = max(len(fraud), len(real))
                sim = 1.0 - (dist / max_len if max_len > 0 else 0.0)
                y_scores.append(sim)
                y_true.append(score)
                
        else:
            raise ValueError(f"Unknown method type: {self.type}")
        
        y_scores = np.asarray(y_scores)
        y_true = np.asarray(y_true)
        
        return self._evaluate(similarities=y_scores, labels=y_true, plot_roc=plot_roc)
        
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
        
        # # Predictions using optimal threshold
        predictions = (similarities >= optimal_threshold).astype(int)

        # Calculate metrics
        accuracy = accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions, zero_division=0)
        recall = recall_score(labels, predictions, zero_division=0)
        f1 = f1_score(labels, predictions, zero_division=0)
        
        metrics = {
            'threshold': optimal_threshold,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'roc_curve': (fpr, tpr)
        }
        
        if plot_roc:
            self._plot_results(similarities, labels, predictions, roc_auc, optimal_threshold, fpr, tpr)
        
        return metrics
        
    
    
        

        