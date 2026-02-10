"""
Attention CNN Evaluator - evaluates attention cnn on image-based text similarity.

Similar to the text evaluator but works with attention cnn and images.
"""

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, precision_score, recall_score, accuracy_score, roc_auc_score, auc

from utils.evals import find_best_threshold_youden, find_best_threshold_accuracy


import tensorflow as tf
from tensorflow import keras
from PIL import Image, ImageDraw, ImageFont, ImageEnhance

class AttentionCNNEvaluator:
    
    def __init__(self, model, batchsize = 32):
        self.model = model
        self.batch_size = batchsize
        self.font = ImageFont.truetype("/home/valxin/Itau-group-2/arial.ttf", 10)
        
    def text_to_img(self, text):
        # render glyph
        img = Image.new("RGB", (150, 12), (0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.text((0, 0), text, fill=(255, 255, 255), font=self.font)

        img = ImageEnhance.Contrast(img).enhance(1.5)

        # resize to model input
        img = img.resize((256, 256))
        img = np.array(img, dtype=np.float32)
        img /= 255.0

        # add batch dim
        img = np.expand_dims(img, axis=0)  # (1,256,256,3)

        return img
        
    def test_pairs(self, test_filepath, plot=False):
        
        # read data
        if test_filepath.endswith('.csv'):
            df = pd.read_csv(test_filepath)
        else:
            df = pd.read_parquet(test_filepath)
        
        df = df.head(1024)
        real = [df['real_name'][i] for i in range(len(df)) if df['label'][i] == 0.0]
        fake = [df['fraudulent_name'][i] for i in range(len(df)) if df['label'][i] == 1.0]
        
        # remove repeats
        uniq_real = list(set(real))
        uniq_fake = list(set(fake))
        
        y_true = []
        y_score = []
        
        def run_batches(texts, label, tag):
            for start in range(0, len(texts), self.batch_size):
                end = min(start + self.batch_size, len(texts))

                if start % (self.batch_size * 10) == 0:
                    print(f"Transformed {start}/{len(texts)} {tag} spoofs", flush=True)

                batch_texts = texts[start:end]

                # build image batch
                imgs = np.concatenate(
                    [self.text_to_img(t) for t in batch_texts],
                    axis=0
                )  # (B,256,256,3)

                probs = self.model.predict(imgs, verbose=0).reshape(-1)
                tf.keras.backend.clear_session()

                for text, prob in zip(batch_texts, probs):
                    y_true.append(label)
                    y_score.append(float(prob))

        # run real and fake batches
        run_batches(uniq_real, 0, "real")
        run_batches(uniq_fake, 1, "fake")

        return np.array(y_true), np.array(y_score)
        
    def compute_metrics(self, y_true, y_scores):
        """
        Compute evaluation metrics from results.
        Optionally plot ROC curve and confusion matrix.
        Args:
            results_df: DataFrame with test results
            plot (bool): If True, plot ROC curve and confusion matrices. If False, do not plot anything.
        Returns:
            dict: Dictionary of metrics
        """
        print('Computing metrics...')
        # Compute ROC curve ONCE and reuse
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        print(f"ROC AUC: {roc_auc:.4f}")
        
        # Find thresholds using the already computed ROC curve
        youden_thresh = find_best_threshold_youden(fpr, tpr, thresholds) # prints best thresh
        best_acc, best_acc_threshold = find_best_threshold_accuracy(y_true, y_scores, thresholds) # prints acc and thresh
        
        # Compute predictions
        y_pred = (y_scores > youden_thresh).astype(int)
        
        # Build metrics dict (at Youden's threshold)
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
    
        
        return metrics
        
            