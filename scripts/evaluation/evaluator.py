import torch
import pandas as pd
import torch.nn.functional as F
import pandas as pd
from sklearn.metrics import roc_curve, precision_score, recall_score, accuracy_score, roc_auc_score
from utils.evals import find_best_threshold_youden, plot_roc_curve, plot_confusion_matrix, find_best_threshold_accuracy
from utils.embeddings import EmbeddingExtractor, SupConEmbeddingExtractor, batched_embedding
import numpy as np
from sklearn.metrics import auc

class Evaluator:
    """
    Unified evaluation interface for model testing and metrics computation (pairwise only).
    """
    def __init__(self, model, batch_size=32, model_type=None):
        self.model = model
        self.batch_size = batch_size
        self.model_type = model_type
        # Only use embedding extractor
        if model_type in ['supcon', 'infonce']:
            print("USING SUPCON EMBEDDING EXTRACTOR")
            self.extractor = SupConEmbeddingExtractor(model)
        else:
            print("USING STANDARD EMBEDDING EXTRACTOR")
            self.extractor = EmbeddingExtractor(model)

    def compute_metrics(self, results_df, plot=False):
        """
        Compute evaluation metrics from results.
        Optionally plot ROC curve and confusion matrix.
        Args:
            results_df: DataFrame with test results
            plot (bool): If True, plot ROC curve and confusion matrices. If False, do not plot anything.
        Returns:
            dict: Dictionary of metrics
        """
        y_true = results_df['label']
        y_scores = results_df['max_similarity']
        
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
        
        # Plot if requested (using already computed values)
        if plot:
            plot_roc_curve(results_df)  # This will recompute ROC for plotting, but that's OK for visualization
            print(f"Plotting confusion matrix at Youden's threshold: {youden_thresh:.3f}")
            plot_confusion_matrix(y_true, y_scores, youden_thresh)
            print(f"Best Accuracy: {best_acc:.4f} at Threshold: {best_acc_threshold:.3f}")
            plot_confusion_matrix(y_true, y_scores, best_acc_threshold)
        
        return metrics

    def evaluate(self, test_filepath, embeddings_flag=True,plot=False):
        """
        Evaluate model on a file of (fraudulent_name, real_name, label) pairs.
        Args:
            test_filepath: Path to test data (CSV or PARQUET with fraudulent_name, real_name, label)
            plot (bool): Whether to plot ROC/confusion matrix
        Returns:
            tuple: (results_df, metrics)
        """
        if embeddings_flag:
            self.create_embeddings(test_filepath)
        return self.test_pairs(test_filepath, plot=plot)

    def test_pairs(self, test_filepath, plot=False):
        if test_filepath.endswith('.csv'):
            df = pd.read_csv(test_filepath)
        else:
            df = pd.read_parquet(test_filepath)
        
        df = df.head(1024)
        fraud_names = df['fraudulent_name'].astype(str).tolist()
        real_names = df['real_name'].astype(str).tolist()
        labels = df['label'].astype(float).tolist()
        
        fraud_embs = batched_embedding(self.extractor, fraud_names, self.batch_size)
        real_embs = batched_embedding(self.extractor, real_names, self.batch_size)
        
        similarities = F.cosine_similarity(fraud_embs, real_embs, dim=1).detach().cpu().numpy()
        results_df = pd.DataFrame({
            'fraudulent_name': fraud_names,
            'real_name': real_names,
            'label': labels,
            'max_similarity': similarities
        })
        
        metrics = self.compute_metrics(results_df, plot=plot)
        return results_df, metrics
    
    def create_embeddings(self, test_filepath):
        """
        Before embeddings are the text embeddings from the backbone encoder
        After embeddings are the final embeddings produced by VATE training has reshaped the embedding space.
        """

        if test_filepath.endswith('.csv'):
            df = pd.read_csv(test_filepath)
        else:
            df = pd.read_parquet(test_filepath)

        fraud_names = df['fraudulent_name'].astype(str).tolist()
        real_names  = df['real_name'].astype(str).tolist()
        labels      = df['label'].astype(float).tolist()

        self.model.eval()

        before_fraud, before_real = [], []
        after_fraud, after_real = [], []

        bs = self.batch_size

        with torch.no_grad():
            for start in range(0, len(df), bs):
                end = start + bs

                fraud_batch = fraud_names[start:end]
                real_batch  = real_names[start:end]

                # BEFORE: backbone embeddings
                f_before = self.model.backbone.encode_text(fraud_batch).detach().cpu()
                r_before = self.model.backbone.encode_text(real_batch).detach().cpu()

                before_fraud.append(f_before)
                before_real.append(r_before)

                # AFTER: VA-TE embeddings
                f_after = self.model.encode(fraud_batch).detach().cpu()
                r_after = self.model.encode(real_batch).detach().cpu()

                after_fraud.append(f_after)
                after_real.append(r_after)

        before_fraud = torch.cat(before_fraud).numpy()
        before_real  = torch.cat(before_real).numpy()
        after_fraud  = torch.cat(after_fraud).numpy()
        after_real   = torch.cat(after_real).numpy()

        #before save
        before_df = pd.DataFrame({
            'fraudulent_name': fraud_names,
            'real_name': real_names,
            'label': labels
        })

        for i in range(before_fraud.shape[1]):
            before_df[f'fraud_emb_{i}'] = before_fraud[:, i]
            before_df[f'real_emb_{i}']  = before_real[:, i]

        before_df.to_csv("embeddings/embeddings_before.csv", index=False)

        #after save
        after_df = pd.DataFrame({
            'fraudulent_name': fraud_names,
            'real_name': real_names,
            'label': labels
        })

        for i in range(after_fraud.shape[1]):
            after_df[f'fraud_emb_{i}'] = after_fraud[:, i]
            after_df[f'real_emb_{i}']  = after_real[:, i]

        after_df.to_csv("embeddings/embeddings_after.csv", index=False)