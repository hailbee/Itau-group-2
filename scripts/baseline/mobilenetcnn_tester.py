"""
MobileNet CNN Tester

Thin wrapper that runs evaluation using MobileNetCNNEvaluator.
"""

import pandas as pd
from scripts.evaluation.mobilenetcnn_evaluator import MobileNetEvaluator
import timm

class MobileNetCNNTester:
    """
    Tester for MobileNet CNN–based glyph similarity model.
    """

    def __init__(
        self,
        batch_size=32,
        image_size=(224, 224)
    ):
        """
        Args:
            keras_model: Keras MobileNet model (outputs embeddings)
            batch_size: Batch size for inference
            image_size: Input image size for MobileNet
        """
        self.model = timm.create_model("hf_hub:timm/mobilenetv4_hybrid_large.ix_e600_r384_in1k", pretrained=True)
        self.evaluator = MobileNetEvaluator(
            model=self.model,
            batch_size=batch_size,
            image_size=image_size
        )

    def test(self, filepath):
        """
        Run full evaluation on test file.

        Args:
            test_filepath: CSV or Parquet with
                - fraudulent_name
                - real_name
                - label
            plot: Whether to plot ROC / confusion matrices

        Returns:
            dict: metrics
        """
        print('testing...')
        y_true, y_scores = self.predict_scores(filepath)
        metrics = self.evaluator.compute_metrics(y_true, y_scores)
        return metrics

    def predict_scores(self, filepath):
        """
        Predict cosine similarity scores only (no metrics).

        Args:
            filepath: CSV or Parquet

        Returns:
            pd.DataFrame with similarity scores
        """
        y_true, y_scores = self.evaluator.predict_scores(filepath)
        return y_true, y_scores
