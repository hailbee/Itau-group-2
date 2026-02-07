"""AttentionCNN Tester - tests attentioncnn on image-based spoof detection"""

from tensorflow import keras
from scripts.evaluation.attentioncnn_evaluator import AttentionCNNEvaluator

class AttentionCNNTester:
    def __init__(
        self,
        model_path: str
    ):
        self.model = keras.models.load_model(
            model_path,
            compile=False,
            safe_mode=False
        )

        self.evaluator = AttentionCNNEvaluator(
            model=self.model
        )

    def test(self, test_filepath: str):
        y_true, y_scores = self.evaluator.test_pairs(test_filepath)
        metrics = self.evaluator.compute_metrics(y_true, y_scores)
        return metrics
